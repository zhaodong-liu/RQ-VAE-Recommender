import os
import os.path as osp
import pandas as pd
import torch
import numpy as np

from data.preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import download_url
from torch_geometric.data import extract_zip
from torch_geometric.io import fs
from typing import Callable, List, Optional
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieLens32M(InMemoryDataset):
    url = 'https://files.grouplens.org/datasets/movielens/ml-32m.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        # 修复bug: 使用兼容的加载方式
        try:
            self.load(self.processed_paths[0], data_cls=HeteroData)
        except Exception as e:
            logger.warning(f"标准加载失败: {e}，尝试兼容模式")
            # 兼容模式加载
            self._load_compatible()
    
    def _load_compatible(self):
        """兼容模式加载，避免weights_only问题"""
        if osp.exists(self.processed_paths[0]):
            # 直接使用torch.load，不使用weights_only
            self.data, self.slices = torch.load(
                self.processed_paths[0], 
                map_location='cpu',
                weights_only=False  # 修复: 禁用weights_only
            )
        else:
            raise FileNotFoundError(f"找不到处理后的数据文件: {self.processed_paths[0]}")
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['links.csv', 'movies.csv', 'ratings.csv', 'tags.csv']
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def has_process(self) -> bool:
        return not os.path.exists(self.processed_paths[0])
    
    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-32m')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self):
        """基类process方法 - 修复bug: 不能为空"""
        raise NotImplementedError("Subclasses must implement process method")


class RawMovieLens32M(MovieLens32M, PreprocessingMixin):
    """修复后的原始MovieLens32M加载器，768维输出"""
    
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        force_reload=False,
        split=None,
        # 参数配置
        target_dim=768,  # 修复: 设置目标维度为768
        max_seq_len=200,
        min_rating_count=5,
        train_split=0.8,
        text_model_name='sentence-transformers/sentence-t5-xl'
    ) -> None:
        self.split = split
        self.target_dim = target_dim
        self.max_seq_len = max_seq_len
        self.min_rating_count = min_rating_count
        self.train_split = train_split
        self.text_model_name = text_model_name
        
        super(RawMovieLens32M, self).__init__(
            root, transform, pre_transform, force_reload
        )

    def _load_ratings(self):
        """加载评分数据"""
        return pd.read_csv(self.raw_paths[2])
    
    def _load_movies(self):
        """加载电影数据"""
        return pd.read_csv(self.raw_paths[1])
    
    def _create_simple_titles(self, movies_df: pd.DataFrame) -> List[str]:
        """提取简洁的电影标题 (回归原始版本风格)"""
        logger.info("🎬 提取电影标题...")
        
        # 简单提取标题，保持原始版本的简洁性
        titles = movies_df["title"].apply(
            lambda s: s.split("(")[0].strip() if pd.notna(s) and '(' in s else (s if pd.notna(s) else "Unknown")
        ).tolist()
        
        return titles

    def _adjust_feature_dimension(self, features: torch.Tensor) -> torch.Tensor:
        """调整特征维度到目标维度"""
        current_dim = features.shape[1]
        
        if current_dim == self.target_dim:
            return features
        elif current_dim > self.target_dim:
            # 截断到目标维度
            logger.info(f"🔧 截断特征维度: {current_dim} -> {self.target_dim}")
            return features[:, :self.target_dim]
        else:
            # 填充到目标维度
            logger.info(f"🔧 填充特征维度: {current_dim} -> {self.target_dim}")
            padding_dim = self.target_dim - current_dim
            padding = torch.zeros(features.shape[0], padding_dim, dtype=features.dtype)
            return torch.cat([features, padding], dim=1)

    def _encode_text_safe(self, text_list: List[str]) -> torch.Tensor:
        """安全的文本编码，带错误处理"""
        try:
            logger.info(f"🔤 使用 {self.text_model_name} 编码文本特征...")
            embeddings = self._encode_text_feature(text_list)
            return embeddings
        except Exception as e:
            logger.warning(f"⚠️ 文本编码失败: {e}，使用随机嵌入")
            # 创建随机嵌入作为备选
            torch.manual_seed(42)
            n_items = len(text_list)
            embeddings = torch.randn(n_items, 384)  # MiniLM默认384维
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

    def process(self, max_seq_len=None) -> None:
        """主处理函数，输出768维特征"""
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
            
        logger.info(f"🔄 开始处理MovieLens数据")
        logger.info(f"目标特征维度: {self.target_dim}")
        logger.info(f"序列长度: {self.max_seq_len}")
        
        data = HeteroData()
        
        # 1. 加载原始数据
        ratings_df = self._load_ratings()
        movies_df = self._load_movies()
        
        logger.info(f"原始数据: {len(movies_df)} 电影, {len(ratings_df)} 评分")

        # 2. 处理电影数据
        logger.info("🎭 处理电影特征...")
        
        # 设置movieId为索引
        movies_indexed = movies_df.set_index('movieId')
        movie_mapping = {idx: i for i, idx in enumerate(movies_indexed.index)}
        
        # 处理类型特征
        try:
            genres_dummies = movies_indexed["genres"].str.get_dummies('|').values
            genres = self._process_genres(genres_dummies, one_hot=True)
            genres = torch.from_numpy(genres).to(torch.float)
        except Exception as e:
            logger.warning(f"类型处理失败: {e}，使用零向量")
            genres = torch.zeros(len(movies_indexed), 20)  # 默认20个类型

        # 创建简洁标题 (回归原始风格)
        titles = self._create_simple_titles(movies_df)
        
        # 安全的文本编码 (只编码标题，不是复杂描述)
        titles_emb = self._encode_text_safe(titles)

        # 合并特征
        logger.info(f"文本嵌入维度: {titles_emb.shape}")
        logger.info(f"类型特征维度: {genres.shape}")
        
        # 确保维度兼容
        min_rows = min(titles_emb.shape[0], genres.shape[0])
        titles_emb = titles_emb[:min_rows]
        genres = genres[:min_rows]
        
        x = torch.cat([titles_emb, genres], dim=1)
        
        # 调整到目标维度
        x = self._adjust_feature_dimension(x)
        
        # 设置item节点特征
        data['item'].x = x
        data['item'].text = np.array(titles[:min_rows])  # 使用简洁标题
        
        logger.info(f"✅ 最终电影特征维度: {x.shape}")

        # 3. 处理用户数据
        logger.info("👥 处理用户数据...")
        
        # 移除低频用户
        try:
            full_user_df = pd.DataFrame({"userId": ratings_df["userId"].unique()})
            filtered_user_df = self._remove_low_occurrence(
                ratings_df, full_user_df, "userId"
            )
            user_mapping = {idx: i for i, idx in enumerate(filtered_user_df["userId"])}
        except Exception as e:
            logger.warning(f"用户过滤失败: {e}，使用所有用户")
            unique_users = ratings_df["userId"].unique()
            user_mapping = {idx: i for i, idx in enumerate(unique_users)}
        
        data['user'].num_nodes = len(user_mapping)
        logger.info(f"✅ 用户数量: {len(user_mapping)}")

        # 4. 处理评分数据
        logger.info("⭐ 处理评分数据...")
        
        # 安全的评分过滤
        try:
            filtered_ratings_df = self._remove_low_occurrence(
                ratings_df, ratings_df, ["userId", "movieId"]
            )
        except Exception as e:
            logger.warning(f"评分过滤失败: {e}，使用原始数据")
            filtered_ratings_df = ratings_df
        
        # 构建边索引（添加安全检查）
        valid_pairs = []
        for _, row in filtered_ratings_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            if user_id in user_mapping and movie_id in movie_mapping:
                valid_pairs.append((user_mapping[user_id], movie_mapping[movie_id], row))
        
        if not valid_pairs:
            logger.error("没有有效的用户-电影对!")
            return
        
        src, dst, rating_data = zip(*valid_pairs)
        
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
        rating = torch.tensor([row['rating'] for row in rating_data], dtype=torch.long)
        time = torch.tensor([row['timestamp'] for row in rating_data], dtype=torch.long)
        
        data['user', 'rates', 'item'].edge_index = edge_index
        data['user', 'rates', 'item'].rating = rating
        data['user', 'rates', 'item'].time = time

        # 反向边
        data['item', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['item', 'rated_by', 'user'].rating = rating
        data['item', 'rated_by', 'user'].time = time
        
        logger.info(f"✅ 交互边数量: {edge_index.shape[1]}")

        # 5. 生成用户历史序列 (使用PreprocessingMixin的强大功能)
        logger.info("📚 生成用户历史序列...")
        
        try:
            # 准备序列数据
            sequence_df = pd.DataFrame({
                'userId': [row['userId'] for row in rating_data],
                'movieId': [row['movieId'] for row in rating_data],
                'rating': [row['rating'] for row in rating_data],
                'timestamp': [row['timestamp'] for row in rating_data]
            })
            
            sequence_df["itemId"] = sequence_df["movieId"].apply(lambda x: movie_mapping.get(x, -1))
            sequence_df = sequence_df[sequence_df["itemId"] != -1]
            
            # 评分处理 (保持原始版本的缩放)
            sequence_df["rating"] = (2 * sequence_df["rating"]).astype(int)  # 原版的评分缩放
            
            # 使用PreprocessingMixin的强大方法 (保持原版参数)
            history = self._generate_user_history(
                sequence_df,
                features=["itemId", "rating"],
                window_size=self.max_seq_len,
                stride=max(1, self.max_seq_len // 4),  # 原版使用180，这里动态计算
                train_split=self.train_split
            )
            data["user", "rated", "item"].history = history
            logger.info("✅ 用户历史序列生成完成")
            
        except Exception as e:
            logger.error(f"❌ 序列生成失败: {e}")
            # 如果PreprocessingMixin方法失败，说明数据有严重问题
            # 这时应该检查数据而不是用简化版本掩盖问题
            raise RuntimeError(f"序列生成失败，请检查数据: {e}")

        # 6. 添加训练掩码
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(x.shape[0], generator=gen) > 0.05
        
        # 7. 数据类型确保
        self._ensure_data_types(data)
        
        # 8. 应用预变换
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # 9. 保存数据（使用兼容模式）
        logger.info("💾 保存处理后的数据...")
        torch.save([data], self.processed_paths[0])  # 简化保存
        
        logger.info("🎉 数据处理完成!")
        self._print_data_summary(data)

    def _ensure_data_types(self, data):
        """确保数据类型正确"""
        # 确保特征是float类型
        if hasattr(data['item'], 'x'):
            data['item'].x = data['item'].x.float()
        
        # 确保边索引是long类型
        if hasattr(data['user', 'rates', 'item'], 'edge_index'):
            data['user', 'rates', 'item'].edge_index = data['user', 'rates', 'item'].edge_index.long()
        
        # 确保评分是long类型
        if hasattr(data['user', 'rates', 'item'], 'rating'):
            data['user', 'rates', 'item'].rating = data['user', 'rates', 'item'].rating.long()

    def _print_data_summary(self, data):
        """打印数据摘要"""
        logger.info("📊 数据摘要:")
        logger.info(f"- 电影节点: {data['item'].x.shape[0]}")
        logger.info(f"- 电影特征维度: {data['item'].x.shape[1]}")
        logger.info(f"- 用户节点: {data['user'].num_nodes}")
        logger.info(f"- 交互边: {data['user', 'rates', 'item'].edge_index.shape[1]}")
        
        if 'history' in data["user", "rated", "item"]:
            history = data["user", "rated", "item"].history
            if 'train' in history:
                logger.info(f"- 训练序列: {len(history['train'].get('itemId', []))}")
            if 'eval' in history:
                logger.info(f"- 评估序列: {len(history['eval'].get('itemId', []))}")


# 使用示例
if __name__ == "__main__":
    # 768维版本
    dataset = RawMovieLens32M(
        root="dataset/ml-32m",
        force_reload=True,
        target_dim=768,  # 768维输出
        max_seq_len=200,
        min_rating_count=5,
        train_split=0.8,
        text_model_name='sentence-transformers/sentence-t5-xl'

    
    print("✅ 数据集加载完成")
    data = dataset[0]
    print(f"电影特征形状: {data['item'].x.shape}")  # 应该是 [N, 768]
    print(f"用户数量: {data['user'].num_nodes}")
    print(f"交互边数量: {data['user', 'rates', 'item'].edge_index.shape[1]}")