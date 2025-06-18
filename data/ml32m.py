import os
import os.path as osp
import pandas as pd
import polars as pl
import torch
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler

from collections import defaultdict
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import download_url
from torch_geometric.data import extract_zip
from torch_geometric.io import fs
from typing import Callable, List, Optional


class MovieLens32M(InMemoryDataset):
    url = 'https://files.grouplens.org/datasets/movielens/ml-32m.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)
    
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

    def process():
        pass


class RawMovieLens32M(MovieLens32M, PreprocessingMixin):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        force_reload=False,
        split=None,
        # 新增降维参数
        dim_reduction_method='pca',  # 'pca', 'svd', 'random_projection', 'linear', 'none'
        target_dim=768,  # 目标维度，与Amazon对齐
        normalize_features=True,  # 是否标准化特征
        random_state=42
    ) -> None:
        self.dim_reduction_method = dim_reduction_method
        self.target_dim = target_dim
        self.normalize_features = normalize_features
        self.random_state = random_state
        
        super(RawMovieLens32M, self).__init__(
            root, transform, pre_transform, force_reload
        )

    def _load_ratings(self):
        return pd.read_csv(self.raw_paths[2])
    
    def _apply_dimensionality_reduction(self, features: torch.Tensor, method: str, target_dim: int) -> torch.Tensor:
        """
        应用降维操作
        
        Args:
            features: 输入特征张量 (n_items, original_dim)
            method: 降维方法
            target_dim: 目标维度
            
        Returns:
            降维后的特征张量 (n_items, target_dim)
        """
        original_features = features.numpy()
        original_dim = original_features.shape[1]
        
        print(f"[降维] 原始特征维度: {original_dim}, 目标维度: {target_dim}")
        
        if target_dim >= original_dim:
            print(f"[降维] 目标维度({target_dim}) >= 原始维度({original_dim})，跳过降维")
            return features
        
        if method == 'none':
            print(f"[降维] 跳过降维操作")
            return features
            
        # 特征标准化（推荐用于PCA和随机投影）
        if self.normalize_features and method in ['pca', 'random_projection']:
            print(f"[降维] 应用特征标准化...")
            scaler = StandardScaler()
            original_features = scaler.fit_transform(original_features)
        
        if method == 'pca':
            print(f"[降维] 使用PCA降维到{target_dim}维...")
            reducer = PCA(n_components=target_dim, random_state=self.random_state)
            reduced_features = reducer.fit_transform(original_features)
            
            # 打印主成分信息
            explained_variance_ratio = reducer.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            print(f"[降维] PCA解释方差比例: 前10个主成分 {explained_variance_ratio[:10]}")
            print(f"[降维] PCA累积解释方差: {cumulative_variance[target_dim-1]:.4f}")
            
        elif method == 'svd':
            print(f"[降维] 使用截断SVD降维到{target_dim}维...")
            reducer = TruncatedSVD(n_components=target_dim, random_state=self.random_state)
            reduced_features = reducer.fit_transform(original_features)
            
            # 打印SVD信息
            explained_variance_ratio = reducer.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            print(f"[降维] SVD解释方差比例: 前10个成分 {explained_variance_ratio[:10]}")
            print(f"[降维] SVD累积解释方差: {cumulative_variance[target_dim-1]:.4f}")
            
        elif method == 'random_projection':
            print(f"[降维] 使用高斯随机投影降维到{target_dim}维...")
            reducer = GaussianRandomProjection(
                n_components=target_dim, 
                random_state=self.random_state
            )
            reduced_features = reducer.fit_transform(original_features)
            print(f"[降维] 随机投影变换矩阵形状: {reducer.components_.shape}")
            
        elif method == 'linear':
            print(f"[降维] 使用线性变换降维到{target_dim}维...")
            # 简单的线性投影：使用随机初始化的权重矩阵
            np.random.seed(self.random_state)
            weight_matrix = np.random.randn(original_dim, target_dim) * 0.1
            reduced_features = original_features @ weight_matrix
            print(f"[降维] 线性变换权重矩阵形状: {weight_matrix.shape}")
            
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        # 转换回PyTorch张量
        reduced_tensor = torch.from_numpy(reduced_features).float()
        
        print(f"[降维] 降维完成: {original_features.shape} -> {reduced_tensor.shape}")
        
        # 验证降维效果
        if method in ['pca', 'svd']:
            print(f"[降维] 信息保留率: {cumulative_variance[target_dim-1]:.2%}")
        
        return reduced_tensor
    
    def _manual_train_test_split(self, df, max_seq_len):
        """类似Amazon Reviews的序列分割方法，包含rating信息"""
        print(f"[DEBUG] Starting manual train test split with max_seq_len={max_seq_len}")
        print(f"[DEBUG] Input dataframe shape: {df.shape}")
        print(f"[DEBUG] Input dataframe columns: {list(df.columns)}")
        
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        
        # 按用户分组并按时间排序
        user_groups = df.sort_values('timestamp').groupby('userId')
        print(f"[DEBUG] Number of user groups: {len(user_groups)}")
        
        user_ids = []
        processed_users = 0
        skipped_users = 0
        
        for user_id, user_df in user_groups:
            items = user_df['itemId'].tolist()
            ratings = user_df['rating'].tolist()
            
            if len(items) < 3:  # 至少需要3个交互
                skipped_users += 1
                continue
            
            processed_users += 1
            user_ids.append(user_id)
            
            if processed_users <= 5:  # 只打印前5个用户的详细信息
                print(f"[DEBUG] User {user_id}: {len(items)} interactions")
                print(f"[DEBUG] Items: {items[:10]}...")  # 只显示前10个
                print(f"[DEBUG] Ratings: {ratings[:10]}...")
            
            # 训练集：除了最后两个的所有交互
            train_items = items[:-2]
            train_ratings = ratings[:-2]
            sequences["train"]["itemId"].append(train_items)
            sequences["train"]["rating"].append(train_ratings)
            sequences["train"]["itemId_fut"].append(items[-2])
            sequences["train"]["rating_fut"].append(ratings[-2])  # 添加rating_fut
            
            # 验证集
            eval_items = items[-(max_seq_len+2):-2] if len(items) >= max_seq_len+2 else items[:-2]
            eval_ratings = ratings[-(max_seq_len+2):-2] if len(ratings) >= max_seq_len+2 else ratings[:-2]
            pad_len = max(0, max_seq_len - len(eval_items))
            sequences["eval"]["itemId"].append(eval_items + [-1] * pad_len)
            sequences["eval"]["rating"].append(eval_ratings + [0] * pad_len)
            sequences["eval"]["itemId_fut"].append(items[-2])
            sequences["eval"]["rating_fut"].append(ratings[-2])  # 添加rating_fut
            
            # 测试集
            test_items = items[-(max_seq_len+1):-1] if len(items) >= max_seq_len+1 else items[:-1]
            test_ratings = ratings[-(max_seq_len+1):-1] if len(ratings) >= max_seq_len+1 else ratings[:-1]
            pad_len = max(0, max_seq_len - len(test_items))
            sequences["test"]["itemId"].append(test_items + [-1] * pad_len)
            sequences["test"]["rating"].append(test_ratings + [0] * pad_len)
            sequences["test"]["itemId_fut"].append(items[-1])
            sequences["test"]["rating_fut"].append(ratings[-1])  # 添加rating_fut
        
        print(f"[DEBUG] Processed {processed_users} users, skipped {skipped_users} users")
        
        # 为每个分割添加用户ID - 确保长度匹配
        user_count = len(sequences["train"]["itemId"])
        print(f"[DEBUG] Total sequences generated: {user_count}")
        
        for sp in splits:
            sequences[sp]["userId"] = user_ids[:user_count]
            print(f"[DEBUG] Before polars conversion - {sp}:")
            print(f"[DEBUG]   itemId: {len(sequences[sp]['itemId'])} sequences")
            print(f"[DEBUG]   rating: {len(sequences[sp]['rating'])} sequences")
            print(f"[DEBUG]   itemId_fut: {len(sequences[sp]['itemId_fut'])} items")
            print(f"[DEBUG]   rating_fut: {len(sequences[sp]['rating_fut'])} ratings")
            print(f"[DEBUG]   userId: {len(sequences[sp]['userId'])} users")
            
            # 转换为polars DataFrame
            try:
                sequences[sp] = pl.from_dict(sequences[sp])
                print(f"[DEBUG] Successfully converted {sp} to polars DataFrame")
                print(f"[DEBUG] {sp} shape: {sequences[sp].shape}")
                print(f"[DEBUG] {sp} columns: {sequences[sp].columns}")
            except Exception as e:
                print(f"[DEBUG] Error converting {sp} to polars: {e}")
                print(f"[DEBUG] {sp} dict keys: {list(sequences[sp].keys())}")
                for key, value in sequences[sp].items():
                    print(f"[DEBUG] {sp}[{key}]: length={len(value)}, type={type(value)}")
                    if len(value) > 0:
                        print(f"[DEBUG] {sp}[{key}][0]: {value[0]}, type={type(value[0])}")
                raise
        
        print(f"[DEBUG] Manual train test split completed")
        return sequences
    
    def process(self, max_seq_len=None) -> None:
        data = HeteroData()
        ratings_df = self._load_ratings()
        
        print(f"[DEBUG] Loading ratings: {len(ratings_df)} total ratings")

        # TODO: Extract actor name tag from tag dataset
        # TODO: Maybe use links to extract more item features
        # Process movie data:
        df = pd.read_csv(self.raw_paths[1], index_col='movieId')
        
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}
        print(f"[DEBUG] Movie mapping created: {len(movie_mapping)} movies")

        genres = self._process_genres(df["genres"].str.get_dummies('|').values, one_hot=True)
        genres = torch.from_numpy(genres).to(torch.float)

        titles_text = df["title"].apply(lambda s: s.split("(")[0].strip()).tolist()
        print(f"[DEBUG] Processing {len(titles_text)} movie titles")
        
        titles_emb = self._encode_text_feature(titles_text)
        print(f"[DEBUG] Title embeddings shape: {titles_emb.shape}")

        # 拼接特征
        x = torch.cat([titles_emb, genres], axis=1)
        print(f"[DEBUG] 拼接后特征形状: {x.shape}")

        # 🔥 应用降维操作
        if self.dim_reduction_method != 'none':
            print(f"[DEBUG] 开始降维操作...")
            print(f"[DEBUG] 降维方法: {self.dim_reduction_method}")
            print(f"[DEBUG] 目标维度: {self.target_dim}")
            
            x_reduced = self._apply_dimensionality_reduction(
                x, 
                self.dim_reduction_method, 
                self.target_dim
            )
            
            # 特征对比
            print(f"[DEBUG] 降维前特征统计:")
            print(f"[DEBUG]   形状: {x.shape}")
            print(f"[DEBUG]   均值: {x.mean().item():.6f}")
            print(f"[DEBUG]   标准差: {x.std().item():.6f}")
            print(f"[DEBUG]   最小值: {x.min().item():.6f}")
            print(f"[DEBUG]   最大值: {x.max().item():.6f}")
            
            print(f"[DEBUG] 降维后特征统计:")
            print(f"[DEBUG]   形状: {x_reduced.shape}")
            print(f"[DEBUG]   均值: {x_reduced.mean().item():.6f}")
            print(f"[DEBUG]   标准差: {x_reduced.std().item():.6f}")
            print(f"[DEBUG]   最小值: {x_reduced.min().item():.6f}")
            print(f"[DEBUG]   最大值: {x_reduced.max().item():.6f}")
            
            x = x_reduced
        else:
            print(f"[DEBUG] 跳过降维，保持原始特征维度: {x.shape}")

        print(f"[DEBUG] Final item features shape: {x.shape}")

        data['item'].x = x
        # 添加text属性 - 与Amazon Reviews保持一致
        data['item'].text = np.array(titles_text)
        
        # 添加is_train标记 - 95%的电影用于训练，5%用于测试
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(x.shape[0], generator=gen) > 0.05
        print(f"[DEBUG] Item train/test split: {data['item'].is_train.sum().item()}/{(~data['item'].is_train).sum().item()}")
        
        # Process user data:
        full_df = pd.DataFrame({"userId": ratings_df["userId"].unique()})
        df = self._remove_low_occurrence(ratings_df, full_df, "userId")
        user_mapping = {idx: i for i, idx in enumerate(df["userId"])}
        self.int_user_data = df
        print(f"[DEBUG] User mapping created: {len(user_mapping)} users after filtering")

        # Process rating data:
        df = self._remove_low_occurrence(
            ratings_df,
            ratings_df,
            ["userId", "movieId"]
        )
        print(f"[DEBUG] After removing low occurrence: {len(df)} ratings")
        
        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])
        data['user', 'rates', 'item'].edge_index = edge_index

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'item'].rating = rating

        time = torch.from_numpy(df['timestamp'].values)
        data['user', 'rates', 'item'].time = time

        data['item', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['item', 'rated_by', 'user'].rating = rating
        data['item', 'rated_by', 'user'].time = time

        df["itemId"] = df["movieId"].apply(lambda x: movie_mapping[x])
        df["rating"] = (2*df["rating"]).astype(int)
        
        print(f"[DEBUG] Starting sequence generation with max_seq_len={max_seq_len if max_seq_len is not None else 200}")
        
        # 使用自定义的序列生成方法，包含rating信息
        sequences = self._manual_train_test_split(df, max_seq_len if max_seq_len is not None else 200)
        
        print(f"[DEBUG] Sequence generation completed")
        print(f"[DEBUG] Available splits: {list(sequences.keys())}")
        
        for split_name, split_data in sequences.items():
            if hasattr(split_data, 'shape'):
                print(f"[DEBUG] {split_name}: shape {split_data.shape}")
            elif hasattr(split_data, '__len__'):
                print(f"[DEBUG] {split_name}: {len(split_data)} samples")
            else:
                print(f"[DEBUG] {split_name}: {type(split_data)}")
            
            # 检查split_data的列
            if hasattr(split_data, 'columns'):
                print(f"[DEBUG] {split_name} columns: {split_data.columns}")
        
        print(f"[DEBUG] Converting sequences to tensor format...")
        
        try:
            # 转换为包含rating信息的格式
            history_data = {}
            for k, v in sequences.items():
                print(f"[DEBUG] Processing split: {k}")
                print(f"[DEBUG] Split data type: {type(v)}")
                if hasattr(v, 'columns'):
                    print(f"[DEBUG] Split columns: {list(v.columns)}")
                
                # 检查是否包含所有必需的列
                expected_columns = ['itemId', 'rating', 'itemId_fut', 'rating_fut', 'userId']
                missing_columns = [col for col in expected_columns if col not in v.columns]
                if missing_columns:
                    print(f"[DEBUG] Warning: Missing columns in {k}: {missing_columns}")
                
                # 传递itemId和rating两个特征
                tensor_dict = self._df_to_tensor_dict(v, ["itemId", "rating"])
                history_data[k] = tensor_dict
                print(f"[DEBUG] Converted {k} to tensor dict")
                
                # 打印tensor dict的内容
                for tensor_key, tensor_value in tensor_dict.items():
                    if hasattr(tensor_value, 'shape'):
                        print(f"[DEBUG] {k}[{tensor_key}] shape: {tensor_value.shape}")
                    else:
                        print(f"[DEBUG] {k}[{tensor_key}] type: {type(tensor_value)}")
            
            data["user", "rated", "item"].history = history_data
            print(f"[DEBUG] History data assignment completed")
            
        except Exception as e:
            print(f"[DEBUG] Error in tensor conversion: {e}")
            print(f"[DEBUG] Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise

        # 验证最终的history结构
        print(f"[DEBUG] Final history keys: {list(data['user', 'rated', 'item'].history.keys())}")
        for key, value in data["user", "rated", "item"].history.items():
            print(f"[DEBUG] History[{key}] type: {type(value)}")
            if hasattr(value, 'keys'):
                print(f"[DEBUG] History[{key}] keys: {list(value.keys())}")

        if self.pre_transform is not None:
            print(f"[DEBUG] Applying pre_transform...")
            data = self.pre_transform(data)

        print(f"[DEBUG] Saving processed data...")
        self.save([data], self.processed_paths[0])
        print(f"[DEBUG] Processing completed successfully!")


# 使用示例和测试代码
def test_dimensionality_reduction():
    """测试不同的降维方法"""
    print("🧪 测试降维方法")
    print("=" * 50)
    
    # 创建测试数据
    np.random.seed(42)
    test_features = torch.randn(100, 788)  # 模拟ML32M的原始特征
    
    methods = ['pca', 'svd', 'random_projection', 'linear']
    target_dims = [768, 512, 256]
    
    for method in methods:
        print(f"\n📊 测试方法: {method.upper()}")
        for target_dim in target_dims:
            # 创建临时的RawMovieLens32M实例（仅用于测试降维方法）
            class TempML32M:
                def __init__(self):
                    self.dim_reduction_method = method
                    self.target_dim = target_dim
                    self.normalize_features = True
                    self.random_state = 42
                
                def _apply_dimensionality_reduction(self, features, method, target_dim):
                    # 复制原始方法的逻辑
                    original_features = features.numpy()
                    original_dim = original_features.shape[1]
                    
                    if target_dim >= original_dim:
                        return features
                    
                    if method == 'pca':
                        from sklearn.decomposition import PCA
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        original_features = scaler.fit_transform(original_features)
                        reducer = PCA(n_components=target_dim, random_state=42)
                        reduced_features = reducer.fit_transform(original_features)
                        variance_explained = reducer.explained_variance_ratio_.sum()
                        print(f"    {target_dim}维: 解释方差={variance_explained:.3f}")
                    elif method == 'svd':
                        from sklearn.decomposition import TruncatedSVD
                        reducer = TruncatedSVD(n_components=target_dim, random_state=42)
                        reduced_features = reducer.fit_transform(original_features)
                        variance_explained = reducer.explained_variance_ratio_.sum()
                        print(f"    {target_dim}维: 解释方差={variance_explained:.3f}")
                    elif method == 'random_projection':
                        from sklearn.random_projection import GaussianRandomProjection
                        reducer = GaussianRandomProjection(n_components=target_dim, random_state=42)
                        reduced_features = reducer.fit_transform(original_features)
                        print(f"    {target_dim}维: 随机投影完成")
                    elif method == 'linear':
                        np.random.seed(42)
                        weight_matrix = np.random.randn(original_dim, target_dim) * 0.1
                        reduced_features = original_features @ weight_matrix
                        print(f"    {target_dim}维: 线性变换完成")
                    
                    return torch.from_numpy(reduced_features).float()
            
            temp_ml32m = TempML32M()
            reduced = temp_ml32m._apply_dimensionality_reduction(test_features, method, target_dim)

if __name__ == "__main__":
    # 运行测试
    test_dimensionality_reduction()
    
    print("\n" + "="*60)
    print("📋 使用说明:")
    print("="*60)
    print("1. PCA降维 (推荐):")
    print("   dataset = RawMovieLens32M(root='...', dim_reduction_method='pca', target_dim=768)")
    print()
    print("2. SVD降维:")
    print("   dataset = RawMovieLens32M(root='...', dim_reduction_method='svd', target_dim=768)")
    print()
    print("3. 随机投影:")
    print("   dataset = RawMovieLens32M(root='...', dim_reduction_method='random_projection', target_dim=768)")
    print()
    print("4. 不降维:")
    print("   dataset = RawMovieLens32M(root='...', dim_reduction_method='none')")
    print()
    print("💡 建议使用PCA降维到768维，与Amazon数据集对齐")