import os
import os.path as osp
import pandas as pd
import torch
import numpy as np
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
        split=None
    ) -> None:
        super(RawMovieLens32M, self).__init__(
            root, transform, pre_transform, force_reload
        )

    def _load_ratings(self):
        return pd.read_csv(self.raw_paths[2])
    
    def _remap_ids(self, x):
        """Remap IDs to start from 0"""
        return x - 1

    def optimized_amazon_style_train_test_split(self, ratings_df, user_mapping, movie_mapping, max_seq_len=200):
        """
        优化版本的Amazon风格序列分割，使用向量化操作提高速度
        """
        print("   使用优化的Amazon风格序列分割...")
        
        # 首先过滤出有效的用户和电影
        print("   过滤有效的评分数据...")
        valid_ratings = ratings_df[
            (ratings_df['userId'].isin(user_mapping.keys())) & 
            (ratings_df['movieId'].isin(movie_mapping.keys()))
        ].copy()
        
        # 映射ID
        print("   映射用户和电影ID...")
        valid_ratings['mapped_userId'] = valid_ratings['userId'].map(user_mapping)
        valid_ratings['mapped_movieId'] = valid_ratings['movieId'].map(movie_mapping)
        
        # 按用户和时间排序
        print("   按用户和时间排序...")
        valid_ratings = valid_ratings.sort_values(['mapped_userId', 'timestamp'])
        
        # 使用pandas groupby进行向量化处理
        print("   按用户分组...")
        user_groups = valid_ratings.groupby('mapped_userId')
        
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        
        print("   处理用户序列...")
        processed_count = 0
        total_users = len(user_groups)
        
        for user_id, group in user_groups:
            if processed_count % 10000 == 0:
                print(f"   处理进度: {processed_count:,}/{total_users:,} ({processed_count/total_users*100:.1f}%)")
            
            # 获取该用户的电影序列
            items = group['mapped_movieId'].tolist()
            ratings = group['rating'].tolist()
            
            # 只处理至少有3个交互的用户
            if len(items) >= 3:
                # Amazon风格分割
                train_items = items[:-2]
                eval_items = items[-(max_seq_len+2):-2] if len(items) >= max_seq_len+2 else items[:-2]
                test_items = items[-(max_seq_len+1):-1] if len(items) >= max_seq_len+1 else items[:-1]
                
                # 填充或截断序列
                def pad_sequence(seq, target_len):
                    if len(seq) > target_len:
                        return seq[-target_len:]
                    else:
                        return seq + [-1] * (target_len - len(seq))
                
                # 添加到训练集
                if len(train_items) > 0:
                    sequences["train"]["itemId"].append(pad_sequence(train_items, max_seq_len))
                    sequences["train"]["itemId_fut"].append(items[-2] if len(items) >= 2 else -1)
                    sequences["train"]["userId"].append(user_id)
                
                # 添加到评估集
                if len(eval_items) > 0:
                    sequences["eval"]["itemId"].append(pad_sequence(eval_items, max_seq_len))
                    sequences["eval"]["itemId_fut"].append(items[-2] if len(items) >= 2 else -1)
                    sequences["eval"]["userId"].append(user_id)
                
                # 添加到测试集
                if len(test_items) > 0:
                    sequences["test"]["itemId"].append(pad_sequence(test_items, max_seq_len))
                    sequences["test"]["itemId_fut"].append(items[-1] if len(items) >= 1 else -1)
                    sequences["test"]["userId"].append(user_id)
            
            processed_count += 1
        
        print(f"   处理完成! 有效用户: {processed_count:,}")
        print(f"   训练序列数: {len(sequences['train']['userId']):,}")
        print(f"   评估序列数: {len(sequences['eval']['userId']):,}")
        print(f"   测试序列数: {len(sequences['test']['userId']):,}")
        
        # 转换为polars DataFrames
        print("   转换为polars DataFrame...")
        import polars as pl
        for sp in splits:
            if sequences[sp]["userId"]:
                sequences[sp] = pl.from_dict(sequences[sp])
                print(f"   {sp} DataFrame shape: {sequences[sp].shape}")
            else:
                sequences[sp] = pl.DataFrame({
                    "userId": [],
                    "itemId": [],
                    "itemId_fut": []
                })
        
        return sequences

    def process(self, max_seq_len=200) -> None:
        print("=" * 60)
        print("开始处理ML-32M数据集 (优化版Amazon格式)")
        print("=" * 60)
        
        data = HeteroData()
        
        print("1. 加载评分数据...")
        ratings_df = self._load_ratings()
        print(f"   原始评分数: {len(ratings_df):,}")
        print(f"   用户数: {ratings_df['userId'].nunique():,}")
        print(f"   电影数: {ratings_df['movieId'].nunique():,}")

        # 可选：为了测试，可以只使用部分数据
        # 取消注释下面的行来使用10%的数据进行快速测试
        # print("   ⚠️  使用10%数据进行快速测试...")
        # ratings_df = ratings_df.sample(frac=0.1, random_state=42)
        # print(f"   采样后评分数: {len(ratings_df):,}")

        print("\n2. 处理电影数据...")
        movies_df = pd.read_csv(self.raw_paths[1], index_col='movieId')
        print(f"   原始电影数: {len(movies_df):,}")
        
        # Remove low occurrence movies
        print("   移除低频电影...")
        movies_df = self._remove_low_occurrence(ratings_df, movies_df, "movieId")
        print(f"   过滤后电影数: {len(movies_df):,}")
        movie_mapping = {idx: i for i, idx in enumerate(movies_df.index)}

        # Process titles and create Amazon-style text descriptions
        print("   创建Amazon风格文本描述...")
        sentences = movies_df.apply(
            lambda row: f"Title: {row['title'].split('(')[0].strip()}; Genres: {row['genres']};",
            axis=1
        ).tolist()
        print(f"   示例文本: {sentences[0]}")
        
        # Create embeddings
        print("   生成文本嵌入 (这可能需要几分钟)...")
        item_emb = self._encode_text_feature(sentences)
        print(f"   文本嵌入形状: {item_emb.shape}")
        
        # Store item features
        data['item'].x = item_emb
        data['item'].text = np.array(sentences)
        
        # Create is_train split
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05
        train_items = data['item'].is_train.sum().item()
        test_items = (~data['item'].is_train).sum().item()
        print(f"   物品分割: {train_items:,} 训练, {test_items:,} 测试")
        
        print("\n3. 处理用户数据...")
        user_df = pd.DataFrame({"userId": ratings_df["userId"].unique()})
        print(f"   原始用户数: {len(user_df):,}")
        user_df = self._remove_low_occurrence(ratings_df, user_df, "userId")
        print(f"   过滤后用户数: {len(user_df):,}")
        user_mapping = {idx: i for i, idx in enumerate(user_df["userId"])}

        print("\n4. 处理评分数据...")
        filtered_ratings = self._remove_low_occurrence(
            ratings_df, ratings_df, ["userId", "movieId"]
        )
        print(f"   过滤后评分数: {len(filtered_ratings):,}")
        
        print("\n5. 创建优化的Amazon风格序列历史...")
        # 使用优化版本的序列分割
        sequences = self.optimized_amazon_style_train_test_split(
            filtered_ratings, user_mapping, movie_mapping, max_seq_len
        )
        
        print("\n6. 转换为张量格式...")
        history = {}
        for split_name, split_data in sequences.items():
            print(f"   处理{split_name}分割...")
            if len(split_data) > 0:
                history[split_name] = {
                    "userId": torch.tensor(split_data.get_column("userId").to_list()),
                    "itemId": torch.tensor(split_data.get_column("itemId").to_list()),
                    "itemId_fut": torch.tensor(split_data.get_column("itemId_fut").to_list()),
                }
                print(f"   {split_name} 张量形状: userId {history[split_name]['userId'].shape}, itemId {history[split_name]['itemId'].shape}")
            else:
                history[split_name] = {
                    "userId": torch.tensor([]),
                    "itemId": torch.tensor([]).reshape(0, max_seq_len),
                    "itemId_fut": torch.tensor([]),
                }
        
        data["user", "rated", "item"].history = history

        print("\n7. 保存数据...")
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
        
        print("\n" + "=" * 60)
        print("ML-32M数据集处理完成! (优化版Amazon格式)")
        print("=" * 60)
        print("数据集统计:")
        print(f"  电影总数: {item_emb.shape[0]:,}")
        print(f"  训练电影: {train_items:,} ({train_items/item_emb.shape[0]*100:.1f}%)")
        print(f"  测试电影: {test_items:,} ({test_items/item_emb.shape[0]*100:.1f}%)")
        print(f"  训练序列: {len(history['train']['userId']):,}")
        print(f"  评估序列: {len(history['eval']['userId']):,}")
        print(f"  测试序列: {len(history['test']['userId']):,}")
        print(f"  序列长度: {max_seq_len}")
        print("=" * 60)