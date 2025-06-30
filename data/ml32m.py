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

from collections import defaultdict

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

    def amazon_style_train_test_split(self, ratings_df, user_mapping, movie_mapping, max_seq_len=200):
        """
        Amazon-style sequence splitting: each user sequence is split into train/eval/test
        based on fixed positions rather than time-based global split
        """
        print("   使用Amazon风格的序列分割...")
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        
        # Group ratings by user and sort by timestamp
        user_sequences = []
        processed_users = 0
        total_users = len(ratings_df['userId'].unique())
        
        for i, user_id in enumerate(ratings_df['userId'].unique()):
            if i % 10000 == 0:
                print(f"   处理用户进度: {i:,}/{total_users:,} ({i/total_users*100:.1f}%)")
                
            if user_id in user_mapping:
                user_ratings = ratings_df[ratings_df['userId'] == user_id].sort_values('timestamp')
                items = [movie_mapping[mid] for mid in user_ratings['movieId'] if mid in movie_mapping]
                ratings = user_ratings['rating'].tolist()
                
                if len(items) >= 3:  # Need at least 3 interactions for splitting
                    user_sequences.append({
                        'userId': user_mapping[user_id],
                        'items': items,
                        'ratings': ratings
                    })
                    processed_users += 1
        
        print(f"   有效用户序列数: {len(user_sequences):,}")
        
        # Split each user sequence (Amazon style)
        print("   执行Amazon风格分割...")
        for i, seq_data in enumerate(user_sequences):
            if i % 5000 == 0:
                print(f"   分割进度: {i:,}/{len(user_sequences):,} ({i/len(user_sequences)*100:.1f}%)")
                
            user_id = seq_data['userId']
            items = seq_data['items']
            ratings = seq_data['ratings']
            
            # Amazon-style splitting
            train_items = items[:-2]
            train_ratings = ratings[:-2] if len(ratings) > 2 else []
            
            eval_items = items[-(max_seq_len+2):-2] if len(items) >= max_seq_len+2 else items[:-2]
            eval_ratings = ratings[-(max_seq_len+2):-2] if len(ratings) >= max_seq_len+2 else ratings[:-2]
            
            test_items = items[-(max_seq_len+1):-1] if len(items) >= max_seq_len+1 else items[:-1]
            test_ratings = ratings[-(max_seq_len+1):-1] if len(ratings) >= max_seq_len+1 else ratings[:-1]
            
            # Pad sequences to max_seq_len
            def pad_sequence(seq, target_len):
                if len(seq) > target_len:
                    return seq[-target_len:]
                else:
                    return seq + [-1] * (target_len - len(seq))
            
            # Store train data
            if len(train_items) > 0:
                sequences["train"]["itemId"].append(pad_sequence(train_items, max_seq_len))
                sequences["train"]["itemId_fut"].append(items[-2] if len(items) >= 2 else -1)
                sequences["train"]["rating"].append(pad_sequence(train_ratings, max_seq_len))
                sequences["train"]["userId"].append(user_id)
            
            # Store eval data
            if len(eval_items) > 0:
                sequences["eval"]["itemId"].append(pad_sequence(eval_items, max_seq_len))
                sequences["eval"]["itemId_fut"].append(items[-2] if len(items) >= 2 else -1)
                sequences["eval"]["rating"].append(pad_sequence(eval_ratings, max_seq_len))
                sequences["eval"]["userId"].append(user_id)
            
            # Store test data  
            if len(test_items) > 0:
                sequences["test"]["itemId"].append(pad_sequence(test_items, max_seq_len))
                sequences["test"]["itemId_fut"].append(items[-1] if len(items) >= 1 else -1)
                sequences["test"]["rating"].append(pad_sequence(test_ratings, max_seq_len))
                sequences["test"]["userId"].append(user_id)
        
        print(f"   训练序列数: {len(sequences['train']['userId']):,}")
        print(f"   评估序列数: {len(sequences['eval']['userId']):,}")
        print(f"   测试序列数: {len(sequences['test']['userId']):,}")
        
        # Convert to polars DataFrames (exactly like Amazon)
        print("   转换为polars DataFrame...")
        import polars as pl
        for sp in splits:
            if sequences[sp]["userId"]:  # Only create if not empty
                sequences[sp] = pl.from_dict(sequences[sp])
                print(f"   {sp} DataFrame shape: {sequences[sp].shape}")
            else:
                # Create empty DataFrame with correct schema
                sequences[sp] = pl.DataFrame({
                    "userId": [],
                    "itemId": [],
                    "itemId_fut": [],
                    "rating": []
                })
        
        return sequences

    def process(self, max_seq_len=200) -> None:  # 改为Amazon的默认序列长度20
        print("=" * 60)
        print("开始处理ML-32M数据集 (Amazon格式)")
        print("=" * 60)
        
        data = HeteroData()
        
        print("1. 加载评分数据...")
        ratings_df = self._load_ratings()
        print(f"   原始评分数: {len(ratings_df):,}")
        print(f"   用户数: {ratings_df['userId'].nunique():,}")
        print(f"   电影数: {ratings_df['movieId'].nunique():,}")

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
        titles_text = movies_df["title"].apply(lambda s: s.split("(")[0].strip()).tolist()
        
        # Create Amazon-style text descriptions (exactly like Amazon)
        sentences = movies_df.apply(
            lambda row: f"Title: {row['title'].split('(')[0].strip()}; Genres: {row['genres']};",
            axis=1
        ).tolist()
        print(f"   示例文本: {sentences[0]}")
        
        # Create embeddings
        print("   生成文本嵌入 (这可能需要几分钟)...")
        item_emb = self._encode_text_feature(sentences)  # 使用Amazon的变量名
        print(f"   文本嵌入形状: {item_emb.shape}")
        
        # Store item features (Amazon-style format - exactly the same)
        data['item'].x = item_emb
        data['item'].text = np.array(sentences)
        
        # Create Amazon-style is_train split (95% train, 5% test)
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
        
        print("\n5. 创建Amazon风格序列历史...")
        # Use Amazon-style sequence splitting
        sequences = self.amazon_style_train_test_split(
            filtered_ratings, user_mapping, movie_mapping, max_seq_len
        )
        
        print("\n6. 转换为张量格式 (Amazon风格)...")
        # Convert to tensor format (exactly like Amazon)
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
                # Empty tensors for empty splits
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
        print("ML-32M数据集处理完成! (Amazon格式)")
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