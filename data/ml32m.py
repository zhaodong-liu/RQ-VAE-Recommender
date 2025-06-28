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
import polars as pl


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
        """Remap IDs to start from 0 (similar to Amazon dataset)"""
        return x - 1

    def amazon_style_train_test_split(self, ratings_df, user_mapping, movie_mapping, max_seq_len=20):
        """
        Amazon-style sequence splitting: each user sequence is split into train/eval/test
        based on fixed positions rather than time-based global split
        """
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []
        
        # Group ratings by user and sort by timestamp
        user_sequences = []
        for user_id in ratings_df['userId'].unique():
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
        
        # Split each user sequence
        for seq_data in user_sequences:
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
        
        # Convert to polars DataFrames (similar to Amazon)
        for sp in splits:
            if sequences[sp]["userId"]:  # Only create if not empty
                sequences[sp] = pl.from_dict(sequences[sp])
            else:
                # Create empty DataFrame with correct schema
                sequences[sp] = pl.DataFrame({
                    "userId": [],
                    "itemId": [],
                    "itemId_fut": [],
                    "rating": []
                })
        
        return sequences

    def process(self, max_seq_len=20) -> None:
        data = HeteroData()
        ratings_df = self._load_ratings()

        # Process movie data (similar to original but save text descriptions)
        movies_df = pd.read_csv(self.raw_paths[1], index_col='movieId')
        
        # Remove low occurrence movies
        movies_df = self._remove_low_occurrence(ratings_df, movies_df, "movieId")
        movie_mapping = {idx: i for i, idx in enumerate(movies_df.index)}

        # Process genres
        genres = self._process_genres(movies_df["genres"].str.get_dummies('|').values, one_hot=True)
        genres = torch.from_numpy(genres).to(torch.float)

        # Process titles and create text descriptions (Amazon-style)
        titles_text = movies_df["title"].apply(lambda s: s.split("(")[0].strip()).tolist()
        
        # Create Amazon-style text descriptions
        sentences = movies_df.apply(
            lambda row: f"Title: {row['title'].split('(')[0].strip()}; Genres: {row['genres']};",
            axis=1
        ).tolist()
        
        # Create embeddings
        titles_emb = self._encode_text_feature(sentences)
        
        # Store item features (Amazon-style format)
        data['item'].x = titles_emb  # Only use text embeddings for consistency with Amazon
        data['item'].text = np.array(sentences)  # ← Amazon-style text storage
        
        # Create Amazon-style is_train split (95% train, 5% test)
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(titles_emb.shape[0], generator=gen) > 0.05
        
        # Process user data
        user_df = pd.DataFrame({"userId": ratings_df["userId"].unique()})
        user_df = self._remove_low_occurrence(ratings_df, user_df, "userId")
        user_mapping = {idx: i for i, idx in enumerate(user_df["userId"])}

        # Process rating data (keep for compatibility)
        filtered_ratings = self._remove_low_occurrence(
            ratings_df, ratings_df, ["userId", "movieId"]
        )
        
        src = [user_mapping[idx] for idx in filtered_ratings['userId'] if idx in user_mapping]
        dst = [movie_mapping[idx] for idx in filtered_ratings['movieId'] if idx in movie_mapping]
        edge_index = torch.tensor([src, dst])
        
        data['user', 'rates', 'item'].edge_index = edge_index
        data['user', 'rates', 'item'].rating = torch.from_numpy(
            filtered_ratings['rating'].values
        ).to(torch.long)
        data['user', 'rates', 'item'].time = torch.from_numpy(
            filtered_ratings['timestamp'].values
        )
        
        # Create Amazon-style sequence history
        sequences = self.amazon_style_train_test_split(
            filtered_ratings, user_mapping, movie_mapping, max_seq_len
        )
        
        # Convert to tensor format (similar to Amazon)
        history = {}
        for split_name, split_data in sequences.items():
            if len(split_data) > 0:
                history[split_name] = {
                    "userId": torch.tensor(split_data.get_column("userId").to_list()),
                    "itemId": torch.tensor(split_data.get_column("itemId").to_list()),
                    "itemId_fut": torch.tensor(split_data.get_column("itemId_fut").to_list()),
                }
            else:
                # Empty tensors for empty splits
                history[split_name] = {
                    "userId": torch.tensor([]),
                    "itemId": torch.tensor([]).reshape(0, max_seq_len),
                    "itemId_fut": torch.tensor([]),
                }
        
        data["user", "rated", "item"].history = history

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])