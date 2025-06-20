import os
import os.path as osp
import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.data import download_url, extract_zip
from torch_geometric.io import fs
from typing import Callable, List, Optional, Dict, Any

from tqdm import tqdm


class MovieLens32M(InMemoryDataset):
    url = 'https://files.grouplens.org/datasets/movielens/ml-32m.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        **kwargs
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['links.csv', 'movies.csv', 'ratings.csv', 'tags.csv']
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-32m')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

class RawMovieLens32M(MovieLens32M):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        dim_reduction_method: str = 'pca',
        target_dim: int = 768,
        normalize_features: bool = True,
        random_state: int = 42,
        pad_value: int = 0,
        max_seq_len: int = 200,
        split: Optional[str] = None,
        **kwargs
    ) -> None:
        self.dim_reduction_method = dim_reduction_method
        self.target_dim = target_dim
        self.normalize_features = normalize_features
        self.random_state = random_state
        self.pad_value = pad_value
        self.max_seq_len = max_seq_len  # 初始化时设置默认值
        self.split = split
        
        super().__init__(root, transform, pre_transform, force_reload, **kwargs)


    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有原始数据"""
        files = ['movies.csv', 'ratings.csv', 'tags.csv', 'links.csv']
        data = {}
        print("📥 加载原始数据中...")
        for fname in tqdm(files, desc="读取CSV"):
            key = fname.replace('.csv', '')
            data[key] = pd.read_csv(osp.join(self.raw_dir, fname))
        return data

    def _process_features(self, df: pd.DataFrame) -> tuple:
        """处理电影特征 - 已修复维度问题"""
        # 确保movieId是连续的
        df['movieId'] = df['movieId'].astype('category').cat.codes + 1
        
        # 处理电影类型 - 形状为 [n_movies, n_genres]
        genres = df["genres"].str.get_dummies('|').values
        genres = torch.from_numpy(genres).float()
        
        # 处理标题文本
        titles = df["title"].apply(lambda s: s.split("(")[0].strip()).tolist()
        
        # 添加年份特征 - 保持形状为 [n_movies, 1]
        years = df["title"].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
        years = torch.from_numpy(years.values).float()  # 👈 不再 unsqueeze

        # 特征拼接 - 两者都是 [n_movies, feature_dim]
        x = torch.cat([genres, years], dim=1)
        return x, titles


    def _apply_dimensionality_reduction(self, features: torch.Tensor) -> torch.Tensor:
        """应用降维"""
        if self.dim_reduction_method == 'none' or features.shape[1] <= self.target_dim:
            return features
            
        original = features.numpy()
        if self.normalize_features:
            original = StandardScaler().fit_transform(original)
        
        if self.dim_reduction_method == 'pca':
            reducer = PCA(n_components=self.target_dim, random_state=self.random_state)
        elif self.dim_reduction_method == 'svd':
            reducer = TruncatedSVD(n_components=self.target_dim, random_state=self.random_state)
        elif self.dim_reduction_method == 'random_projection':
            reducer = GaussianRandomProjection(n_components=self.target_dim, random_state=self.random_state)
        else:
            raise ValueError(f"不支持的降维方法: {self.dim_reduction_method}")
        
        reduced = reducer.fit_transform(original)
        return torch.from_numpy(reduced).float()

    def _create_sequences(self, df: pd.DataFrame, movie_mapping: Dict) -> Dict:
        """创建用户交互序列 - 使用self.max_seq_len"""
        sequences = defaultdict(lambda: defaultdict(list))
        
        if self.split and self.split in ['train', 'eval', 'test']:
            unique_users = df['userId'].unique()
            np.random.seed(self.random_state)
            mask = np.random.rand(len(unique_users)) < 0.8
            split_users = {
                'train': unique_users[mask],
                'eval': unique_users[~mask][:len(unique_users[~mask])//2],
                'test': unique_users[~mask][len(unique_users[~mask])//2:]
            }
            df = df[df['userId'].isin(split_users[self.split])]

        print(f"🧩 正在构建用户序列数据...")
        for user_id, group in tqdm(df.sort_values('timestamp').groupby('userId'), desc="处理用户"):
            items = group['movieId'].map(movie_mapping).tolist()
            ratings = group['rating'].tolist()
            
            if len(items) < 3:
                continue
            
            sequences['train']['itemId'].append(items[:-2])
            sequences['train']['rating'].append(ratings[:-2])
            sequences['train']['itemId_fut'].append(items[-2])
            sequences['train']['rating_fut'].append(ratings[-2])
            
            for split, end_idx in [('eval', -2), ('test', -1)]:
                seq_len = self.max_seq_len if split == 'test' else self.max_seq_len // 2
                start_idx = max(0, end_idx - seq_len)
                
                seq_items = items[start_idx:end_idx]
                seq_ratings = ratings[start_idx:end_idx]
                pad_len = max(0, seq_len - len(seq_items))
                
                sequences[split]['itemId'].append(seq_items + [self.pad_value] * pad_len)
                sequences[split]['rating'].append(seq_ratings + [0] * pad_len)
                sequences[split]['itemId_fut'].append(items[end_idx])
                sequences[split]['rating_fut'].append(ratings[end_idx])
                sequences[split]['userId'].append(user_id)
        
        return sequences


    def process(self, max_seq_len: Optional[int] = None) -> None:
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
        print(f"[数据处理] 使用序列长度: {self.max_seq_len}")
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA错误
        
        data = HeteroData()
        raw_data = self._load_data()
        
        # 1. 处理电影数据
        x, titles = self._process_features(raw_data['movies'])
        x = self._apply_dimensionality_reduction(x)
        
        # 创建电影映射 (保留0给填充值)
        movie_mapping = {mid: i+1 for i, mid in enumerate(raw_data['movies']['movieId'].unique())}
        movie_mapping[self.pad_value] = 0
        x = torch.cat([torch.zeros(1, x.size(1)), x])  # 填充值特征
        
        data['item'].x = x
        data['item'].text = np.array([""] + titles)
        
        # 2. 处理评分数据
        ratings = raw_data['ratings']
        ratings = ratings[ratings['movieId'].isin(raw_data['movies']['movieId'])]
        
        # 3. 创建用户映射
        user_mapping = {uid: i+1 for i, uid in enumerate(ratings['userId'].unique())}
        user_mapping[self.pad_value] = 0
        data['user'].num_nodes = len(user_mapping)
        
        # 4. 创建边
        src = torch.tensor([user_mapping[uid] for uid in ratings['userId']], dtype=torch.long)
        dst = torch.tensor([movie_mapping[mid] for mid in ratings['movieId']], dtype=torch.long)
        
        assert src.min() >= 0 and src.max() < len(user_mapping), "用户索引越界"
        assert dst.min() >= 0 and dst.max() < len(movie_mapping), "电影索引越界"
        
        edge_index = torch.stack([src, dst])
        data['user', 'rates', 'item'].edge_index = edge_index
        data['user', 'rates', 'item'].rating = torch.from_numpy(ratings['rating'].values).long()
        

        # 5. 创建序列数据 - 使用self.max_seq_len
        sequences = self._create_sequences(ratings, movie_mapping)

        # 转换为PyTorch张量
        history_data = {}
        print("📦 转换为PyTorch张量中...")
        for split, split_data in tqdm(sequences.items(), desc="处理Split"):
            history_data[split] = {
                'itemId': torch.tensor(split_data['itemId'], dtype=torch.long),
                'rating': torch.tensor(split_data['rating'], dtype=torch.long),
                'itemId_fut': torch.tensor(split_data['itemId_fut'], dtype=torch.long),
                'rating_fut': torch.tensor(split_data['rating_fut'], dtype=torch.long),
                'userId': torch.tensor(split_data['userId'], dtype=torch.long)
            }

        
        data['user', 'rated', 'item'].history = history_data
        
        # 6. 保存处理后的数据
        self.save([data], self.processed_paths[0])
        
        print(f"数据处理完成! 统计信息:")
        print(f"- 电影数量: {len(movie_mapping)} (包含填充值)")
        print(f"- 用户数量: {len(user_mapping)} (包含填充值)")
        print(f"- 交互数量: {len(ratings)}")
        print(f"- 特征维度: {x.size(1)}")
        print(f"- 最大序列长度: {self.max_seq_len}")
        if self.split:
            print(f"- 当前分割: {self.split}")