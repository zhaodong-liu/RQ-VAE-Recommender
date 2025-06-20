import os
import os.path as osp
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
from torch.utils.data import Dataset
from torch_geometric.data import download_url, extract_zip
from torch_geometric.io import fs
from typing import Callable, List, Optional, Dict, Any, Tuple
import logging
from tqdm import tqdm
from enum import Enum

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecDataset(Enum):
    """推荐数据集枚举"""
    ML_1M = "ml-1m"
    ML_32M = "ml-32m"

class ItemData(Dataset):
    """适配RQ-VAE的物品数据集类"""
    
    # 数据集URL映射
    DATASET_URLS = {
        RecDataset.ML_1M: 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        RecDataset.ML_32M: 'https://files.grouplens.org/datasets/movielens/ml-32m.zip'
    }
    
    def __init__(
        self,
        root: str,
        dataset: RecDataset = RecDataset.ML_1M,
        force_process: bool = False,
        train_test_split: str = "train",  # "train", "eval", "test", "all"
        input_dim: int = 18,
        min_interactions: int = 5,
        random_state: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        normalize_features: bool = True,
        **kwargs
    ):
        self.root = root
        self.dataset = dataset
        self.force_process = force_process
        self.train_test_split = train_test_split
        self.input_dim = input_dim
        self.min_interactions = min_interactions
        self.random_state = random_state
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalize_features = normalize_features
        
        # 设置路径
        self.raw_dir = osp.join(root, 'raw')
        self.processed_dir = osp.join(root, 'processed')
        
        # 确保目录存在
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # 处理后的文件路径
        self.processed_file = osp.join(
            self.processed_dir, 
            f'{dataset.value}_{train_test_split}_{input_dim}d.pt'
        )
        
        # 加载或处理数据
        if force_process or not osp.exists(self.processed_file):
            self._download_and_process()
        
        # 加载处理后的数据
        self.data = torch.load(self.processed_file)
        self.items = self.data['items']
        self.item_features = self.data['item_features']
        self.metadata = self.data['metadata']
        
        logger.info(f"✅ 数据集加载完成: {len(self.items)} 个样本")

    def _download_and_process(self):
        """下载并处理数据"""
        logger.info(f"🔄 开始处理 {self.dataset.value} 数据集...")
        
        # 下载数据
        if not self._check_raw_files():
            self._download()
        
        # 处理数据
        self._process()

    def _check_raw_files(self) -> bool:
        """检查原始文件是否存在"""
        required_files = ['movies.csv', 'ratings.csv']
        if self.dataset == RecDataset.ML_1M:
            required_files = ['movies.dat', 'ratings.dat', 'users.dat']
        
        return all(osp.exists(osp.join(self.raw_dir, f)) for f in required_files)

    def _download(self):
        """下载数据集"""
        logger.info(f"📥 下载 {self.dataset.value} 数据集...")
        
        url = self.DATASET_URLS[self.dataset]
        path = download_url(url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        
        # 移动文件到raw目录
        extracted_folder = osp.join(self.root, self.dataset.value)
        if osp.exists(extracted_folder):
            # 移动所有文件
            import shutil
            for file in os.listdir(extracted_folder):
                shutil.move(
                    osp.join(extracted_folder, file),
                    osp.join(self.raw_dir, file)
                )
            os.rmdir(extracted_folder)

    def _load_movielens_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载MovieLens数据"""
        if self.dataset == RecDataset.ML_1M:
            # ML-1M使用dat文件和::分隔符
            movies = pd.read_csv(
                osp.join(self.raw_dir, 'movies.dat'),
                sep='::',
                names=['MovieID', 'Title', 'Genres'],
                engine='python',
                encoding='latin-1'
            )
            ratings = pd.read_csv(
                osp.join(self.raw_dir, 'ratings.dat'),
                sep='::',
                names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                engine='python'
            )
            users = pd.read_csv(
                osp.join(self.raw_dir, 'users.dat'),
                sep='::',
                names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                engine='python'
            )
        else:
            # ML-32M使用csv文件
            movies = pd.read_csv(osp.join(self.raw_dir, 'movies.csv'))
            movies.rename(columns={'movieId': 'MovieID', 'title': 'Title', 'genres': 'Genres'}, inplace=True)
            
            ratings = pd.read_csv(osp.join(self.raw_dir, 'ratings.csv'))
            ratings.rename(columns={
                'userId': 'UserID', 
                'movieId': 'MovieID', 
                'rating': 'Rating', 
                'timestamp': 'Timestamp'
            }, inplace=True)
            
            # ML-32M没有用户信息，创建虚拟用户数据
            unique_users = ratings['UserID'].unique()
            users = pd.DataFrame({
                'UserID': unique_users,
                'Gender': 'M',  # 默认值
                'Age': 25,      # 默认值
                'Occupation': 0, # 默认值
                'Zip-code': '00000'  # 默认值
            })
        
        return movies, ratings, users

    def _create_item_features(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> torch.Tensor:
        """创建物品特征向量"""
        logger.info("🎬 创建物品特征...")
        
        # 确保有足够的电影
        movie_ids = movies['MovieID'].unique()
        n_movies = len(movie_ids)
        
        # 创建电影ID到索引的映射
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(sorted(movie_ids))}
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}
        
        features = []
        
        for movie_id in tqdm(sorted(movie_ids), desc="处理电影特征"):
            movie_row = movies[movies['MovieID'] == movie_id].iloc[0]
            movie_ratings = ratings[ratings['MovieID'] == movie_id]
            
            feature_vector = []
            
            # 1. 类型特征 (多热编码，取前8个类型)
            genres = movie_row['Genres'].split('|') if pd.notna(movie_row['Genres']) else []
            all_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                         'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                         'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                         'Thriller', 'War', 'Western']
            
            # 取前8个类型作为特征
            genre_features = [1.0 if genre in genres else 0.0 for genre in all_genres[:8]]
            feature_vector.extend(genre_features)
            
            # 2. 统计特征 (10个特征)
            if len(movie_ratings) > 0:
                feature_vector.extend([
                    len(movie_ratings),  # 评分数量
                    movie_ratings['Rating'].mean(),  # 平均评分
                    movie_ratings['Rating'].std() if len(movie_ratings) > 1 else 0.0,  # 评分标准差
                    movie_ratings['Rating'].min(),  # 最低评分
                    movie_ratings['Rating'].max(),  # 最高评分
                    (movie_ratings['Rating'] >= 4).sum(),  # 高评分数量
                    (movie_ratings['Rating'] <= 2).sum(),  # 低评分数量
                    movie_ratings['Rating'].median(),  # 中位数评分
                    movie_ratings['Rating'].quantile(0.25),  # 25%分位数
                    movie_ratings['Rating'].quantile(0.75),  # 75%分位数
                ])
            else:
                feature_vector.extend([0.0] * 10)
            
            features.append(feature_vector)
        
        # 转换为张量
        features = torch.tensor(features, dtype=torch.float32)
        
        # 如果特征维度不等于input_dim，进行调整
        if features.shape[1] != self.input_dim:
            if features.shape[1] > self.input_dim:
                # 截断
                features = features[:, :self.input_dim]
            else:
                # 填充
                padding = torch.zeros(features.shape[0], self.input_dim - features.shape[1])
                features = torch.cat([features, padding], dim=1)
        
        # 标准化
        if self.normalize_features:
            scaler = StandardScaler()
            features = torch.tensor(
                scaler.fit_transform(features.numpy()), 
                dtype=torch.float32
            )
        
        logger.info(f"✅ 物品特征维度: {features.shape}")
        return features

    def _split_data(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """分割数据集"""
        # 过滤交互数不足的用户
        user_counts = ratings['UserID'].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        ratings_filtered = ratings[ratings['UserID'].isin(valid_users)]
        
        if self.train_test_split == "all":
            return ratings_filtered
        
        # 用户级别分割
        np.random.seed(self.random_state)
        unique_users = valid_users.values
        shuffled_users = np.random.permutation(unique_users)
        
        n_users = len(shuffled_users)
        n_test = int(n_users * self.test_ratio)
        n_val = int(n_users * self.val_ratio)
        
        if self.train_test_split == "train":
            target_users = shuffled_users[:-n_test-n_val]
        elif self.train_test_split == "eval":
            target_users = shuffled_users[-n_test-n_val:-n_test]
        elif self.train_test_split == "test":
            target_users = shuffled_users[-n_test:]
        else:
            raise ValueError(f"不支持的分割类型: {self.train_test_split}")
        
        return ratings_filtered[ratings_filtered['UserID'].isin(target_users)]

    def _process(self):
        """处理数据的主函数"""
        # 加载原始数据
        movies, ratings, users = self._load_movielens_data()
        
        logger.info(f"原始数据: {len(movies)} 电影, {len(ratings)} 评分, {len(users)} 用户")
        
        # 创建物品特征
        item_features = self._create_item_features(movies, ratings)
        
        # 分割数据
        split_ratings = self._split_data(ratings)
        
        # 创建物品交互数据
        items = []
        for movie_id in tqdm(sorted(movies['MovieID'].unique()), desc="创建物品数据"):
            movie_ratings = split_ratings[split_ratings['MovieID'] == movie_id]
            if len(movie_ratings) > 0:
                movie_idx = self.movie_id_to_idx[movie_id]
                
                # 物品的用户交互信息
                item_data = {
                    'movie_id': movie_id,
                    'movie_idx': movie_idx,
                    'features': item_features[movie_idx],
                    'user_ratings': movie_ratings[['UserID', 'Rating']].values,
                    'n_ratings': len(movie_ratings),
                    'avg_rating': movie_ratings['Rating'].mean()
                }
                items.append(item_data)
        
        # 保存处理后的数据
        processed_data = {
            'items': items,
            'item_features': item_features,
            'metadata': {
                'n_movies': len(movies),
                'n_users': len(users),
                'n_ratings': len(split_ratings),
                'input_dim': self.input_dim,
                'split': self.train_test_split,
                'movie_id_to_idx': self.movie_id_to_idx,
                'idx_to_movie_id': self.idx_to_movie_id
            }
        }
        
        torch.save(processed_data, self.processed_file)
        logger.info(f"✅ 数据处理完成，保存至: {self.processed_file}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """获取单个样本 - 返回物品特征向量"""
        if isinstance(idx, torch.Tensor):
            # 如果idx是张量，处理批量索引
            return self.item_features[idx]
        else:
            # 单个索引
            return self.item_features[idx]

    def get_item_by_movie_id(self, movie_id: int) -> Optional[torch.Tensor]:
        """根据电影ID获取特征"""
        if movie_id in self.movie_id_to_idx:
            idx = self.movie_id_to_idx[movie_id]
            return self.item_features[idx]
        return None

    def get_all_features(self) -> torch.Tensor:
        """获取所有物品特征"""
        return self.item_features

    def get_random_batch(self, batch_size: int) -> torch.Tensor:
        """获取随机批次"""
        indices = torch.randint(0, len(self), (batch_size,))
        return self.item_features[indices]


# 数据工具函数 - 与训练代码兼容
def batch_to(batch, device):
    """将批次数据移动到指定设备"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return [batch_to(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {key: batch_to(value, device) for key, value in batch.items()}
    else:
        return batch

def cycle(dataloader):
    """无限循环数据加载器"""
    while True:
        for batch in dataloader:
            yield batch

def next_batch(dataloader, device):
    """获取下一个批次并移动到设备"""
    batch = next(dataloader)
    return batch_to(batch, device)


# 使用示例
if __name__ == "__main__":
    # 创建数据集
    dataset = ItemData(
        root="dataset/ml-1m",
        dataset=RecDataset.ML_1M,
        train_test_split="train",
        input_dim=18,
        force_process=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"特征维度: {dataset.item_features.shape}")
    print(f"样本特征: {dataset[0]}")