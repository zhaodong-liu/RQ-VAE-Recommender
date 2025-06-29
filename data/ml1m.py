import pandas as pd
import torch
import numpy as np

from data.preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData
from torch_geometric.datasets import MovieLens1M


class RawMovieLens1M(MovieLens1M, PreprocessingMixin):
    MOVIE_HEADERS = ["movieId", "title", "genres"]
    USER_HEADERS = ["userId", "gender", "age", "occupation", "zipCode"]
    RATING_HEADERS = ['userId', 'movieId', 'rating', 'timestamp']

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        force_reload=False,
        split=None
    ) -> None:
        super(RawMovieLens1M, self).__init__(
            root, transform, pre_transform, force_reload
        )

    def _load_ratings(self):
        return pd.read_csv(
            self.raw_paths[2],
            sep='::',
            header=None,
            names=self.RATING_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )
    
    def process(self, max_seq_len=200) -> None:
        data = HeteroData()
        ratings_df = self._load_ratings()

        # Process movie data (Amazon-style)
        full_df = pd.read_csv(
            self.raw_paths[0],
            sep='::',
            header=None,
            index_col='movieId',
            names=self.MOVIE_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )
        df = self._remove_low_occurrence(ratings_df, full_df, "movieId")
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        # Process titles and create Amazon-style text descriptions
        titles_text = df["title"].apply(lambda s: s.split("(")[0].strip()).tolist()
        
        # Create Amazon-style text descriptions
        sentences = df.apply(
            lambda row: f"Title: {row['title'].split('(')[0].strip()}; Genres: {row['genres']};",
            axis=1
        ).tolist()
        
        # Create embeddings (Amazon-style: only text embeddings)
        titles_emb = self._encode_text_feature(sentences)
        
        # Store item features (Amazon-style format)
        data['item'].x = titles_emb  # Only use text embeddings
        data['item'].text = np.array(sentences)  # Amazon-style text storage
        
        # Create Amazon-style is_train split (95% train, 5% test)
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(titles_emb.shape[0], generator=gen) > 0.05

        # Process user data for mapping only
        user_df = pd.read_csv(
            self.raw_paths[1],
            sep='::',
            header=None,
            index_col='userId',
            names=self.USER_HEADERS,
            dtype='str',
            encoding='ISO-8859-1',
            engine='python',
        )
        user_df = self._remove_low_occurrence(ratings_df, user_df, "userId")
        user_mapping = {idx: i for i, idx in enumerate(user_df.index)}

        # Process rating data for sequence generation only
        filtered_ratings = self._remove_low_occurrence(
            ratings_df,
            ratings_df,
            ["userId", "movieId"]
        )
        
        # 直接创建兼容的历史数据格式，避免_generate_user_history的复杂性
        # 按用户分组并按时间排序创建序列
        user_sequences = []
        for user_id in filtered_ratings['userId'].unique():
            if user_id in user_mapping:
                user_ratings = filtered_ratings[filtered_ratings['userId'] == user_id].sort_values('timestamp')
                items = [movie_mapping[mid] for mid in user_ratings['movieId'] if mid in movie_mapping]
                
                if len(items) >= 3:  # 需要至少3个交互
                    user_sequences.append({
                        'mapped_user_id': user_mapping[user_id],
                        'items': items
                    })
        
        # 按时间分割序列 (80% train, 20% eval)
        train_data = {'userId': [], 'itemId': [], 'itemId_fut': []}
        eval_data = {'userId': [], 'itemId': [], 'itemId_fut': []}
        
        for seq_data in user_sequences:
            user_id = seq_data['mapped_user_id']
            items = seq_data['items']
            
            # 80%时间点作为分割点
            split_point = int(len(items) * 0.8)
            split_point = max(1, min(split_point, len(items) - 2))  # 确保有效分割
            
            # 训练序列：前split_point个物品
            train_items = items[:split_point]
            if len(train_items) > 0:
                # 截断或填充到max_seq_len
                if len(train_items) > max_seq_len:
                    train_items = train_items[-max_seq_len:]
                else:
                    train_items = train_items + [-1] * (max_seq_len - len(train_items))
                
                train_data['userId'].append(user_id)
                train_data['itemId'].append(train_items)
                train_data['itemId_fut'].append(-1)  # 训练时无目标
            
            # 评估序列：从split_point开始的物品
            eval_items = items[split_point:]
            if len(eval_items) >= 2:  # 需要至少2个物品（序列+目标）
                eval_seq = eval_items[:-1]  # 序列部分
                eval_target = eval_items[-1]  # 目标部分
                
                # 截断或填充序列
                if len(eval_seq) > max_seq_len:
                    eval_seq = eval_seq[-max_seq_len:]
                else:
                    eval_seq = eval_seq + [-1] * (max_seq_len - len(eval_seq))
                
                eval_data['userId'].append(user_id)
                eval_data['itemId'].append(eval_seq)
                eval_data['itemId_fut'].append(eval_target)
        
        # 转换为张量格式
        history = {}
        
        # 训练集
        if len(train_data['userId']) > 0:
            history["train"] = {
                "userId": torch.tensor(train_data['userId'], dtype=torch.long),
                "itemId": torch.tensor(train_data['itemId'], dtype=torch.long),
                "itemId_fut": torch.tensor(train_data['itemId_fut'], dtype=torch.long),
            }
        else:
            history["train"] = {
                "userId": torch.tensor([], dtype=torch.long),
                "itemId": torch.tensor([], dtype=torch.long).reshape(0, max_seq_len),
                "itemId_fut": torch.tensor([], dtype=torch.long),
            }
        
        # 将评估数据随机分为eval和test
        if len(eval_data['userId']) > 0:
            eval_size = len(eval_data['userId'])
            indices = torch.randperm(eval_size)
            split_point = max(1, eval_size // 2)
            
            eval_indices = indices[:split_point].tolist()
            test_indices = indices[split_point:].tolist()
            
            # eval分割
            history["eval"] = {
                "userId": torch.tensor([eval_data['userId'][i] for i in eval_indices], dtype=torch.long),
                "itemId": torch.tensor([eval_data['itemId'][i] for i in eval_indices], dtype=torch.long),
                "itemId_fut": torch.tensor([eval_data['itemId_fut'][i] for i in eval_indices], dtype=torch.long),
            }
            
            # test分割
            history["test"] = {
                "userId": torch.tensor([eval_data['userId'][i] for i in test_indices], dtype=torch.long),
                "itemId": torch.tensor([eval_data['itemId'][i] for i in test_indices], dtype=torch.long),
                "itemId_fut": torch.tensor([eval_data['itemId_fut'][i] for i in test_indices], dtype=torch.long),
            }
        else:
            # 空的eval和test
            history["eval"] = {
                "userId": torch.tensor([], dtype=torch.long),
                "itemId": torch.tensor([], dtype=torch.long).reshape(0, max_seq_len),
                "itemId_fut": torch.tensor([], dtype=torch.long),
            }
            history["test"] = {
                "userId": torch.tensor([], dtype=torch.long),
                "itemId": torch.tensor([], dtype=torch.long).reshape(0, max_seq_len),
                "itemId_fut": torch.tensor([], dtype=torch.long),
            }
        
        data["user", "rated", "item"].history = history

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])