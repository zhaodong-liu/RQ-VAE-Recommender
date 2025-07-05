import gin
import os
import random
import torch

from data.amazon import AmazonReviews
from data.ml1m import RawMovieLens1M
from data.ml32m import RawMovieLens32M
from data.schemas import SeqBatch
from enum import Enum
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional

PROCESSED_MOVIE_LENS_SUFFIX = "/processed/data.pt"


@gin.constants_from_enum
class RecDataset(Enum):
    AMAZON = 1
    ML_1M = 2
    ML_32M = 3


DATASET_NAME_TO_RAW_DATASET = {
    RecDataset.AMAZON: AmazonReviews,
    RecDataset.ML_1M: RawMovieLens1M,
    RecDataset.ML_32M: RawMovieLens32M
}


DATASET_NAME_TO_MAX_SEQ_LEN = {
    RecDataset.AMAZON: 20,
    RecDataset.ML_1M: 20,
    RecDataset.ML_32M: 50
}


class ItemData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        train_test_split: str = "all",
        **kwargs
    ) -> None:
        
        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

        raw_data = raw_dataset_class(root=root, *args, **kwargs)
        
        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)
        
        if train_test_split == "train":
            filt = raw_data.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_data.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones_like(raw_data.data["item"]["x"][:,0], dtype=bool)

        self.item_data, self.item_text = raw_data.data["item"]["x"][filt], raw_data.data["item"]["text"][filt]

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_data[idx, :768]
        
        # 确保所有返回值的形状都与Amazon SeqData一致
        return SeqBatch(
            user_ids=torch.tensor([-1]),  # 形状 (1,)
            ids=item_ids,                 # 形状 (1,)
            ids_fut=torch.tensor([-1]),   # 形状 (1,) 
            x=x,                          # 形状 (768,)
            x_fut=torch.full((1, 768), -1.0, dtype=torch.float32),  # 形状 (1, 768), float32类型
            seq_mask=torch.ones_like(item_ids, dtype=bool)  # 形状 (1,)
        )


class SeqData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        is_train: bool = True,
        subsample: bool = False,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        **kwargs
    ) -> None:
        
        assert (not subsample) or is_train, "Can only subsample on training split."

        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

        raw_data = raw_dataset_class(root=root, *args, **kwargs)

        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)

        split = "train" if is_train else "test"
        self.subsample = subsample
        self.dataset = dataset
        self.sequence_data = raw_data.data[("user", "rated", "item")]["history"][split]

        # 统一数据格式处理
        if not self.subsample:
            # 确保所有数据集的itemId都是张量格式，并且形状一致
            if isinstance(self.sequence_data["itemId"], list):
                # 对于list格式（如ML数据集），转换为张量
                self.sequence_data["itemId"] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(l[-max_seq_len:]) for l in self.sequence_data["itemId"]],
                    batch_first=True,
                    padding_value=-1
                )

        self._max_seq_len = max_seq_len
        self.item_data = raw_data.data["item"]["x"]
        self.split = split
    
    @property
    def max_seq_len(self):
        return self._max_seq_len

    def __len__(self):
        return self.sequence_data["userId"].shape[0]
  
    def __getitem__(self, idx):
        # 统一处理 user_ids - 确保所有数据集都返回形状为 (1,) 的张量
        user_ids_raw = self.sequence_data["userId"][idx]
        if isinstance(user_ids_raw, torch.Tensor):
            if user_ids_raw.dim() == 0:  # 标量张量
                user_ids = user_ids_raw.unsqueeze(0)  # 转换为 (1,)
            else:
                user_ids = user_ids_raw
        else:
            user_ids = torch.tensor([user_ids_raw])  # 确保是 (1,) 形状
        
        if self.subsample:
            # 获取完整序列 - 确保所有操作都使用 list
            seq_items = self.sequence_data["itemId"][idx] 
            fut_item = self.sequence_data["itemId_fut"][idx]
            
            # 转换为 list 并合并
            if isinstance(seq_items, torch.Tensor):
                seq_items = seq_items.tolist()
            elif isinstance(seq_items, list):
                # 如果是嵌套列表，则取第一个元素
                if len(seq_items) > 0 and isinstance(seq_items[0], list):
                    seq_items = seq_items[0]
            else:
                seq_items = list(seq_items)
                
            if isinstance(fut_item, torch.Tensor):
                if fut_item.dim() == 0:  # 标量张量
                    fut_item = fut_item.item()
                else:
                    fut_item = fut_item.tolist()
                    if isinstance(fut_item, list) and len(fut_item) == 1:
                        fut_item = fut_item[0]
            elif isinstance(fut_item, list):
                if len(fut_item) == 1:
                    fut_item = fut_item[0]
            
            # 创建完整序列
            seq = seq_items + [fut_item]
            
            # 随机采样子序列
            start_idx = random.randint(0, max(0, len(seq)-3))
            end_idx = random.randint(start_idx+3, start_idx+self.max_seq_len+1)
            sample = seq[start_idx:end_idx]
            
            # 创建 item_ids 和 item_ids_fut
            item_ids_list = sample[:-1] + [-1] * (self.max_seq_len - len(sample[:-1]))
            item_ids = torch.tensor(item_ids_list)
            item_ids_fut = torch.tensor([sample[-1]])  # 确保是 (1,) 形状

        else:
            item_ids = self.sequence_data["itemId"][idx]
            item_ids_fut_raw = self.sequence_data["itemId_fut"][idx]
            
            # 统一处理 item_ids_fut - 确保形状为 (1,)
            if isinstance(item_ids_fut_raw, torch.Tensor):
                if item_ids_fut_raw.dim() == 0:  # 标量张量
                    item_ids_fut = item_ids_fut_raw.unsqueeze(0)  # 转换为 (1,)
                else:
                    item_ids_fut = item_ids_fut_raw
            else:
                item_ids_fut = torch.tensor([item_ids_fut_raw])  # 确保是 (1,) 形状
        
        # 添加严格的数据验证
        max_item_id = self.item_data.shape[0] - 1
        
        # 验证item_ids: 确保所有非-1的ID都在有效范围内
        valid_ids_mask = (item_ids >= 0) & (item_ids <= max_item_id)
        invalid_ids_mask = (item_ids >= 0) & (item_ids > max_item_id)
        if invalid_ids_mask.any():
            print(f"Warning: Found invalid item IDs: {item_ids[invalid_ids_mask]}, max allowed: {max_item_id}")
            item_ids[invalid_ids_mask] = -1  # 将无效ID设为-1
        
        # 验证item_ids_fut
        valid_fut_mask = (item_ids_fut >= 0) & (item_ids_fut <= max_item_id)
        invalid_fut_mask = (item_ids_fut >= 0) & (item_ids_fut > max_item_id)
        if invalid_fut_mask.any():
            print(f"Warning: Found invalid future item IDs: {item_ids_fut[invalid_fut_mask]}, max allowed: {max_item_id}")
            item_ids_fut[invalid_fut_mask] = -1  # 将无效ID设为-1
        
        assert (item_ids >= -1).all(), f"Invalid movie id found: min={item_ids.min()}, max={item_ids.max()}"
        assert (item_ids_fut >= -1).all(), f"Invalid future movie id found: min={item_ids_fut.min()}, max={item_ids_fut.max()}"
        
        x = self.item_data[item_ids, :768]
        x[item_ids == -1] = -1

        # 统一处理 x_fut - 确保形状为 (1, 768) 和正确的数据类型
        x_fut = self.item_data[item_ids_fut, :768]
        x_fut[item_ids_fut == -1] = -1.0  # 确保是浮点数
        if x_fut.dim() == 1:  # 如果是 (768,)，扩展为 (1, 768)
            x_fut = x_fut.unsqueeze(0)
        
        # 确保x_fut是float32类型，与x保持一致
        x_fut = x_fut.float()

        return SeqBatch(
            user_ids=user_ids,
            ids=item_ids,
            ids_fut=item_ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=(item_ids >= 0)
        )


if __name__ == "__main__":
    dataset = ItemData("dataset/amazon", dataset=RecDataset.AMAZON, split="beauty", force_process=True)
    dataset[0]
    import pdb; pdb.set_trace()