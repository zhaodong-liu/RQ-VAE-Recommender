import torch

from data.schemas import TokenizedSeqBatch
from torch import nn
from torch import Tensor
from typing import NamedTuple


class SemIdEmbeddingBatch(NamedTuple):
    seq: Tensor
    fut: Tensor


class SemIdEmbedder(nn.Module):
    def __init__(self, num_embeddings, sem_ids_dim, embeddings_dim) -> None:
        super().__init__()
        
        self.sem_ids_dim = sem_ids_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = sem_ids_dim*num_embeddings
        
        self.emb = nn.Embedding(
            num_embeddings=num_embeddings*self.sem_ids_dim+1,
            embedding_dim=embeddings_dim,
            padding_idx=self.padding_idx
        )
    
    def forward(self, batch: TokenizedSeqBatch) -> SemIdEmbeddingBatch:
        # 确保 token_type_ids 在有效范围内
        token_type_ids_clamped = torch.clamp(batch.token_type_ids, 0, self.sem_ids_dim - 1)
        
        # 计算实际的embedding索引
        sem_ids = token_type_ids_clamped * self.num_embeddings + batch.sem_ids
        
        # 检查并修复越界的索引
        max_valid_idx = self.num_embeddings * self.sem_ids_dim - 1
        
        # 将超出范围的索引设为padding_idx
        sem_ids = torch.where(
            (batch.sem_ids >= 0) & (batch.sem_ids < self.num_embeddings) & 
            (token_type_ids_clamped >= 0) & (token_type_ids_clamped < self.sem_ids_dim),
            sem_ids,
            self.padding_idx
        )
        
        # 对于无效位置，使用padding_idx
        sem_ids[~batch.seq_mask] = self.padding_idx

        if batch.sem_ids_fut is not None:
            # 同样处理future token_type_ids
            token_type_ids_fut_clamped = torch.clamp(batch.token_type_ids_fut, 0, self.sem_ids_dim - 1)
            sem_ids_fut = token_type_ids_fut_clamped * self.num_embeddings + batch.sem_ids_fut
            
            # 同样检查future语义ID的边界
            sem_ids_fut = torch.where(
                (batch.sem_ids_fut >= 0) & (batch.sem_ids_fut < self.num_embeddings) &
                (token_type_ids_fut_clamped >= 0) & (token_type_ids_fut_clamped < self.sem_ids_dim),
                sem_ids_fut,
                self.padding_idx
            )
            
            sem_ids_fut = self.emb(sem_ids_fut)
        else:
            sem_ids_fut = None
            
        return SemIdEmbeddingBatch(
            seq=self.emb(sem_ids),
            fut=sem_ids_fut
        ) 
    

class UserIdEmbedder(nn.Module):
    # TODO: Implement hashing trick embedding for user id
    def __init__(self, num_buckets, embedding_dim) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, embedding_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        hashed_indices = x % self.num_buckets
        # hashed_indices = torch.tensor([hash(token) % self.num_buckets for token in x], device=x.device)
        return self.emb(hashed_indices)