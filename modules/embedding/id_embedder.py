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
    
    # def forward(self, batch: TokenizedSeqBatch) -> SemIdEmbeddingBatch:
    #     # 确保 token_type_ids 在有效范围内
    #     token_type_ids_clamped = torch.clamp(batch.token_type_ids, 0, self.sem_ids_dim - 1)
        
    #     # 计算实际的embedding索引
    #     sem_ids = token_type_ids_clamped * self.num_embeddings + batch.sem_ids
        
    #     # 检查并修复越界的索引
    #     max_valid_idx = self.num_embeddings * self.sem_ids_dim - 1
        
    #     # 将超出范围的索引设为padding_idx
    #     sem_ids = torch.where(
    #         (batch.sem_ids >= 0) & (batch.sem_ids < self.num_embeddings) & 
    #         (token_type_ids_clamped >= 0) & (token_type_ids_clamped < self.sem_ids_dim),
    #         sem_ids,
    #         self.padding_idx
    #     )
        
    #     # 对于无效位置，使用padding_idx
    #     sem_ids[~batch.seq_mask] = self.padding_idx

    #     if batch.sem_ids_fut is not None:
    #         # 同样处理future token_type_ids
    #         token_type_ids_fut_clamped = torch.clamp(batch.token_type_ids_fut, 0, self.sem_ids_dim - 1)
    #         sem_ids_fut = token_type_ids_fut_clamped * self.num_embeddings + batch.sem_ids_fut
            
    #         # 同样检查future语义ID的边界
    #         sem_ids_fut = torch.where(
    #             (batch.sem_ids_fut >= 0) & (batch.sem_ids_fut < self.num_embeddings) &
    #             (token_type_ids_fut_clamped >= 0) & (token_type_ids_fut_clamped < self.sem_ids_dim),
    #             sem_ids_fut,
    #             self.padding_idx
    #         )
            
    #         sem_ids_fut = self.emb(sem_ids_fut)
    #     else:
    #         sem_ids_fut = None
            
    #     return SemIdEmbeddingBatch(
    #         seq=self.emb(sem_ids),
    #         fut=sem_ids_fut
    #     ) 

    def forward(self, batch: TokenizedSeqBatch) -> SemIdEmbeddingBatch:
        print(f"\n🔍 SemIdEmbedder调试:")
        print(f"  self.num_embeddings: {self.num_embeddings}")
        print(f"  self.sem_ids_dim: {self.sem_ids_dim}")
        print(f"  self.padding_idx: {self.padding_idx}")
        
        # 详细检查每个输入tensor
        print(f"  输入检查:")
        print(f"    sem_ids: shape={batch.sem_ids.shape}, range=[{batch.sem_ids.min()}, {batch.sem_ids.max()}]")
        print(f"    token_type_ids: shape={batch.token_type_ids.shape}, range=[{batch.token_type_ids.min()}, {batch.token_type_ids.max()}]")
        
        # 检查边界条件
        invalid_sem_ids = (batch.sem_ids >= self.num_embeddings) & (batch.sem_ids != -1)
        if invalid_sem_ids.any():
            print(f"    ❌ 发现无效sem_ids: {batch.sem_ids[invalid_sem_ids].unique()}")
            
        invalid_token_type = batch.token_type_ids >= self.sem_ids_dim
        if invalid_token_type.any():
            print(f"    ❌ 发现无效token_type_ids: {batch.token_type_ids[invalid_token_type].unique()}")

        # 夹取边界
        token_type_ids_clamped = torch.clamp(batch.token_type_ids, 0, self.sem_ids_dim - 1)
        
        # 计算embedding索引
        sem_ids = token_type_ids_clamped * self.num_embeddings + batch.sem_ids
        
        # 检查计算后的索引
        max_valid_idx = self.num_embeddings * self.sem_ids_dim - 1
        print(f"  计算后的索引:")
        print(f"    sem_ids range: [{sem_ids.min()}, {sem_ids.max()}]")
        print(f"    max_valid_idx: {max_valid_idx}")
        
        # 检查最终索引是否越界
        final_invalid = (sem_ids > max_valid_idx) | (sem_ids < 0)
        final_invalid = final_invalid & (batch.sem_ids != -1)  # 排除padding
        
        if final_invalid.any():
            print(f"    ❌ 最终索引越界: {sem_ids[final_invalid].unique()}")
            sem_ids = torch.where(final_invalid, self.padding_idx, sem_ids)
            print(f"    🔧 已修复为padding_idx")

        # 处理seq_mask
        if batch.seq_mask is not None:
            sem_ids[~batch.seq_mask] = self.padding_idx

        # 处理future数据
        sem_ids_fut = None
        if batch.sem_ids_fut is not None:
            print(f"  处理future数据:")
            print(f"    sem_ids_fut: shape={batch.sem_ids_fut.shape}, range=[{batch.sem_ids_fut.min()}, {batch.sem_ids_fut.max()}]")
            print(f"    token_type_ids_fut: shape={batch.token_type_ids_fut.shape}, range=[{batch.token_type_ids_fut.min()}, {batch.token_type_ids_fut.max()}]")
            
            # 同样的检查和修复流程
            invalid_fut_sem = (batch.sem_ids_fut >= self.num_embeddings) & (batch.sem_ids_fut != -1)
            if invalid_fut_sem.any():
                print(f"    ❌ 发现无效sem_ids_fut: {batch.sem_ids_fut[invalid_fut_sem].unique()}")
                
            invalid_fut_type = batch.token_type_ids_fut >= self.sem_ids_dim
            if invalid_fut_type.any():
                print(f"    ❌ 发现无效token_type_ids_fut: {batch.token_type_ids_fut[invalid_fut_type].unique()}")

            token_type_ids_fut_clamped = torch.clamp(batch.token_type_ids_fut, 0, self.sem_ids_dim - 1)
            sem_ids_fut_computed = token_type_ids_fut_clamped * self.num_embeddings + batch.sem_ids_fut
            
            print(f"    计算后sem_ids_fut range: [{sem_ids_fut_computed.min()}, {sem_ids_fut_computed.max()}]")
            
            fut_final_invalid = (sem_ids_fut_computed > max_valid_idx) | (sem_ids_fut_computed < 0)
            fut_final_invalid = fut_final_invalid & (batch.sem_ids_fut != -1)
            
            if fut_final_invalid.any():
                print(f"    ❌ future最终索引越界: {sem_ids_fut_computed[fut_final_invalid].unique()}")
                sem_ids_fut_computed = torch.where(fut_final_invalid, self.padding_idx, sem_ids_fut_computed)
                print(f"    🔧 已修复future索引")
            
            try:
                sem_ids_fut = self.emb(sem_ids_fut_computed)
                print(f"    ✅ future embedding成功")
            except Exception as e:
                print(f"    ❌ future embedding失败: {e}")
                raise e

        # 主要的embedding查找
        try:
            print(f"  执行主embedding查找...")
            seq_emb = self.emb(sem_ids)
            print(f"    ✅ 主embedding成功, shape: {seq_emb.shape}")
        except Exception as e:
            print(f"    ❌ 主embedding失败: {e}")
            print(f"    sem_ids详细信息: min={sem_ids.min()}, max={sem_ids.max()}, shape={sem_ids.shape}")
            print(f"    embedding层大小: {self.emb.num_embeddings}")
            raise e

        return SemIdEmbeddingBatch(seq=seq_emb, fut=sem_ids_fut)

    

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