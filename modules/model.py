import gin
import torch

from einops import rearrange
from enum import Enum
from data.schemas import TokenizedSeqBatch
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.model import TransformerDecoder
from modules.transformer.model import TransformerEncoderDecoder
from modules.utils import eval_mode
from modules.utils import maybe_repeat_interleave
from modules.utils import reset_encoder_cache
from modules.utils import reset_kv_cache
from modules.utils import select_columns_per_row
from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F

# 禁用动态编译以避免Triton错误
# torch._dynamo.config.suppress_errors = True
# torch.set_float32_matmul_precision('high')


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


class EncoderDecoderRetrievalModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_id_dim,
        inference_verifier_fn,
        max_pos=2048,
        jagged_mode: bool = True,
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False

        self.bos_emb = nn.Parameter(torch.rand(embedding_dim))
        self.norm = RMSNorm(embedding_dim)
        self.norm_cxt = RMSNorm(embedding_dim)
        self.do = nn.Dropout(p=0.5)

        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim
        )
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        
        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)
        self.tte_fut = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)

        self.transformer = TransformerEncoderDecoder(
            d_in=attn_dim,
            d_out=attn_dim,
            dropout=dropout,
            num_heads=num_heads,
            encoder_layers=n_layers // 2,
            decoder_layers=n_layers // 2
        ) if self.jagged_mode else nn.Transformer(
            d_model=attn_dim,
            nhead=num_heads,
            num_encoder_layers=n_layers // 2,
            num_decoder_layers=n_layers // 2,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)
    
    # def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
    #     user_emb = self.user_id_embedder(batch.user_ids)
    #     sem_ids_emb = self.sem_id_embedder(batch)
    #     sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
    #     seq_lengths = batch.seq_mask.sum(axis=1)
        
    #     B, N, D = sem_ids_emb.shape

    #     pos_max = N // self.sem_id_dim
    #     # pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat_interleave(self.sem_id_dim)
          
    #     pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
    #     wpe = self.wpe(pos)

    #     input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
    #     input_embedding_fut = self.bos_emb.repeat(B, 1, 1)
    #     if sem_ids_emb_fut is not None:
    #         tte_fut = self.tte(batch.token_type_ids_fut)
    #         input_embedding_fut = torch.cat([
    #             input_embedding_fut, 
    #             sem_ids_emb_fut + tte_fut
    #             ], axis=1
    #         )

    #     if self.jagged_mode:
    #         input_embedding = padded_to_jagged_tensor(input_embedding, lengths=seq_lengths+1, max_len=input_embedding.shape[1])

    #         seq_lengths_fut = torch.tensor(input_embedding_fut.shape[1], device=input_embedding_fut.device, dtype=torch.int64).repeat(B)
    #         input_embedding_fut = padded_to_jagged_tensor(input_embedding_fut, lengths=seq_lengths_fut, max_len=input_embedding_fut.shape[1])
    #     else:
    #         mem_mask = torch.cat([
    #             torch.ones(B, 1, dtype=torch.bool, device=batch.seq_mask.device),
    #             batch.seq_mask
    #         ], axis=1)
    #         f_mask = torch.zeros_like(mem_mask, dtype=torch.float32)
    #         f_mask[~mem_mask] = float("-inf")
        
    #     transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
    #     transformer_input = self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))
        
    #     if self.jagged_mode:
    #         transformer_output = self.transformer(x=transformer_input, context=transformer_context, padding_mask=batch.seq_mask, jagged=self.jagged_mode)
    #     else:
    #         causal_mask = nn.Transformer.generate_square_subsequent_mask(transformer_input.shape[1])
    #         transformer_output = self.transformer(src=transformer_context, tgt=transformer_input, tgt_is_causal=True, tgt_mask=causal_mask, src_key_padding_mask=f_mask, memory_key_padding_mask=f_mask)

    #     return transformer_output
    def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
        # print(f"\n🔍 _predict 方法开始调试:")
        # print(f"  输入batch信息:")
        # print(f"    user_ids: {batch.user_ids.shape if batch.user_ids is not None else None}")
        # print(f"    sem_ids: {batch.sem_ids.shape}, range=[{batch.sem_ids.min()}, {batch.sem_ids.max()}]")
        # print(f"    sem_ids_fut: {batch.sem_ids_fut.shape if batch.sem_ids_fut is not None else None}")
        # if batch.sem_ids_fut is not None:
        #     print(f"    sem_ids_fut range: [{batch.sem_ids_fut.min()}, {batch.sem_ids_fut.max()}]")
        # print(f"    token_type_ids: {batch.token_type_ids.shape}, range=[{batch.token_type_ids.min()}, {batch.token_type_ids.max()}]")
        # print(f"    seq_mask: {batch.seq_mask.shape}, sum={batch.seq_mask.sum()}")

        # # Step 1: User embedding
        # try:
        #     print(f"  Step 1: User embedding...")
        #     user_emb = self.user_id_embedder(batch.user_ids)
        #     print(f"    ✅ user_emb shape: {user_emb.shape}")
        # except Exception as e:
        #     print(f"    ❌ User embedding失败: {e}")
        #     raise e

        # # Step 2: Semantic ID embedding
        # try:
        #     print(f"  Step 2: Semantic ID embedding...")
        #     sem_ids_emb = self.sem_id_embedder(batch)
        #     sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
        #     print(f"    ✅ sem_ids_emb shape: {sem_ids_emb.shape}")
        #     if sem_ids_emb_fut is not None:
        #         print(f"    ✅ sem_ids_emb_fut shape: {sem_ids_emb_fut.shape}")
        # except Exception as e:
        #     print(f"    ❌ Semantic ID embedding失败: {e}")
        #     raise e

        # seq_lengths = batch.seq_mask.sum(axis=1)
        # B, N, D = sem_ids_emb.shape
        # print(f"  Tensor shapes: B={B}, N={N}, D={D}")

        # # Step 3: Position encoding
        # try:
        #     print(f"  Step 3: Position encoding...")
        #     pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
        #     print(f"    pos shape: {pos.shape}, range=[{pos.min()}, {pos.max()}]")
            
        #     # 检查position embedding的范围
        #     if pos.max() >= self.wpe.num_embeddings:
        #         print(f"    ❌ Position超出范围: {pos.max()} >= {self.wpe.num_embeddings}")
        #         raise RuntimeError(f"Position index {pos.max()} >= {self.wpe.num_embeddings}")
            
        #     wpe = self.wpe(pos)
        #     print(f"    ✅ wpe shape: {wpe.shape}")
        # except Exception as e:
        #     print(f"    ❌ Position encoding失败: {e}")
        #     raise e

        # # Step 4: Input embedding construction
        # try:
        #     print(f"  Step 4: Input embedding construction...")
        #     input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
        #     print(f"    ✅ input_embedding shape: {input_embedding.shape}")
            
        #     input_embedding_fut = self.bos_emb.repeat(B, 1, 1)
        #     print(f"    ✅ input_embedding_fut initial shape: {input_embedding_fut.shape}")
            
        #     if sem_ids_emb_fut is not None:
        #         # 检查token_type_ids_fut的范围
        #         if batch.token_type_ids_fut.max() >= self.tte_fut.num_embeddings:
        #             print(f"    ❌ token_type_ids_fut超出范围: {batch.token_type_ids_fut.max()} >= {self.tte_fut.num_embeddings}")
        #             raise RuntimeError(f"token_type_ids_fut {batch.token_type_ids_fut.max()} >= {self.tte_fut.num_embeddings}")
                
        #         tte_fut = self.tte_fut(batch.token_type_ids_fut)
        #         print(f"    ✅ tte_fut shape: {tte_fut.shape}")
                
        #         input_embedding_fut = torch.cat([
        #             input_embedding_fut, 
        #             sem_ids_emb_fut + tte_fut
        #         ], axis=1)
        #         print(f"    ✅ input_embedding_fut final shape: {input_embedding_fut.shape}")
        # except Exception as e:
        #     print(f"    ❌ Input embedding construction失败: {e}")
        #     raise e

        # # Step 5: 处理jagged/padding
        # if self.jagged_mode:
        #     print(f"  Step 5: Jagged tensor conversion...")
        #     try:
        #         input_embedding = padded_to_jagged_tensor(input_embedding, lengths=seq_lengths+1, max_len=input_embedding.shape[1])
        #         print(f"    ✅ input_embedding jagged conversion成功")

        #         seq_lengths_fut = torch.tensor(input_embedding_fut.shape[1], device=input_embedding_fut.device, dtype=torch.int64).repeat(B)
        #         input_embedding_fut = padded_to_jagged_tensor(input_embedding_fut, lengths=seq_lengths_fut, max_len=input_embedding_fut.shape[1])
        #         print(f"    ✅ input_embedding_fut jagged conversion成功")
        #     except Exception as e:
        #         print(f"    ❌ Jagged conversion失败: {e}")
        #         raise e
        # else:
        #     print(f"  Step 5: Standard tensor processing...")
        #     mem_mask = torch.cat([
        #         torch.ones(B, 1, dtype=torch.bool, device=batch.seq_mask.device),
        #         batch.seq_mask
        #     ], axis=1)
        #     f_mask = torch.zeros_like(mem_mask, dtype=torch.float32)
        #     f_mask[~mem_mask] = float("-inf")
        #     print(f"    ✅ mask创建成功: f_mask shape={f_mask.shape}")

        # # Step 6: Transformer projections
        # try:
        #     print(f"  Step 6: Transformer projections...")
        #     transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        #     transformer_input = self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))
        #     print(f"    ✅ transformer_context shape: {transformer_context.shape if hasattr(transformer_context, 'shape') else 'jagged'}")
        #     print(f"    ✅ transformer_input shape: {transformer_input.shape if hasattr(transformer_input, 'shape') else 'jagged'}")
        # except Exception as e:
        #     print(f"    ❌ Transformer projections失败: {e}")
        #     raise e

        # # Step 7: Transformer forward
        # try:
        #     print(f"  Step 7: Transformer forward...")
        #     if self.jagged_mode:
        #         transformer_output = self.transformer(x=transformer_input, context=transformer_context, padding_mask=batch.seq_mask, jagged=self.jagged_mode)
        #     else:
        #         causal_mask = nn.Transformer.generate_square_subsequent_mask(transformer_input.shape[1], device=transformer_input.device)
        #         print(f"    causal_mask shape: {causal_mask.shape}")
        #         transformer_output = self.transformer(src=transformer_context, tgt=transformer_input, tgt_is_causal=True, tgt_mask=causal_mask, src_key_padding_mask=f_mask, memory_key_padding_mask=f_mask)
            
        #     print(f"    ✅ transformer_output shape: {transformer_output.shape if hasattr(transformer_output, 'shape') else 'jagged'}")
        #     print(f"  🎉 _predict方法成功完成!")
        #     return transformer_output
            
        # except Exception as e:
        #     print(f"    ❌ Transformer forward失败: {e}")
        #     print(f"    错误类型: {type(e).__name__}")
        #     if "illegal memory access" in str(e):
        #         print(f"    💥 这是CUDA内存访问错误!")
        #         # 打印所有相关tensor的详细信息
        #         print(f"    调试信息:")
        #         if not self.jagged_mode:
        #             print(f"      transformer_context shape: {transformer_context.shape}")
        #             print(f"      transformer_input shape: {transformer_input.shape}")
        #             print(f"      causal_mask shape: {causal_mask.shape}")
        #             print(f"      f_mask shape: {f_mask.shape}")
        #             print(f"      f_mask inf count: {torch.isinf(f_mask).sum()}")
        #     raise e

    @eval_mode
    @reset_encoder_cache
    @torch.no_grad
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        temperature: int = 1,
        top_k: bool = True
    ) -> GenerationOutput:
        
        assert self.enable_generation, "Model generation is not enabled"

        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 32 if top_k else 1
        n_top_k_candidates = 200 if top_k else 1

        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None
        )

        for i in range(self.sem_id_dim):
            logits = self.forward(input_batch).logits
            probas_batched = F.softmax(logits / temperature, dim=-1)
            samples_batched = torch.multinomial(probas_batched, num_samples=n_top_k_candidates)

            if generated is None:
                is_valid_prefix = self.inference_verifier_fn(samples_batched.unsqueeze(-1))
            else:
                prefix = torch.cat([generated.flatten(0,1).unsqueeze(1).repeat_interleave(n_top_k_candidates, axis=1), samples_batched.unsqueeze(-1)], axis=-1)
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)
            
            sampled_log_probas = torch.log(torch.gather(probas_batched, 1, samples_batched)).reshape(B, -1)
            samples = samples_batched.reshape(B, -1)

            # Get top-K:
            sorted_log_probas, sorted_indices = (
                -10000*(~is_valid_prefix) +
                sampled_log_probas +
                maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
            ).sort(-1, descending=True)

            top_k_log_probas, top_k_indices = sorted_log_probas[:, :k], sorted_indices[:, :k]
            top_k_samples = torch.gather(samples, 1, top_k_indices)
            
            if generated is not None:
                parent_id = torch.gather(generated, 1, (top_k_indices // n_top_k_candidates).unsqueeze(2).expand(-1,-1,i))
                top_k_samples = torch.cat([parent_id, top_k_samples.unsqueeze(-1)], axis=-1)

                next_sem_ids = top_k_samples.flatten(end_dim=1)

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids,
                    sem_ids=input_batch.sem_ids,
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.arange(next_sem_ids.shape[1], device=next_sem_ids.device).repeat(next_sem_ids.shape[0], 1),
                    seq_mask=input_batch.seq_mask,
                    token_type_ids=input_batch.token_type_ids
                )

                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)

                # Explode encoder cache on dim 0 to match input size B*k
                # TODO: Figure out how to avoid jagged - padded conversions 
                # (E.g. Implement repeat_interleave jagged kernel)
                if self.jagged_mode:
                    cache = torch.zeros(input_batch.sem_ids.shape[0], input_batch.sem_ids.shape[1]+1, self.attn_dim, device=input_batch.sem_ids.device)
                    cache_mask = torch.cat([torch.ones(input_batch.sem_ids.shape[0], 1, dtype=bool, device=input_batch.seq_mask.device), input_batch.seq_mask], axis=1)
                    cache[cache_mask] = self.transformer.cached_enc_output.values()
                    lengths = self.transformer.cached_enc_output.offsets().diff().repeat_interleave(k)
                    cache = cache.repeat_interleave(k, dim=0)
                    self.transformer.cached_enc_output = padded_to_jagged_tensor(cache, lengths, max_len=cache.shape[1])

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids.repeat_interleave(k, dim=0),
                    sem_ids=input_batch.sem_ids.repeat_interleave(k, dim=0),
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.zeros_like(next_sem_ids),
                    seq_mask=input_batch.seq_mask.repeat_interleave(k, dim=0),
                    token_type_ids=input_batch.token_type_ids.repeat_interleave(k, dim=0)
                )

                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())
        
        return GenerationOutput(
            sem_ids=generated.squeeze(),
            log_probas=log_probas.squeeze()
        )
            
    # 注释掉 @torch.compile 来避免 Triton 错误
    # @torch.compile
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        trnsf_out = self._predict(batch)
        
        if self.training or not self.enable_generation:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                # This works because batch.sem_ids_fut is fixed length, no padding.
                logits = rearrange(jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=B)[:,:-1,:].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                
                # 添加目标值的边界检查和修复
                valid_mask = (target >= 0) & (target < self.num_embeddings)
                invalid_mask = (target >= 0) & (target >= self.num_embeddings)
                if invalid_mask.any():
                    print(f"Warning: Found invalid target values: {target[invalid_mask].unique()}, max allowed: {self.num_embeddings-1}")
                    target = torch.where(invalid_mask, -1, target)  # 将无效目标设为ignore_index
                
                unred_loss = rearrange(F.cross_entropy(logits, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B)
                loss = unred_loss.sum(axis=1).mean()
                loss_d = unred_loss.mean(axis=0)
            else:
                logits = predict_out
                out = logits[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                
                # 添加目标值的边界检查和修复
                valid_mask = (target >= 0) & (target < self.num_embeddings)
                invalid_mask = (target >= 0) & (target >= self.num_embeddings)
                if invalid_mask.any():
                    print(f"Warning: Found invalid target values: {target[invalid_mask].unique()}, max allowed: {self.num_embeddings-1}")
                    target = torch.where(invalid_mask, -1, target)  # 将无效目标设为ignore_index
                
                unred_loss = rearrange(F.cross_entropy(out, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B)
                loss = unred_loss.sum(axis=1).mean()
                loss_d = unred_loss.mean(axis=0)
            
            if not self.training and self.jagged_mode:
                self.transformer.cached_enc_output = None
                
        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            trnsf_out_flattened = rearrange(jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=B)[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None
        else:
            trnsf_out_flattened = trnsf_out[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None

        return ModelOutput(loss=loss, logits=logits, loss_d=loss_d)