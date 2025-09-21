import argparse
import os
import gin
import torch
import wandb

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
# from evaluate.metrics import TopKAccumulator

from evaluate.enhanced_metrics import EnhancedMetricsAccumulator as TopKAccumulator

from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm
from torch.nn import functional as F  # 如果还没有的话
from data.schemas import TokenizedSeqBatch  # 如果还没有的话

def debug_generation_step_by_step(model, tokenized_data):
    """一步步调试generation过程"""
    print(f"\n🧪 开始逐步调试generation...")
    
    model.eval()
    model.enable_generation = True
    
    # 测试基本forward
    print(f"1. 测试基本forward...")
    try:
        with torch.no_grad():
            basic_output = model(tokenized_data)
            if basic_output.loss is not None:
                print(f"   ✅ 基本forward成功, loss: {basic_output.loss:.4f}")
            else:
                print(f"   ✅ 基本forward成功, loss: None (evaluation模式)")
    except Exception as e:
        print(f"   ❌ 基本forward失败: {e}")
        return False
    
    # 测试简化的generation
    print(f"2. 测试简化generation...")
    try:
        # 只做第一步generation
        input_batch = TokenizedSeqBatch(
            user_ids=tokenized_data.user_ids,
            sem_ids=tokenized_data.sem_ids,
            sem_ids_fut=None,
            seq_mask=tokenized_data.seq_mask,
            token_type_ids=tokenized_data.token_type_ids,
            token_type_ids_fut=None
        )
        
        with torch.no_grad():
            first_step_output = model.forward(input_batch)
            print(f"   ✅ 第一步generation forward成功")
            
            # 测试sampling
            logits = first_step_output.logits
            if logits is None:
                print(f"   ❌ logits是None")
                return False
                
            probas = F.softmax(logits, dim=-1)
            samples = torch.multinomial(probas, num_samples=10)
            print(f"   ✅ sampling成功, samples range: [{samples.min()}, {samples.max()}]")
            
            # 检查采样结果是否在有效范围内
            if samples.max() >= model.num_embeddings:
                print(f"   ❌ 采样结果越界: {samples.max()} >= {model.num_embeddings}")
                return False
            
            # 测试第二步
            next_sem_ids = samples[:, 0:1]  # 只取第一个sample
            print(f"   准备第二步: next_sem_ids = {next_sem_ids.flatten()[:5].tolist()}...")
            
            input_batch_step2 = TokenizedSeqBatch(
                user_ids=tokenized_data.user_ids,
                sem_ids=tokenized_data.sem_ids,
                sem_ids_fut=next_sem_ids,
                seq_mask=tokenized_data.seq_mask,
                token_type_ids=tokenized_data.token_type_ids,
                token_type_ids_fut=torch.zeros_like(next_sem_ids)
            )
            
            second_step_output = model.forward(input_batch_step2)
            print(f"   ✅ 第二步generation forward成功")
            
    except Exception as e:
        print(f"   ❌ 简化generation失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"3. 基础测试都通过，问题可能在复杂的beam search逻辑中")
    return True

@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    push_vae_to_hf=False,
    train_data_subsample=True,
    model_jagged_mode=True,
    vae_hf_model_name="/rqvae-ml32m"
):  
    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="gen-retrieval-decoder-training",
            config=params
        )
    
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
    )
    train_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=True, 
        subsample=train_data_subsample, 
    )
    eval_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=False, 
        subsample=False, 
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq
    )

    # 在 train_decoder_ml32m.py 中，创建tokenizer后添加：

    print(f"\n🔧 配置匹配检查:")
    print(f"配置中的codebook_size: {vae_codebook_size}")
    print(f"tokenizer中的codebook_size: {tokenizer.codebook_size}")
    print(f"RQ-VAE中的codebook_size: {tokenizer.rq_vae.codebook_size}")
    # print(f"模型中的num_embeddings: {t.num_embeddings}")

    # 检查embedding层的实际大小
    for i, layer in enumerate(tokenizer.rq_vae.layers):
        actual_size = layer.embedding.weight.shape[0]
        print(f"第{i}层embedding实际大小: {actual_size} (期望: {vae_codebook_size})")
        if actual_size != vae_codebook_size:
            print(f"  ❌ 大小不匹配!")
            
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)


    # 在 train_decoder_ml32m.py 中，tokenizer.precompute_corpus_ids(item_dataset) 之后添加：

    print(f"\n📊 预计算corpus检查:")
    print(f"corpus形状: {tokenizer.cached_ids.shape}")
    print(f"corpus语义ID范围: [{tokenizer.cached_ids[:, :-1].min()}, {tokenizer.cached_ids[:, :-1].max()}]")
    print(f"去重维度范围: [{tokenizer.cached_ids[:, -1].min()}, {tokenizer.cached_ids[:, -1].max()}]")

    # 检查是否有超出4096的值
    for i in range(3):  # 检查前3层
        layer_data = tokenizer.cached_ids[:, i]
        layer_max = layer_data.max().item()
        if layer_max >= 4096:
            print(f"❌ 第{i}层语义ID超出范围: max={layer_max}")
            over_count = (layer_data >= 4096).sum().item()
            print(f"   超出数量: {over_count}/{layer_data.numel()}")
    
    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    model = EncoderDecoderRetrievalModel(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
        sem_id_dim=tokenizer.sem_ids_dim,
        max_pos=train_dataset.max_seq_len*tokenizer.sem_ids_dim,
        jagged_mode=model_jagged_mode
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    lr_scheduler = InverseSquareRootScheduler(
        optimizer=optimizer,
        warmup_steps=10000
    )
    
    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params:,}")
    
    with tqdm(initial=start_iter, total=start_iter + iterations,
              disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)


                # 在 train_decoder_ml32m.py 的训练循环中，tokenized_data = tokenizer(data) 之后添加：

                # 简单检查
                max_sem_id = tokenized_data.sem_ids.max().item()
                max_fut_id = tokenized_data.sem_ids_fut.max().item() if tokenized_data.sem_ids_fut is not None else -1

                # if max_sem_id >= 4096 or max_fut_id >= 4096:
                #     print(f"\n❌ 发现语义ID越界 (iteration {iter}):")
                #     print(f"  sem_ids.max(): {max_sem_id} (应该 < 4096)")
                #     print(f"  sem_ids_fut.max(): {max_fut_id} (应该 < 4096)")
                #     print(f"  原始batch.ids.max(): {data.ids.max().item()}")
                #     print(f"  tokenizer.codebook_size: {tokenizer.codebook_size}")
                    
                    # # 打印一些具体的越界值
                    # if max_sem_id >= 4096:
                    #     over_values = tokenized_data.sem_ids[tokenized_data.sem_ids >= 4096].unique()[:10]
                    #     print(f"  具体越界sem_ids: {over_values.tolist()}")

                with accelerator.autocast():
                    model_output = model(tokenized_data)
                    loss = model_output.loss / gradient_accumulate_every
                    total_loss += loss
                
                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data, model_output)

                accelerator.backward(loss)
                assert model.sem_id_embedder.emb.weight.grad is not None

            pbar.set_description(f'loss: {total_loss.item():.4f}')

            accelerator.wait_for_everyone()

            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()

            # if (iter+1) % partial_eval_every == 0:
            #     model.eval()
            #     model.enable_generation = False
            #     for batch in eval_dataloader:
            #         data = batch_to(batch, device)
            #         tokenized_data = tokenizer(data)

            #         with torch.no_grad():
            #             model_output_eval = model(tokenized_data)

            #         if wandb_logging and accelerator.is_main_process:
            #             eval_debug_metrics = compute_debug_metrics(tokenized_data, model_output_eval, "eval")
            #             eval_debug_metrics["eval_loss"] = model_output_eval.loss.detach().cpu().item()
            #             wandb.log(eval_debug_metrics)

            # if (iter+1) % full_eval_every == 0:
            #     model.eval()
            #     model.enable_generation = True
            #     with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
            #         for batch in pbar_eval:
            #             data = batch_to(batch, device)
            #             tokenized_data = tokenizer(data)

            #             generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
            #             actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids

            #             metrics_accumulator.accumulate(actual=actual, top_k=top_k)

            #             if accelerator.is_main_process and wandb_logging:
            #                 wandb.log(eval_debug_metrics)
                
            #     eval_metrics = metrics_accumulator.reduce()
                
            #     print(eval_metrics)
            #     if accelerator.is_main_process and wandb_logging:
            #         wandb.log(eval_metrics)
                
            #     metrics_accumulator.reset()


            if (iter+1) % full_eval_every == 0:
                print(f"\n🔧 开始generation evaluation调试 (iteration {iter+1})...")
                model.eval()
                
                # 获取第一个batch进行详细调试
                try:
                    first_batch = next(eval_dataloader.__iter__())
                    data = batch_to(first_batch, device)
                    tokenized_data = tokenizer(data)
                    
                    # 执行逐步调试
                    success = debug_generation_step_by_step(model, tokenized_data)
                    
                    if success:
                        try:
                            model.enable_generation = True
                            generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                            
                            if generated is not None:
                                actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
                                metrics_accumulator.accumulate(actual=actual, top_k=top_k)
                                
                                # 如果第一个batch成功，尝试更多batch
                                successful_batches = 1
                                total_batches = 1
                                
                                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', 
                                        disable=not accelerator.is_main_process, 
                                        initial=1) as pbar_eval:
                                    for batch_idx, batch in enumerate(pbar_eval):
                                        # 限制evaluation数量
                                        if batch_idx >= 19:  # 总共20个batch (包括第一个)
                                            break
                                        
                                        try:
                                            data = batch_to(batch, device)
                                            tokenized_data = tokenizer(data)
                                            
                                            # 快速检查
                                            if tokenized_data.sem_ids.max() >= model.num_embeddings:
                                                continue
                                            
                                            generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                                            
                                            if generated is not None:
                                                actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
                                                metrics_accumulator.accumulate(actual=actual, top_k=top_k)
                                                successful_batches += 1
                                            
                                            total_batches += 1
                                            
                                        except RuntimeError as e:
                                            if "illegal memory access" in str(e):
                                                print(f"❌ batch {batch_idx+1} CUDA错误，跳过")
                                                continue
                                            else:
                                                print(f"❌ batch {batch_idx+1} 其他错误: {e}")
                                                break
                                        except Exception as e:
                                            print(f"❌ batch {batch_idx+1} 意外错误: {e}")
                                            continue
                                
                                if successful_batches > 0:
                                    # 计算所有指标
                                    eval_metrics = metrics_accumulator.reduce()
                                    
                                    # 打印完整指标字典（用于调试）
                                    print("\n📊 All Evaluation Metrics:")
                                    print(eval_metrics)
                                    
                                    # 格式化打印关键指标
                                    print("\n" + "="*70)
                                    print(f"🎯 Evaluation Results - Iteration {iter+1}")
                                    print("="*70)
                                    print(f"Batches processed: {successful_batches}/{total_batches}")
                                    print("-"*70)
                                    
                                    # 打印 Recall 指标
                                    print("📈 Recall Metrics:")
                                    for k in [1, 5, 10]:
                                        recall_key = f"recall@{k}"
                                        if recall_key in eval_metrics:
                                            print(f"  Recall@{k:<2}: {eval_metrics[recall_key]:.4f} ({eval_metrics[recall_key]*100:.2f}%)")
                                        else:
                                            print(f"  Recall@{k:<2}: N/A")
                                    
                                    print("-"*35)
                                    
                                    # 打印 NDCG 指标
                                    print("📊 NDCG Metrics:")
                                    for k in [1, 5, 10]:
                                        ndcg_key = f"ndcg@{k}"
                                        if ndcg_key in eval_metrics:
                                            print(f"  NDCG@{k:<2}  : {eval_metrics[ndcg_key]:.4f}")
                                        else:
                                            print(f"  NDCG@{k:<2}  : N/A")
                                    
                                    print("-"*35)
                                    
                                    # 打印主要的 Hit 指标（选择性显示）
                                    print("🎯 Hit Metrics (Selected):")
                                    # 只显示完整序列的 Hit 指标
                                    for k in [1, 5, 10]:
                                        hit_key = f"h@{k}_slice_:4"  # 假设语义ID长度为4
                                        if hit_key in eval_metrics:
                                            print(f"  Hit@{k:<2}   : {eval_metrics[hit_key]:.4f} ({eval_metrics[hit_key]*100:.2f}%)")
                                    
                                    print("="*70)
                                    
                                    # 检查新指标是否存在
                                    if "recall@5" not in eval_metrics and "ndcg@5" not in eval_metrics:
                                        print("\n⚠️  Warning: NDCG and Recall metrics not found!")
                                        print("   Please check if EnhancedMetricsAccumulator is properly imported")
                                        print(f"   Current accumulator type: {type(metrics_accumulator).__name__}")
                                    
                                    # 创建一个包含关键指标的简化字典用于 wandb
                                    key_metrics = {
                                        "iteration": iter + 1,
                                        "eval_batches": successful_batches,
                                    }
                                    
                                    # 添加 Recall 和 NDCG 到关键指标
                                    for k in [5, 10]:
                                        recall_key = f"recall@{k}"
                                        ndcg_key = f"ndcg@{k}"
                                        if recall_key in eval_metrics:
                                            key_metrics[recall_key] = eval_metrics[recall_key]
                                        if ndcg_key in eval_metrics:
                                            key_metrics[ndcg_key] = eval_metrics[ndcg_key]
                                    
                                    # 添加主要的 Hit 指标
                                    for k in [5, 10]:
                                        hit_key = f"h@{k}_slice_:4"
                                        if hit_key in eval_metrics:
                                            key_metrics[f"hit@{k}_full"] = eval_metrics[hit_key]
                                    
                                    # Log to wandb
                                    if accelerator.is_main_process and wandb_logging:
                                        # Log 所有指标
                                        wandb.log(eval_metrics)
                                        # Log 关键指标（带有特殊前缀以便在 wandb 中更容易找到）
                                        wandb.log({f"eval/{k}": v for k, v in key_metrics.items()})
                                        
                                        # 创建 wandb 表格显示结果
                                        if "recall@5" in eval_metrics:
                                            wandb.log({
                                                "eval_summary": wandb.Table(
                                                    columns=["Metric", "Value"],
                                                    data=[
                                                        ["Recall@5", f"{eval_metrics['recall@5']:.4f}"],
                                                        ["Recall@10", f"{eval_metrics.get('recall@10', 0):.4f}"],
                                                        ["NDCG@5", f"{eval_metrics.get('ndcg@5', 0):.4f}"],
                                                        ["NDCG@10", f"{eval_metrics.get('ndcg@10', 0):.4f}"],
                                                    ]
                                                )
                                            })
                                    
                                    # 保存最佳结果（可选）
                                    if "recall@10" in eval_metrics:
                                        current_recall_10 = eval_metrics["recall@10"]
                                        # 如果这是最佳结果，保存模型
                                        if not hasattr(train, 'best_recall_10') or current_recall_10 > train.best_recall_10:
                                            train.best_recall_10 = current_recall_10
                                            print(f"\n🏆 New best Recall@10: {current_recall_10:.4f}")
                                            # 可以在这里保存最佳模型
                                            if accelerator.is_main_process:
                                                best_checkpoint = {
                                                    "model": accelerator.unwrap_model(model).state_dict(),
                                                    "optimizer": optimizer.state_dict(),
                                                    "scheduler": lr_scheduler.state_dict(),
                                                    "iter": iter,
                                                    "best_recall_10": current_recall_10,
                                                    "eval_metrics": eval_metrics
                                                }
                                                torch.save(best_checkpoint, f"{save_dir_root}/best_model.pt")
                                                print(f"💾 Saved best model with Recall@10: {current_recall_10:.4f}")
                                
                            else:
                                print(f"⚠️ generation返回None")
                                
                        except RuntimeError as e:
                            if "illegal memory access" in str(e):
                                print(f"❌ 完整generation出现CUDA错误:")
                                print(f"   错误信息: {e}")
                                print(f"   这个错误需要进一步调试...")
                            else:
                                print(f"❌ 完整generation出现其他错误: {e}")
                                raise e
                        except Exception as e:
                            print(f"❌ 完整generation出现意外错误: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"❌ 基础测试失败，无法进行generation")
                        
                except Exception as e:
                    print(f"❌ evaluation batch准备失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 重置accumulator为下次评估做准备
                metrics_accumulator.reset()


            if accelerator.is_main_process:
                if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")
                
                if wandb_logging:
                    wandb.log({
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "total_loss": total_loss.cpu().item(),
                        **train_debug_metrics
                    })

            pbar.update(1)
    
    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()