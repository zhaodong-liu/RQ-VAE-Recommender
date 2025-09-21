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


from torch.nn import functional as F
from data.schemas import TokenizedSeqBatch
from data.schemas import SeqBatch

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


def analyze_amazon_dataset_info(item_dataset, train_dataset, eval_dataset):
    """分析Amazon数据集的详细信息"""
    print(f"\n📊 Amazon数据集详细分析:")
    
    # 物品数据集信息
    print(f"🏷️  物品数据集信息:")
    print(f"  物品总数: {len(item_dataset):,}")
    print(f"  物品特征维度: {item_dataset.item_data.shape}")
    
    # 训练数据集信息
    print(f"🚂 训练数据集信息:")
    print(f"  训练序列数: {len(train_dataset):,}")
    print(f"  最大序列长度: {train_dataset.max_seq_len}")
    
    # 抽样检查几个训练样本
    sample_indices = [0, len(train_dataset)//2, len(train_dataset)-1]
    print(f"  训练样本检查:")
    for i in sample_indices:
        sample = train_dataset[i]
        print(f"    样本{i}: user_ids={sample.user_ids.shape}, ids={sample.ids.shape}, seq_mask_sum={sample.seq_mask.sum()}")
    
    # 评估数据集信息
    print(f"🔍 评估数据集信息:")
    print(f"  评估序列数: {len(eval_dataset):,}")
    
    # 抽样检查几个评估样本
    print(f"  评估样本检查:")
    for i in sample_indices[:2]:  # 只检查前两个
        if i < len(eval_dataset):
            sample = eval_dataset[i]
            print(f"    样本{i}: user_ids={sample.user_ids.shape}, ids={sample.ids.shape}, seq_mask_sum={sample.seq_mask.sum()}")


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
    dataset_split="beauty",
    push_vae_to_hf=False,
    train_data_subsample=True,
    model_jagged_mode=True,
    vae_hf_model_name="edobotta/rqvae-amazon-beauty"
):  
    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")

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
        split=dataset_split
    )
    train_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=True, 
        subsample=train_data_subsample, 
        split=dataset_split
    )
    eval_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=False, 
        subsample=False, 
        split=dataset_split
    )

    # print(f"\n" + "="*60)
    # print("Amazon数据集信息分析")
    # print("="*60)
    # analyze_amazon_dataset_info(item_dataset, train_dataset, eval_dataset)
    # print("="*60)

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
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)

    # print(f"\n📊 Amazon预计算corpus检查:")
    # print(f"corpus形状: {tokenizer.cached_ids.shape}")
    # print(f"corpus语义ID范围: [{tokenizer.cached_ids[:, :-1].min()}, {tokenizer.cached_ids[:, :-1].max()}]")
    # print(f"去重维度范围: [{tokenizer.cached_ids[:, -1].min()}, {tokenizer.cached_ids[:, -1].max()}]")

    # 检查是否有超出codebook_size的值
    for i in range(tokenizer.cached_ids.shape[1] - 1):  # 检查前3层
        layer_data = tokenizer.cached_ids[:, i]
        layer_max = layer_data.max().item()
        layer_min = layer_data.min().item()
        # print(f"第{i}层语义ID范围: [{layer_min}, {layer_max}]")
        if layer_max >= vae_codebook_size:
            print(f"❌ 第{i}层语义ID超出范围: max={layer_max} >= codebook_size={vae_codebook_size}")
            over_count = (layer_data >= vae_codebook_size).sum().item()
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
    print(f"Device: {device}, Num Parameters: {num_params}")
    with tqdm(initial=start_iter, total=start_iter + iterations,
              disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    model_output = model(tokenized_data)
                    loss = model_output.loss / gradient_accumulate_every
                    total_loss += loss
                
                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data, model_output)

                accelerator.backward(total_loss)
                assert model.sem_id_embedder.emb.weight.grad is not None

            pbar.set_description(f'loss: {total_loss.item():.4f}')

            accelerator.wait_for_everyone()

            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()

            if (iter+1) % partial_eval_every == 0:
                model.eval()
                model.enable_generation = False
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)

                    with torch.no_grad():
                        model_output_eval = model(tokenized_data)

                    if wandb_logging and accelerator.is_main_process:
                        eval_debug_metrics = compute_debug_metrics(tokenized_data, model_output_eval, "eval")
                        eval_debug_metrics["eval_loss"] = model_output_eval.loss.detach().cpu().item()
                        wandb.log(eval_debug_metrics)

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
                print(f"\n🔧 开始Amazon generation evaluation (iteration {iter+1})...")
                model.eval()
                
                # 重置metrics accumulator
                metrics_accumulator.reset()
                successful_batches = 0
                total_batches = 0
                max_eval_batches = 10  # 限制评估批次数量
                
                # 遍历evaluation dataloader
                for batch_idx, batch in enumerate(eval_dataloader):
                    if batch_idx >= max_eval_batches:
                        break
                        
                    try:
                        data = batch_to(batch, device)
                        
                        # 对于Amazon数据集，限制batch size以避免内存问题
                        test_batch_size = data.user_ids.shape[0]
                        print(f"🔍 Batch {batch_idx+1}/{max_eval_batches}: 处理 {test_batch_size} 条数据")
                        small_data = SeqBatch(
                            user_ids=data.user_ids[:test_batch_size],
                            ids=data.ids[:test_batch_size],
                            ids_fut=data.ids_fut[:test_batch_size],
                            x=data.x[:test_batch_size] if data.x is not None else None,
                            x_fut=data.x_fut[:test_batch_size] if data.x_fut is not None else None,
                            seq_mask=data.seq_mask[:test_batch_size] if data.seq_mask is not None else None
                        )
                        
                        tokenized_data = tokenizer(small_data)
                        
                        # 检查数据完整性
                        if tokenized_data.sem_ids.max() >= model.num_embeddings:
                            print(f"⚠️ Batch {batch_idx+1}: sem_ids越界，跳过")
                            continue
                        
                        if hasattr(tokenized_data, 'token_type_ids') and tokenized_data.token_type_ids.max() >= model.sem_id_dim:
                            print(f"⚠️ Batch {batch_idx+1}: token_type_ids越界，跳过")
                            continue
                        
                        # 第一个batch进行详细调试
                        if batch_idx == 0:
                            success = debug_generation_step_by_step(model, tokenized_data)
                            if not success:
                                print(f"❌ 基础测试失败，跳过evaluation")
                                break
                        
                        # 执行generation
                        model.enable_generation = True
                        generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1.0)
                        
                        if generated is not None and generated.sem_ids is not None:
                            actual = tokenized_data.sem_ids_fut
                            top_k = generated.sem_ids
                            
                            # 累积指标
                            metrics_accumulator.accumulate(actual=actual, top_k=top_k)
                            successful_batches += 1
                        
                        total_batches += 1
                        
                    except RuntimeError as e:
                        if "illegal memory access" in str(e):
                            print(f"❌ Batch {batch_idx+1}: CUDA错误，停止evaluation")
                            break
                        else:
                            print(f"❌ Batch {batch_idx+1}: Runtime错误 - {str(e)[:100]}")
                            continue
                    except Exception as e:
                        print(f"❌ Batch {batch_idx+1}: 意外错误 - {str(e)[:100]}")
                        continue
                
                # 计算和显示指标
                if successful_batches > 0:
                    # 计算所有指标
                    eval_metrics = metrics_accumulator.reduce()
                    
                    # 打印原始指标字典（用于调试）
                    print("\n📊 All Amazon Evaluation Metrics:")
                    print(eval_metrics)
                    
                    # 格式化打印
                    print("\n" + "="*70)
                    print(f"🎯 Amazon Evaluation Results - Iteration {iter+1}")
                    print("="*70)
                    print(f"Dataset: Amazon {dataset_split}")
                    print(f"Batches processed: {successful_batches}/{total_batches} (max: {max_eval_batches})")
                    print("-"*70)
                    
                    # 打印 Recall 指标
                    print("📈 Recall Metrics:")
                    recall_values = []
                    for k in [1, 5, 10]:
                        recall_key = f"recall@{k}"
                        if recall_key in eval_metrics:
                            value = eval_metrics[recall_key]
                            recall_values.append((k, value))
                            print(f"  Recall@{k:<2}: {value:.4f} ({value*100:.2f}%)")
                        else:
                            print(f"  Recall@{k:<2}: N/A")
                    
                    print("-"*35)
                    
                    # 打印 NDCG 指标
                    print("📊 NDCG Metrics:")
                    ndcg_values = []
                    for k in [1, 5, 10]:
                        ndcg_key = f"ndcg@{k}"
                        if ndcg_key in eval_metrics:
                            value = eval_metrics[ndcg_key]
                            ndcg_values.append((k, value))
                            print(f"  NDCG@{k:<2}  : {value:.4f}")
                        else:
                            print(f"  NDCG@{k:<2}  : N/A")
                    
                    print("-"*35)
                    
                    # 打印主要的 Hit 指标（针对Amazon的语义ID长度）
                    print("🎯 Hit Metrics (Complete Sequence):")
                    # 找出语义ID的长度
                    sem_id_length = 4  # Amazon通常是4
                    for k in [1, 5, 10]:
                        hit_key = f"h@{k}_slice_:{sem_id_length}"
                        if hit_key in eval_metrics:
                            value = eval_metrics[hit_key]
                            print(f"  Hit@{k:<2}   : {value:.4f} ({value*100:.2f}%)")
                        elif f"h@{k}_slice_:3" in eval_metrics:  # 如果长度是3
                            value = eval_metrics[f"h@{k}_slice_:3"]
                            print(f"  Hit@{k:<2}   : {value:.4f} ({value*100:.2f}%)")
                    
                    print("="*70)
                    
                    # 显示改进趋势（如果有历史数据）
                    if hasattr(train, 'prev_recall_10'):
                        current_recall_10 = eval_metrics.get('recall@10', 0)
                        improvement = current_recall_10 - train.prev_recall_10
                        if improvement > 0:
                            print(f"📈 Improvement: Recall@10 +{improvement:.4f} from last eval")
                        elif improvement < 0:
                            print(f"📉 Decline: Recall@10 {improvement:.4f} from last eval")
                        train.prev_recall_10 = current_recall_10
                    else:
                        train.prev_recall_10 = eval_metrics.get('recall@10', 0)
                    
                    # 检查是否有新指标
                    if "recall@5" not in eval_metrics and "ndcg@5" not in eval_metrics:
                        print("\n⚠️  Warning: NDCG and Recall metrics not found!")
                        print("   Please verify EnhancedMetricsAccumulator is properly imported")
                        print(f"   Current accumulator type: {type(metrics_accumulator).__name__}")
                        print("   Expected: EnhancedMetricsAccumulator or TopKAccumulator")
                    
                    # 为wandb准备关键指标
                    key_metrics = {
                        "iteration": iter + 1,
                        "dataset": f"amazon_{dataset_split}",
                        "eval_batches": successful_batches,
                    }
                    
                    # 添加所有Recall和NDCG指标
                    for k in [5, 10]:
                        for metric_type in ['recall', 'ndcg']:
                            metric_key = f"{metric_type}@{k}"
                            if metric_key in eval_metrics:
                                key_metrics[f"amazon_{metric_key}"] = eval_metrics[metric_key]
                    
                    # Log to wandb
                    if accelerator.is_main_process and wandb_logging:
                        # Log所有指标
                        wandb.log(eval_metrics)
                        
                        # Log关键指标with prefix
                        wandb.log({f"eval/{k}": v for k, v in key_metrics.items()})
                        
                        # 创建summary表格
                        if "recall@5" in eval_metrics:
                            summary_data = []
                            for k in [5, 10]:
                                summary_data.append([
                                    f"@{k}",
                                    f"{eval_metrics.get(f'recall@{k}', 0):.4f}",
                                    f"{eval_metrics.get(f'ndcg@{k}', 0):.4f}"
                                ])
                            
                            wandb.log({
                                "amazon_eval_summary": wandb.Table(
                                    columns=["K", "Recall", "NDCG"],
                                    data=summary_data
                                )
                            })
                    
                    # 保存最佳模型（基于Recall@10）
                    if "recall@10" in eval_metrics:
                        current_recall_10 = eval_metrics["recall@10"]
                        
                        # 初始化或更新最佳分数
                        if not hasattr(train, 'best_amazon_recall_10'):
                            train.best_amazon_recall_10 = 0
                        
                        if current_recall_10 > train.best_amazon_recall_10:
                            train.best_amazon_recall_10 = current_recall_10
                            print(f"\n🏆 New best Amazon Recall@10: {current_recall_10:.4f}")
                            
                            # 保存最佳模型
                            if accelerator.is_main_process:
                                best_checkpoint = {
                                    "model": accelerator.unwrap_model(model).state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": lr_scheduler.state_dict(),
                                    "iter": iter,
                                    "best_recall_10": current_recall_10,
                                    "dataset": f"amazon_{dataset_split}",
                                    "eval_metrics": eval_metrics
                                }
                                save_path = f"{save_dir_root}/best_amazon_model.pt"
                                torch.save(best_checkpoint, save_path)
                                print(f"💾 Saved best Amazon model: {save_path}")
                                print(f"   Best Recall@10: {current_recall_10:.4f}")
                else:
                    print(f"\n❌ No successful evaluation batches for Amazon dataset")
                    print(f"   Total attempted: {total_batches}")
                
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
