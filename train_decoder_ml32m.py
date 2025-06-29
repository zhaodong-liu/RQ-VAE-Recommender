import argparse
import os
import gin
import torch
import wandb
import time
import gc
import traceback

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from evaluate.metrics import TopKAccumulator
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


def print_memory_usage():
    """打印内存使用情况"""
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB")
        print(f"GPU Memory cached: {torch.cuda.memory_cached()/1e9:.2f}GB")
    
    import psutil
    process = psutil.Process(os.getpid())
    print(f"CPU Memory: {process.memory_info().rss/1e9:.2f}GB")


def debug_batch_content(batch, batch_idx=0):
    """调试batch内容"""
    print(f"\n=== Batch {batch_idx} Debug Info ===")
    print(f"user_ids shape: {batch.user_ids.shape}, dtype: {batch.user_ids.dtype}")
    print(f"ids shape: {batch.ids.shape}, dtype: {batch.ids.dtype}")
    print(f"ids_fut shape: {batch.ids_fut.shape}, dtype: {batch.ids_fut.dtype}")
    print(f"x shape: {batch.x.shape}, dtype: {batch.x.dtype}")
    print(f"x_fut shape: {batch.x_fut.shape}, dtype: {batch.x_fut.dtype}")
    print(f"seq_mask shape: {batch.seq_mask.shape}, dtype: {batch.seq_mask.dtype}")
    
    # 检查数据范围
    print(f"ids range: [{batch.ids.min().item()}, {batch.ids.max().item()}]")
    print(f"ids_fut range: [{batch.ids_fut.min().item()}, {batch.ids_fut.max().item()}]")
    print(f"user_ids range: [{batch.user_ids.min().item()}, {batch.user_ids.max().item()}]")
    
    # 检查mask情况
    valid_positions = batch.seq_mask.sum().item()
    total_positions = batch.seq_mask.numel()
    print(f"Valid positions: {valid_positions}/{total_positions} ({valid_positions/total_positions*100:.1f}%)")
    
    # 检查序列长度分布
    seq_lengths = batch.seq_mask.sum(dim=1)
    print(f"Sequence lengths - min: {seq_lengths.min().item()}, max: {seq_lengths.max().item()}, mean: {seq_lengths.float().mean().item():.1f}")
    print("=" * 50)


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
    print("=" * 80)
    print("开始训练解码器模型")
    print("=" * 80)

    if wandb_logging:
        params = locals()
        print("wandb_logging已启用")

    print("初始化Accelerator...")
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )
    device = accelerator.device
    print(f"设备: {device}")
    print(f"混合精度: {mixed_precision_type if amp else 'disabled'}")

    if wandb_logging and accelerator.is_main_process:
        print("初始化wandb...")
        wandb.login()
        run = wandb.init(
            project="gen-retrieval-decoder-training",
            config=params
        )
        print("wandb初始化完成")
    
    print("\n" + "=" * 50)
    print("加载数据集...")
    print("=" * 50)
    
    print(f"数据集文件夹: {dataset_folder}")
    print(f"数据集类型: {dataset}")
    print(f"强制重新处理: {force_dataset_process}")
    
    try:
        print("加载物品数据集...")
        start_time = time.time()
        item_dataset = ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=force_dataset_process,
        )
        print(f"物品数据集加载完成，耗时 {time.time() - start_time:.2f}秒")
        print(f"物品总数: {len(item_dataset)}")
        print_memory_usage()
        
        print("\n加载训练序列数据集...")
        start_time = time.time()
        train_dataset = SeqData(
            root=dataset_folder, 
            dataset=dataset, 
            is_train=True, 
            subsample=train_data_subsample, 
        )
        print(f"训练序列数据集加载完成，耗时 {time.time() - start_time:.2f}秒")
        print(f"训练序列数: {len(train_dataset)}")
        print(f"最大序列长度: {train_dataset.max_seq_len}")
        print_memory_usage()
        
        print("\n加载评估序列数据集...")
        start_time = time.time()
        eval_dataset = SeqData(
            root=dataset_folder, 
            dataset=dataset, 
            is_train=False, 
            subsample=False, 
        )
        print(f"评估序列数据集加载完成，耗时 {time.time() - start_time:.2f}秒")
        print(f"评估序列数: {len(eval_dataset)}")
        print_memory_usage()
        
    except Exception as e:
        print(f"数据集加载失败: {e}")
        traceback.print_exc()
        return

    print("\n" + "=" * 50)
    print("创建数据加载器...")
    print("=" * 50)
    
    print(f"批量大小: {batch_size}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"训练数据加载器创建完成，总批次数: {len(train_dataloader)}")
    
    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    print(f"评估数据加载器创建完成，总批次数: {len(eval_dataloader)}")
    
    print("准备数据加载器...")
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )
    print("数据加载器准备完成")

    print("\n" + "=" * 50)
    print("初始化tokenizer...")
    print("=" * 50)
    
    print("tokenizer参数:")
    print(f"  input_dim: {vae_input_dim}")
    print(f"  hidden_dims: {vae_hidden_dims}")
    print(f"  output_dim: {vae_embed_dim}")
    print(f"  codebook_size: {vae_codebook_size}")
    print(f"  n_layers: {vae_n_layers}")
    print(f"  n_cat_feats: {vae_n_cat_feats}")
    print(f"  pretrained_path: {pretrained_rqvae_path}")
    
    try:
        start_time = time.time()
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
        print(f"tokenizer初始化完成，耗时 {time.time() - start_time:.2f}秒")
        print(f"语义ID维度: {tokenizer.sem_ids_dim}")
        
        print("准备tokenizer...")
        tokenizer = accelerator.prepare(tokenizer)
        print("tokenizer准备完成")
        
        print("开始预计算语义ID...")
        start_time = time.time()
        corpus_ids = tokenizer.precompute_corpus_ids(item_dataset)
        print(f"语义ID预计算完成，耗时 {time.time() - start_time:.2f}秒")
        print(f"语义ID缓存形状: {corpus_ids.shape}")
        print_memory_usage()
        
    except Exception as e:
        print(f"tokenizer初始化失败: {e}")
        traceback.print_exc()
        return
    
    if push_vae_to_hf:
        print("推送VAE到HuggingFace...")
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)
        print("推送完成")

    print("\n" + "=" * 50)
    print("初始化模型...")
    print("=" * 50)
    
    print("模型参数:")
    print(f"  embedding_dim: {decoder_embed_dim}")
    print(f"  attn_dim: {attn_embed_dim}")
    print(f"  dropout: {dropout_p}")
    print(f"  num_heads: {attn_heads}")
    print(f"  n_layers: {attn_layers}")
    print(f"  num_embeddings: {vae_codebook_size}")
    print(f"  sem_id_dim: {tokenizer.sem_ids_dim}")
    print(f"  max_pos: {train_dataset.max_seq_len*tokenizer.sem_ids_dim}")
    print(f"  jagged_mode: {model_jagged_mode}")
    
    try:
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
        print("模型初始化完成")
        
    except Exception as e:
        print(f"模型初始化失败: {e}")
        traceback.print_exc()
        return

    print("\n初始化优化器和调度器...")
    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    print(f"优化器: AdamW, lr={learning_rate}, weight_decay={weight_decay}")

    lr_scheduler = InverseSquareRootScheduler(
        optimizer=optimizer,
        warmup_steps=10000
    )
    print("学习率调度器: InverseSquareRoot, warmup_steps=10000")
    
    start_iter = 0
    if pretrained_decoder_path is not None:
        print(f"加载预训练解码器: {pretrained_decoder_path}")
        try:
            checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["scheduler"])
            start_iter = checkpoint["iter"] + 1
            print(f"预训练解码器加载完成，从第 {start_iter} 轮开始")
        except Exception as e:
            print(f"预训练解码器加载失败: {e}")
            traceback.print_exc()

    print("准备模型、优化器和调度器...")
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    print("准备完成")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {num_params:,}")
    print_memory_usage()

    print("\n" + "=" * 50)
    print("开始训练循环...")
    print("=" * 50)
    print(f"总迭代数: {iterations}")
    print(f"起始迭代: {start_iter}")
    print(f"梯度累积步数: {gradient_accumulate_every}")
    print(f"部分评估间隔: {partial_eval_every}")
    print(f"完整评估间隔: {full_eval_every}")
    print(f"模型保存间隔: {save_model_every}")

    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])
    
    with tqdm(initial=start_iter, total=start_iter + iterations,
              disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            print(f"\n--- 迭代 {iter+1}/{iterations} ---")
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            
            # 梯度累积循环
            for accumulate_step in range(gradient_accumulate_every):
                print(f"梯度累积步骤 {accumulate_step+1}/{gradient_accumulate_every}")
                
                try:
                    print("获取下一个batch...")
                    start_time = time.time()
                    data = next_batch(train_dataloader, device)
                    print(f"batch获取完成，耗时 {time.time() - start_time:.3f}秒")
                    
                    if accumulate_step == 0:  # 只在第一步打印详细信息
                        debug_batch_content(data, iter)
                    
                    print("tokenize数据...")
                    start_time = time.time()
                    tokenized_data = tokenizer(data)
                    print(f"tokenize完成，耗时 {time.time() - start_time:.3f}秒")
                    
                    if accumulate_step == 0:
                        print(f"tokenized数据形状:")
                        print(f"  sem_ids: {tokenized_data.sem_ids.shape}")
                        print(f"  sem_ids_fut: {tokenized_data.sem_ids_fut.shape}")
                        print(f"  token_type_ids: {tokenized_data.token_type_ids.shape}")
                        print(f"  seq_mask: {tokenized_data.seq_mask.shape}")

                    print("前向传播...")
                    start_time = time.time()
                    with accelerator.autocast():
                        model_output = model(tokenized_data)
                        loss = model_output.loss / gradient_accumulate_every
                        total_loss += loss
                    print(f"前向传播完成，耗时 {time.time() - start_time:.3f}秒，loss: {loss.item():.4f}")
                    
                    if wandb_logging and accelerator.is_main_process and accumulate_step == 0:
                        train_debug_metrics = compute_debug_metrics(tokenized_data, model_output)
                        print(f"调试指标: {train_debug_metrics}")

                    print("反向传播...")
                    start_time = time.time()
                    accelerator.backward(loss)
                    print(f"反向传播完成，耗时 {time.time() - start_time:.3f}秒")
                    
                    # 检查梯度
                    if hasattr(model, 'sem_id_embedder'):
                        grad_norm = model.sem_id_embedder.emb.weight.grad.norm().item()
                        print(f"梯度范数: {grad_norm:.6f}")
                        assert model.sem_id_embedder.emb.weight.grad is not None, "梯度为空!"
                    
                    print_memory_usage()
                    
                except Exception as e:
                    print(f"梯度累积步骤 {accumulate_step+1} 失败: {e}")
                    traceback.print_exc()
                    return

            pbar.set_description(f'loss: {total_loss.item():.4f}')
            print(f"总损失: {total_loss.item():.4f}")

            print("等待所有进程...")
            accelerator.wait_for_everyone()

            print("优化器步骤...")
            start_time = time.time()
            optimizer.step()
            lr_scheduler.step()
            print(f"优化器步骤完成，耗时 {time.time() - start_time:.3f}秒")
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

            accelerator.wait_for_everyone()

            # 部分评估
            if (iter+1) % partial_eval_every == 0:
                print(f"\n执行部分评估 (迭代 {iter+1})...")
                model.eval()
                model.enable_generation = False
                
                eval_count = 0
                try:
                    for batch in eval_dataloader:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        with torch.no_grad():
                            model_output_eval = model(tokenized_data)

                        if wandb_logging and accelerator.is_main_process:
                            eval_debug_metrics = compute_debug_metrics(tokenized_data, model_output_eval, "eval")
                            eval_debug_metrics["eval_loss"] = model_output_eval.loss.detach().cpu().item()
                            wandb.log(eval_debug_metrics)
                            print(f"评估指标: {eval_debug_metrics}")
                        
                        eval_count += 1
                        if eval_count >= 5:  # 只评估前5个batch
                            break
                    
                    print(f"部分评估完成，评估了 {eval_count} 个batch")
                    
                except Exception as e:
                    print(f"部分评估失败: {e}")
                    traceback.print_exc()

            # 完整评估
            if (iter+1) % full_eval_every == 0:
                print(f"\n执行完整评估 (迭代 {iter+1})...")
                model.eval()
                model.enable_generation = True
                
                try:
                    with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                        eval_count = 0
                        for batch in pbar_eval:
                            data = batch_to(batch, device)
                            tokenized_data = tokenizer(data)

                            print(f"生成语义ID (评估batch {eval_count+1})...")
                            start_time = time.time()
                            generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                            print(f"生成完成，耗时 {time.time() - start_time:.3f}秒")
                            
                            actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
                            metrics_accumulator.accumulate(actual=actual, top_k=top_k)

                            if accelerator.is_main_process and wandb_logging:
                                wandb.log(eval_debug_metrics)
                            
                            eval_count += 1
                            if eval_count >= 10:  # 限制评估batch数
                                break
                    
                    eval_metrics = metrics_accumulator.reduce()
                    print(f"完整评估结果: {eval_metrics}")
                    
                    if accelerator.is_main_process and wandb_logging:
                        wandb.log(eval_metrics)
                    
                    metrics_accumulator.reset()
                    print("完整评估完成")
                    
                except Exception as e:
                    print(f"完整评估失败: {e}")
                    traceback.print_exc()

            # 保存模型
            if accelerator.is_main_process:
                if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                    print(f"保存模型 (迭代 {iter+1})...")
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    save_path = save_dir_root + f"checkpoint_{iter}.pt"
                    torch.save(state, save_path)
                    print(f"模型已保存到: {save_path}")
                
                if wandb_logging:
                    log_data = {
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "total_loss": total_loss.cpu().item(),
                        **train_debug_metrics
                    }
                    wandb.log(log_data)
                    print(f"wandb日志: {log_data}")

            # 清理内存
            if iter % 10 == 0:
                print("清理GPU缓存...")
                torch.cuda.empty_cache()
                gc.collect()

            pbar.update(1)
            print(f"迭代 {iter+1} 完成\n")
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    
    if wandb_logging:
        wandb.finish()
        print("wandb会话结束")


if __name__ == "__main__":
    try:
        print("解析配置文件...")
        parse_config()
        print("配置文件解析完成")
        
        print("开始训练...")
        train()
        print("训练脚本执行完成")
        
    except Exception as e:
        print(f"训练脚本执行失败: {e}")
        traceback.print_exc()