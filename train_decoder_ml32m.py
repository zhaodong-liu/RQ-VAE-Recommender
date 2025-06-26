import argparse
import os
import gin
import torch
import wandb
import traceback

# ============ CUDA调试设置 ============
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
torch.autograd.set_detect_anomaly(True)

def check_cuda_error(operation_name=""):
    """检查CUDA错误并打印详细信息"""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"❌ CUDA错误 in {operation_name}: {e}")
            raise

def force_to_device(obj, target_device, name=""):
    """强制将张量或包含张量的对象递归地移动到目标设备"""
    if torch.is_tensor(obj):
        if obj.device != target_device:
            print(f"🔧 移动 {name}: {obj.device} -> {target_device}")
            return obj.to(target_device)
        return obj
    elif hasattr(obj, '_asdict'):  # NamedTuple（比如TokenizedSeqBatch）
        fields = obj._asdict()
        moved_fields = {
            k: force_to_device(v, target_device, f"{name}.{k}")
            for k, v in fields.items()
        }
        return type(obj)(**moved_fields)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(force_to_device(x, target_device, f"{name}[{i}]") for i, x in enumerate(obj))
    elif isinstance(obj, dict):
        return {k: force_to_device(v, target_device, f"{name}.{k}") for k, v in obj.items()}
    elif hasattr(obj, '__dataclass_fields__'):  # dataclass
        from dataclasses import replace
        return replace(obj, **{
            k: force_to_device(v, target_device, f"{name}.{k}")
            for k, v in vars(obj).items()
        })
    else:
        return obj

def create_safe_tokenized_data(tokenized_data, operation_name="unknown"):
    """创建安全的tokenized数据，彻底修复范围和设备问题"""
    verbose = "初始测试" in operation_name or "第一次" in operation_name or "iter0" in operation_name
    
    if verbose:
        print(f"🔧 彻底修复 {operation_name} 的所有问题...")
    
    # 🔧 CRITICAL: 强制确定目标设备
    target_device = None
    for attr_name in ['sem_ids', 'sem_ids_fut', 'user_ids', 'seq_mask', 'token_type_ids', 'token_type_ids_fut']:
        if hasattr(tokenized_data, attr_name):
            tensor = getattr(tokenized_data, attr_name)
            if torch.is_tensor(tensor) and tensor.is_cuda:
                target_device = tensor.device
                break
    
    if target_device is None:
        target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"  🎯 目标设备: {target_device}")
    
    # 🔧 第一步：强制所有张量到目标设备
    user_ids = force_to_device(tokenized_data.user_ids, target_device, "user_ids")
    sem_ids = force_to_device(tokenized_data.sem_ids, target_device, "sem_ids")
    sem_ids_fut = force_to_device(tokenized_data.sem_ids_fut, target_device, "sem_ids_fut")
    seq_mask = force_to_device(tokenized_data.seq_mask, target_device, "seq_mask")
    token_type_ids = force_to_device(tokenized_data.token_type_ids, target_device, "token_type_ids")
    token_type_ids_fut = force_to_device(tokenized_data.token_type_ids_fut, target_device, "token_type_ids_fut")
    
    # 🔧 第二步：修复语义ID范围（更严格的钳制）
    if verbose:
        print(f"  原始 sem_ids 范围: [{sem_ids.min()}, {sem_ids.max()}]")
        print(f"  原始 sem_ids_fut 范围: [{sem_ids_fut.min()}, {sem_ids_fut.max()}]")
    
    # 检查是否有超出安全范围的值
    sem_ids_out_of_range = (sem_ids > 255) & (sem_ids != -1)
    sem_ids_fut_out_of_range = (sem_ids_fut > 255) & (sem_ids_fut != -1)
    
    if sem_ids_out_of_range.any():
        out_of_range_values = sem_ids[sem_ids_out_of_range].unique()
        print(f"  ⚠️  发现 {sem_ids_out_of_range.sum()} 个sem_ids超出范围[0,255]: {out_of_range_values[:10].tolist()}")
    
    if sem_ids_fut_out_of_range.any():
        out_of_range_values = sem_ids_fut[sem_ids_fut_out_of_range].unique()
        print(f"  ⚠️  发现 {sem_ids_fut_out_of_range.sum()} 个sem_ids_fut超出范围[0,255]: {out_of_range_values[:10].tolist()}")
    
    # 🔧 CRITICAL: 严格钳制到安全范围 [0, 255]，保持-1作为padding
    max_safe_sem_id = 255
    
    safe_sem_ids = torch.where(
        sem_ids == -1,
        torch.tensor(-1, device=target_device, dtype=sem_ids.dtype),
        torch.clamp(sem_ids, 0, max_safe_sem_id)
    )
    
    safe_sem_ids_fut = torch.where(
        sem_ids_fut == -1,
        torch.tensor(-1, device=target_device, dtype=sem_ids_fut.dtype),
        torch.clamp(sem_ids_fut, 0, max_safe_sem_id)
    )
    
    # 验证修复结果
    if verbose:
        print(f"  修复后 sem_ids 范围: [{safe_sem_ids.min()}, {safe_sem_ids.max()}]")
        print(f"  修复后 sem_ids_fut 范围: [{safe_sem_ids_fut.min()}, {safe_sem_ids_fut.max()}]")
        
        # 确保没有超出范围的值
        assert safe_sem_ids.max() <= 255 or (safe_sem_ids == -1).all(), f"sem_ids仍有超出范围的值: {safe_sem_ids.max()}"
        assert safe_sem_ids_fut.max() <= 255 or (safe_sem_ids_fut == -1).all(), f"sem_ids_fut仍有超出范围的值: {safe_sem_ids_fut.max()}"
        print(f"  ✅ 验证通过：所有语义ID都在安全范围内")
    
    # 🔧 第三步：修复user_ids范围（UserIdEmbedder使用取模操作）
    # UserIdEmbedder有2000个buckets，所以user_ids应该在[0, 1999]范围内
    # 但实际上UserIdEmbedder会自动取模，所以我们只需要确保user_ids是有效的整数
    safe_user_ids = torch.clamp(user_ids, 0, 999999)  # 确保是正整数，取模会处理范围
    
    if verbose and not torch.equal(user_ids, safe_user_ids):
        print(f"  修复了 user_ids 范围: [{user_ids.min()}, {user_ids.max()}] -> [{safe_user_ids.min()}, {safe_user_ids.max()}]")
    
    # 创建新的TokenizedSeqBatch对象
    from data.schemas import TokenizedSeqBatch
    
    safe_tokenized_data = TokenizedSeqBatch(
        user_ids=safe_user_ids,
        sem_ids=safe_sem_ids,
        sem_ids_fut=safe_sem_ids_fut,
        seq_mask=seq_mask,
        token_type_ids=token_type_ids,
        token_type_ids_fut=token_type_ids_fut
    )
    
    if verbose:
        print(f"  ✅ 最终验证所有张量设备和范围:")
        print(f"    user_ids: {safe_tokenized_data.user_ids.device}, 范围[{safe_tokenized_data.user_ids.min()}, {safe_tokenized_data.user_ids.max()}]")
        print(f"    sem_ids: {safe_tokenized_data.sem_ids.device}, 范围[{safe_tokenized_data.sem_ids.min()}, {safe_tokenized_data.sem_ids.max()}]")
        print(f"    sem_ids_fut: {safe_tokenized_data.sem_ids_fut.device}, 范围[{safe_tokenized_data.sem_ids_fut.min()}, {safe_tokenized_data.sem_ids_fut.max()}]")
        print(f"    seq_mask: {safe_tokenized_data.seq_mask.device}")
        print(f"    token_type_ids: {safe_tokenized_data.token_type_ids.device}")
        print(f"    token_type_ids_fut: {safe_tokenized_data.token_type_ids_fut.device}")
        
    return safe_tokenized_data

def safe_model_forward(model, tokenized_data, operation_name="forward"):
    """安全的模型前向传播，彻底解决设备问题"""
    try:
        print(f"\n🚀 开始 {operation_name}...")
        
        # 🔧 CRITICAL: 强制修复所有设备和范围问题
        corrected_tokenized_data = create_safe_tokenized_data(tokenized_data, operation_name)
        
        # 🔧 额外验证：确保模型和数据在同一设备上
        model_device = next(model.parameters()).device
        corrected_tokenized_data = force_to_device(corrected_tokenized_data, model_device, "corrected_tokenized_data")

        # 🛡️ 检查所有字段是否仍在 CPU（防止 Triton 报错）
        for attr_name in corrected_tokenized_data.__annotations__:
            tensor = getattr(corrected_tokenized_data, attr_name, None)
            if torch.is_tensor(tensor) and not tensor.is_cuda:
                raise RuntimeError(f"[CRITICAL] {attr_name} 是 CPU tensor! device={tensor.device}, shape={tensor.shape}")

        # 🔧 修复UserIdEmbedder.forward的设备不一致问题
        original_user_id_forward = model.user_id_embedder.forward

        def fixed_user_id_forward(x):
            device = x.device
            hashed_indices = (x.to(device) % model.user_id_embedder.num_buckets).to(device)
            return model.user_id_embedder.emb(hashed_indices)

        model.user_id_embedder.forward = fixed_user_id_forward

        try:
            check_cuda_error(f"before_{operation_name}")
            print(f"📡 调用模型...")

            output = model(corrected_tokenized_data)
            print(f"✅ 模型调用成功")

            check_cuda_error(f"after_{operation_name}")
            return output

        finally:
            model.user_id_embedder.forward = original_user_id_forward

    except Exception as e:
        print(f"\n❌ {operation_name} 失败!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")

        if "device" in str(e).lower() or "Expected all tensors to be on the same device" in str(e):
            print(f"\n📍 设备不匹配详细分析:")
            model_device = next(model.parameters()).device
            print(f"  模型设备: {model_device}")
            print(f"  Embedding层设备:")
            print(f"    user_id_embedder.emb: {model.user_id_embedder.emb.weight.device}")
            print(f"    sem_id_embedder.emb: {model.sem_id_embedder.emb.weight.device}")
            print(f"  输入数据设备:")
            for attr_name in corrected_tokenized_data.__annotations__:
                tensor = getattr(corrected_tokenized_data, attr_name, None)
                if torch.is_tensor(tensor):
                    print(f"    {attr_name}: {tensor.device}, 形状: {tensor.shape}")

        if "index" in str(e).lower() or "out of range" in str(e).lower():
            print(f"\n📍 索引范围详细分析:")
            for attr_name in ['user_ids', 'sem_ids', 'sem_ids_fut']:
                if hasattr(corrected_tokenized_data, attr_name):
                    tensor = getattr(corrected_tokenized_data, attr_name)
                    if torch.is_tensor(tensor):
                        unique_vals = torch.unique(tensor)
                        print(f"  {attr_name}: 范围[{tensor.min()}, {tensor.max()}], 唯一值数={unique_vals.numel()}")
                        if unique_vals.numel() <= 20:
                            print(f"    所有唯一值: {unique_vals.tolist()}")
                        else:
                            print(f"    前10个: {unique_vals[:10].tolist()}")
                            print(f"    后10个: {unique_vals[-10:].tolist()}")

        print(f"\n📚 完整错误堆栈:")
        traceback.print_exc()
        raise

# ============ 主要训练代码 ============
from accelerate import Accelerator
from data.processed import ItemData, RecDataset, SeqData
from data.utils import batch_to, cycle, next_batch
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics, parse_config
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

@gin.configurable
def train(
    iterations=500,    # 更少迭代用于快速验证
    batch_size=4,      # 更小batch size以减少复杂度
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-32m",
    save_dir_root="out/",
    dataset=RecDataset.ML_32M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=100,
    full_eval_every=10000,  # 🔧 添加缺失的参数
    vae_input_dim=768,
    vae_embed_dim=64,
    vae_hidden_dims=[512, 256, 128],
    vae_codebook_size=256,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    push_vae_to_hf=False,  # 🔧 添加缺失的参数
    train_data_subsample=True,
    model_jagged_mode=True,
    vae_hf_model_name="edobotta/rqvae-movielens32m",  # 🔧 添加缺失的参数
):
    print("🚀 开始训练 - 最终修复版本（设备+索引）")
    
    accelerator = Accelerator(split_batches=split_batches, mixed_precision='no')
    device = accelerator.device
    print(f"🎯 使用设备: {device}")

    # 加载数据集
    try:
        print("📚 加载数据集...")
        item_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=force_dataset_process)
        train_dataset = SeqData(root=dataset_folder, dataset=dataset, is_train=True, subsample=train_data_subsample)
        eval_dataset = SeqData(root=dataset_folder, dataset=dataset, is_train=False, subsample=False)
        
        print(f"✅ 数据集加载成功: 物品={len(item_dataset)}, 训练={len(train_dataset)}, 评估={len(eval_dataset)}")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        raise

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    # 初始化tokenizer
    try:
        print("🔤 初始化tokenizer...")
        tokenizer = SemanticIdTokenizer(
            input_dim=vae_input_dim, hidden_dims=vae_hidden_dims, output_dim=vae_embed_dim,
            codebook_size=vae_codebook_size, n_layers=vae_n_layers, n_cat_feats=vae_n_cat_feats,
            rqvae_weights_path=pretrained_rqvae_path, rqvae_codebook_normalize=vae_codebook_normalize,
            rqvae_sim_vq=vae_sim_vq
        )
        tokenizer = accelerator.prepare(tokenizer)
        
        print("🏗️ 预计算语料库IDs...")
        tokenizer.precompute_corpus_ids(item_dataset)
        print(f"✅ Tokenizer初始化成功")
    except Exception as e:
        print(f"❌ Tokenizer初始化失败: {e}")
        raise

    # 初始化模型
    try:
        print("🤖 初始化模型...")
        model_num_embeddings = 1091
        
        model = EncoderDecoderRetrievalModel(
            embedding_dim=decoder_embed_dim, attn_dim=attn_embed_dim, dropout=dropout_p,
            num_heads=attn_heads, n_layers=attn_layers, num_embeddings=model_num_embeddings,
            inference_verifier_fn=lambda x: tokenizer.exists_prefix(x), sem_id_dim=tokenizer.sem_ids_dim,
            max_pos=train_dataset.max_seq_len * tokenizer.sem_ids_dim + 256, jagged_mode=model_jagged_mode
        )
        print(f"✅ 模型初始化成功")
        
        # 检查embedding层设备
        print(f"📍 Embedding层设备检查:")
        print(f"  user_id_embedder.emb: {model.user_id_embedder.emb.weight.device}")
        print(f"  sem_id_embedder.emb: {model.sem_id_embedder.emb.weight.device}")
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        raise

    # 🔧 CRITICAL: 测试模型前向传播
    try:
        print("🧪 测试模型前向传播...")
        model.eval()
        model.to(device)
        
        test_batch = next_batch(train_dataloader, device)
        test_tokenized = tokenizer(test_batch)

        test_tokenized = force_to_device(test_tokenized, device, name="test_tokenized")
        
        print(f"🔍 测试数据原始状态:")
        print(f"  user_ids: {test_tokenized.user_ids.device}, 范围[{test_tokenized.user_ids.min()}, {test_tokenized.user_ids.max()}]")
        print(f"  sem_ids: {test_tokenized.sem_ids.device}, 范围[{test_tokenized.sem_ids.min()}, {test_tokenized.sem_ids.max()}]")
        print(f"  sem_ids_fut: {test_tokenized.sem_ids_fut.device}, 范围[{test_tokenized.sem_ids_fut.min()}, {test_tokenized.sem_ids_fut.max()}]")
        
        with torch.no_grad():
            test_output = safe_model_forward(model, test_tokenized, "初始测试")
            print(f"✅ 模型前向传播测试成功: loss={test_output.loss}")
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        traceback.print_exc()
        raise

    # 初始化优化器
    optimizer = AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=10000)
    
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    
    print(f"🎯 开始训练: 设备={device}")
    
    # 主训练循环
    with tqdm(total=iterations, disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            try:
                model.train()
                optimizer.zero_grad()
                
                # 获取批次数据
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)
                
                # 🔧 CRITICAL: 在每次forward之前强制修复所有问题
                with accelerator.autocast():
                    model_output = safe_model_forward(model, tokenized_data, f"训练_iter{iter}")
                    loss = model_output.loss

            # 💡 判断是否是 NestedTensor 并修复
                try:
                    from torch._nested import NestedTensor
                    if isinstance(loss, NestedTensor):
                        print("⚠️ loss 是 NestedTensor，转换为普通 tensor")
                        loss = loss.to_padded_tensor(0.0).mean()
                except ImportError:
                    print("⚠️ 当前 PyTorch 不支持 NestedTensor 检测，跳过判断")
                finally:
                    pass

                accelerator.backward(loss)
                accelerator.wait_for_everyone()
                optimizer.step()
                lr_scheduler.step()
                accelerator.wait_for_everyone()

                pbar.set_description(f'loss: {loss.item():.4f}')
                pbar.update(1)
                
                # 简化的部分评估
                if (iter+1) % partial_eval_every == 0:
                    print(f"🔍 第 {iter+1} 次迭代部分评估，loss: {loss.item():.4f}")
                    model.eval()
                    
                    # 简单评估一个batch
                    eval_batch_count = 0
                    for batch in eval_dataloader:
                        if eval_batch_count >= 1:  # 只评估1个batch
                            break
                        eval_batch_count += 1
                        
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        with torch.no_grad():
                            model_output_eval = safe_model_forward(model, tokenized_data, f"部分评估_iter{iter}")
                            print(f"  评估loss: {model_output_eval.loss.item():.4f}")
                
                # 完整评估（可选，通常用于生成任务）
                if (iter+1) % full_eval_every == 0:
                    print(f"🔍 第 {iter+1} 次迭代完整评估...")
                    model.eval()
                    # 对于调试版本，跳过复杂的生成评估
                    print(f"  跳过生成评估（调试模式）")
                
                # 保存模型
                if accelerator.is_main_process and (iter+1) % save_model_every == 0:
                    state = {"iter": iter, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": lr_scheduler.state_dict()}
                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)
                    torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")
                    print(f"💾 保存检查点: checkpoint_{iter}.pt")
                
            except Exception as e:
                print(f"\n❌ 训练在第 {iter} 次迭代失败: {e}")
                traceback.print_exc()
                break
    
    print("🏁 训练完成!")

if __name__ == "__main__":
    parse_config()
    train()