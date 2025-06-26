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


@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-32m",  # Changed: default to ml-32m
    save_dir_root="out/",
    dataset=RecDataset.ML_32M,  # Changed: default to ML_32M
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
    vae_input_dim=768,  # Updated for MovieLens (text embeddings + genres)
    vae_embed_dim=64,   # Updated based on checkpoint
    vae_hidden_dims=[512, 256, 128],  # Updated based on checkpoint
    vae_codebook_size=256,  # Updated based on checkpoint
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,  # Updated for MovieLens genres
    vae_n_layers=3,      # Updated based on checkpoint
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    push_vae_to_hf=False,
    train_data_subsample=True,
    model_jagged_mode=True,
    vae_hf_model_name="edobotta/rqvae-movielens32m"  # Changed: updated model name for MovieLens
):  
    # Removed: Dataset restriction for AMAZON only
    # Now supports ML_32M and other datasets
    
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
    
    # Fixed: ItemData now always uses all items (train_test_split parameter is ignored)
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        train_test_split="all"  # Added: explicit parameter (though ignored internally)
    )
    
    # Fixed: SeqData uses sequence-level is_train for proper train/eval split
    train_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=True,  # Training sequences from history["train"]
        subsample=train_data_subsample
    )
    eval_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=False,  # Evaluation sequences from history["eval"]
        subsample=False
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
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)
    
    # Debug: Check data consistency
    print(f"Number of items: {len(item_dataset)}")
    print(f"Codebook size: {vae_codebook_size}")
    print(f"Tokenizer semantic ID dimension: {tokenizer.sem_ids_dim}")
    
    # Validate a few samples to check for index issues
    print("Checking sample data ranges...")
    for i in range(min(5, len(train_dataset))):
        sample_data = train_dataset[i]
        valid_ids = sample_data.ids[sample_data.ids >= 0]  # Exclude padding (-1)
        if len(valid_ids) > 0:
            print(f"Sample {i}: item IDs range [{valid_ids.min()}, {valid_ids.max()}], max allowed: {len(item_dataset)-1}")
            if valid_ids.max() >= len(item_dataset):
                raise ValueError(f"❌ Item ID {valid_ids.max()} exceeds dataset size {len(item_dataset)}")
        
        # Check future IDs too
        if hasattr(sample_data, 'ids_fut') and sample_data.ids_fut.numel() > 0:
            fut_ids = sample_data.ids_fut[sample_data.ids_fut >= 0]
            if len(fut_ids) > 0:
                print(f"Sample {i}: future item IDs range [{fut_ids.min()}, {fut_ids.max()}]")
                if fut_ids.max() >= len(item_dataset):
                    raise ValueError(f"❌ Future item ID {fut_ids.max()} exceeds dataset size {len(item_dataset)}")
    
    print("✅ Data validation passed!")
    
    # Additional debug: Check semantic IDs range
    print("Checking semantic IDs...")
    
    # Fix: Create a proper batch from single sample
    sample = train_dataset[0]
    
    # Debug: Check the raw sample data
    print(f"Raw sample data:")
    print(f"  user_ids: {sample.user_ids}")
    print(f"  ids shape: {sample.ids.shape}, range: [{sample.ids.min()}, {sample.ids.max()}]")
    print(f"  ids (first 10): {sample.ids[:10]}")
    print(f"  ids_fut: {sample.ids_fut}")
    print(f"  seq_mask sum: {sample.seq_mask.sum()} / {len(sample.seq_mask)}")
    
    # Convert single sample to batch format by adding batch dimension
    from types import SimpleNamespace
    batched_sample = SimpleNamespace(
        user_ids=sample.user_ids.unsqueeze(0),  # Add batch dim: [1]
        ids=sample.ids.unsqueeze(0),            # Add batch dim: [1, seq_len]
        ids_fut=sample.ids_fut.unsqueeze(0),    # Add batch dim: [1, 1] or [1]
        x=sample.x.unsqueeze(0),                # Add batch dim: [1, seq_len, feat_dim]
        x_fut=sample.x_fut.unsqueeze(0),        # Add batch dim: [1, 1, feat_dim] or [1, feat_dim]
        seq_mask=sample.seq_mask.unsqueeze(0)   # Add batch dim: [1, seq_len]
    )
    
    print(f"\nSample shapes after batching:")
    print(f"  ids: {batched_sample.ids.shape}")
    print(f"  ids_fut: {batched_sample.ids_fut.shape}")
    print(f"  x: {batched_sample.x.shape}")
    
    try:
        # Debug tokenizer step by step
        print(f"\nTokenizer input validation:")
        print(f"  Valid (non-padding) IDs: {(batched_sample.ids >= 0).sum()}")
        print(f"  Padding IDs: {(batched_sample.ids == -1).sum()}")
        
        sample_tokenized = tokenizer(batched_sample)
        print(f"\nTokenizer output:")
        print(f"  Semantic IDs shape: {sample_tokenized.sem_ids.shape}")
        print(f"  Semantic IDs range: [{sample_tokenized.sem_ids.min()}, {sample_tokenized.sem_ids.max()}]")
        print(f"  Non-padding semantic IDs: {(sample_tokenized.sem_ids != -1).sum()}")
        print(f"  Expected range: [0, {vae_codebook_size-1}]")
        
        # Check a few actual values
        unique_sem_ids = torch.unique(sample_tokenized.sem_ids)
        print(f"  Unique semantic IDs: {unique_sem_ids[:10]}...")  # Show first 10
        
        if sample_tokenized.sem_ids.max() >= vae_codebook_size:
            raise ValueError(f"❌ Semantic ID {sample_tokenized.sem_ids.max()} exceeds codebook size {vae_codebook_size}")
        
        # Check if ALL semantic IDs are -1 (this would cause the CUDA error)
        if (sample_tokenized.sem_ids == -1).all():
            print("❌ WARNING: ALL semantic IDs are -1! This will cause indexing errors.")
            print("   This suggests the tokenizer is not properly encoding non-padding items.")
            
            # Let's check what happens if we force some valid item IDs
            print("   Trying with a simple test case...")
            test_sample = SimpleNamespace(
                user_ids=torch.tensor([0]),
                ids=torch.tensor([[0, 1, 2, 3, 4, -1, -1, -1]]),  # Simple test with valid IDs
                ids_fut=torch.tensor([[5]]),
                x=torch.randn(1, 8, 768),
                x_fut=torch.randn(1, 1, 768),
                seq_mask=torch.tensor([[True, True, True, True, True, False, False, False]])
            )
            
            test_tokenized = tokenizer(test_sample)
            print(f"   Test semantic IDs range: [{test_tokenized.sem_ids.min()}, {test_tokenized.sem_ids.max()}]")
            
        else:
            print("✅ Semantic IDs validation passed!")
        
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        print("This indicates a tokenizer configuration issue.")
        import traceback
        traceback.print_exc()
        raise
    
    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    # Potential fix for embedding size mismatch
    # The error boundary 1025 suggests the model expects 1024 embeddings
    # This might be due to special tokens or different architecture requirements
    
    # Try to determine correct num_embeddings based on error analysis
    # Option 1: Use codebook size (256) - original approach
    # Option 2: Use 1024 based on error boundary (1025 = 1024 + 1)
    # Option 3: Use codebook_size + special tokens
    
    # Start with the original approach, but add debugging
    model_num_embeddings = vae_codebook_size  # 256
    
    print(f"Attempting model initialization with num_embeddings={model_num_embeddings}")
    
    try:
        model = EncoderDecoderRetrievalModel(
            embedding_dim=decoder_embed_dim,
            attn_dim=attn_embed_dim,
            dropout=dropout_p,
            num_heads=attn_heads,
            n_layers=attn_layers,
            num_embeddings=model_num_embeddings,
            inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
            sem_id_dim=tokenizer.sem_ids_dim,
            max_pos=train_dataset.max_seq_len*tokenizer.sem_ids_dim,
            jagged_mode=model_jagged_mode
        )
        print(f"✅ Model initialized successfully with num_embeddings={model_num_embeddings}")
    except Exception as e:
        print(f"❌ Model initialization failed with num_embeddings={model_num_embeddings}: {e}")
        print("Trying with num_embeddings=1024 based on error analysis...")
        
        model_num_embeddings = 1024
        model = EncoderDecoderRetrievalModel(
            embedding_dim=decoder_embed_dim,
            attn_dim=attn_embed_dim,
            dropout=dropout_p,
            num_heads=attn_heads,
            n_layers=attn_layers,
            num_embeddings=model_num_embeddings,
            inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
            sem_id_dim=tokenizer.sem_ids_dim,
            max_pos=train_dataset.max_seq_len*tokenizer.sem_ids_dim,
            jagged_mode=model_jagged_mode
        )
        print(f"✅ Model initialized with fallback num_embeddings={model_num_embeddings}")
    
    print(f"Final model configuration:")
    print(f"  num_embeddings: {model_num_embeddings}")
    print(f"  sem_id_dim: {tokenizer.sem_ids_dim}")
    print(f"  max_seq_len: {train_dataset.max_seq_len}")
    print(f"  max_pos: {train_dataset.max_seq_len*tokenizer.sem_ids_dim}")

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

            if (iter+1) % full_eval_every == 0:
                model.eval()
                model.enable_generation = True
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                        actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids

                        metrics_accumulator.accumulate(actual=actual, top_k=top_k)

                        if accelerator.is_main_process and wandb_logging:
                            wandb.log(eval_debug_metrics)
                
                eval_metrics = metrics_accumulator.reduce()
                
                print(eval_metrics)
                if accelerator.is_main_process and wandb_logging:
                    wandb.log(eval_metrics)
                
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