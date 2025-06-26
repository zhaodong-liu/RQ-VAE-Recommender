import torch

def check_rqvae_config():
    """Check the actual configuration of the trained RQ-VAE model"""
    
    checkpoint_path = "trained_models/rqvae_ml32m/checkpoint_high_entropy.pt"
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("=== RQ-VAE Checkpoint Info ===")
        print(f"Available keys: {list(checkpoint.keys())}")
        
        # Check model config if available
        if "model_config" in checkpoint:
            config = checkpoint["model_config"]
            print(f"\nModel Config: {config}")
        else:
            print("\nNo model_config found in checkpoint")
        
        # Check model state dict for clues about architecture
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            
            # Look for codebook related parameters
            codebook_keys = [k for k in state_dict.keys() if 'codebook' in k.lower()]
            print(f"\nCodebook related keys: {codebook_keys}")
            
            # Check codebook sizes
            for key in codebook_keys:
                if 'weight' in key or 'embed' in key:
                    shape = state_dict[key].shape
                    print(f"{key}: {shape}")
            
            # Look for layer structure
            layer_keys = [k for k in state_dict.keys() if 'layer' in k.lower() or 'quantize' in k.lower()]
            print(f"\nLayer/Quantization keys: {layer_keys[:10]}...")  # Show first 10
            
            # Check specific parameters that might give us clues
            important_keys = [
                'input_dim', 'embed_dim', 'hidden_dims', 'codebook_size', 
                'n_layers', 'n_cat_features'
            ]
            
            for key in state_dict.keys():
                for important in important_keys:
                    if important in key.lower():
                        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
                        print(f"{key}: {shape}")
        
        # Try to extract actual codebook size from embeddings
        print("\n=== Attempting to extract actual codebook size ===")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            
            # Look for RQ-VAE quantizer embeddings
            for key, param in state_dict.items():
                if ('quantize' in key.lower() and 'embed' in key.lower()) or ('codebook' in key.lower()):
                    if hasattr(param, 'shape') and len(param.shape) >= 2:
                        print(f"{key}: shape={param.shape} -> possible codebook_size={param.shape[0]}")
        
        return checkpoint
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

if __name__ == "__main__":
    checkpoint = check_rqvae_config()