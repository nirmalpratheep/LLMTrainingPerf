"""Load QWEN 1.5B model from HuggingFace."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def load_qwen_model(model_name: str = "Qwen/Qwen1.5-1.8B", device: str = "cpu"):
    """
    Load QWEN 1.5B model and tokenizer from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on initially (before FSDP wrapping)
    
    Returns:
        tuple: (model, tokenizer, config)
    """
    print(f"Loading model: {model_name}")
    
    # Load configuration
    config = AutoConfig.from_pretrained(model_name)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with configurations suitable for FSDP
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,  # Use BF16 for training
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    print(f"Model loaded successfully!")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Max position embeddings: {config.max_position_embeddings}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return model, tokenizer, config


if __name__ == "__main__":
    # Test the model loader
    model, tokenizer, config = load_qwen_model()
    print("\nModel loading test successful!")
    print(f"Sequence length: {config.max_position_embeddings}")
