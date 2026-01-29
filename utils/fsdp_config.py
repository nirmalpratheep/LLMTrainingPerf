"""FSDP configuration utilities."""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen2 import Qwen2DecoderLayer
from functools import partial


def get_mixed_precision_policy(dtype=torch.bfloat16):
    """
    Get mixed precision policy for FSDP.
    
    Args:
        dtype: Data type for mixed precision (bfloat16 or float16)
    
    Returns:
        MixedPrecision policy
    """
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def get_auto_wrap_policy():
    """
    Get auto-wrap policy for FSDP.
    Wraps each transformer layer individually.
    
    Returns:
        Auto-wrap policy function
    """
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )


def get_fsdp_config(
    strategy: str = "zero2",
    use_cpu_offload: bool = False,
    dtype=torch.bfloat16,
):
    """
    Get FSDP configuration for a given strategy.
    
    Args:
        strategy: "zero2" or "zero3"
        use_cpu_offload: Whether to offload to CPU
        dtype: Data type for mixed precision
    
    Returns:
        Dictionary of FSDP kwargs
    """
    # Base configuration
    config = {
        "auto_wrap_policy": get_auto_wrap_policy(),
        "mixed_precision": get_mixed_precision_policy(dtype),
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "device_id": torch.cuda.current_device(),
        "limit_all_gathers": True,
        "use_orig_params": True,  # Required for torch.compile
    }
    
    # Sharding strategy
    if strategy.lower() == "zero2":
        config["sharding_strategy"] = ShardingStrategy.SHARD_GRAD_OP
        print("Using FSDP Zero2 strategy (SHARD_GRAD_OP)")
    elif strategy.lower() == "zero3":
        config["sharding_strategy"] = ShardingStrategy.FULL_SHARD
        print("Using FSDP Zero3 strategy (FULL_SHARD)")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # CPU offload
    if use_cpu_offload:
        config["cpu_offload"] = CPUOffload(offload_params=True)
        print("CPU offload enabled")
    
    return config


def setup_fused_optimizer(model, learning_rate: float = 1e-5):
    """
    Setup fused AdamW optimizer.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
    
    Returns:
        Optimizer
    """
    # Use fused AdamW for better performance
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,  # Use fused kernel
    )
    
    print(f"Using fused AdamW optimizer with lr={learning_rate}")
    return optimizer


def init_distributed():
    """
    Initialize distributed process group.
    
    Returns:
        tuple: (rank, world_size)
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"Distributed training initialized: {world_size} GPUs")
    
    return rank, world_size


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # Test FSDP config
    print("Testing FSDP configurations...\n")
    
    print("=" * 50)
    config_zero2 = get_fsdp_config(strategy="zero2")
    print(f"Zero2 config keys: {config_zero2.keys()}")
    
    print("\n" + "=" * 50)
    config_zero3 = get_fsdp_config(strategy="zero3")
    print(f"Zero3 config keys: {config_zero3.keys()}")
    
    print("\nFSDP configuration test successful!")
