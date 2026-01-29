"""FSDP Zero2 training loop with optimizations."""

import os
import time
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.profiler import profile, ProfilerActivity, schedule
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_loader import load_qwen_model
from utils.data_loader import load_training_data
from utils.fsdp_config import (
    get_fsdp_config,
    setup_fused_optimizer,
    init_distributed,
    cleanup_distributed,
)
from utils.metrics_tracker import MetricsTracker


def train_zero2(
    data_folder: str = "data",
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    num_steps: int = 50,
    learning_rate: float = 1e-5,
    max_length: int = 2048,
    output_dir: str = "results/zero2",
    use_torch_compile: bool = True,
    use_profiler: bool = False,
):
    """
    Train with FSDP Zero2 strategy.
    
    Args:
        data_folder: Path to data folder
        batch_size: Batch size per GPU
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_steps: Number of training steps
        learning_rate: Learning rate
        max_length: Maximum sequence length
        output_dir: Output directory for results
        use_torch_compile: Whether to use torch.compile
        use_profiler: Whether to use PyTorch profiler
    """
    # Initialize distributed
    rank, world_size = init_distributed()
    
    print(f"[Rank {rank}] Starting Zero2 training...")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Num steps: {num_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps * world_size}")
    
    # Load model and tokenizer
    model, tokenizer, config = load_qwen_model()
    
    # Wrap with FSDP Zero2
    fsdp_config = get_fsdp_config(strategy="zero2", dtype=torch.bfloat16)
    model = FSDP(model, **fsdp_config)
    
    if rank == 0:
        print(f"Model wrapped with FSDP Zero2")
    
    # Apply torch.compile for optimization
    if use_torch_compile:
        if rank == 0:
            print("Applying torch.compile (this may take a moment)...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Setup fused optimizer
    optimizer = setup_fused_optimizer(model, learning_rate)
    
    # Load data
    dataloader = load_training_data(
        data_folder=data_folder,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        rank=rank,
        world_size=world_size,
    )
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(rank=rank, world_size=world_size)
    
    # Training loop
    model.train()
    optimizer.zero_grad()
    
    # Setup profiler if requested
    prof = None
    if use_profiler and rank == 0:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(f"{output_dir}/trace_zero2.json"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
    
    data_iter = iter(dataloader)
    
    if rank == 0:
        pbar = tqdm(total=num_steps, desc="Training Zero2")
    
    for step in range(num_steps):
        metrics_tracker.start_step()
        
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Move to GPU
        batch = {k: v.cuda() for k, v in batch.items()}
        metrics_tracker.mark_data_loaded()
        
        # Forward pass with mixed precision
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        comm_start = time.time()
        loss.backward()
        comm_time = time.time() - comm_start
        metrics_tracker.record_comm_time(comm_time)
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate tokens processed
        num_tokens = batch["input_ids"].numel() * world_size
        
        # Record metrics
        metrics_tracker.end_step(
            loss=loss.item() * gradient_accumulation_steps,
            num_tokens=num_tokens,
            step=step,
        )
        
        if rank == 0:
            pbar.update(1)
        
        if prof:
            prof.step()
    
    if rank == 0:
        pbar.close()
    
    if prof:
        prof.stop()
    
    # Save metrics
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        metrics_tracker.save(
            f"{output_dir}/metrics.json",
            config={
                "strategy": "zero2",
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_steps": num_steps,
                "learning_rate": learning_rate,
                "world_size": world_size,
            }
        )
        
        summary = metrics_tracker.get_summary()
        print("\n" + "=" * 70)
        print("ZERO2 TRAINING SUMMARY")
        print("=" * 70)
        print(f"Average tokens/sec: {summary['tokens_per_sec']['mean']:.1f} ± {summary['tokens_per_sec']['std']:.1f}")
        print(f"Average GPU utilization: {summary['gpu_utilization_pct']['mean']:.1f}%")
        print(f"Peak GPU memory: {summary['gpu_memory_peak_gb']['max']:.2f} GB")
        print(f"Average communication overhead: {summary['comm_overhead_pct']['mean']:.1f}%")
        print(f"Average CPU overhead: {summary['cpu_overhead_pct']['mean']:.1f}%")
        print("=" * 70)
    
    # Cleanup
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="FSDP Zero2 Training")
    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="results/zero2")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--use_profiler", action="store_true", help="Enable PyTorch profiler")
    
    args = parser.parse_args()
    
    train_zero2(
        data_folder=args.data_folder,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        output_dir=args.output_dir,
        use_torch_compile=not args.no_compile,
        use_profiler=args.use_profiler,
    )


if __name__ == "__main__":
    main()
