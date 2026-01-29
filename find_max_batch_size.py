"""Find maximum batch size for FSDP strategies before OOM."""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def test_batch_size(
    strategy: str,
    batch_size: int,
    nproc_per_node: int,
    max_length: int = 2048,
    num_steps: int = 10,
) -> bool:
    """
    Test if a batch size works without OOM.
    
    Args:
        strategy: "zero2" or "zero3"
        batch_size: Batch size to test
        nproc_per_node: Number of GPUs
        max_length: Sequence length
        num_steps: Number of test steps
    
    Returns:
        True if successful, False if OOM
    """
    script_path = f"training/train_{strategy}.py"
    output_dir = f"results/batchsize_test_{strategy}"
    
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        script_path,
        "--data_folder=data",
        f"--batch_size={batch_size}",
        f"--num_steps={num_steps}",
        "--warmup_steps=2",  # Minimal warmup for testing
        f"--max_length={max_length}",
        f"--output_dir={output_dir}",
        "--no_compile",  # Disable compile for faster testing
    ]
    
    print(f"\n{'='*60}")
    print(f"Testing {strategy.upper()} with batch_size={batch_size}")
    print(f"{'='*60}")
    
    try:
        # Run with timeout and capture output
        result = subprocess.run(
            cmd,
            timeout=300,  # 5 minute timeout
            capture_output=True,
            text=True,
        )
        
        # Check for OOM in output
        output = result.stdout + result.stderr
        
        if "out of memory" in output.lower() or "oom" in output.lower():
            print(f"❌ OOM detected for batch_size={batch_size}")
            return False
        
        if result.returncode != 0:
            print(f"❌ Failed with return code {result.returncode}")
            return False
        
        print(f"✅ Success with batch_size={batch_size}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout (treating as failure)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def binary_search_max_batch_size(
    strategy: str,
    min_bs: int,
    max_bs: int,
    nproc_per_node: int,
    max_length: int,
) -> int:
    """
    Use binary search to find maximum working batch size.
    
    Args:
        strategy: "zero2" or "zero3"
        min_bs: Minimum batch size to test
        max_bs: Maximum batch size to test
        nproc_per_node: Number of GPUs
        max_length: Sequence length
    
    Returns:
        Maximum working batch size
    """
    print(f"\n{'#'*60}")
    print(f"# Finding maximum batch size for {strategy.upper()}")
    print(f"# Search range: [{min_bs}, {max_bs}]")
    print(f"{'#'*60}")
    
    last_working = min_bs
    left, right = min_bs, max_bs
    
    while left <= right:
        mid = (left + right) // 2
        
        if test_batch_size(strategy, mid, nproc_per_node, max_length):
            last_working = mid
            left = mid + 1  # Try larger
        else:
            right = mid - 1  # Try smaller
    
    return last_working


def linear_search_max_batch_size(
    strategy: str,
    start_bs: int,
    step: int,
    nproc_per_node: int,
    max_length: int,
) -> int:
    """
    Use linear search to find maximum working batch size.
    Faster when you don't know the upper bound.
    
    Args:
        strategy: "zero2" or "zero3"
        start_bs: Starting batch size
        step: Increment step
        nproc_per_node: Number of GPUs
        max_length: Sequence length
    
    Returns:
        Maximum working batch size
    """
    print(f"\n{'#'*60}")
    print(f"# Finding maximum batch size for {strategy.upper()}")
    print(f"# Starting from {start_bs}, step size: {step}")
    print(f"{'#'*60}")
    
    current_bs = start_bs
    last_working = start_bs
    
    while True:
        if test_batch_size(strategy, current_bs, nproc_per_node, max_length):
            last_working = current_bs
            current_bs += step
        else:
            # Hit OOM, refine with binary search in last interval
            if step > 1:
                print(f"\nRefining in range [{last_working}, {current_bs}]...")
                return binary_search_max_batch_size(
                    strategy,
                    last_working,
                    current_bs - 1,
                    nproc_per_node,
                    max_length,
                )
            else:
                return last_working


def main():
    parser = argparse.ArgumentParser(description="Find maximum batch size for FSDP strategies")
    parser.add_argument("--strategy", type=str, choices=["zero2", "zero3", "both"], default="both")
    parser.add_argument("--nproc_per_node", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--max_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--search_method", type=str, choices=["linear", "binary"], default="linear")
    parser.add_argument("--min_bs", type=int, default=1, help="Minimum batch size (binary search)")
    parser.add_argument("--max_bs", type=int, default=64, help="Maximum batch size (binary search)")
    parser.add_argument("--start_bs", type=int, default=2, help="Starting batch size (linear search)")
    parser.add_argument("--step", type=int, default=2, help="Step size (linear search)")
    
    args = parser.parse_args()
    
    strategies = ["zero2", "zero3"] if args.strategy == "both" else [args.strategy]
    results = {}
    
    for strategy in strategies:
        if args.search_method == "binary":
            max_bs = binary_search_max_batch_size(
                strategy,
                args.min_bs,
                args.max_bs,
                args.nproc_per_node,
                args.max_length,
            )
        else:  # linear
            max_bs = linear_search_max_batch_size(
                strategy,
                args.start_bs,
                args.step,
                args.nproc_per_node,
                args.max_length,
            )
        
        results[strategy] = max_bs
    
    # Print summary
    print("\n" + "="*60)
    print("MAXIMUM BATCH SIZE RESULTS")
    print("="*60)
    
    for strategy, max_bs in results.items():
        print(f"{strategy.upper()}: {max_bs}")
    
    if len(results) == 2:
        ratio = results["zero2"] / results["zero3"] if results["zero3"] > 0 else float('inf')
        print(f"\nZero2/Zero3 ratio: {ratio:.2f}x")
        print(f"Zero3 saves ~{(1 - 1/ratio)*100:.1f}% memory" if ratio > 1 else "")
    
    print("="*60)
    
    # Save results
    output_file = "results/max_batch_sizes.txt"
    os.makedirs("results", exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write("Maximum Batch Size Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Sequence Length: {args.max_length}\n")
        f.write(f"Number of GPUs: {args.nproc_per_node}\n\n")
        
        for strategy, max_bs in results.items():
            f.write(f"{strategy.upper()}: {max_bs}\n")
        
        if len(results) == 2:
            ratio = results["zero2"] / results["zero3"] if results["zero3"] > 0 else float('inf')
            f.write(f"\nZero2/Zero3 ratio: {ratio:.2f}x\n")
    
    print(f"\nResults saved to {output_file}")
    
    # Suggest comparison command with max batch sizes
    if len(results) == 2:
        print("\n" + "="*60)
        print("SUGGESTED COMPARISON COMMAND:")
        print("="*60)
        print(f"python compare_strategies.py \\")
        print(f"    --batch_sizes {results['zero3']} {results['zero2']} \\")
        print(f"    --grad_accum_steps 1 2 4 \\")
        print(f"    --num_steps=50 \\")
        print(f"    --nproc_per_node={args.nproc_per_node}")
        print("="*60)


if __name__ == "__main__":
    main()
