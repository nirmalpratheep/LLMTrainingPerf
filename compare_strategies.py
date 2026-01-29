"""Top-level comparison script for FSDP Zero2 vs Zero3."""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def run_training(
    strategy: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_steps: int,
    data_folder: str,
    nproc_per_node: int,
):
    """
    Run training with specified configuration.
    
    Args:
        strategy: "zero2" or "zero3"
        batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        num_steps: Number of training steps
        data_folder: Path to data folder
        nproc_per_node: Number of GPUs to use
    
    Returns:
        Path to metrics file
    """
    output_dir = f"results/{strategy}_bs{batch_size}_ga{gradient_accumulation_steps}"
    
    script_path = f"training/train_{strategy}.py"
    
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        script_path,
        f"--data_folder={data_folder}",
        f"--batch_size={batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--num_steps={num_steps}",
        f"--output_dir={output_dir}",
    ]
    
    print(f"\n{'='*70}")
    print(f"Running {strategy.upper()} with batch_size={batch_size}, grad_accum={gradient_accumulation_steps}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run training
    result = subprocess.run(cmd, check=True)
    
    return f"{output_dir}/metrics.json"


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, "r") as f:
        return json.load(f)


def create_comparison_plots(all_results: List[Dict], output_dir: str):
    """
    Create visualization plots comparing Zero2 vs Zero3.
    
    Args:
        all_results: List of results dictionaries
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    data = []
    for result in all_results:
        config = result["config"]
        summary = result["summary"]
        
        data.append({
            "Strategy": config["strategy"].upper(),
            "Batch Size": config["batch_size"],
            "Grad Accum": config["gradient_accumulation_steps"],
            "Effective BS": config["batch_size"] * config["gradient_accumulation_steps"],
            "Tokens/sec": summary["tokens_per_sec"]["mean"],
            "Tokens/sec Std": summary["tokens_per_sec"]["std"],
            "GPU Util %": summary["gpu_utilization_pct"]["mean"],
            "Peak Memory GB": summary["gpu_memory_peak_gb"]["max"],
            "Comm Overhead %": summary["comm_overhead_pct"]["mean"],
            "CPU Overhead %": summary["cpu_overhead_pct"]["mean"],
        })
    
    df = pd.DataFrame(data)
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 1. Tokens/sec vs Batch Size
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy in ["ZERO2", "ZERO3"]:
        strategy_df = df[df["Strategy"] == strategy]
        ax.plot(
            strategy_df["Effective BS"],
            strategy_df["Tokens/sec"],
            marker="o",
            label=strategy,
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Effective Batch Size", fontsize=12)
    ax.set_ylabel("Tokens per Second", fontsize=12)
    ax.set_title("Throughput Comparison: Zero2 vs Zero3", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tokens_per_sec.png", dpi=300)
    plt.close()
    
    # 2. GPU Memory vs Batch Size
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df["Effective BS"].unique()))
    width = 0.35
    
    zero2_df = df[df["Strategy"] == "ZERO2"].sort_values("Effective BS")
    zero3_df = df[df["Strategy"] == "ZERO3"].sort_values("Effective BS")
    
    ax.bar(x - width/2, zero2_df["Peak Memory GB"], width, label="ZERO2", alpha=0.8)
    ax.bar(x + width/2, zero3_df["Peak Memory GB"], width, label="ZERO3", alpha=0.8)
    
    ax.set_xlabel("Effective Batch Size", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title("Memory Usage Comparison: Zero2 vs Zero3", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(zero2_df["Effective BS"].values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gpu_memory.png", dpi=300)
    plt.close()
    
    # 3. Communication Overhead Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, zero2_df["Comm Overhead %"], width, label="ZERO2", alpha=0.8)
    ax.bar(x + width/2, zero3_df["Comm Overhead %"], width, label="ZERO3", alpha=0.8)
    
    ax.set_xlabel("Effective Batch Size", fontsize=12)
    ax.set_ylabel("Communication Overhead (%)", fontsize=12)
    ax.set_title("Communication Overhead: Zero2 vs Zero3", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(zero2_df["Effective BS"].values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comm_overhead.png", dpi=300)
    plt.close()
    
    # 4. GPU Utilization Heatmap
    pivot_util = df.pivot_table(
        values="GPU Util %",
        index="Strategy",
        columns="Effective BS",
        aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot_util,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "GPU Utilization (%)"},
        ax=ax,
    )
    ax.set_title("GPU Utilization Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Effective Batch Size", fontsize=12)
    ax.set_ylabel("Strategy", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gpu_utilization_heatmap.png", dpi=300)
    plt.close()
    
    # 5. Combined Overhead Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, strategy in enumerate(["ZERO2", "ZERO3"]):
        strategy_df = df[df["Strategy"] == strategy].sort_values("Effective BS")
        
        x_pos = np.arange(len(strategy_df))
        
        axes[idx].bar(x_pos, strategy_df["CPU Overhead %"], label="CPU Overhead", alpha=0.8)
        axes[idx].bar(
            x_pos,
            strategy_df["Comm Overhead %"],
            bottom=strategy_df["CPU Overhead %"],
            label="Comm Overhead",
            alpha=0.8
        )
        
        axes[idx].set_xlabel("Effective Batch Size", fontsize=11)
        axes[idx].set_ylabel("Overhead (%)", fontsize=11)
        axes[idx].set_title(f"{strategy} Overhead Breakdown", fontsize=12, fontweight="bold")
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(strategy_df["Effective BS"].values)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overhead_breakdown.png", dpi=300)
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


def create_comparison_table(all_results: List[Dict], output_dir: str):
    """Create and save comparison table."""
    data = []
    for result in all_results:
        config = result["config"]
        summary = result["summary"]
        
        data.append({
            "Strategy": config["strategy"].upper(),
            "Batch Size": config["batch_size"],
            "Grad Accum": config["gradient_accumulation_steps"],
            "Effective BS": config["batch_size"] * config["gradient_accumulation_steps"],
            "Tokens/sec (mean±std)": f"{summary['tokens_per_sec']['mean']:.1f}±{summary['tokens_per_sec']['std']:.1f}",
            "GPU Util %": f"{summary['gpu_utilization_pct']['mean']:.1f}",
            "Peak Memory GB": f"{summary['gpu_memory_peak_gb']['max']:.2f}",
            "Comm OH %": f"{summary['comm_overhead_pct']['mean']:.1f}",
            "CPU OH %": f"{summary['cpu_overhead_pct']['mean']:.1f}",
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = f"{output_dir}/comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to {csv_path}")
    
    # Print to console
    print("\n" + "="*120)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description="Compare FSDP Zero2 vs Zero3")
    parser.add_argument("--data_folder", type=str, default="data", help="Path to data folder")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 8, 16], help="Batch sizes to test")
    parser.add_argument("--grad_accum_steps", type=int, nargs="+", default=[1, 2], help="Gradient accumulation steps to test")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of training steps per config")
    parser.add_argument("--nproc_per_node", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="Output directory")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only generate plots from existing results")
    
    args = parser.parse_args()
    
    all_results = []
    
    if not args.skip_training:
        # Run all configurations
        for batch_size in args.batch_sizes:
            for grad_accum in args.grad_accum_steps:
                for strategy in ["zero2", "zero3"]:
                    try:
                        metrics_path = run_training(
                            strategy=strategy,
                            batch_size=batch_size,
                            gradient_accumulation_steps=grad_accum,
                            num_steps=args.num_steps,
                            data_folder=args.data_folder,
                            nproc_per_node=args.nproc_per_node,
                        )
                        
                        # Load metrics
                        metrics = load_metrics(metrics_path)
                        all_results.append(metrics)
                        
                    except Exception as e:
                        print(f"Error running {strategy} with bs={batch_size}, ga={grad_accum}: {e}")
                        continue
        
        # Save all results
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        # Load existing results
        print("Loading existing results...")
        with open(f"{args.output_dir}/all_results.json", "r") as f:
            all_results = json.load(f)
    
    # Create comparison visualizations
    print("\nGenerating comparison plots...")
    create_comparison_plots(all_results, args.output_dir)
    
    # Create comparison table
    create_comparison_table(all_results, args.output_dir)
    
    print(f"\n✅ Comparison complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
