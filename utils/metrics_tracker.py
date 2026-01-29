"""Metrics tracking for training performance."""

import time
import json
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import psutil


try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: nvidia-ml-py not available, GPU utilization tracking disabled")


class MetricsTracker:
    """Track comprehensive training metrics."""
    
    def __init__(self, rank: int = 0, world_size: int = 1):
        """
        Initialize metrics tracker.
        
        Args:
            rank: Current process rank
            world_size: Total number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.metrics: Dict[str, List] = defaultdict(list)
        self.step_times: List[float] = []
        self.data_load_times: List[float] = []
        self.comm_times: List[float] = []
        
        # Initialize NVML for GPU monitoring
        if NVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
                self.nvml_enabled = True
            except:
                self.nvml_enabled = False
                print(f"Rank {rank}: Could not initialize NVML")
        else:
            self.nvml_enabled = False
    
    def start_step(self):
        """Mark the start of a training step."""
        self.step_start_time = time.time()
        self.data_load_start_time = time.time()
    
    def mark_data_loaded(self):
        """Mark that data loading is complete."""
        self.data_load_time = time.time() - self.data_load_start_time
        self.data_load_times.append(self.data_load_time)
    
    def record_comm_time(self, comm_time: float):
        """Record communication time for this step."""
        self.comm_times.append(comm_time)
    
    def end_step(
        self,
        loss: float,
        num_tokens: int,
        step: int,
    ):
        """
        Mark the end of a training step and record metrics.
        
        Args:
            loss: Training loss
            num_tokens: Number of tokens processed
            step: Current step number
        """
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        # Calculate tokens per second
        tokens_per_sec = num_tokens / step_time if step_time > 0 else 0
        
        # Get GPU metrics
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB
        gpu_memory_peak = torch.cuda.max_memory_allocated() / 1e9  # GB
        
        # Get GPU utilization
        gpu_utilization = self._get_gpu_utilization()
        
        # Calculate overhead percentages
        cpu_overhead_pct = (self.data_load_time / step_time * 100) if step_time > 0 else 0
        comm_overhead_pct = (self.comm_times[-1] / step_time * 100) if self.comm_times and step_time > 0 else 0
        
        # Record metrics
        self.metrics["step"].append(step)
        self.metrics["loss"].append(loss)
        self.metrics["tokens_per_sec"].append(tokens_per_sec)
        self.metrics["step_time"].append(step_time)
        self.metrics["gpu_memory_allocated_gb"].append(gpu_memory_allocated)
        self.metrics["gpu_memory_reserved_gb"].append(gpu_memory_reserved)
        self.metrics["gpu_memory_peak_gb"].append(gpu_memory_peak)
        self.metrics["gpu_utilization_pct"].append(gpu_utilization)
        self.metrics["cpu_overhead_pct"].append(cpu_overhead_pct)
        self.metrics["comm_overhead_pct"].append(comm_overhead_pct)
        self.metrics["data_load_time"].append(self.data_load_time)
        
        # Log on rank 0
        if self.rank == 0 and step % 10 == 0:
            print(f"Step {step}: Loss={loss:.4f}, Tokens/s={tokens_per_sec:.1f}, "
                  f"GPU Mem={gpu_memory_allocated:.2f}GB, GPU Util={gpu_utilization:.1f}%, "
                  f"CPU OH={cpu_overhead_pct:.1f}%, Comm OH={comm_overhead_pct:.1f}%")
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        if not self.nvml_enabled:
            return 0.0
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            return float(util.gpu)
        except:
            return 0.0
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of all metrics.
        
        Returns:
            Dictionary of metric summaries
        """
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values and isinstance(values[0], (int, float)):
                import numpy as np
                summary[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        
        return summary
    
    def save(self, filepath: str, config: Optional[Dict] = None):
        """
        Save metrics to JSON file.
        
        Args:
            filepath: Path to save metrics
            config: Optional configuration dictionary to save
        """
        if self.rank != 0:
            return  # Only rank 0 saves
        
        # Convert defaultdict to regular dict
        metrics_dict = {k: v for k, v in self.metrics.items()}
        
        output = {
            "config": config or {},
            "metrics": metrics_dict,
            "summary": self.get_summary(),
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
    
    def reset_peak_memory(self):
        """Reset peak memory stats."""
        torch.cuda.reset_peak_memory_stats()
    
    def __del__(self):
        """Cleanup NVML on deletion."""
        if self.nvml_enabled:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


if __name__ == "__main__":
    # Test metrics tracker
    print("Testing MetricsTracker...\n")
    
    if torch.cuda.is_available():
        tracker = MetricsTracker(rank=0, world_size=1)
        
        # Simulate a few training steps
        for step in range(5):
            tracker.start_step()
            time.sleep(0.01)  # Simulate data loading
            tracker.mark_data_loaded()
            
            # Simulate training
            time.sleep(0.02)
            
            # Simulate communication
            tracker.record_comm_time(0.005)
            
            # End step
            tracker.end_step(
                loss=2.5 - step * 0.1,
                num_tokens=4096,
                step=step,
            )
        
        # Get summary
        summary = tracker.get_summary()
        print("\nMetrics Summary:")
        for metric, stats in summary.items():
            print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        # Save metrics
        tracker.save("test_metrics.json", config={"test": True})
        print("\nMetricsTracker test successful!")
    else:
        print("CUDA not available, skipping test")
