# FSDP Training Performance Comparison

A comprehensive framework for comparing FSDP Zero2 vs Zero3 sharding strategies when training the QWEN 1.5B language model. This project measures and analyzes performance across multiple dimensions including throughput, GPU utilization, memory usage, and communication overhead.

## 🎯 Objectives

- **Compare FSDP Strategies**: Analyze Zero2 (SHARD_GRAD_OP) vs Zero3 (FULL_SHARD) performance
- **Comprehensive Metrics**: Track tokens/sec, GPU utilization, memory usage, communication and CPU overhead
- **GPU Optimizations**: Apply common optimizations (torch.compile, mixed precision, fused kernels, gradient accumulation)
- **Batch Size Analysis**: Test performance across varying batch sizes and gradient accumulation configurations

## 📊 What is FSDP?

**Fully Sharded Data Parallel (FSDP)** is PyTorch's distributed training strategy that enables training large models across multiple GPUs:

- **Zero2 (SHARD_GRAD_OP)**: Shards gradients and optimizer states across GPUs, but keeps model parameters replicated
- **Zero3 (FULL_SHARD)**: Shards model parameters, gradients, and optimizer states for maximum memory efficiency

### Zero2 vs Zero3 Trade-offs

| Aspect | Zero2 | Zero3 |
|--------|-------|-------|
| **Memory Usage** | Higher (parameters replicated) | Lower (parameters sharded) |
| **Communication** | Less (only gradients) | More (all-gather parameters) |
| **Throughput** | Generally faster | Slightly slower |
| **Best For** | Models that fit in GPU memory | Very large models |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- 2+ GPUs (recommended for effective FSDP testing)

### Installation

```bash
# Install dependencies
pip install -e .
```

### Running the Comparison

```bash
# Using bash (Linux/Mac)
./launch_comparison.sh 2  # for 2 GPUs

# Using PowerShell (Windows)
.\launch_comparison.ps1 2  # for 2 GPUs

# Or manually with torchrun
torchrun --nproc_per_node=2 compare_strategies.py
```

### Individual Strategy Testing

```bash
# Test Zero2 only
torchrun --nproc_per_node=2 training/train_zero2.py --batch_size=4 --num_steps=50

# Test Zero3 only
torchrun --nproc_per_node=2 training/train_zero3.py --batch_size=4 --num_steps=50
```

## 📁 Project Structure

```
LLMTrainingPerf/
├── utils/
│   ├── model_loader.py          # QWEN 1.5B model loading
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── fsdp_config.py            # FSDP Zero2/Zero3 configurations
│   └── metrics_tracker.py        # Comprehensive metrics collection
├── training/
│   ├── train_zero2.py            # Zero2 training loop
│   └── train_zero3.py            # Zero3 training loop
├── data/
│   └── sample_data.txt           # Sample training data
├── compare_strategies.py         # Main comparison script
├── launch_comparison.sh          # Bash launch script
└── launch_comparison.ps1         # PowerShell launch script
```

## 🔧 GPU Optimizations (Common to Both Strategies)

All training runs use these optimizations for fair comparison:

1. **torch.compile()**: JIT compilation for faster execution
2. **Mixed Precision (BF16)**: Reduced memory usage and faster computation
3. **Fused AdamW**: Optimized optimizer with fused kernels
4. **Gradient Accumulation**: Simulate larger batch sizes efficiently

## 📈 Metrics Tracked

The framework collects comprehensive performance metrics:

- **Tokens per Second**: Primary throughput metric
- **GPU Utilization**: Percentage of GPU compute used
- **GPU Memory**: Peak and average memory consumption per GPU
- **Communication Overhead**: Time spent in collective operations (all-reduce, all-gather)
- **CPU Overhead**: Time spent in data loading and preprocessing
- **End-to-End Training Time**: Total time per batch/epoch

## 📊 Output and Visualization

The comparison generates:

1. **Performance Plots**:
   - Tokens/sec vs Batch Size
   - GPU Memory vs Batch Size
   - Communication Overhead Comparison
   - GPU Utilization Heatmap
   - CPU/Communication Overhead Breakdown

2. **Comparison Table** (CSV): Detailed metrics for each configuration

3. **Raw Metrics** (JSON): Complete metrics for custom analysis

All results are saved to `comparison_results/`

## 🎛️ Configuration Options

### Comparison Script

```bash
python compare_strategies.py \
    --data_folder=data \
    --batch_sizes 2 4 8 16 \
    --grad_accum_steps 1 2 4 \
    --num_steps=50 \
    --nproc_per_node=2 \
    --output_dir=comparison_results
```

### Individual Training

```bash
torchrun --nproc_per_node=2 training/train_zero2.py \
    --batch_size=4 \
    --gradient_accumulation_steps=2 \
    --num_steps=100 \
    --learning_rate=1e-5 \
    --max_length=2048 \
    --output_dir=results/zero2 \
    --use_profiler  # Enable PyTorch profiler
```

## 📝 Adding Your Own Data

Place your training data in the `data/` folder. Supported formats:

- **Text files** (`.txt`): Plain text
- **JSON** (`.json`): `{"text": "your content"}`
- **JSONL** (`.jsonl`): One JSON object per line with `text` field

The data loader will automatically discover and load all files.

## 🔍 Interpreting Results

### Expected Patterns

- **Throughput**: Should increase with batch size until memory limits
- **Memory**: Zero3 uses ~40-60% less memory than Zero2
- **Communication**: Zero3 has higher overhead (~2-3x) due to parameter sharding
- **GPU Utilization**: Should be >70% for well-optimized training

### Performance Tips

- **For maximum throughput**: Use Zero2 with largest possible batch size
- **For memory efficiency**: Use Zero3 to fit larger models or batch sizes
- **Balance**: Use gradient accumulation to increase effective batch size without exceeding memory

## 🛠️ Troubleshooting

### Out of Memory

- Reduce batch size
- Increase gradient accumulation steps
- Use Zero3 instead of Zero2
- Reduce sequence length (`--max_length`)

### Low GPU Utilization

- Increase batch size
- Check data loading (should be <10% overhead)
- Verify torch.compile is working

### High Communication Overhead

- Expected for Zero3 (parameter sharding requires communication)
- Ensure fast GPU interconnect (NVLink, InfiniBand)
- Consider Zero2 if communication is bottleneck

## 📚 References

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [QWEN Model](https://huggingface.co/Qwen)
- [ZeRO Paper (Microsoft)](https://arxiv.org/abs/1910.02054)

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Feel free to open issues or submit pull requests.
