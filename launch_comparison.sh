#!/bin/bash

# Launch FSDP comparison with torchrun
# Usage: ./launch_comparison.sh [num_gpus]

NUM_GPUS=${1:-2}

echo "========================================"
echo "FSDP Zero2 vs Zero3 Comparison"
echo "========================================"
echo "GPUs: $NUM_GPUS"
echo "Warmup steps: 5"
echo "Training steps: 50 per config"
echo "Batch sizes: 2, 4, 8"
echo "Gradient accumulation: 1, 2"
echo "========================================"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    compare_strategies.py \
    --data_folder=data \
    --batch_sizes 2 4 8 \
    --grad_accum_steps 1 2 \
    --num_steps=50 \
    --warmup_steps=5 \
    --nproc_per_node=$NUM_GPUS

echo ""
echo "========================================"
echo "Comparison complete!"
echo "Results saved to: comparison_results/"
echo "========================================"
