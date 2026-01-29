#!/bin/bash

# Launch FSDP comparison with torchrun
# Usage: ./launch_comparison.sh [num_gpus]

NUM_GPUS=${1:-2}

echo "Launching FSDP comparison with $NUM_GPUS GPUs..."

torchrun \
    --nproc_per_node=$NUM_GPUS \
    compare_strategies.py \
    --data_folder=data \
    --batch_sizes 2 4 8 \
    --grad_accum_steps 1 2 \
    --num_steps=50 \
    --nproc_per_node=$NUM_GPUS

echo "Comparison complete! Check comparison_results/ for outputs."
