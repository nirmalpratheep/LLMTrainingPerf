#!/bin/bash

# Complete workflow: Find max batch sizes then run comparison
# Usage: ./run_full_comparison.sh [num_gpus]

NUM_GPUS=${1:-2}

echo "========================================"
echo "FSDP Complete Comparison Workflow"
echo "========================================"
echo "GPUs: $NUM_GPUS"
echo ""
echo "Step 1: Finding maximum batch sizes..."
echo "Step 2: Running full comparison"
echo "========================================"
echo ""

# Step 1: Find maximum batch sizes
echo ""
echo "🔍 Finding maximum batch sizes for each strategy..."
echo ""

python find_max_batch_size.py \
    --nproc_per_node=$NUM_GPUS \
    --search_method=linear \
    --start_bs=2 \
    --step=2

# Check if batch size finding was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to find maximum batch sizes"
    exit 1
fi

echo ""
read -p "Press Enter to continue with full comparison (or Ctrl+C to exit)..."
echo ""

# Step 2: Run comparison with discovered batch sizes
# Read the max batch sizes from results
if [ -f "results/max_batch_sizes.txt" ]; then
    ZERO3_MAX=$(grep "ZERO3:" results/max_batch_sizes.txt | awk '{print $2}')
    ZERO2_MAX=$(grep "ZERO2:" results/max_batch_sizes.txt | awk '{print $2}')
    
    echo "🚀 Running comparison with discovered max batch sizes..."
    echo "   Zero3 max: $ZERO3_MAX"
    echo "   Zero2 max: $ZERO2_MAX"
    echo ""
    
    # Use a range of batch sizes up to the max
    BATCH_SIZES="2 4 $ZERO3_MAX"
    if [ "$ZERO2_MAX" != "$ZERO3_MAX" ]; then
        BATCH_SIZES="$BATCH_SIZES $ZERO2_MAX"
    fi
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        compare_strategies.py \
        --data_folder=data \
        --batch_sizes $BATCH_SIZES \
        --grad_accum_steps 1 2 \
        --num_steps=50 \
        --warmup_steps=5 \
        --nproc_per_node=$NUM_GPUS
else
    echo "⚠️  Could not find max batch sizes, running with defaults..."
    ./launch_comparison.sh $NUM_GPUS
fi

echo ""
echo "========================================"
echo "✅ Complete comparison finished!"
echo "========================================"
echo "Results:"
echo "  - Max batch sizes: results/max_batch_sizes.txt"
echo "  - Comparison plots: comparison_results/"
echo "  - Comparison table: comparison_results/comparison_table.csv"
echo "========================================"
