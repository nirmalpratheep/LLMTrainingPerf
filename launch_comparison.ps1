# Launch FSDP comparison with torchrun (PowerShell)
# Usage: .\launch_comparison.ps1 [num_gpus]

param(
    [int]$NumGpus = 2
)

Write-Host "Launching FSDP comparison with $NumGpus GPUs..." -ForegroundColor Green

torchrun `
    --nproc_per_node=$NumGpus `
    compare_strategies.py `
    --data_folder=data `
    --batch_sizes 2 4 8 `
    --grad_accum_steps 1 2 `
    --num_steps=50 `
    --nproc_per_node=$NumGpus

Write-Host "Comparison complete! Check comparison_results/ for outputs." -ForegroundColor Green
