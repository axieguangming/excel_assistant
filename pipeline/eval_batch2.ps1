# Evaluate v4 and DPO round 2 on batch 2 (clean holdout test set, 799 items).
#
# Why this batch is required:
#   Round 1 DPO took its 44 preference pairs from batch-1 holdout failures, then
#   reported scores on that same batch. The entire gain came from memorization
#   (44 trained samples: +11; 352 untouched samples: -2).
#   Batch 2 is fully independent: zero overlap with training tables, batch 1, or
#   any preference pair.
#
# Sequential execution: only one GPU is available.
# Estimate: 799 items x 15.8 s ~ 3.5 h per model, ~7 h total.

$ErrorActionPreference = 'Continue'
$py = 'D:\Develops\miniforge\envs\excel_assistant\python.exe'
$root = 'D:\Projects\git repository\excel_assistant'
Set-Location $root

$jobs = @(
    @{ tag = 'a-v4';   sft = 'datasets-holdout2/sft-a'; tables = 'datasets-holdout2/tables-a'; adapter = 'build/lora-v4' },
    @{ tag = 'b-v4';   sft = 'datasets-holdout2/sft-b'; tables = 'datasets-holdout2/tables-b'; adapter = 'build/lora-v4' },
    @{ tag = 'a-dpo2'; sft = 'datasets-holdout2/sft-a'; tables = 'datasets-holdout2/tables-a'; adapter = 'build/lora-v4-dpo2' },
    @{ tag = 'b-dpo2'; sft = 'datasets-holdout2/sft-b'; tables = 'datasets-holdout2/tables-b'; adapter = 'build/lora-v4-dpo2' }
)

foreach ($job in $jobs) {
    $out = "build/exp2-$($job.tag).json"
    if (Test-Path $out) {
        Write-Host "[skip] $out already exists" -ForegroundColor Yellow
        continue
    }
    Write-Host "`n===== START $($job.tag)  $(Get-Date -Format 'HH:mm:ss') =====" -ForegroundColor Cyan
    & $py -m finetune.eval_sft `
        --sft $job.sft `
        --split test `
        --tables-dir $job.tables `
        --adapter $job.adapter `
        --out $out
    Write-Host "===== DONE $($job.tag)  $(Get-Date -Format 'HH:mm:ss') =====" -ForegroundColor Green
}

Write-Host "`nAll evaluations finished $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
