# Evaluate the strong model on batch 2 (clean holdout test set, 799 items).
#
# Purpose:
#   dpo2 scored 89.5% on batch 2 while v4 scored 82.7% (same batch, +6.8 points,
#   p < 0.0001). The strong model scored 83.8% on batch 1, where v4 scored 82.1%.
#   Since v4 differs by only 0.7 points between the two batches, the batches are
#   equally hard. Measuring the strong model on batch 2 turns the cross-batch
#   inference into a direct comparison.
#
# Remote API (-base-url): does not use the local GPU.
# Estimate: 799 items, ~3.5 h.

$ErrorActionPreference = 'Continue'
$py = 'D:\Develops\miniforge\envs\excel_assistant\python.exe'
$root = 'D:\Projects\git repository\excel_assistant'
Set-Location $root

$baseUrl = 'https://api.deepseek.com'

$jobs = @(
    @{ tag = 'a-strong'; sft = 'datasets-holdout2/sft-a'; tables = 'datasets-holdout2/tables-a' },
    @{ tag = 'b-strong'; sft = 'datasets-holdout2/sft-b'; tables = 'datasets-holdout2/tables-b' }
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
        --base-url $baseUrl `
        --out $out
    Write-Host "===== DONE $($job.tag)  $(Get-Date -Format 'HH:mm:ss') =====" -ForegroundColor Green
}

Write-Host "`nAll strong-model evaluations finished $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
