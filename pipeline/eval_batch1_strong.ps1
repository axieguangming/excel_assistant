# Re-evaluate the strong model on batch 1 (396 items) with the CURRENT code.
#
# Why this is needed:
#   The stored batch-1 strong-model results (build/exp-{a,b}-strong.json) were
#   produced on 2026-11-02. Two relevant source files changed afterwards:
#       finetune/eval_sft.py   2026-11-04   (judging: read-only alignment, etc.)
#       excel_ops.py           2026-11-05   (describe_data became read-only)
#   Comparing 83.8% (old code, batch 1) with 78.7% (new code, batch 2) mixes
#   two variables: the dataset AND the code version. Re-running batch 1 with the
#   current code removes the code-version confounder.
#
# Outputs go to build/exp1b-*.json so the original files are preserved for
# comparison and audit trail.
#
# Remote API: does not use the local GPU. Estimate: 396 items, ~1.7 h.

$ErrorActionPreference = 'Continue'
$py = 'D:\Develops\miniforge\envs\excel_assistant\python.exe'
$root = 'D:\Projects\git repository\excel_assistant'
Set-Location $root

$baseUrl = 'https://api.deepseek.com'

$jobs = @(
    @{ tag = 'a-strong'; sft = 'datasets-holdout/sft-a'; tables = 'datasets-holdout/tables-a' },
    @{ tag = 'b-strong'; sft = 'datasets-holdout/sft-b'; tables = 'datasets-holdout/tables-b' }
)

foreach ($job in $jobs) {
    $out = "build/exp1b-$($job.tag).json"
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

Write-Host "`nAll batch-1 strong re-evaluations finished $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
