param(
    [string]$CohortName = "leaderboard-expanded-2026-09-25",
    [ValidateRange(250, 10000)][int]$DelayMs = 750
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$CohortDirectory = Join-Path $ProjectRoot "data\verse-history\cohorts\$CohortName"

& node (Join-Path $PSScriptRoot "collect-leaderboard-cohort.mjs") `
    --cohort-dir $CohortDirectory `
    --delay-ms $DelayMs

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
