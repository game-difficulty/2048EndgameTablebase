param(
    [string]$CohortName = "leaderboard-expanded-2026-09-25",
    [ValidateRange(250, 10000)][int]$DelayMs = 500,
    [ValidateRange(0, 60000)][int]$BetweenPlayersMs = 3000,
    [int]$Limit = 0,
    [string]$Only,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$CohortDirectory = Join-Path $ProjectRoot "data\verse-history\cohorts\$CohortName"
$Arguments = @(
    (Join-Path $PSScriptRoot "bulk-import-verse-history.mjs"),
    "--cohort", (Join-Path $CohortDirectory "cohort.json"),
    "--output-root", (Join-Path $ProjectRoot "data\verse-history"),
    "--delay-ms", $DelayMs,
    "--between-players-ms", $BetweenPlayersMs
)
if ($Limit -gt 0) { $Arguments += @("--limit", $Limit) }
if ($Only) { $Arguments += @("--only", $Only) }
if ($DryRun) { $Arguments += "--dry-run" }

& node @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
