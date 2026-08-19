param(
    [string]$ConfigPath = "docs_and_configs/remote_worker.local.json",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ResolvedConfig = if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath
} else {
    Join-Path $ProjectRoot $ConfigPath
}

if (-not (Test-Path -LiteralPath $ResolvedConfig -PathType Leaf)) {
    throw "Worker config not found: $ResolvedConfig"
}
if (-not $env:TABLEBASE_WORKER_TOKEN) {
    $env:TABLEBASE_WORKER_TOKEN = [System.Environment]::GetEnvironmentVariable(
        "TABLEBASE_WORKER_TOKEN",
        "User"
    )
}
if (-not $env:TABLEBASE_WORKER_TOKEN) {
    $env:TABLEBASE_WORKER_TOKEN = [System.Environment]::GetEnvironmentVariable(
        "TABLEBASE_WORKER_TOKEN",
        "Machine"
    )
}
if (-not $env:TABLEBASE_WORKER_TOKEN) {
    throw "TABLEBASE_WORKER_TOKEN is not set in the process, user, or machine environment."
}

Set-Location -LiteralPath $ProjectRoot
if ($PythonExe) {
    & $PythonExe -m tools.tablebase_worker --config $ResolvedConfig
} else {
    $ProjectPython = Join-Path $ProjectRoot "..\myenv\Scripts\python.exe"
    if (Test-Path -LiteralPath $ProjectPython -PathType Leaf) {
        & $ProjectPython -m tools.tablebase_worker --config $ResolvedConfig
    } else {
        & py -3.12 -m tools.tablebase_worker --config $ResolvedConfig
    }
}
exit $LASTEXITCODE
