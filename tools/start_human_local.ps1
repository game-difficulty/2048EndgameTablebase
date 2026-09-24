param(
    [string]$Python = 'C:/Anaconda/python.exe',
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [switch]$SkipBuild,
    [switch]$Stop
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot
$localDir = Join-Path $projectRoot 'tmp/human-local'
$pidFile = Join-Path $localDir 'server.pid'
New-Item -ItemType Directory -Force -Path $localDir | Out-Null
$existing = $null
if (Test-Path -LiteralPath $pidFile) {
    $serverPid = 0
    if ([int]::TryParse((Get-Content -Raw -LiteralPath $pidFile).Trim(), [ref]$serverPid)) {
        $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid"
        if ($existing -and $existing.CommandLine -notmatch 'uvicorn backend\.human_play\.local_app:app') {
            throw 'The recorded PID belongs to another process. No process was stopped.'
        }
    }
}
if ($Stop) {
    if ($existing) { Stop-Process -Id $existing.ProcessId; Write-Output 'Local human site stopped.' }
    else { Write-Output 'Local human site is not running.' }
    exit 0
}
if ($existing) { Write-Output "Local human site already running (PID $($existing.ProcessId)). Use -Stop before rebuilding."; exit 0 }
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { throw "Port $Port is already in use." }
if (!$SkipBuild) {
    Push-Location (Join-Path $projectRoot 'frontend')
    try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' } }
    finally { Pop-Location }
}
if (!(Test-Path -LiteralPath (Join-Path $projectRoot 'frontend/dist/human/index.html'))) { throw 'Build frontend first.' }
$previousLocalDir = $env:HUMAN_LOCAL_DIR
try {
    $env:HUMAN_LOCAL_DIR = $localDir
    $process = Start-Process -FilePath $Python -ArgumentList @('-m', 'uvicorn', 'backend.human_play.local_app:app', '--host', '127.0.0.1', '--port', "$Port", '--log-level', 'warning') `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $localDir 'server.log') -RedirectStandardError (Join-Path $localDir 'server.err.log')
    Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii
} finally { $env:HUMAN_LOCAL_DIR = $previousLocalDir }
$url = "http://127.0.0.1:$Port/human/"
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 300
    if ($process.HasExited) { throw "Server exited. Read $localDir/server.err.log" }
    try { $ready = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2; if ($ready.StatusCode -eq 200) { Write-Output "Local human site ready: $url (PID $($process.Id))"; exit 0 } }
    catch { if ($attempt -eq 29) { throw } }
}
