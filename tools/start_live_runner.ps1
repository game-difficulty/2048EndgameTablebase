param(
    [string]$Python = 'C:/Anaconda/python.exe',
    [string]$EngineRoot = 'C:/Apps/2048endgameTablebase/src',
    [string]$SecretFile = "$env:USERPROFILE/.config/2048tables/live.env"
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '[\\/]tools[\\/]live_runner\.py'
}
if ($running) { throw "Live runner already active: $($running.ProcessId -join ', ')" }
$line = Get-Content -LiteralPath $SecretFile | Where-Object { $_.StartsWith('LIVE_PUBLISH_TOKEN=') } | Select-Object -First 1
if (!$line -or $line.Substring(19).Length -lt 32) { throw 'Missing publisher credential' }
$data = Join-Path $root 'data'
New-Item -ItemType Directory -Force $data | Out-Null
$previous = $env:LIVE_PUBLISH_TOKEN
try {
    $env:LIVE_PUBLISH_TOKEN = $line.Substring(19)
    $arguments = @(
        '-u', ('"' + (Join-Path $PSScriptRoot 'live_runner.py') + '"'),
        '--engine-root', ('"' + $EngineRoot + '"'),
        '--interval', '0.08', '--threads', '1',
        '--log-file', ('"' + (Join-Path $data 'live-runner.log') + '"')
    )
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $data 'live-runner.stdout.log') `
        -RedirectStandardError (Join-Path $data 'live-runner.stderr.log')
    $process.PriorityClass = 'BelowNormal'
    Write-Output "Live runner started: PID $($process.Id), interval >=80ms, one thread, BelowNormal priority."
} finally { $env:LIVE_PUBLISH_TOKEN = $previous }
