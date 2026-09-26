[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$Username,

    [string]$OutputRoot,

    [ValidateRange(0, 10000)]
    [int]$DelayMs = 250,

    [ValidateRange(0, 8)]
    [int]$Retries = 3,

    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$toolDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $toolDirectory)
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $projectRoot 'data\verse-history'
}

$importerPath = Join-Path $toolDirectory 'import-verse-history-api.mjs'
$arguments = @(
    $importerPath,
    '--username', $Username,
    '--output-root', [IO.Path]::GetFullPath($OutputRoot),
    '--delay-ms', [string]$DelayMs,
    '--retries', [string]$Retries
)
if ($Force) { $arguments += '--force' }

& node @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Verse API history import failed for $Username"
}
