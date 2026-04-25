param(
    [string]$BuildDir = "",
    [int]$Jobs = 4,
    [string]$PythonExe = "python",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $Root
$BuildDir = if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    Join-Path $Root "build-formation"
} else {
    $BuildDir
}
$Cache = Join-Path $BuildDir "CMakeCache.txt"
$NanobindSrc = Join-Path $Root "extern\nanobind\src"
$OriginalPythonPath = $env:PYTHONPATH

function Invoke-StubGen([string]$ModuleName) {
    $stubgenCode = @'
import sys

module_name = sys.argv[1]
output_dir = sys.argv[2]
vendored_src = sys.argv[3]

if vendored_src:
    sys.path.insert(0, vendored_src)
    import stubgen as nb_stubgen
else:
    import nanobind.stubgen as nb_stubgen

nb_stubgen.main(['-m', module_name, '-O', output_dir])
'@

    & $PythonExe -c $stubgenCode $ModuleName $Root $NanobindSrc
    if ($LASTEXITCODE -ne 0) {
        throw "Stub generation failed for $ModuleName with exit code $LASTEXITCODE"
    }
}

if (-not $SkipBuild) {
    if (-not (Test-Path -LiteralPath $Cache)) {
        throw "Missing CMake cache: $Cache. Configure native_core/build-formation first."
    }

    & cmake --build $BuildDir --target ai_core formation_core -j $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake build failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $NanobindSrc)) {
    $NanobindSrc = ""
}

Push-Location $RepoRoot
try {
    Invoke-StubGen "native_core.ai_core"
    Invoke-StubGen "native_core.formation_core"
} finally {
    Pop-Location
    $env:PYTHONPATH = $OriginalPythonPath
}

foreach ($stub in @("ai_core.pyi", "formation_core.pyi")) {
    $stubPath = Join-Path $Root $stub
    if (-not (Test-Path -LiteralPath $stubPath)) {
        throw "Expected stub was not produced: $stubPath"
    }
}

Write-Host "Updated ai_core.pyi and formation_core.pyi in $Root"
