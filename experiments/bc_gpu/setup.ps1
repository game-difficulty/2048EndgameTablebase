$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot '../..')).Path
Push-Location $repositoryRoot
try {
    if (-not (Test-Path -LiteralPath "$experimentRoot/.venv/Scripts/python.exe")) {
        python -m venv "$experimentRoot/.venv"
        if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
    }
    & "$experimentRoot/.venv/Scripts/python.exe" -m pip install -r "$experimentRoot/requirements.txt"
    if ($LASTEXITCODE -ne 0) { throw 'dependency installation failed' }
    New-Item -ItemType Directory -Force "$experimentRoot/build" | Out-Null
    g++ -std=c++17 -O3 -march=native -fopenmp -I native_core/include experiments/bc_gpu/native_fixture.cpp native_core/src/CanonicalBatch.cpp -o experiments/bc_gpu/build/native_fixture.exe
    if ($LASTEXITCODE -ne 0) { throw 'native fixture build failed' }
} finally {
    Pop-Location
}
