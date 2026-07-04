param(
    [string]$Output = "$env:TEMP\2048tables-cloud-src.tgz"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (Test-Path $Output) {
        Remove-Item $Output -Force
    }

    tar -czf $Output `
        --exclude .git `
        --exclude frontend/node_modules `
        --exclude frontend/.vite `
        --exclude native_core/build-* `
        --exclude native_core/x86simdsort/x86-simd-sort/builddir `
        --exclude __pycache__ `
        --exclude '*.pyc' `
        --exclude cloud_server.log `
        --exclude cloud_server.err.log `
        .

    Get-Item $Output | Select-Object FullName, Length
} finally {
    Pop-Location
}

