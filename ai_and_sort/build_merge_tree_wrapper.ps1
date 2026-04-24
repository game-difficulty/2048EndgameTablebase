param(
    [string]$Compiler = "C:\Apps\mingw64\bin\g++.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Compiler)) {
    throw "Compiler not found: $Compiler"
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $Root "src\merge_tree_wrapper.cpp"
$Include = Join-Path $Root "include"
$Output = Join-Path $Root "merge_tree_wrapper.dll"

$Args = @(
    "-O3",
    "-march=x86-64-v3",
    "-std=c++17",
    "-shared",
    "-fopenmp",
    "-ffunction-sections",
    "-fdata-sections",
    "-static-libstdc++",
    "-static-libgcc",
    "-I",
    $Include,
    $Source,
    "-Wl,--gc-sections",
    "-s",
    "-o",
    $Output
)

& $Compiler @Args

if ($LASTEXITCODE -ne 0) {
    throw "Compiler failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $Output)) {
    throw "Build finished without producing $Output"
}

Write-Host "Built $Output"
