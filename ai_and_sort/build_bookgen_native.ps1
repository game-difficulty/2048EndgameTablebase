param(
    [string]$Compiler = "C:\Apps\mingw64\bin\g++.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Compiler)) {
    throw "Compiler not found: $Compiler"
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Include = Join-Path $Root "include"
$Output = Join-Path $Root "bookgen_native.dll"
$SortLibrary = Join-Path $Root "libx86simdsortcpp.a"

$Sources = @(
    (Join-Path $Root "xss_wrapper.cpp"),
    (Join-Path $Root "src\bookgen_native_wrapper.cpp"),
    (Join-Path $Root "src\booksolver_utils_wrapper.cpp"),
    (Join-Path $Root "src\unique_wrapper.cpp"),
    (Join-Path $Root "src\merge_tree_wrapper.cpp")
)

$Args = @(
    "-O3",
    "-march=x86-64-v2",
    "-std=c++17",
    "-shared",
    "-fopenmp",
    "-ffunction-sections",
    "-fdata-sections",
    "-static-libstdc++",
    "-static-libgcc",
    "-I",
    $Root,
    "-I",
    $Include
)

$Args += $Sources
$Args += @(
    $SortLibrary,
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
