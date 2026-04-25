param(
    [string]$PythonExe = "",
    [string]$AppName = "2048EndgameTablebase",
    [string]$DistDir = "dist",
    [string]$WorkDir = "build\pyinstaller\windows"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$NativeDir = Join-Path $Root "native_core"
$FrontendDist = Join-Path $Root "frontend\dist"
$DistPath = Join-Path $Root $DistDir
$WorkPath = Join-Path $Root $WorkDir

function Resolve-Python {
    param([string]$Requested)

    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        return $Requested
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    throw "Python interpreter not found. Pass -PythonExe explicitly."
}

function Assert-Path {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing $Label: $Path"
    }
}

function Resolve-NativeExtension {
    param([string]$ModuleName)

    $matches = Get-ChildItem -Path $NativeDir -Filter "$ModuleName*.pyd" -File
    if ($matches.Count -eq 0) {
        throw "Missing native extension for $ModuleName under $NativeDir"
    }
    if ($matches.Count -gt 1) {
        throw "Multiple native extensions found for $ModuleName under $NativeDir"
    }
    return $matches[0].FullName
}

function Add-DataArg {
    param(
        [System.Collections.Generic.List[string]]$Args,
        [string]$Source,
        [string]$Dest
    )

    $Args.Add("--add-data")
    $Args.Add("$Source;$Dest")
}

function Add-BinaryArg {
    param(
        [System.Collections.Generic.List[string]]$Args,
        [string]$Source,
        [string]$Dest
    )

    $Args.Add("--add-binary")
    $Args.Add("$Source;$Dest")
}

$PythonExe = Resolve-Python -Requested $PythonExe

Assert-Path -Path $FrontendDist -Label "frontend build output"
Assert-Path -Path (Join-Path $Root "docs_and_configs\default_patterns.json") -Label "default pattern config"
Assert-Path -Path (Join-Path $Root "docs_and_configs\themes.json") -Label "theme config"
Assert-Path -Path (Join-Path $Root "docs_and_configs\help") -Label "help docs"
Assert-Path -Path (Join-Path $Root "pic") -Label "picture assets"
Assert-Path -Path (Join-Path $Root "font") -Label "font directory"
Assert-Path -Path (Join-Path $Root "mathjax") -Label "mathjax directory"
Assert-Path -Path (Join-Path $Root "favicon.ico") -Label "favicon"
Assert-Path -Path (Join-Path $Root "backend_server.py") -Label "backend entrypoint"
Assert-Path -Path (Join-Path $Root "7zip\7z.dll") -Label "7zip dll"
Assert-Path -Path (Join-Path $Root "7zip\7z.exe") -Label "7zip executable"

$AiCore = Resolve-NativeExtension -ModuleName "ai_core"
$MoverCore = Resolve-NativeExtension -ModuleName "mover_core"
$FormationCore = Resolve-NativeExtension -ModuleName "formation_core"
$BookgenNative = Join-Path $NativeDir "bookgen_native.dll"
$Libgcc = Join-Path $NativeDir "libgcc_s_seh-1.dll"
$Libgomp = Join-Path $NativeDir "libgomp-1.dll"
$Libwinpthread = Join-Path $NativeDir "libwinpthread-1.dll"
$IconPath = Join-Path $Root "pic\2048_2.ico"

Assert-Path -Path $BookgenNative -Label "bookgen native DLL"
Assert-Path -Path $Libgcc -Label "libgcc runtime DLL"
Assert-Path -Path $Libgomp -Label "libgomp runtime DLL"
Assert-Path -Path $Libwinpthread -Label "libwinpthread runtime DLL"

& $PythonExe -m PyInstaller --version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not available for $PythonExe"
}

$PyArgs = [System.Collections.Generic.List[string]]::new()
$PyArgs.AddRange(@(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--windowed",
    "--name", $AppName,
    "--paths", $Root,
    "--distpath", $DistPath,
    "--workpath", $WorkPath,
    "--specpath", $WorkPath,
    "--hidden-import", "native_core.ai_core",
    "--hidden-import", "native_core.mover_core",
    "--hidden-import", "native_core.formation_core",
    "--collect-submodules", "uvicorn",
    "--collect-submodules", "webview"
))

Add-DataArg -Args $PyArgs -Source (Join-Path $Root "docs_and_configs\default_patterns.json") -Dest "docs_and_configs"
Add-DataArg -Args $PyArgs -Source (Join-Path $Root "docs_and_configs\themes.json") -Dest "docs_and_configs"
Add-DataArg -Args $PyArgs -Source (Join-Path $Root "docs_and_configs\help") -Dest "docs_and_configs/help"
Add-DataArg -Args $PyArgs -Source (Join-Path $Root "pic") -Dest "pic"
Add-DataArg -Args $PyArgs -Source (Join-Path $Root "font") -Dest "font"
Add-DataArg -Args $PyArgs -Source (Join-Path $Root "favicon.ico") -Dest "."
Add-DataArg -Args $PyArgs -Source (Join-Path $Root "mathjax") -Dest "mathjax"
Add-DataArg -Args $PyArgs -Source $FrontendDist -Dest "frontend/dist"
Add-DataArg -Args $PyArgs -Source (Join-Path $Root "7zip\7z.dll") -Dest "."
Add-DataArg -Args $PyArgs -Source (Join-Path $Root "7zip\7z.exe") -Dest "."

Add-BinaryArg -Args $PyArgs -Source $BookgenNative -Dest "native_core"
Add-BinaryArg -Args $PyArgs -Source $Libgcc -Dest "native_core"
Add-BinaryArg -Args $PyArgs -Source $Libgomp -Dest "native_core"
Add-BinaryArg -Args $PyArgs -Source $Libwinpthread -Dest "native_core"

if (Test-Path -LiteralPath $IconPath) {
    $PyArgs.Add("--icon")
    $PyArgs.Add($IconPath)
}

$PyArgs.Add((Join-Path $Root "backend_server.py"))

& $PythonExe @PyArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller packaging failed with exit code $LASTEXITCODE"
}

Write-Host "Windows package created at $(Join-Path $DistPath $AppName)"
