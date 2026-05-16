# Packaging and Distribution Guide

This guide records the current Windows packaging command. The checked-in
`2048EndgameTablebase.spec` is the source of truth; use it instead of copying a
long ad hoc PyInstaller command.

## 1. Prerequisites

Build the frontend:

```powershell
cd C:\Apps\2048endgameTablebase\src\frontend
npm install
npm run build
```

Build the native modules:

```powershell
cd C:\Apps\2048endgameTablebase\src
cmake -S .\native_core -B .\native_core\build-formation -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build .\native_core\build-formation --config Release --target ai_core mover_core formation_core bookgen_native -j
```

The package expects these native runtime files under `src\native_core`:

- `ai_core*.pyd`
- `mover_core*.pyd`
- `formation_core*.pyd`
- `bookgen_native.dll`
- `libgcc_s_seh-1.dll`
- `libgomp-1.dll`
- `libwinpthread-1.dll`

The package also includes `src\7zip\7z.exe` and `src\7zip\7z.dll`, which are
used by the compressed temporary-file option.

## 2. Windows Package

From `C:\Apps\2048endgameTablebase`:

```powershell
.\myenv\Scripts\activate
pyinstaller --noconfirm --clean 2048EndgameTablebase.spec
```

The onedir output is:

```text
C:\Apps\2048endgameTablebase\dist\2048EndgameTablebase
```

If a timestamped archive is needed:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
.\src\7zip\7z.exe a -t7z -mx=7 ".\dist\2048EndgameTablebase-windows-$stamp.7z" ".\dist\2048EndgameTablebase"
```

## 3. Linux Package

Use the Docker packaging workflow:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy Bypass -File "C:\Apps\2048endgameTablebase\linux_build\build_linux_docker.ps1"
```

Use `-SkipImageBuild` only when files under `linux_build\docker` have not
changed:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy Bypass -File "C:\Apps\2048endgameTablebase\linux_build\build_linux_docker.ps1" -SkipImageBuild
```

Linux artifacts are written to:

```text
C:\Apps\2048endgameTablebase\linux_build\artifacts
```

## 4. Packaged Data Checklist

The Windows and Linux packages must include:

- `docs_and_configs/default_patterns.json`
- `docs_and_configs/patterns_config.json`
- `docs_and_configs/themes.json`
- `docs_and_configs/runtime_deletion_threshold.txt`
- `docs_and_configs/help`
- `frontend/dist`
- `pic`
- `font`
- `favicon.ico`
- `mathjax`
- native `ai_core`, `mover_core`, `formation_core`, and `bookgen_native`

Runtime-created files such as `docs_and_configs/config`,
`docs_and_configs/mistakes_book.pkl`, and `logger.txt` should stay outside
version control.
