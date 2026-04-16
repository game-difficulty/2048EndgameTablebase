# Packaging and Distribution Guide: PyInstaller

This guide explains how to bundle the desktop application with PyInstaller.

## 1. Prerequisites

1. Build the Vue frontend:
   ```bash
   cd frontend
   npm install
   npm run build
   ```

2. Ensure native runtime binaries already exist in `ai_and_sort/`:
   - `ai_core*.pyd`
   - `mover_core*.pyd`
   - `sort_wrapper.dll`
   - required runtime DLLs such as `libgcc_s_seh-1.dll`, `libgomp-1.dll`, `libwinpthread-1.dll`

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

## 2. Recommended PyInstaller command

```powershell
pyinstaller --noconfirm --onedir --windowed `
    --name "2048EndgameTablebase" `
    --add-data "src/docs_and_configs/default_patterns.json;docs_and_configs" `
    --add-data "src/docs_and_configs/themes.json;docs_and_configs" `
    --add-data "src/docs_and_configs/help;docs_and_configs/help" `
    --add-data "src/pic;pic" `
    --add-data "src/font;font" `
    --add-data "src/favicon.ico;." `
    --add-data "src/mathjax;mathjax" `
    --add-data "src/minigames/pic;minigames/pic" `
    --add-data "src/frontend/dist;frontend/dist" `
    --add-binary "src/ai_and_sort/libgcc_s_seh-1.dll;ai_and_sort" `
    --add-binary "src/ai_and_sort/libgomp-1.dll;ai_and_sort" `
    --add-binary "src/ai_and_sort/libwinpthread-1.dll;ai_and_sort" `
    --add-binary "src/ai_and_sort/sort_wrapper.dll;ai_and_sort" `
    --add-data "src/7zip/7z.dll;." `
    --add-data "src/7zip/7z.exe;." `
    --icon "src/pic/2048_2.ico" `
    src/backend_server.py
```

## 3. Source layout assumptions

The repository now uses `ai_and_sort/` as the only native-source directory.
`AIPlayer_cpp/` is treated as a local legacy workspace and is not part of the packaged build.

## 4. Final distribution notes

The packaged application creates these files at runtime:
- `docs_and_configs/config`
- `docs_and_configs/mistakes_book.pkl`
- `logger.txt`

These should stay outside version control.

## 5. Troubleshooting

### Missing native module or DLL
Check that the following runtime files exist in `dist/2048EndgameTablebase/ai_and_sort/`:
- `ai_core*.pyd`
- `mover_core*.pyd`
- `sort_wrapper.dll`
- `libgcc_s_seh-1.dll`
- `libgomp-1.dll`
- `libwinpthread-1.dll`

### Frozen startup path issues
`backend_server.py` resolves resources using runtime-relative paths and now also handles bundled MotW cleanup and backend child-process shutdown. Keep distribution files together inside the unpacked directory.
