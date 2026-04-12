# Packaging and Distribution Guide: PyInstaller

This guide explains how to bundle the **2048 Endgame Tablebase** application into a standalone executable for distribution using PyInstaller.

## 1. Prerequisites

1. **Frontend Build**: Ensure the Vue 3 frontend is compiled.
   ```bash
   cd frontend
   npm install
   npm run build
   ```
   This creates the `frontend/dist/` directory which the backend serves as static files.

2. **Python Environment**: Install all dependencies from the provided requirements file.
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

---

## 2. Packaging Steps

We recommend using the **Directory Mode** (`--onedir`) for faster startup and easier debugging of asset paths.

### Recommended Command

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

### Key Flags Explained:
- `--onedir` / `-D`: Creates a folder containing the executable and all dependencies (recommended).
- `--windowed` / `-w`: Prevents a console window from appearing when running the app.
- `--add-data`: Bundles non-binary files (configs, frontend, assets).
- `--add-binary`: Bundles DLLs or shared libraries into specific subfolders.
- `backend_server.py`: The main entry point script.

---

## 3. Final Distribution Structure

After running PyInstaller, the `dist/2048EndgameTablebase/` folder will be created. This is what you distribute to users.

```text
2048EndgameTablebase/            # The main distribution folder
├── 2048EndgameTablebase.exe     # The main executable
├── _internal/                   # Python runtime, DLLs, and bundled libraries
│   ├── egtb_core/               # Bundled core logic
│   ├── backend/
│   ├── numpy/
│   ├── numba/
│   └── ...
├── docs_and_configs/            # Essential configs and help files
│   ├── help/
│   ├── patterns_config.json
│   ├── default_patterns.json
│   └── themes.json
├── frontend/
│   └── dist/                    # Compiled Vue static files
├── pic/                         # UI Assets
├── font/                        # Typography
└── favicon.ico                  # Icon
```

### User-Generated Files
When the user runs the `.exe`, the following files will be automatically created in the folder (if they don't exist):
- `docs_and_configs/config`: User settings.
- `docs_and_configs/mistakes_book.pkl`: Saved mistakes.
- `logger.txt`: Runtime logs.

---

## 4. Troubleshooting

### 1. Numba Cache Issues
Numba may try to write to the `_internal` directory which might be read-only. We handle this in `Config.py` by ensuring configurations are loaded relative to the executable path.

### 2. File Path Resolution
Ensure all paths in your Python code use `os.path.dirname(__file__)` or `sys._MEIPASS` (PyInstaller's temp directory) to resolve bundled assets. 

Example:
```python
import sys
import os

if getattr(sys, 'frozen', False):
    # Running in a bundle
    base_path = sys._MEIPASS
else:
    # Running in normal Python
    base_path = os.path.dirname(__file__)
```

### 3. Missing Dependencies
If the app crashes with `ModuleNotFoundError`, you may need to add `--hidden-import` for dynamic imports or libraries like `cpuinfo` that PyInstaller might miss.
