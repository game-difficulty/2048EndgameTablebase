# GitHub Upload Guide: 2048 Endgame Tablebase

This document outlines the recommended inclusion list for publishing the repository to GitHub.
The goal is to keep all source needed for building the app while excluding local state, table data, and generated binaries.

## 1. Upload these directories and files

### Application source
- `backend_server.py`
- `Config.py`, `SignalHub.py`, `error_bridge.py`
- `backend/`
- `egtb_core/`
- `minigames/`
- `frontend/`

### Native source
- `ai_and_sort/`
  - Include:
    - `src/`
    - `include/`
    - `x86simdsort/x86-simd-sort/`
    - `egtb_data.7z`
    - `CMakeLists.txt`
    - `make.sh`
    - `README.md`
    - `ai_core.pyi`
  - Exclude local WIP files:
    - `include/BookGenerator.h`
    - `include/BookGeneratorUtils.h`
    - `src/BookGenerator.cpp`
    - `src/BookGeneratorUtils.cpp`
  - Exclude generated outputs:
    - `*.pyd`, `*.so`, `*.dll`, `*.a`, `build/`, `x86simdsort/x86-simd-sort/builddir/`
  - Exclude local-only large generated sources restored from the archive:
    - `src/egtb_data_1256.cpp`
    - `src/egtb_data_256.cpp`
    - `src/egtb_data_512.cpp`

### Documentation and static assets
- `docs_and_configs/help/`
- `docs_and_configs/code_structure.txt`
- `docs_and_configs/github_upload_guide.md`
- `docs_and_configs/packaging_guide.md`
- `docs_and_configs/default_patterns.json`
- `docs_and_configs/patterns_config.json`
- `docs_and_configs/themes.json`
- `pic/`, `font/`, `favicon.ico`, `mathjax/`

## 2. Do not upload these files

- `docs_and_configs/config`
- `docs_and_configs/mistakes_book.pkl`
- `docs_and_configs/color_schemes.txt`
- `logger.txt`
- any local tablebase folders
- `*.book`, `*.z`
- `frontend/node_modules/`, `frontend/dist/`
- `AIPlayer_cpp/` legacy local workspace

## 3. Recommended repository structure

```text
2048EndgameTablebase/
├─ backend/
├─ egtb_core/
├─ ai_and_sort/
│  ├─ src/
│  ├─ include/
│  ├─ x86simdsort/
│  │  └─ x86-simd-sort/
│  ├─ CMakeLists.txt
│  ├─ make.sh
│  └─ README.md
├─ frontend/
├─ minigames/
├─ docs_and_configs/
├─ pic/
├─ font/
├─ mathjax/
├─ backend_server.py
├─ Config.py
├─ SignalHub.py
├─ error_bridge.py
└─ .gitignore
```

## 4. Build guidance for users

### Windows users
- Extract `ai_and_sort/egtb_data.7z` into `ai_and_sort/src/`
- Build `ai_core` and `mover_core` with CMake from `ai_and_sort/`
- Build `sort_wrapper.dll` from `ai_and_sort/x86simdsort/x86-simd-sort/`
- Runtime binaries should end up in `ai_and_sort/`

### Linux users
- Install `nanobind`, `cmake`, `meson`, `ninja`, and OpenMP
- Run `ai_and_sort/make.sh`
- The script builds `ai_core*.so`, `mover_core*.so`, and `sort_wrapper.so`

## 5. Third-party source note

The modified `x86-simd-sort` subtree is now part of `ai_and_sort/x86simdsort/x86-simd-sort/`.
Keep its upstream license and README when publishing the repository.
