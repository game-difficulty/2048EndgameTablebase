# GitHub Upload Guide: 2048 Endgame Tablebase

This document outlines the recommended inclusion list for publishing the repository to GitHub.
The goal is to keep all source needed for building the app while excluding local state, table data, and generated binaries.

## 1. Upload these directories and files

### Application source
- `backend_server.py`
- `Config.py`, `SignalHub.py`, `error_bridge.py`
- `backend/`
- `engine_core/`
- `assets/`
- `frontend/`

### Native source
- `native_core/`
  - Include:
    - `src/`
    - `include/`
    - `extern/nanobind/`
    - `x86simdsort/x86-simd-sort/`
    - `egtb_data.7z`
    - `CMakeLists.txt`
    - `make.sh`
    - `README.md`
    - `update_stubs.ps1`
    - `__init__.py`
    - `ai_core.pyi`
    - `formation_core.pyi`
  - Exclude generated outputs:
    - `*.pyd`, `*.so`, `*.dll`, `*.a`, `build/`, `build-formation/`, `x86simdsort/x86-simd-sort/builddir/`
  - Exclude local-only large generated sources restored from the archive:
    - `src/egtb_data_1256.cpp`
    - `src/egtb_data_256.cpp`
    - `src/egtb_data_512.cpp`

### Documentation and static assets
- `docs/`
- `docs_and_configs/help/`
- `docs_and_configs/code_structure.txt`
- `docs_and_configs/github_upload_guide.md`
- `docs_and_configs/packaging_guide.md`
- `docs_and_configs/default_patterns.json`
- `docs_and_configs/patterns_config.json`
- `docs_and_configs/themes.json`
- `pic/`, `assets/`, `font/`, `favicon.ico`, `mathjax/`

## 2. Do not upload these files

- `docs_and_configs/config`
- `docs_and_configs/mistakes_book.pkl`
- `docs_and_configs/color_schemes.txt`
- `logger.txt`
- any local tablebase folders
- `*.book`, `*.z`
- `frontend/node_modules/`, `frontend/dist/`

## 3. Recommended repository structure

```text
2048EndgameTablebase/
├─ backend/
├─ engine_core/
├─ native_core/
│  ├─ src/
│  ├─ include/
│  ├─ x86simdsort/
│  │  └─ x86-simd-sort/
│  ├─ CMakeLists.txt
│  ├─ make.sh
│  └─ README.md
├─ frontend/
├─ assets/
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
- Extract `native_core/egtb_data.7z` into `native_core/src/`
- Build `libx86simdsortcpp.a` from `native_core/x86simdsort/x86-simd-sort/` and copy it to `native_core/`
- Build `ai_core`, `mover_core`, `formation_core`, and `bookgen_native` with CMake from `native_core/`
- Runtime binaries should end up in `native_core/`

### Linux users
- Install `nanobind`, `cmake`, `meson`, `ninja`, and OpenMP
- Run `native_core/make.sh`
- The script builds `ai_core*.so`, `mover_core*.so`, `formation_core*.so`, and `bookgen_native.so`

## 5. Third-party source note

The modified `x86-simd-sort` subtree is now part of `native_core/x86simdsort/x86-simd-sort/`.
Keep its upstream license and README when publishing the repository.

## 6. WASM Demo Layout

- Keep the canonical WASM source files in `native_core/`:
  - `include/AIPlayer - wasm.h`
  - `src/AIPlayer - wasm.cpp`
  - `src/bindings_wasm.cpp`
- Publish the browser demo from the repository root `docs/` directory:
  - `docs/index.html`
  - `docs/main.js`
  - `docs/worker.js`
  - `docs/ai_core.js`
  - `docs/ai_core.wasm`
- Keep the WASM build scaffold in `docs/wasm_build/`.
