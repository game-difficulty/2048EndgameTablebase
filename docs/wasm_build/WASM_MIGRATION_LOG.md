# 2048 AI Core - WASM Migration Log

This note documents the current browser-demo build layout after the native-source refactor.

## Current layout

- Canonical native sources live in `native_core/`.
- The WASM build scaffold lives in `docs/wasm_build/`.
- Generated browser assets are written to the repository root `docs/` directory.

## Source of truth

The WASM target builds directly from these files under `native_core/`:

- `include/AIPlayer - wasm.h`
- `src/AIPlayer - wasm.cpp`
- `src/bindings_wasm.cpp`
- `src/egtb_query.cpp`
- `src/egtb_data_512.cpp`
- `src/egtb_data_256.cpp`
- `src/egtb_data_1256.cpp`

If the extracted `egtb_data_*.cpp` files are missing, restore them locally from `native_core/egtb_data.7z` before building.

## Build model

- The desktop native modules use `nanobind`, but the browser build does not.
- `docs/wasm_build/CMakeLists.txt` is the dedicated Emscripten entry point.
- `bindings_wasm.cpp` exports the JS-facing API through Embind.
- `docs/wasm_build/omp.h` acts as a local single-thread OpenMP stub so the browser build does not depend on `SharedArrayBuffer` deployment requirements.

## Recommended build steps

From the repository root:

```powershell
cd docs/wasm_build
emcmake cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target ai_core_wasm
```

## Output files

The build writes these files into `docs/`:

- `docs/ai_core.js`
- `docs/ai_core.wasm`

These are the published browser assets consumed by the static demo.

## Notes

- `docs/wasm_build/` is a build scaffold, not a second native workspace.
- `native_core/` remains the only canonical source tree for native and WASM builds.
- If you update the WASM-facing C++ sources, rebuild from `docs/wasm_build/` so the generated assets in `docs/` stay in sync.