# WASM Build Scaffold

Use it only for the browser-demo build recipe and auxiliary WASM build files. The canonical sources stay in:

- `native_core/src/AIPlayer - wasm.cpp`
- `native_core/src/bindings_wasm.cpp`
- `native_core/include/AIPlayer - wasm.h`

The published static demo assets stay in the `docs/` root:

- `docs/index.html`
- `docs/main.js`
- `docs/worker.js`
- `docs/ai_core.js`
- `docs/ai_core.wasm`

`docs/wasm_build/CMakeLists.txt` builds from `native_core/` and writes the generated `ai_core.js` / `ai_core.wasm` back into the `docs/` root.
