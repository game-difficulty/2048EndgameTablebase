# WASM Build Scaffold

Use it only for the browser-demo build recipe and auxiliary WASM build files. The canonical sources stay in:

- `ai_and_sort/src/AIPlayer - wasm.cpp`
- `ai_and_sort/src/bindings_wasm.cpp`
- `ai_and_sort/include/AIPlayer - wasm.h`

The published static demo assets stay in the `docs/` root:

- `docs/index.html`
- `docs/main.js`
- `docs/worker.js`
- `docs/ai_core.js`
- `docs/ai_core.wasm`

`docs/wasm_build/CMakeLists.txt` builds from `ai_and_sort/` and writes the generated `ai_core.js` / `ai_core.wasm` back into the `docs/` root.
