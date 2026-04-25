# ai_and_sort

This directory is the canonical native-core workspace for the project.

It contains:

- `src/` and `include/`: the native sources for `ai_core`, `mover_core`, `formation_core`, and `bookgen_native`
- `x86simdsort/x86-simd-sort/`: the modified SIMD sorting source used to build `libx86simdsortcpp.a`
- `egtb_data.7z`: compressed large generated sources that are extracted locally before building
- `CMakeLists.txt`: native module build entry
- `make.sh`: Linux-oriented convenience build script

Generated artifacts such as `ai_core*.pyd`, `mover_core*.pyd`, `formation_core*.pyd`,
`bookgen_native.dll` / `bookgen_native.so`, and `libx86simdsortcpp.a` are intentionally not tracked.
The large `egtb_data_*.cpp` files are also intentionally kept out of Git and restored locally from `egtb_data.7z`.

## Build prerequisites

- Python 3.8+
- CMake 3.18+
- A C++17 compiler
- OpenMP
- `nanobind`
- For Linux/macOS shell builds: `meson`, `ninja`

Recommended dependency install:

```bash
pip install nanobind
```

If you prefer a vendored dependency, `extern/nanobind` is also supported when present.

## Windows build

From inside `ai_and_sort/`, first build the SIMD static library:

```powershell
cd x86simdsort\x86-simd-sort
meson setup builddir --buildtype=release -Duse_openmp=true
meson compile -C builddir
Copy-Item .\builddir\libx86simdsortcpp.a ..\..\libx86simdsortcpp.a -Force
```

Then build the native targets:

```powershell
cd ..\..
..\7zip\7z.exe x -y -o.\src .\egtb_data.7z
cmake -S . -B build-formation -DCMAKE_BUILD_TYPE=Release
cmake --build .\build-formation --target ai_core mover_core formation_core bookgen_native -j 4
```

The runtime expects generated files in the `ai_and_sort/` root:

- `ai_core*.pyd`
- `mover_core*.pyd`
- `formation_core*.pyd`
- `bookgen_native.dll`
- `libgcc_s_seh-1.dll`
- `libgomp-1.dll`
- `libwinpthread-1.dll`

If you only need the native DLL after the build tree is configured:

```powershell
cmake --build .\build-formation --target bookgen_native -j 4
```

If you want to rebuild `ai_core` / `formation_core` and refresh their `.pyi` stubs:

```powershell
.\update_stubs.ps1
```

## Linux build

Use the helper script:

```bash
cd ai_and_sort
chmod +x make.sh
./make.sh
```

This builds:

- `ai_core*.so`
- `mover_core*.so`
- `formation_core*.so`
- `bookgen_native.so`

## Notes

- `egtb_data.7z` is part of the source distribution. It exists to keep the generated `egtb_data_*.cpp` out of normal GitHub language and line-count statistics while still letting users rebuild locally.
- `x86simdsort/x86-simd-sort/` is a modified upstream codebase; keep its README and license when publishing.
- `src/AIPlayer - wasm.cpp`, `src/bindings_wasm.cpp`, and `include/AIPlayer - wasm.h` are intentionally retained as the canonical WASM source set for the repository's web-demo AI flow.
- Published browser assets belong in the repository root `docs/` directory: `docs/index.html`, `docs/main.js`, `docs/worker.js`, `docs/ai_core.js`, and `docs/ai_core.wasm`.
- The WASM build scaffold lives in `docs/wasm_build/` and should not be treated as a second native workspace.
