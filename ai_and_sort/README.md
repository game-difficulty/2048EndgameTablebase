# ai_and_sort

This directory is the canonical native-core workspace for the project.
`AIPlayer_cpp/` is now only a local legacy/reference workspace and should not be used for normal builds or GitHub uploads.

It contains:

- `src/` and `include/`: the nanobind-based native modules for `ai_core` and `mover_core`
- `x86simdsort/x86-simd-sort/`: the modified SIMD sorting source used to build `sort_wrapper`
- `egtb_data.7z`: compressed large generated sources that are extracted locally before building
- `CMakeLists.txt`: native module build entry
- `make.sh`: Linux-oriented convenience build script

Generated artifacts such as `ai_core*.pyd`, `mover_core*.so`, `sort_wrapper.dll`, and static libraries are intentionally not tracked. The large `egtb_data_*.cpp` files are also intentionally kept out of Git and restored locally from `egtb_data.7z`.

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

Build the Python extension modules:

```powershell
cd ai_and_sort
..\7zip\7z.exe x -y -o.\src .\egtb_data.7z
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Build the modified x86-simd-sort wrapper:

```powershell
cd ai_and_sort\x86simdsort\x86-simd-sort
meson setup builddir --buildtype=release -Duse_openmp=true
meson compile -C builddir
g++ -O3 -march=x86-64-v3 -flto -shared -fPIC -ffunction-sections -fdata-sections xss_wrapper.cpp .\builddir\libx86simdsortcpp.a -I.\lib -I.\src -static-libstdc++ -static-libgcc -fopenmp -Wl,--gc-sections -s -o ..\..\sort_wrapper.dll
```

The runtime expects generated files in the `ai_and_sort/` root:

- `ai_core*.pyd`
- `mover_core*.pyd`
- `sort_wrapper.dll`

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
- `sort_wrapper.so`

## Notes

- `egtb_data.7z` is part of the source distribution. It exists to keep the generated `egtb_data_*.cpp` out of normal GitHub language and line-count statistics while still letting users rebuild locally.
- `x86simdsort/x86-simd-sort/` is a modified upstream codebase; keep its README and license when publishing.
