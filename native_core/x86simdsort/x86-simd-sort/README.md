# Simplified x86-simd-sort (uint64_t only)

This directory contains the modified SIMD sorting source vendored into `native_core/`.
It is the source used to build the project's `sort_wrapper` shared library.

## What changed from upstream

- Only the `uint64_t` quicksort path is kept.
- Unused data types and algorithms were removed.
- The code is organized for direct integration into `native_core` instead of being built as a standalone package.
- `xss_wrapper.cpp` provides the small C ABI that the Python layer loads dynamically.

## Build

### Windows (MinGW)

```powershell
meson setup builddir --buildtype=release -Duse_openmp=true
meson compile -C builddir
g++ -O3 -march=x86-64-v3 -flto -shared -fPIC -ffunction-sections -fdata-sections xss_wrapper.cpp .\builddir\libx86simdsortcpp.a -I.\lib -I.\src -static-libstdc++ -static-libgcc -fopenmp -Wl,--gc-sections -s -o ..\..\sort_wrapper.dll
```

### Linux

```bash
meson setup builddir --buildtype=release -Duse_openmp=true
meson compile -C builddir
g++ -O3 -march=x86-64-v3 -flto -shared -fPIC -ffunction-sections -fdata-sections xss_wrapper.cpp ./builddir/libx86simdsortcpp.a -I./lib -I./src -fopenmp -Wl,--gc-sections -s -o ../../sort_wrapper.so
```

## Exported interface

```cpp
extern "C" {
    void sort_uint64(uint64_t *arr, size_t arrsize, bool descending);
}
```

## License

Keep the upstream BSD 3-Clause license when redistributing this subtree.
