# Re-pruned x86-simd-sort (`uint32_t` / `uint64_t` only)

This directory vendors the upstream `numpy/x86-simd-sort` codebase into `native_core/`
and then re-prunes it for the project's current needs.

## Kept from upstream

- `uint32_t` quicksort
- `uint64_t` quicksort
- `uint32_t` argsort
- `uint64_t` argsort
- AVX2 and AVX-512 SKX dispatch paths
- Scalar fallback implementations inside the library
- OpenMP support for larger sorts

## Removed from upstream

- 16-bit, floating-point, signed 32-bit, signed 64-bit, and double specializations
- `qselect`, `partial_qsort`, `argselect`
- key-value sort/select/partial-sort APIs
- object sort API
- ICL and SPR specific translation units
- benchmark, test, and packaging-oriented Meson options

## Local build contract

The project still builds this vendored library in two steps:

1. Run Meson in this directory to produce `builddir/libx86simdsortcpp.a`
2. Copy that archive to `native_core/libx86simdsortcpp.a`
3. Link `native_core/bookgen_native.dll` or `bookgen_native.so` against it from CMake

### Windows / MinGW

```powershell
meson setup builddir --buildtype=release -Duse_openmp=true
meson compile -C builddir
Copy-Item .\builddir\libx86simdsortcpp.a ..\..\libx86simdsortcpp.a -Force
```

### Linux

```bash
meson setup builddir --buildtype=release -Duse_openmp=true
meson compile -C builddir
cp ./builddir/libx86simdsortcpp.a ../../libx86simdsortcpp.a
```

## C ABI exposed by `native_core/xss_wrapper.cpp`

```cpp
extern "C" {
    void sort_uint32(uint32_t *arr, size_t size, bool descending);
    void sort_uint64(uint64_t *arr, size_t size, bool descending);
    void argsort_uint32(const uint32_t *arr, size_t size, size_t *indices, bool descending);
    void argsort_uint64(const uint64_t *arr, size_t size, size_t *indices, bool descending);
}
```

Keep the upstream BSD 3-Clause license when redistributing this subtree.
