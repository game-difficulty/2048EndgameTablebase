#include "lib/x86simdsort.h" // 根据你的结构，头文件在 lib 文件夹下
#include <cstdint>

extern "C" {
    void sort_uint64(uint64_t *arr, size_t arrsize, bool descending) {
        x86simdsort::qsort<uint64_t>(arr, arrsize, false, descending);
    }
}

// meson setup builddir --buildtype=release -Duse_openmp=true
// meson compile -C builddir
// g++ --% -O3 -march=native -flto -shared -fPIC -ffunction-sections -fdata-sections xss_wrapper.cpp "./builddir/libx86simdsortcpp.a" -I"./lib" -I"./src" -static-libstdc++ -static-libgcc -fopenmp -Wl,--gc-sections -s -o sort_wrapper.dll