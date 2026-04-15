#!/bin/bash

if [ ! -e ./x86-simd-sort ]; then
  git clone --depth=1 https://github.com/numpy/x86-simd-sort.git
fi

7z x -y ./egtb_data.7z

cmake -B build
cmake --build build -j --config Release
cp ./build/ai_core.*.so .

cd ./x86-simd-sort || exit
meson setup build --buildtype=release -Duse_openmp=true -Dlib_type=static
meson compile -C build
cd ..

g++ -O3 -march=native -flto -shared -fPIC \
        -ffunction-sections -fdata-sections \
        xss_wrapper.cpp ./x86-simd-sort/build/libx86simdsortcpp.a \
        -I./x86-simd-sort \
        -static-libstdc++ -static-libgcc -fopenmp \
        -Wl,--gc-sections -s \
        -o sort_wrapper.so

