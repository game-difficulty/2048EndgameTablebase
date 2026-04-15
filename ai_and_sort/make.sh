#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
XSS_DIR="$ROOT_DIR/x86simdsort/x86-simd-sort"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

cd "$ROOT_DIR"

if [ ! -d "$XSS_DIR" ]; then
  echo "Missing x86-simd-sort sources at: $XSS_DIR" >&2
  exit 1
fi

need_cmd cmake
need_cmd meson
need_cmd g++

if [ -f "$ROOT_DIR/egtb_data.7z" ]; then
  need_cmd 7z
  7z x -y "-o$ROOT_DIR/src" "$ROOT_DIR/egtb_data.7z"
fi

cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT_DIR/build" -j --config Release

find "$ROOT_DIR/build" -maxdepth 1 \( -name 'ai_core*.so' -o -name 'ai_core*.pyd' -o -name 'mover_core*.so' -o -name 'mover_core*.pyd' \) -exec cp {} "$ROOT_DIR" \;

meson setup "$XSS_DIR/builddir" "$XSS_DIR" --buildtype=release -Duse_openmp=true
meson compile -C "$XSS_DIR/builddir"

g++ -O3 -march=native -flto -shared -fPIC \
    -ffunction-sections -fdata-sections \
    "$XSS_DIR/xss_wrapper.cpp" "$XSS_DIR/builddir/libx86simdsortcpp.a" \
    -I"$XSS_DIR/lib" -I"$XSS_DIR/src" \
    -fopenmp -Wl,--gc-sections -s \
    -o "$ROOT_DIR/sort_wrapper.so"
