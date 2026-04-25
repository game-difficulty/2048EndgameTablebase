#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
XSS_DIR="$ROOT_DIR/x86simdsort/x86-simd-sort"
XSS_BUILD_DIR="$XSS_DIR/builddir"
BUILD_DIR="$ROOT_DIR/build-formation"

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
need_cmd cp

if [ -f "$ROOT_DIR/egtb_data.7z" ]; then
  need_cmd 7z
  7z x -y "-o$ROOT_DIR/src" "$ROOT_DIR/egtb_data.7z"
fi

if [ -d "$XSS_BUILD_DIR" ]; then
  meson setup "$XSS_BUILD_DIR" "$XSS_DIR" --buildtype=release -Duse_openmp=true --reconfigure
else
  meson setup "$XSS_BUILD_DIR" "$XSS_DIR" --buildtype=release -Duse_openmp=true
fi
meson compile -C "$XSS_BUILD_DIR"
cp "$XSS_BUILD_DIR/libx86simdsortcpp.a" "$ROOT_DIR/libx86simdsortcpp.a"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release --target ai_core mover_core formation_core bookgen_native -j
