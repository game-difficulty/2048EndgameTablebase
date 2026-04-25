#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="${APP_NAME:-2048EndgameTablebase}"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/build/pyinstaller/linux}"
NATIVE_DIR="$ROOT_DIR/native_core"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

pick_python() {
  if [ -n "${PYTHON_EXE:-}" ]; then
    echo "$PYTHON_EXE"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi
  echo "Missing Python interpreter (python3 or python)." >&2
  exit 1
}

add_data_arg() {
  PY_ARGS+=(--add-data "$1:$2")
}

add_binary_arg() {
  PY_ARGS+=(--add-binary "$1:$2")
}

PYTHON_BIN="$(pick_python)"

require_path "$ROOT_DIR/frontend/dist" "frontend build output"
require_path "$ROOT_DIR/docs_and_configs/default_patterns.json" "default pattern config"
require_path "$ROOT_DIR/docs_and_configs/themes.json" "theme config"
require_path "$ROOT_DIR/docs_and_configs/help" "help docs"
require_path "$ROOT_DIR/pic" "picture assets"
require_path "$ROOT_DIR/font" "font directory"
require_path "$ROOT_DIR/mathjax" "mathjax directory"
require_path "$ROOT_DIR/favicon.ico" "favicon"
require_path "$ROOT_DIR/backend_server.py" "backend entrypoint"

shopt -s nullglob
ai_core_matches=("$NATIVE_DIR"/ai_core*.so)
mover_core_matches=("$NATIVE_DIR"/mover_core*.so)
formation_core_matches=("$NATIVE_DIR"/formation_core*.so)
shopt -u nullglob

if [ "${#ai_core_matches[@]}" -ne 1 ]; then
  echo "Expected exactly one ai_core*.so under $NATIVE_DIR" >&2
  exit 1
fi
if [ "${#mover_core_matches[@]}" -ne 1 ]; then
  echo "Expected exactly one mover_core*.so under $NATIVE_DIR" >&2
  exit 1
fi
if [ "${#formation_core_matches[@]}" -ne 1 ]; then
  echo "Expected exactly one formation_core*.so under $NATIVE_DIR" >&2
  exit 1
fi

AI_CORE_SO="${ai_core_matches[0]}"
MOVER_CORE_SO="${mover_core_matches[0]}"
FORMATION_CORE_SO="${formation_core_matches[0]}"
BOOKGEN_NATIVE_SO="$NATIVE_DIR/bookgen_native.so"

require_path "$BOOKGEN_NATIVE_SO" "bookgen native shared library"

"$PYTHON_BIN" -m PyInstaller --version >/dev/null

PY_ARGS=(
  -m PyInstaller
  --noconfirm
  --clean
  --onedir
  --windowed
  --name "$APP_NAME"
  --paths "$ROOT_DIR"
  --distpath "$DIST_DIR"
  --workpath "$WORK_DIR"
  --specpath "$WORK_DIR"
  --hidden-import native_core.ai_core
  --hidden-import native_core.mover_core
  --hidden-import native_core.formation_core
  --collect-submodules uvicorn
  --collect-submodules webview
)

add_data_arg "$ROOT_DIR/docs_and_configs/default_patterns.json" "docs_and_configs"
add_data_arg "$ROOT_DIR/docs_and_configs/themes.json" "docs_and_configs"
add_data_arg "$ROOT_DIR/docs_and_configs/help" "docs_and_configs/help"
add_data_arg "$ROOT_DIR/pic" "pic"
add_data_arg "$ROOT_DIR/font" "font"
add_data_arg "$ROOT_DIR/favicon.ico" "."
add_data_arg "$ROOT_DIR/mathjax" "mathjax"
add_data_arg "$ROOT_DIR/frontend/dist" "frontend/dist"

add_binary_arg "$BOOKGEN_NATIVE_SO" "native_core"

if [ -f "$ROOT_DIR/pic/2048_2.ico" ]; then
  PY_ARGS+=(--icon "$ROOT_DIR/pic/2048_2.ico")
fi

PY_ARGS+=("$ROOT_DIR/backend_server.py")

"$PYTHON_BIN" "${PY_ARGS[@]}"

echo "Linux package created at $DIST_DIR/$APP_NAME"
