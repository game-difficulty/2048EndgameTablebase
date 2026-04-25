#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
NATIVE_DIR="$ROOT_DIR/ai_and_sort"

RUN_AFTER_BUILD=false
SKIP_FRONTEND=false
SKIP_NATIVE=false

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Missing Python interpreter (python3 or python)." >&2
  exit 1
fi

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

usage() {
  cat <<'EOF'
Usage: ./build_linux.sh [--run] [--skip-frontend] [--skip-native]

  --run             Launch backend_server.py after the build finishes
  --skip-frontend   Reuse the existing frontend/dist output
  --skip-native     Reuse the existing ai_and_sort native binaries
EOF
}

while (($#)); do
  case "$1" in
    --run)
      RUN_AFTER_BUILD=true
      ;;
    --skip-frontend)
      SKIP_FRONTEND=true
      ;;
    --skip-native)
      SKIP_NATIVE=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [ "$SKIP_FRONTEND" = false ]; then
  need_cmd npm
  echo "[1/3] Building frontend..."
  cd "$FRONTEND_DIR"
  if [ -f package-lock.json ]; then
    npm ci
  else
    npm install
  fi
  npm run build
fi

if [ "$SKIP_NATIVE" = false ]; then
  echo "[2/3] Building native modules..."
  cd "$NATIVE_DIR"
  chmod +x make.sh
  ./make.sh
fi

echo "[3/3] Verifying Python dependencies..."
"$PYTHON_BIN" -c "import fastapi, uvicorn, webview, markdown, numpy, psutil, cpuinfo" >/dev/null

echo "Build completed."
echo "Run the app with: $PYTHON_BIN backend_server.py"

if [ "$RUN_AFTER_BUILD" = true ]; then
  cd "$ROOT_DIR"
  exec "$PYTHON_BIN" backend_server.py
fi
