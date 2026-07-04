#!/usr/bin/env bash

set -euo pipefail

echo "[check] verifying replacement service on 8081..."
if ! ss -ltnp 2>/dev/null | grep -qE '127\.0\.0\.1:8081|0\.0\.0\.0:8081|\[::\]:8081'; then
  echo "8081 is not listening; refusing to stop the legacy 8000 service." >&2
  exit 1
fi

echo "[check] locating legacy process on 8000..."
mapfile -t PIDS < <(
  ss -ltnp 2>/dev/null \
    | awk '/127\.0\.0\.1:8000|0\.0\.0\.0:8000|\[::\]:8000/ {print $0}' \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | sort -u
)

if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "No process is listening on 8000."
  exit 0
fi

for pid in "${PIDS[@]}"; do
  cmd="$(ps -p "$pid" -o cmd= || true)"
  echo "8000 pid=$pid cmd=$cmd"
  if [[ "$cmd" != *"/data/app"* || "$cmd" != *"uvicorn"* ]]; then
    echo "Refusing to stop pid=$pid because it is not the expected /data/app uvicorn process." >&2
    exit 1
  fi
done

echo "[action] stopping legacy /data/app uvicorn on 8000..."
for pid in "${PIDS[@]}"; do
  kill -TERM "$pid"
done

sleep 3

if ss -ltnp 2>/dev/null | grep -qE '127\.0\.0\.1:8000|0\.0\.0\.0:8000|\[::\]:8000'; then
  echo "8000 is still occupied after SIGTERM; inspect manually before using SIGKILL." >&2
  exit 1
fi

echo "8000 released."

