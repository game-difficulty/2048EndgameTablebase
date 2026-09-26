"""Insert the internal tablebase router into an older deployed main app.py.

The Play site has a separate release cadence, so this intentionally changes
only two lines in the currently deployed main application.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


IMPORT_ANCHOR = "from backend.remote_workers import remote_worker_registry\n"
IMPORT_LINE = "from backend.remote_workers.internal_bridge import router as internal_tablebase_router\n"
ROUTER_ANCHOR = "app.include_router(battle_router)\n"
ROUTER_LINE = "app.include_router(internal_tablebase_router)\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("app_py", type=Path)
    parser.add_argument("backup", type=Path)
    args = parser.parse_args()
    original = args.app_py.read_text(encoding="utf-8")
    if IMPORT_LINE in original or ROUTER_LINE in original:
        raise SystemExit("Bridge router is already present; inspect the file before patching.")
    if original.count(IMPORT_ANCHOR) != 1 or original.count(ROUTER_ANCHOR) != 1:
        raise SystemExit("Main app anchors changed; no file was written.")
    updated = original.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE, 1)
    updated = updated.replace(ROUTER_ANCHOR, ROUTER_ANCHOR + ROUTER_LINE, 1)
    ast.parse(updated, filename=str(args.app_py))
    if args.backup.exists():
        raise SystemExit("Backup already exists; no file was written.")
    args.backup.write_text(original, encoding="utf-8")
    temporary = args.app_py.with_suffix(".py.bridge-new")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(args.app_py)
    print(f"Patched {args.app_py}; original saved at {args.backup}")


if __name__ == "__main__":
    main()
