"""Make a consistent online SQLite backup before a schema-changing release."""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not args.source.is_file() or args.destination.exists():
        raise SystemExit("Source missing or destination already exists")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{args.source}?mode=ro", uri=True, timeout=30) as src:
        with sqlite3.connect(args.destination, timeout=30) as dst:
            src.backup(dst, pages=1000, sleep=0.1)
            result = dst.execute("PRAGMA quick_check").fetchone()[0]
            if result != "ok":
                raise RuntimeError(f"Backup integrity check failed: {result}")
    print(f"{args.destination}: {args.destination.stat().st_size} bytes; quick_check ok")


if __name__ == "__main__":
    main()
