"""Package the Play API and its independent frontend without runtime data."""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("backend", "engine_core", "tools/verse-history")
STATIC_DIRS = (
    "frontend/dist/assets",
    "frontend/dist/human",
    "frontend/dist/verse-replay",
    "frontend/dist/compat",
)
SOURCE_FILES = ("Config.py", "SignalHub.py", "error_bridge.py")
STATIC_FILES = ("frontend/dist/favicon.ico",)


def validate_human_entry() -> None:
    """Refuse to package an entry page whose precompressed copy is stale."""
    entry = ROOT / "frontend/dist/human/index.html"
    compressed = entry.with_name(f"{entry.name}.gz")
    if not entry.is_file() or not compressed.is_file():
        raise FileNotFoundError(compressed if entry.is_file() else entry)
    with gzip.open(compressed, "rb") as source:
        compressed_payload = source.read()
    if compressed_payload != entry.read_bytes():
        raise RuntimeError(
            "frontend/dist/human/index.html.gz does not match index.html; "
            "rebuild the precompressed entry before packaging"
        )


def include(path: Path) -> bool:
    return not any(part in {"__pycache__", "node_modules", ".pytest_cache"}
                   for part in path.parts) and path.suffix not in {".pyc", ".pyo"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    validate_human_entry()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as tar:
        for name in SOURCE_DIRS + STATIC_DIRS:
            directory = ROOT / name
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for path in sorted(directory.rglob("*")):
                if path.is_file() and include(path.relative_to(ROOT)):
                    tar.add(path, arcname=path.relative_to(ROOT).as_posix())
        for name in SOURCE_FILES + STATIC_FILES:
            path = ROOT / name
            if not path.is_file():
                raise FileNotFoundError(path)
            tar.add(path, arcname=name)
    print(f"{args.archive} ({args.archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
