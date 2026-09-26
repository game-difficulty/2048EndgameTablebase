"""Package the Play API and its independent frontend without runtime data."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("backend", "engine_core", "tools/verse-history")
STATIC_DIRS = (
    "frontend/dist/assets",
    "frontend/dist/human",
    "frontend/dist/verse-replay",
    "frontend/dist/compat",
    "frontend/dist/payments",
)
SOURCE_FILES = ("Config.py", "SignalHub.py", "error_bridge.py")
STATIC_FILES = ("frontend/dist/favicon.ico", "frontend/dist/release.json")


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
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    built = json.loads((ROOT / "frontend/dist/release.json").read_text(encoding="utf-8"))
    if built["revision"] != revision:
        raise RuntimeError("Frontend build revision does not match backend checkout; rebuild before packaging")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *SOURCE_DIRS, *SOURCE_FILES,
         "frontend/src", "frontend/public", "frontend/scripts", "frontend/vite.config.js"],
        cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("Commit release source changes before packaging from a clean checkout")
    manifest = {"revision": revision, "build_id": built["buildId"], "files": {}}
    def add(tar, path, name):
        manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tar.add(path, arcname=name)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as tar:
        for name in SOURCE_DIRS + STATIC_DIRS:
            directory = ROOT / name
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for path in sorted(directory.rglob("*")):
                if path.is_file() and include(path.relative_to(ROOT)):
                    add(tar, path, path.relative_to(ROOT).as_posix())
        for name in SOURCE_FILES + STATIC_FILES:
            path = ROOT / name
            if not path.is_file():
                raise FileNotFoundError(path)
            add(tar, path, name)
        payload = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo("play-release-manifest.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    print(f"{args.archive} ({args.archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
