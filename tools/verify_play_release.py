"""Verify a staged release and exercise deployed routes without changing user data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def verify_files(root: Path) -> str:
    root = root.resolve()
    manifest = json.loads((root / "play-release-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError(f"Missing release file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Release file mismatch: {name}")
    return manifest["revision"]


def verify_routes(base_url: str) -> None:
    probes = [
        ("GET", "/api/human/health", 200),
        ("DELETE", "/api/human/runs/release-smoke-test/history", 401),
        ("GET", "/api/analysis/library?variant=4x4&grade=invalid", 400),
        ("GET", "/api/analysis/library?variant=4x4&grade=B&limit=1", 200),
        ("POST", "/api/analysis/replays/release-smoke-test/open-link", 404),
    ]
    for method, path, expected in probes:
        request = Request(base_url.rstrip("/") + path, method=method)
        try:
            with urlopen(request, timeout=20) as response:
                status = response.status
                body = json.load(response)
        except HTTPError as error:
            status = error.code
            body = None
        if status != expected:
            raise RuntimeError(f"{method} {path}: expected {expected}, received {status}")
        if path.endswith("/health") and body != {"ok": True}:
            raise RuntimeError("Unexpected health payload")
        if body and "items" in body and any(item.get("grade") != "B" for item in body["items"]):
            raise RuntimeError("Analysis grade filter is not applied")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--base-url")
    args = parser.parse_args()
    if not args.root and not args.base_url:
        parser.error("supply --root and/or --base-url")
    if args.root:
        print("Verified release:", verify_files(args.root))
    if args.base_url:
        verify_routes(args.base_url)
        print("Verified deployed routes:", args.base_url)
