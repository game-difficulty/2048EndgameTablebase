from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "docs_and_configs" / "cloud_tablebases.json"
DEFAULT_ROOT = "/opt/2048tables/tablebases"
TABLE_EXTENSIONS = (
    ".book",
    ".z",
    "b",
    ".zbook",
    ".exzbook",
    ".exadbook",
    ".exadzbook",
    ".bccmp",
)


def _manifest_path() -> Path:
    override = os.getenv("CLOUD_TABLEBASE_MANIFEST")
    return Path(override) if override else DEFAULT_MANIFEST_PATH


def _load_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        return {"root": DEFAULT_ROOT, "tables": []}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid tablebase manifest: {path}")
    return data


def _catalog_root(manifest: dict[str, Any]) -> Path:
    root = os.getenv("CLOUD_TABLEBASE_ROOT") or manifest.get("root") or DEFAULT_ROOT
    return Path(str(root))


def _full_pattern(entry: dict[str, Any]) -> str:
    if entry.get("full_pattern"):
        return str(entry["full_pattern"])
    return f"{entry.get('pattern')}_{entry.get('target')}"


def _entry_path(root: Path, entry: dict[str, Any]) -> Path:
    relative_path = str(entry.get("relative_path") or _full_pattern(entry))
    return root / relative_path


def _path_has_table_file(path: Path, full_pattern: str) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    prefix = f"{full_pattern}_"
    try:
        for item in path.iterdir():
            if not item.is_file():
                continue
            name = item.name
            if name.startswith(prefix) and name.endswith(TABLE_EXTENSIONS):
                return True
    except OSError:
        return False
    return False


def _iter_available_entries() -> list[dict[str, Any]]:
    manifest = _load_manifest()
    root = _catalog_root(manifest)
    entries: list[dict[str, Any]] = []
    for raw_entry in manifest.get("tables", []):
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        full_pattern = _full_pattern(entry)
        path = _entry_path(root, entry)
        if not _path_has_table_file(path, full_pattern):
            continue
        entry["_full_pattern"] = full_pattern
        entry["_absolute_path"] = str(path.resolve())
        entries.append(entry)
    return entries


def get_available_tablebases() -> list[dict[str, Any]]:
    tables = []
    for entry in _iter_available_entries():
        tables.append(
            {
                "pattern": str(entry.get("pattern") or ""),
                "target": str(entry.get("target") or ""),
                "full_pattern": str(entry["_full_pattern"]),
                "dtype": str(entry.get("dtype") or "uint32"),
                "spawn_rate": float(entry.get("spawn_rate", 0.1)),
            }
        )
    return tables


def resolve_tablebase(
    full_pattern: str,
    spawn_rate: float | None = None,
) -> dict[str, Any] | None:
    target_pattern = str(full_pattern or "").strip()
    if not target_pattern:
        return None
    for entry in _iter_available_entries():
        if entry["_full_pattern"] != target_pattern:
            continue
        if spawn_rate is not None:
            entry_spawn_rate = float(entry.get("spawn_rate", 0.1))
            if abs(entry_spawn_rate - float(spawn_rate)) > 1e-4:
                continue
        return entry
    return None


def build_filepath_map_entry(
    full_pattern: str,
    spawn_rate: float | None = None,
) -> list[tuple[str, str]]:
    entry = resolve_tablebase(full_pattern, spawn_rate)
    if not entry:
        return []
    return [(str(entry["_absolute_path"]), str(entry.get("dtype") or "uint32"))]


def get_catalog_target_tiles() -> list[int]:
    targets = set()
    for table in get_available_tablebases():
        try:
            targets.add(int(table["target"]))
        except (TypeError, ValueError):
            continue
    return sorted(targets)


def get_catalog_categories() -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {"cloud": []}
    for table in get_available_tablebases():
        pattern = str(table.get("pattern") or "")
        if pattern and pattern not in categories["cloud"]:
            categories["cloud"].append(pattern)
    return categories
