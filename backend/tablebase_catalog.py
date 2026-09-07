from __future__ import annotations

import json
import hashlib
import os
import time
from pathlib import Path
from typing import Any
from Config import category_info, pattern_32k_tiles_map

from .remote_workers.config import configured_remote_tables
from .remote_workers.registry import remote_worker_registry
from .quota.config import MULTIPLIER_UNIT, table_multiplier_units


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
_CATALOG_VERSION_CACHE: tuple[float, str, int] = (0.0, "", -1)


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


def _iter_local_entries() -> list[dict[str, Any]]:
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
        entry["_provider"] = "local"
        entries.append(entry)
    return entries


def _iter_remote_entries(*, online_only: bool) -> list[dict[str, Any]]:
    online_tables = remote_worker_registry.online_tables()
    entries: list[dict[str, Any]] = []
    for full_pattern, raw_entry in configured_remote_tables().items():
        online = full_pattern in online_tables
        if online_only and not online:
            continue
        entry = dict(raw_entry)
        entry["_full_pattern"] = full_pattern
        entry["_provider"] = "remote"
        entry["_available"] = online
        entries.append(entry)
    return entries


def _iter_available_entries() -> list[dict[str, Any]]:
    return [*_iter_local_entries(), *_iter_remote_entries(online_only=True)]


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
                "guest_available": _entry_guest_available(entry),
                "ai": ai_table_metadata(entry),
            }
        )
    return tables


def ai_table_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    pattern = str(entry.get("pattern") or "")
    parameters = pattern_32k_tiles_map.get(pattern)
    compatible = bool(parameters and pattern not in category_info.get("variant", [])
                      and "_" not in pattern)
    return {"compatible": compatible, "policy_version": 1,
            "large_tiles": int(parameters[0]) if parameters else 0,
            "free_tiles": int(parameters[1]) if parameters else 0}


def _guest_max_multiplier_units() -> int:
    try:
        multiplier = float(os.getenv("GUEST_MAX_TABLE_MULTIPLIER", "5"))
    except ValueError:
        multiplier = 5.0
    return max(0, int(round(multiplier * MULTIPLIER_UNIT)))


def _entry_guest_available(entry: dict[str, Any]) -> bool:
    return (
        entry.get("_provider") == "local"
        and table_multiplier_units(str(entry.get("_full_pattern") or ""))
        <= _guest_max_multiplier_units()
    )


def is_guest_tablebase_available(full_pattern: str) -> bool:
    target = str(full_pattern or "").strip()
    if not target:
        return False
    return any(
        str(entry.get("_full_pattern") or "") == target
        and _entry_guest_available(entry)
        for entry in _iter_local_entries()
    )


def get_catalog_version() -> str:
    global _CATALOG_VERSION_CACHE
    now = time.monotonic()
    remote_epoch = remote_worker_registry.availability_epoch
    expires_at, cached_version, cached_remote_epoch = _CATALOG_VERSION_CACHE
    if (
        cached_version
        and expires_at > now
        and cached_remote_epoch == remote_epoch
    ):
        return cached_version
    manifest_path = _manifest_path()
    version_parts: list[Any] = [get_available_tablebases()]
    try:
        version_parts.append(manifest_path.stat().st_mtime_ns)
    except OSError:
        version_parts.append(0)
    for entry in _iter_local_entries():
        try:
            version_parts.append(
                (entry["_full_pattern"], Path(entry["_absolute_path"]).stat().st_mtime_ns)
            )
        except OSError:
            continue
    version_parts.append(("remote_availability_epoch", remote_epoch))
    encoded = json.dumps(
        version_parts,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    version = hashlib.sha256(encoded).hexdigest()[:16]
    _CATALOG_VERSION_CACHE = (now + 5.0, version, remote_epoch)
    return version


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


def resolve_configured_tablebase(
    full_pattern: str,
    spawn_rate: float | None = None,
) -> dict[str, Any] | None:
    entry = resolve_tablebase(full_pattern, spawn_rate)
    if entry is not None:
        return entry
    target_pattern = str(full_pattern or "").strip()
    for remote_entry in _iter_remote_entries(online_only=False):
        if remote_entry["_full_pattern"] != target_pattern:
            continue
        if spawn_rate is not None and abs(
            float(remote_entry.get("spawn_rate", 0.1)) - float(spawn_rate)
        ) > 1e-4:
            continue
        return remote_entry
    return None


def tablebase_provider_kind(full_pattern: str) -> str | None:
    entry = resolve_configured_tablebase(full_pattern)
    return str(entry.get("_provider")) if entry else None


def build_filepath_map_entry(
    full_pattern: str,
    spawn_rate: float | None = None,
) -> list[tuple[str, str]]:
    entry = resolve_tablebase(full_pattern, spawn_rate)
    if not entry or entry.get("_provider") != "local":
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
