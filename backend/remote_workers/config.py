from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_MANIFEST = (
    PROJECT_ROOT / "docs_and_configs" / "cloud_remote_tablebases.json"
)


def remote_manifest_path() -> Path:
    override = os.getenv("CLOUD_REMOTE_TABLEBASE_MANIFEST")
    return Path(override) if override else DEFAULT_REMOTE_MANIFEST


@lru_cache(maxsize=1)
def load_remote_manifest() -> dict[str, Any]:
    path = remote_manifest_path()
    if not path.exists():
        return {"workers": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("workers", []), list):
        raise ValueError(f"Invalid remote tablebase manifest: {path}")
    return data


def configured_workers() -> dict[str, dict[str, Any]]:
    workers: dict[str, dict[str, Any]] = {}
    for raw_worker in load_remote_manifest().get("workers", []):
        if not isinstance(raw_worker, dict):
            continue
        worker_id = str(raw_worker.get("worker_id") or "").strip()
        if not worker_id:
            continue
        tables: dict[str, dict[str, Any]] = {}
        for raw_table in raw_worker.get("tables", []):
            if not isinstance(raw_table, dict):
                continue
            pattern = str(raw_table.get("pattern") or "").strip()
            target = str(raw_table.get("target") or "").strip()
            full_pattern = str(
                raw_table.get("full_pattern") or f"{pattern}_{target}"
            ).strip()
            if not pattern or not target or not full_pattern:
                continue
            table = dict(raw_table)
            table.update(
                {
                    "worker_id": worker_id,
                    "pattern": pattern,
                    "target": target,
                    "full_pattern": full_pattern,
                    "dtype": str(raw_table.get("dtype") or "uint32"),
                    "spawn_rate": float(raw_table.get("spawn_rate", 0.1)),
                    "resource_group": str(
                        raw_table.get("resource_group") or full_pattern
                    ),
                }
            )
            tables[full_pattern] = table
        workers[worker_id] = {"worker_id": worker_id, "tables": tables}
    return workers


def configured_remote_tables() -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for worker in configured_workers().values():
        for full_pattern, table in worker["tables"].items():
            if full_pattern in tables:
                raise ValueError(
                    f"Remote tablebase is assigned to multiple workers: {full_pattern}"
                )
            tables[full_pattern] = dict(table)
    return tables


def worker_secret() -> str:
    return str(os.getenv("REMOTE_TABLEBASE_WORKER_SECRET") or "")

