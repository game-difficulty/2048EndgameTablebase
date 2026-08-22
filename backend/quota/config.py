from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MULTIPLIERS_PATH = PROJECT_ROOT / "docs_and_configs" / "cloud_table_multipliers.json"
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "docs_and_configs" / "cloud_table_thresholds.json"
DEFAULT_COSTS_PATH = PROJECT_ROOT / "docs_and_configs" / "cloud_token_costs.json"
TOKEN_UNIT = 1000
MULTIPLIER_UNIT = 1000


def _load_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return fallback


@lru_cache(maxsize=1)
def table_multiplier_config() -> dict[str, Any]:
    path = Path(os.getenv("CLOUD_TABLE_MULTIPLIERS") or DEFAULT_MULTIPLIERS_PATH)
    return _load_json(
        path,
        {
            "default_multiplier": 1,
            "rules": [
                {"prefix": "L3", "multiplier": 1},
                {"prefix": "t", "multiplier": 1},
                {"prefix": "442t", "multiplier": 1},
                {"prefix": "2x4", "multiplier": 1},
                {"prefix": "3x3free8", "multiplier": 1},
                {"prefix": "3x3", "multiplier": 1},
                {"prefix": "3x4free9", "multiplier": 3},
                {"prefix": "free9", "multiplier": 3},
                {"prefix": "free10", "multiplier": 8},
                {"prefix": "free11", "multiplier": 50},
                {"prefix": "4421", "multiplier": 5},
                {"prefix": "4431", "multiplier": 8},
                {"prefix": "444", "multiplier": 5},
                {"prefix": "LL", "multiplier": 8},
            ],
        },
    )


@lru_cache(maxsize=1)
def table_threshold_config() -> dict[str, Any]:
    path = Path(os.getenv("CLOUD_TABLE_THRESHOLDS") or DEFAULT_THRESHOLDS_PATH)
    return _load_json(path, {"tables": {}})


@lru_cache(maxsize=1)
def token_cost_config() -> dict[str, Any]:
    path = Path(os.getenv("CLOUD_TOKEN_COSTS") or DEFAULT_COSTS_PATH)
    return _load_json(
        path,
        {
            "trainer_lookup_hit": 1,
            "trainer_lookup_miss": 0.2,
            "tester_lookup_hit": 1,
            "tester_lookup_miss": 0.2,
            "analysis_per_replay": 100,
            "replay_load": 3,
        },
    )


def token_to_units(value: int | float | str) -> int:
    return int(round(float(value) * TOKEN_UNIT))


def multiplier_to_units(value: int | float | str) -> int:
    return int(round(float(value) * MULTIPLIER_UNIT))


def operation_cost_units(operation_key: str) -> int:
    value = token_cost_config().get(operation_key, 0)
    return max(0, token_to_units(value))


def table_multiplier_units(full_pattern: str | None) -> int:
    pattern = str(full_pattern or "").strip()
    base = pattern.split("_", 1)[0]
    config = table_multiplier_config()
    for rule in config.get("rules", []):
        prefix = str(rule.get("prefix") or "").strip()
        if prefix and base.startswith(prefix):
            return max(0, multiplier_to_units(rule.get("multiplier", 1)))
    return max(0, multiplier_to_units(config.get("default_multiplier", 1)))


def apply_multiplier(base_cost_units: int, multiplier_units: int) -> int:
    return int(round(max(0, int(base_cost_units)) * max(0, int(multiplier_units)) / MULTIPLIER_UNIT))
