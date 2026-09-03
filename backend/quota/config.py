from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MULTIPLIERS_PATH = PROJECT_ROOT / "docs_and_configs" / "cloud_table_multipliers.json"
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "docs_and_configs" / "cloud_table_thresholds.json"
DEFAULT_COSTS_PATH = PROJECT_ROOT / "docs_and_configs" / "cloud_token_costs.json"
DEFAULT_PRICING_PATH = PROJECT_ROOT / "docs_and_configs" / "cloud_token_pricing.json"
TOKEN_UNIT = 1000
MULTIPLIER_UNIT = 1000
PRICING_REFRESH_SECONDS = 5.0
MIN_GLOBAL_MULTIPLIER = 0.001
MAX_GLOBAL_MULTIPLIER = 100.0


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PricingSnapshot:
    global_multiplier_units: int
    policy_key: str


_pricing_lock = threading.Lock()
_pricing_state: dict[str, Any] = {
    "source_key": None,
    "fingerprint": None,
    "checked_at": 0.0,
    "config": {"schema_version": 1, "policy_key": "standard", "global_multiplier": 1.0},
    "warning_key": None,
}


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
            "battle_route_generation": 100,
        },
    )


def _pricing_source() -> tuple[Path, str | None]:
    path = Path(os.getenv("CLOUD_TOKEN_PRICING_CONFIG") or DEFAULT_PRICING_PATH)
    override = os.getenv("CLOUD_TOKEN_GLOBAL_MULTIPLIER")
    return path, None if override is None else override.strip()


def _pricing_fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return None


def _validate_pricing_config(raw: Any, *, override: str | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("pricing config must be an object")
    schema_version = int(raw.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError("unsupported pricing config schema")
    raw_multiplier = override if override not in {None, ""} else raw.get("global_multiplier", 1)
    multiplier = float(raw_multiplier)
    if (
        not math.isfinite(multiplier)
        or multiplier < MIN_GLOBAL_MULTIPLIER
        or multiplier > MAX_GLOBAL_MULTIPLIER
    ):
        raise ValueError("global multiplier must be between 0.001 and 100")
    policy_key = str(raw.get("policy_key") or "standard").strip()[:64] or "standard"
    if override not in {None, ""}:
        policy_key = "environment_override"
    return {
        "schema_version": 1,
        "policy_key": policy_key,
        "global_multiplier": multiplier,
    }


def token_pricing_config() -> dict[str, Any]:
    """Return a hot-reloadable, last-known-good global pricing configuration."""
    path, override = _pricing_source()
    source_key = (str(path.resolve()), override)
    now = time.monotonic()
    with _pricing_lock:
        if (
            _pricing_state["source_key"] == source_key
            and now - float(_pricing_state["checked_at"]) < PRICING_REFRESH_SECONDS
        ):
            return _pricing_state["config"]

        fingerprint = _pricing_fingerprint(path)
        if (
            _pricing_state["source_key"] == source_key
            and _pricing_state["fingerprint"] == fingerprint
        ):
            _pricing_state["checked_at"] = now
            return _pricing_state["config"]

        try:
            if override not in {None, ""}:
                raw = {}
            elif path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
            else:
                raw = {}
            config = _validate_pricing_config(raw, override=override)
        except Exception as exc:
            warning_key = (source_key, fingerprint, type(exc).__name__, str(exc))
            if _pricing_state["warning_key"] != warning_key:
                logger.warning("Ignoring invalid token pricing configuration: %s", exc)
                _pricing_state["warning_key"] = warning_key
            _pricing_state["source_key"] = source_key
            _pricing_state["fingerprint"] = fingerprint
            _pricing_state["checked_at"] = now
            return _pricing_state["config"]

        _pricing_state.update(
            {
                "source_key": source_key,
                "fingerprint": fingerprint,
                "checked_at": now,
                "config": config,
                "warning_key": None,
            }
        )
        return config


def clear_token_pricing_cache() -> None:
    with _pricing_lock:
        _pricing_state.update(
            {
                "source_key": None,
                "fingerprint": None,
                "checked_at": 0.0,
                "config": {
                    "schema_version": 1,
                    "policy_key": "standard",
                    "global_multiplier": 1.0,
                },
                "warning_key": None,
            }
        )


def resolve_pricing_snapshot() -> PricingSnapshot:
    config = token_pricing_config()
    return PricingSnapshot(
        global_multiplier_units=multiplier_to_units(config.get("global_multiplier", 1)),
        policy_key=str(config.get("policy_key") or "standard"),
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


def apply_pricing_multipliers(
    base_cost_units: int,
    table_multiplier_units: int,
    global_multiplier_units: int,
) -> int:
    table_cost_units = apply_multiplier(base_cost_units, table_multiplier_units)
    return apply_multiplier(table_cost_units, global_multiplier_units)
