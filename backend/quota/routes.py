from __future__ import annotations

from collections import OrderedDict
from typing import Any

from fastapi import APIRouter

from .config import (
    MULTIPLIER_UNIT,
    TOKEN_UNIT,
    resolve_pricing_snapshot,
    table_multiplier_config,
    table_threshold_config,
    token_cost_config,
)
from .service import (
    INVITED_WEEKLY_GRANT_UNITS,
    PUBLIC_WEEKLY_GRANT_UNITS,
    SUPPORTER_WEEKLY_GRANT_UNITS,
    WEEKLY_GRANT_INTERVAL,
)


router = APIRouter(prefix="/api/quota", tags=["quota"])


def _tokens(units: int) -> int | float:
    value = int(units) / TOKEN_UNIT
    return int(value) if value.is_integer() else value


def _multiplier(units: int) -> int | float:
    value = int(units) / MULTIPLIER_UNIT
    return int(value) if value.is_integer() else value


def public_quota_rules() -> dict[str, Any]:
    pricing = resolve_pricing_snapshot()
    multiplier_config = table_multiplier_config()
    threshold_config = table_threshold_config()
    grouped_rules: OrderedDict[int, list[str]] = OrderedDict()
    for rule in multiplier_config.get("rules", []):
        prefix = str(rule.get("prefix") or "").strip()
        if not prefix:
            continue
        multiplier_units = int(
            round(float(rule.get("multiplier", 1)) * MULTIPLIER_UNIT)
        )
        grouped_rules.setdefault(multiplier_units, []).append(prefix)

    table_groups = [
        {
            "patterns": patterns,
            "multiplier": _multiplier(multiplier_units),
        }
        for multiplier_units, patterns in grouped_rules.items()
    ]
    costs = token_cost_config()
    tablebase_thresholds = []
    for full_pattern, raw_threshold in threshold_config.get("tables", {}).items():
        if not isinstance(raw_threshold, dict):
            continue
        threshold = raw_threshold.get("threshold")
        mode = raw_threshold.get("mode")
        tablebase_thresholds.append(
            {
                "full_pattern": str(full_pattern),
                "threshold": None if threshold is None else float(threshold),
                "mode": str(mode) if mode in {"absolute", "relative"} else None,
            }
        )

    return {
        "weekly_grants": {
            "public": _tokens(PUBLIC_WEEKLY_GRANT_UNITS),
            "invited": _tokens(INVITED_WEEKLY_GRANT_UNITS),
            "supporter": _tokens(SUPPORTER_WEEKLY_GRANT_UNITS),
            "interval_days": int(WEEKLY_GRANT_INTERVAL.days),
        },
        "operation_costs": {
            "trainer_lookup_hit": float(costs.get("trainer_lookup_hit", 0)),
            "trainer_lookup_miss": float(costs.get("trainer_lookup_miss", 0)),
            "tester_lookup_hit": float(costs.get("tester_lookup_hit", 0)),
            "tester_lookup_miss": float(costs.get("tester_lookup_miss", 0)),
            "analysis_per_replay": float(costs.get("analysis_per_replay", 0)),
            "replay_load": float(costs.get("replay_load", 0)),
            "battle_route_generation": float(costs.get("battle_route_generation", 0)),
        },
        "table_groups": table_groups,
        "tablebase_thresholds": tablebase_thresholds,
        "default_multiplier": float(
            multiplier_config.get("default_multiplier", 1)
        ),
        "global_multiplier": _multiplier(pricing.global_multiplier_units),
        "pricing_policy_key": pricing.policy_key,
    }


@router.get("/rules")
async def get_quota_rules() -> dict[str, Any]:
    return public_quota_rules()
