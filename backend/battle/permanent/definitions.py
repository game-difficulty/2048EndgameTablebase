from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "docs_and_configs" / "cloud_battle_permanent_rooms.json"


@dataclass(frozen=True)
class HostPolicy:
    offline_grace_seconds: int = 45
    idle_timeout_seconds: int = 180
    member_offline_grace_seconds: int = 120
    sweep_interval_seconds: int = 10


@dataclass(frozen=True)
class PermanentRoomDefinition:
    template_key: str
    mode_key: str
    full_pattern: str
    initial_board: str
    max_players: int
    step_timeout_seconds: int
    version: int


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def load_definitions() -> tuple[HostPolicy, tuple[PermanentRoomDefinition, ...]]:
    path = Path(os.getenv("BATTLE_PERMANENT_ROOMS_CONFIG") or DEFAULT_CONFIG)
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = _positive_int(raw.get("version"), 1)
    defaults = dict(raw.get("defaults") or {})
    policy_raw = dict(raw.get("host_policy") or {})
    policy = HostPolicy(
        offline_grace_seconds=_positive_int(policy_raw.get("offline_grace_seconds"), 45),
        idle_timeout_seconds=_positive_int(policy_raw.get("idle_timeout_seconds"), 180),
        member_offline_grace_seconds=_positive_int(
            policy_raw.get("member_offline_grace_seconds"), 120
        ),
        sweep_interval_seconds=_positive_int(policy_raw.get("sweep_interval_seconds"), 10),
    )
    definitions: list[PermanentRoomDefinition] = []
    seen: set[str] = set()
    for item in raw.get("rooms") or []:
        template_key = str(item.get("template_key") or "").strip()
        mode_key = str(item.get("mode_key") or "").strip().lower()
        full_pattern = str(item.get("full_pattern") or "").strip()
        initial_board = str(item.get("initial_board") or "").strip().lower()
        if not template_key or template_key in seen:
            raise ValueError("duplicate_or_missing_permanent_template_key")
        if mode_key not in {"goodness", "free_goodness"}:
            raise ValueError(f"unsupported_permanent_mode:{mode_key}")
        if len(initial_board) != 16 or any(c not in "0123456789abcdef" for c in initial_board):
            raise ValueError(f"invalid_permanent_initial_board:{template_key}")
        seen.add(template_key)
        definitions.append(
            PermanentRoomDefinition(
                template_key=template_key,
                mode_key=mode_key,
                full_pattern=full_pattern,
                initial_board=initial_board,
                max_players=_positive_int(
                    item.get("max_players"), defaults.get("max_players", 8)
                ),
                step_timeout_seconds=_positive_int(
                    item.get("step_timeout_seconds"),
                    defaults.get("step_timeout_seconds", 90),
                ),
                version=version,
            )
        )
    return policy, tuple(definitions)
