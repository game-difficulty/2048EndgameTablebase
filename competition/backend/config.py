from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


COMPETITION_ROOT = Path(__file__).resolve().parents[1]


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _csv(name: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, "").split(",") if item.strip())


def _positive_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class CompetitionSettings:
    database_path: Path
    allow_dev_auth: bool
    bootstrap_organizer_ids: frozenset[int]
    room_creator_ids: frozenset[int]
    cors_origins: tuple[str, ...]
    frontend_dist: Path
    draw_reveal_seconds: int
    draft_turn_seconds: int
    c_draw_reveal_seconds: int
    lineup_seconds: int
    team_clock_seconds: int
    test_project_target_tile: int
    live_internal_token: str
    live_result_retention_seconds: int


def load_settings() -> CompetitionSettings:
    database_path = Path(
        os.getenv("COMPETITION_DB")
        or COMPETITION_ROOT / "runtime" / "competition.sqlite3"
    )
    raw_ids = _csv("COMPETITION_BOOTSTRAP_ORGANIZER_IDS")
    organizer_ids = frozenset(int(item) for item in raw_ids if item.isdigit())
    raw_creator_ids = _csv("COMPETITION_ROOM_CREATOR_IDS")
    room_creator_ids = frozenset(int(item) for item in raw_creator_ids if item.isdigit())
    cors_origins = _csv("COMPETITION_CORS_ORIGINS") or (
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    )
    return CompetitionSettings(
        database_path=database_path,
        allow_dev_auth=_flag("COMPETITION_ALLOW_DEV_AUTH", False),
        bootstrap_organizer_ids=organizer_ids,
        room_creator_ids=room_creator_ids,
        cors_origins=cors_origins,
        frontend_dist=COMPETITION_ROOT / "frontend" / "dist",
        draw_reveal_seconds=10,
        draft_turn_seconds=_positive_int("COMPETITION_DRAFT_TURN_SECONDS", 60, 5),
        c_draw_reveal_seconds=10,
        lineup_seconds=_positive_int("COMPETITION_LINEUP_SECONDS", 180, 5),
        team_clock_seconds=_positive_int("COMPETITION_TEAM_CLOCK_SECONDS", 1800, 30),
        test_project_target_tile=_positive_int(
            "COMPETITION_TEST_PROJECT_TARGET_TILE", 2048, 4
        ),
        live_internal_token=os.getenv("COMPETITION_LIVE_INTERNAL_TOKEN", "").strip(),
        live_result_retention_seconds=_positive_int(
            "COMPETITION_LIVE_RESULT_RETENTION_SECONDS", 1800, 60
        ),
    )
