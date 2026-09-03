from __future__ import annotations

from typing import Any

from backend.tablebase_catalog import resolve_tablebase

from ...core.chat_policy import normalize_chat_roles
from ...core.contracts import BattleMode
from ..goodness.mode import VALID_STEP_TIMEOUTS, _normalize_board
from .rules import (
    MOVE_RISK_LIMIT,
    SPAWN_DRAWDOWN_LIMIT,
    SPAWN_RISK_LIMIT,
)


class FreeGoodnessBattleMode(BattleMode):
    key = "free_goodness"
    version = 2
    artifact_kind = "free_goodness_state_v1"
    token_operation_key = "tester_lookup_hit"

    def __init__(self) -> None:
        self._runtime = None

    def bind_runtime(self, runtime) -> None:
        self._runtime = runtime

    @property
    def runtime(self):
        if self._runtime is None:
            raise RuntimeError("free_goodness_runtime_not_bound")
        return self._runtime

    def validate_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        full_pattern = str(payload.get("full_pattern") or "").strip()
        entry = resolve_tablebase(full_pattern)
        if entry is None:
            raise ValueError("table_unavailable")
        target = int(entry.get("target") or 0)
        if target < 2:
            raise ValueError("invalid_target")
        score_step_limit = target // 2
        raw_ranking_min_steps = payload.get("ranking_min_steps")
        if raw_ranking_min_steps in (None, ""):
            ranking_min_steps = score_step_limit
        else:
            if isinstance(raw_ranking_min_steps, bool):
                raise ValueError("invalid_ranking_min_steps")
            ranking_text = str(raw_ranking_min_steps).strip()
            if not ranking_text.isdigit():
                raise ValueError("invalid_ranking_min_steps")
            ranking_min_steps = int(ranking_text)
            if not 1 <= ranking_min_steps <= score_step_limit:
                raise ValueError("invalid_ranking_min_steps")
        timeout = int(payload.get("step_timeout_seconds") or 90)
        if timeout not in VALID_STEP_TIMEOUTS:
            raise ValueError("invalid_step_timeout")
        max_players = int(payload.get("max_players") or 2)
        if not 2 <= max_players <= 8:
            raise ValueError("invalid_max_players")
        initial_board = _normalize_board(payload.get("initial_board"))
        return {
            "full_pattern": full_pattern,
            "pattern": str(entry.get("pattern") or ""),
            "target": target,
            "initial_board": None if initial_board is None else f"{initial_board:016x}",
            "score_step_limit": score_step_limit,
            "ranking_min_steps": ranking_min_steps,
            "step_timeout_seconds": timeout,
            "max_players": max_players,
            "visibility": "public" if bool(payload.get("is_public", True)) else "private",
            "allow_spectators": bool(payload.get("allow_spectators", True)),
            "allow_guest_chat": bool(payload.get("allow_guest_chat", False)),
            "chat_roles": normalize_chat_roles(payload.get("chat_roles")),
            "move_risk_limit": MOVE_RISK_LIMIT,
            "spawn_risk_limit": SPAWN_RISK_LIMIT,
            "spawn_drawdown_limit": SPAWN_DRAWDOWN_LIMIT,
            "rules_version": self.version,
        }

    def repository_fields(self, settings: dict[str, Any]) -> dict[str, Any]:
        return {
            **settings,
            "max_steps": int(settings["score_step_limit"]),
        }

    def public_settings(self, room: dict[str, Any]) -> dict[str, Any]:
        settings = dict(room.get("settings") or {})
        if settings:
            settings.pop("move_risk_min_absolute_increase", None)
            settings["initial_board"] = room.get("initial_board")
            return settings
        return {
            "full_pattern": room.get("full_pattern"),
            "pattern": room.get("pattern"),
            "target": room.get("target"),
            "initial_board": room.get("initial_board"),
            "score_step_limit": int(room.get("max_steps") or 0),
            "ranking_min_steps": int(room.get("max_steps") or 0),
            "step_timeout_seconds": room.get("step_timeout_seconds"),
            "move_risk_limit": MOVE_RISK_LIMIT,
            "spawn_risk_limit": SPAWN_RISK_LIMIT,
            "spawn_drawdown_limit": SPAWN_DRAWDOWN_LIMIT,
            "rules_version": self.version,
        }

    async def create_room(self, **kwargs):
        return await self.runtime.create_room_for_mode(**kwargs)

    async def start_room(self, room_code: str, **kwargs):
        return await self.runtime.start_room_for_mode(room_code, **kwargs)

    async def ensure_permanent_room(self, definition):
        return await self.runtime.ensure_permanent_room(definition)

    async def normalize_lobby_settings_patch(self, room, payload):
        timeout = int(payload.get("step_timeout_seconds") or 0)
        if timeout not in VALID_STEP_TIMEOUTS:
            raise ValueError("invalid_step_timeout")
        target_cap = max(1, int(room.get("target") or 0) // 2)
        score_steps = int(payload.get("score_step_limit") or 0)
        ranking_steps = int(payload.get("ranking_min_steps") or 0)
        if not 1 <= score_steps <= target_cap:
            raise ValueError("invalid_score_step_limit")
        if not 1 <= ranking_steps <= score_steps:
            raise ValueError("invalid_ranking_min_steps")
        board = _normalize_board(payload.get("initial_board"))
        if board is None:
            raise ValueError("invalid_board")
        board_hex = await self.runtime.validate_lobby_initial_board(room, board)
        return {
            "step_timeout_seconds": timeout,
            "initial_board": board_hex,
            "score_step_limit": score_steps,
            "ranking_min_steps": ranking_steps,
        }

    def artifact_payload(self, room_code: str, round_id: str, *, actor_key: str):
        return self.runtime.artifact_payload_for_mode(
            room_code, round_id, actor_key=actor_key
        )

    def handle_action(self, room_code: str, *, actor_key: str, action: str, payload):
        return self.runtime.handle_action_for_mode(
            room_code, actor_key=actor_key, action=action, payload=payload
        )

    async def handle_action_async(
        self, room_code: str, *, actor_key: str, action: str, payload
    ):
        return await self.runtime.handle_action_for_mode(
            room_code, actor_key=actor_key, action=action, payload=payload
        )

    def sanitize_snapshot(
        self,
        payload: dict[str, Any],
        *,
        viewer_actor_key: str,
        viewer_user_id: int | None,
    ):
        return self.runtime.sanitize_snapshot_for_mode(
            payload,
            viewer_actor_key=viewer_actor_key,
            viewer_user_id=viewer_user_id,
        )

    def settle_unstarted_round(self, room_id: str, *, reason: str) -> None:
        self.runtime.settle_unstarted_round_for_mode(room_id, reason=reason)

    def forfeit_round(self, room_code: str, *, actor_key: str, round_id: str):
        return self.runtime.forfeit_round_for_mode(
            room_code, actor_key=actor_key, round_id=round_id
        )

    async def startup(self) -> None:
        await self.runtime.startup()

    async def shutdown(self) -> None:
        await self.runtime.shutdown()
