from __future__ import annotations

from typing import Any

from backend.tablebase_catalog import resolve_tablebase

from ...core.chat_policy import normalize_chat_roles
from ...core.contracts import BattleMode


VALID_STEP_TIMEOUTS = frozenset(range(5, 601, 5))


def _normalize_board(value: Any) -> int | None:
    if value in (None, ""):
        return None
    board = str(value).strip().lower()
    if len(board) != 16 or any(char not in "0123456789abcdef" for char in board):
        raise ValueError("invalid_board")
    return int(board, 16)


class GoodnessBattleMode(BattleMode):
    key = "goodness"
    version = 1
    artifact_kind = "trainer_route_v1"
    token_operation_key = "battle_route_generation"

    def __init__(self) -> None:
        self._runtime = None

    def bind_runtime(self, runtime) -> None:
        self._runtime = runtime

    @property
    def runtime(self):
        if self._runtime is None:
            raise RuntimeError("goodness_runtime_not_bound")
        return self._runtime

    def validate_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        full_pattern = str(payload.get("full_pattern") or "").strip()
        resolver = (
            getattr(self._runtime, "resolve_tablebase", None)
            if self._runtime is not None
            else None
        ) or resolve_tablebase
        entry = resolver(full_pattern)
        if entry is None:
            raise ValueError("table_unavailable")
        max_steps = payload.get("max_steps")
        if max_steps in (None, "", 0, "0"):
            max_steps = None
        else:
            max_steps = int(max_steps)
            if not 1 <= max_steps <= 9_999:
                raise ValueError("invalid_max_steps")
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
            "target": int(entry.get("target") or 0),
            "initial_board": None if initial_board is None else f"{initial_board:016x}",
            "max_steps": max_steps,
            "step_timeout_seconds": timeout,
            "max_players": max_players,
            "visibility": "public" if bool(payload.get("is_public", True)) else "private",
            "allow_spectators": bool(payload.get("allow_spectators", True)),
            "allow_guest_chat": bool(payload.get("allow_guest_chat", False)),
            "chat_roles": normalize_chat_roles(payload.get("chat_roles")),
        }

    def repository_fields(self, settings: dict[str, Any]) -> dict[str, Any]:
        return dict(settings)

    def public_settings(self, room: dict[str, Any]) -> dict[str, Any]:
        settings = dict(room.get("settings") or {})
        if settings:
            return settings
        return {
            "full_pattern": room.get("full_pattern"),
            "pattern": room.get("pattern"),
            "target": room.get("target"),
            "initial_board": room.get("initial_board"),
            "max_steps": room.get("max_steps"),
            "step_timeout_seconds": room.get("step_timeout_seconds"),
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
        settings = {"step_timeout_seconds": timeout}
        if room.get("lifecycle_kind") == "permanent" and "initial_board" in payload:
            board = _normalize_board(payload["initial_board"])
            if board is None:
                raise ValueError("invalid_board")
            settings["initial_board"] = f"{board:016x}"
        return settings

    def artifact_payload(self, room_code: str, round_id: str, *, actor_key: str):
        return self.runtime.artifact_payload_for_mode(
            room_code, round_id, actor_key=actor_key
        )

    def sanitize_snapshot(
        self,
        payload: dict[str, Any],
        *,
        viewer_actor_key: str,
        viewer_user_id: int | None,
    ) -> dict[str, Any]:
        return self.runtime.sanitize_snapshot_for_mode(
            payload,
            viewer_actor_key=viewer_actor_key,
            viewer_user_id=viewer_user_id,
        )

    def handle_action(
        self,
        room_code: str,
        *,
        actor_key: str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self.runtime.handle_action_for_mode(
            room_code,
            actor_key=actor_key,
            action=action,
            payload=payload,
        )

    def settle_unstarted_round(self, room_id: str, *, reason: str) -> None:
        self.runtime.settle_unstarted_round_for_mode(room_id, reason=reason)

    def forfeit_round(
        self,
        room_code: str,
        *,
        actor_key: str,
        round_id: str,
    ) -> dict[str, Any]:
        return self.runtime.forfeit_round_for_mode(
            room_code,
            actor_key=actor_key,
            round_id=round_id,
        )

    async def startup(self) -> None:
        await self.runtime.startup()

    async def shutdown(self) -> None:
        await self.runtime.shutdown()
