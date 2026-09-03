from __future__ import annotations

import hashlib
from typing import Any

from Config import category_info

from backend.auth.db import auth_db

from . import repository
from .actors import coerce_actor
from .core.contracts import BattleModeError
from .core import chat
from .core.errors import BattleServiceError
from .core.lifecycle import (
    broadcast_room,
    current_room,
    join_room,
    kick,
    leave_room,
    list_rooms,
    room_snapshot,
    set_broadcast_callback,
    set_ready,
    set_role,
    update_room_settings,
)
from .core.registry import get_battle_mode, list_battle_modes
from .modes.goodness import runtime as goodness_runtime
from .modes.free_goodness import runtime as free_goodness_runtime  # noqa: F401
from .replay_records import (
    BattleReplayError,
    battle_replay_filename,
    finalize_replay,
)


VALID_STEP_TIMEOUTS = goodness_runtime.VALID_STEP_TIMEOUTS
_route_tasks = goodness_runtime._route_tasks
mark_timeouts = goodness_runtime.mark_timeouts


def _mode_for_room(room_ref: str):
    room = repository.get_room(room_ref)
    try:
        return get_battle_mode(str(room.get("mode_key") or "goodness"))
    except BattleModeError as exc:
        raise BattleServiceError(
            "BATTLE_MODE_UNAVAILABLE",
            "This Battle mode is not available on the server.",
            409,
        ) from exc


async def create_room(
    *,
    user_id: int,
    session_id: int | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    mode_key = str(payload.get("mode_key") or "goodness")
    try:
        mode = get_battle_mode(mode_key)
    except BattleModeError as exc:
        raise BattleServiceError(
            "BATTLE_MODE_UNAVAILABLE", "The selected Battle mode is unavailable.", 409
        ) from exc
    return await mode.create_room(
        user_id=user_id,
        session_id=session_id,
        payload=payload,
    )


async def start_room(
    room_code: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
    session_id: int | None = None,
) -> dict[str, Any]:
    identity = coerce_actor(actor, user_id=user_id)
    from .permanent.service import assert_start_allowed

    assert_start_allowed(room_code, identity)
    return await _mode_for_room(room_code).start_room(
        room_code,
        actor_key=identity.actor_key,
        user_id=identity.user_id,
        session_id=session_id if session_id is not None else (
            int(getattr(actor, "session_id"))
            if actor is not None and getattr(actor, "session_id", None) is not None
            else None
        ),
    )


def route_payload(
    room_code: str,
    round_id: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    identity = coerce_actor(actor, user_id=user_id)
    mode = _mode_for_room(room_code)
    blob, metadata = mode.artifact_payload(
        room_code,
        round_id,
        actor_key=identity.actor_key,
    )
    payload = dict(metadata or {})
    payload.setdefault("artifact_kind", mode.artifact_kind)
    payload.setdefault("artifact_hash", hashlib.sha256(blob).hexdigest())
    return blob, payload


def player_replay_payload(
    room_code: str,
    round_id: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    identity = coerce_actor(actor, user_id=user_id)
    with auth_db() as db:
        row = db.execute(
            """
            SELECT result.replay_blob, result.replay_move_count,
                   result.board_state, result.goodness_of_fit,
                   room.mode_key, room.pattern, room.full_pattern
            FROM battle_player_results AS result
            JOIN battle_rounds AS round ON round.round_id = result.round_id
            JOIN battle_rooms AS room ON room.room_id = round.room_id
            WHERE result.round_id = ? AND result.actor_key = ?
              AND (room.room_id = ? OR room.room_code = ? COLLATE NOCASE)
            """,
            (str(round_id), identity.actor_key, str(room_code), str(room_code)),
        ).fetchone()
    if row is None or int(row["replay_move_count"] or 0) <= 0:
        raise BattleServiceError(
            "BATTLE_REPLAY_NOT_FOUND",
            "No replay is available for this Battle result.",
            404,
        )
    try:
        payload = finalize_replay(
            bytes(row["replay_blob"] or b""),
            terminal_board=int(str(row["board_state"] or "0"), 16),
        )
    except (BattleReplayError, TypeError, ValueError) as exc:
        raise BattleServiceError(
            "BATTLE_REPLAY_INVALID",
            "This Battle replay is unavailable.",
            409,
        ) from exc
    full_pattern = str(row["full_pattern"] or "")
    return payload, {
        "filename": battle_replay_filename(
            mode_key=str(row["mode_key"] or "battle"),
            full_pattern=full_pattern,
            goodness_of_fit=float(row["goodness_of_fit"] or 0.0),
        ),
        "full_pattern": full_pattern,
        "use_variant": str(row["pattern"] or "")
        in category_info.get("variant", []),
        "move_count": int(row["replay_move_count"] or 0),
    }


def record_choice(
    room_code: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
    round_id: str,
    sequence: int,
    route_index: int,
    direction: str,
) -> dict[str, Any]:
    identity = coerce_actor(actor, user_id=user_id)
    mode = _mode_for_room(room_code)
    if mode.key != "goodness":
        raise BattleServiceError(
            "BATTLE_ACTION_UNSUPPORTED",
            "This Battle mode requires the current action protocol.",
            409,
        )
    return mode.handle_action(
        room_code,
        actor_key=identity.actor_key,
        action="move",
        payload={
            "round_id": round_id,
            "sequence": sequence,
            "route_index": route_index,
            "direction": direction,
        },
    )


def handle_mode_action(
    room_code: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
    action: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    identity = coerce_actor(actor, user_id=user_id)
    return _mode_for_room(room_code).handle_action(
        room_code,
        actor_key=identity.actor_key,
        action=action,
        payload=payload,
    )


async def handle_mode_action_async(
    room_code: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
    action: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    identity = coerce_actor(actor, user_id=user_id)
    return await _mode_for_room(room_code).handle_action_async(
        room_code,
        actor_key=identity.actor_key,
        action=action,
        payload=payload,
    )


def forfeit_round(
    room_code: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
    round_id: str,
) -> dict[str, Any]:
    identity = coerce_actor(actor, user_id=user_id)
    return _mode_for_room(room_code).forfeit_round(
        room_code,
        actor_key=identity.actor_key,
        round_id=round_id,
    )


async def startup() -> None:
    repository.init_battle_db()
    await chat.startup()
    for mode in list_battle_modes().values():
        await mode.startup()
    from .permanent import startup as startup_permanent_rooms

    await startup_permanent_rooms()


async def shutdown() -> None:
    from .permanent import shutdown as shutdown_permanent_rooms

    await shutdown_permanent_rooms()
    for mode in reversed(tuple(list_battle_modes().values())):
        await mode.shutdown()
    await chat.shutdown()
