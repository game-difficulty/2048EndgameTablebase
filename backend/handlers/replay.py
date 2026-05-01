from __future__ import annotations

import asyncio
from typing import Any

from Config import category_info
from fastapi import WebSocket
from engine_core.replay_utils import load_replay_file

from ..actions import Action
from ..replay import (
    _replay_load_record,
    _replay_pattern_from_path,
    _replay_reset,
    _replay_sync_step,
    send_replay_state,
)
from ..session import GameSession
from ..tester import LATEST_TESTER_REPLAY
from ..webview_api import Api


def _load_replay_path(session: GameSession, path: str) -> None:
    normalized_path = str(path or "").strip()
    if not normalized_path:
        return
    try:
        record = load_replay_file(normalized_path)
        if len(record) == 0:
            _replay_reset(session, "Recording file corrupted")
            return

        pattern = _replay_pattern_from_path(normalized_path)
        use_variant = pattern.split("_")[0] in category_info.get("variant", [])
        _replay_load_record(session, record, pattern, normalized_path, use_variant)
    except Exception as exc:
        _replay_reset(session, f"Failed to load replay: {exc}")


async def handle_replay_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> bool:
    if action == Action.REPLAY_GET_INIT:
        if not session.replay_loaded and len(LATEST_TESTER_REPLAY["record"]) > 0:
            _replay_load_record(
                session,
                LATEST_TESTER_REPLAY["record"],
                LATEST_TESTER_REPLAY["pattern"],
                LATEST_TESTER_REPLAY["source"],
                LATEST_TESTER_REPLAY["use_variant"],
            )
        elif not session.replay_loaded:
            _replay_reset(session, "No tester replay available yet.")
        await send_replay_state(websocket, session)
        return True

    if action == Action.REPLAY_LOAD_LATEST:
        if len(LATEST_TESTER_REPLAY["record"]) > 0:
            _replay_load_record(
                session,
                LATEST_TESTER_REPLAY["record"],
                LATEST_TESTER_REPLAY["pattern"],
                LATEST_TESTER_REPLAY["source"],
                LATEST_TESTER_REPLAY["use_variant"],
            )
        else:
            _replay_reset(session, "No tester replay available yet.")
        await send_replay_state(websocket, session)
        return True

    if action == Action.REPLAY_LOAD_FILE:
        path = str(payload.get("path") or "").strip()
        if not path:
            return True
        _load_replay_path(session, path)
        await send_replay_state(websocket, session)
        return True

    if action == Action.REPLAY_TRIGGER_OPEN_FILE:
        path = await asyncio.to_thread(Api().select_open_replay_file)
        if path:
            _load_replay_path(session, path)
        await send_replay_state(websocket, session)
        return True

    if action == Action.REPLAY_SET_STEP:
        if not session.replay_loaded:
            return True
        target_step = int(payload.get("step", 0))
        metadata = _replay_sync_step(session, target_step, animate=False)
        await send_replay_state(websocket, session, metadata)
        return True

    if action == Action.REPLAY_STEP:
        if not session.replay_loaded:
            return True
        delta = int(payload.get("delta", 1))
        previous_step = session.replay_current_step
        next_step = previous_step + delta
        metadata = _replay_sync_step(
            session,
            next_step,
            animate=(delta == 1),
            previous_step=previous_step,
        )
        await send_replay_state(websocket, session, metadata)
        return True

    if action == Action.REPLAY_NEXT_POINT:
        if not session.replay_loaded or not session.replay_points_rank:
            return True
        next_point = None
        for point in session.replay_points_rank:
            if point > session.replay_current_step:
                next_point = point
                break
        if next_point is None:
            return True
        metadata = _replay_sync_step(session, next_point, animate=False)
        await send_replay_state(websocket, session, metadata)
        return True

    return False
