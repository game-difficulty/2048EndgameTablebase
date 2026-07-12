from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from Config import category_info
from fastapi import WebSocket
from engine_core.replay_utils import REPLAY_DTYPE, load_replay_file, strip_replay_sentinel

from ..actions import Action
from ..cloud_files import get_upload_record
from ..cloud_safety import is_cloud_mode
from ..quota.config import MULTIPLIER_UNIT
from ..quota.service import consume_operation_tokens
from ..replay import (
    _replay_load_record,
    _replay_pattern_from_path,
    _replay_reset,
    _replay_sync_step,
    send_replay_state,
)
from ..session import GameSession
from ..tester import LATEST_TESTER_REPLAY, get_scoped_latest_tester_replay
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


def _load_replay_upload(session: GameSession, upload_id: str, filename: str = "", pattern: str = "") -> None:
    try:
        upload = get_upload_record(upload_id, user_id=session.user_id)
        raw_bytes = upload.path.read_bytes()
        if len(raw_bytes) == 0 or len(raw_bytes) % REPLAY_DTYPE.itemsize != 0:
            _replay_reset(session, "Recording file corrupted")
            return
        record = np.frombuffer(raw_bytes, dtype=REPLAY_DTYPE).copy()
        record = strip_replay_sentinel(record)
        if len(record) == 0:
            _replay_reset(session, "Recording file corrupted")
            return
        source_name = filename or upload.filename
        resolved_pattern = pattern or _replay_pattern_from_path(source_name)
        use_variant = resolved_pattern.split("_")[0] in category_info.get("variant", [])
        _replay_load_record(session, record, resolved_pattern, source_name, use_variant)
    except Exception as exc:
        _replay_reset(session, f"Failed to load replay upload")


def _get_latest_tester_replay(session: GameSession) -> dict[str, Any]:
    if is_cloud_mode():
        return get_scoped_latest_tester_replay(session)
    return LATEST_TESTER_REPLAY


async def handle_replay_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> bool:
    if action == Action.REPLAY_GET_INIT:
        latest_replay = _get_latest_tester_replay(session)
        if not session.replay_loaded and len(latest_replay["record"]) > 0:
            _replay_load_record(
                session,
                latest_replay["record"],
                latest_replay["pattern"],
                latest_replay["source"],
                latest_replay["use_variant"],
            )
        elif not session.replay_loaded:
            _replay_reset(session, "No tester replay available yet.")
        await send_replay_state(websocket, session)
        return True

    if action == Action.REPLAY_LOAD_LATEST:
        latest_replay = _get_latest_tester_replay(session)
        if len(latest_replay["record"]) > 0:
            _replay_load_record(
                session,
                latest_replay["record"],
                latest_replay["pattern"],
                latest_replay["source"],
                latest_replay["use_variant"],
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

    if action == Action.REPLAY_LOAD_UPLOAD:
        upload_id = str(payload.get("upload_id") or "").strip()
        if not upload_id:
            return True
        consume_operation_tokens(
            user_id=session.user_id,
            session_id=session.auth_session_id,
            operation_key="replay_load",
            full_pattern="",
            multiplier_override_units=MULTIPLIER_UNIT,
            metadata={"upload_id": upload_id},
        )
        _load_replay_upload(
            session,
            upload_id,
            filename=str(payload.get("filename") or ""),
            pattern=str(payload.get("pattern") or ""),
        )
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
