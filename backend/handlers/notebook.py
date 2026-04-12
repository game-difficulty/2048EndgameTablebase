from __future__ import annotations

from typing import Any

from fastapi import WebSocket

from ..actions import Action, Message
from ..notebook import (
    _notebook_delete_current,
    _notebook_next_problem,
    _notebook_pattern_list,
    _notebook_select_pattern,
    _notebook_reset_unseen_boards,
    _notebook_answer,
    send_notebook_state,
)
from ..session import GameSession


async def handle_notebook_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> bool:
    if action == Action.NOTEBOOK_GET_INIT:
        patterns = _notebook_pattern_list()
        if not session.notebook_pattern and patterns:
            _notebook_select_pattern(session, patterns[0])
        await websocket.send_json(
            {
                "action": Message.NOTEBOOK_BOOTSTRAP,
                "data": {
                    "patterns": patterns,
                    "weight_mode": int(session.notebook_weight_mode),
                },
            }
        )
        await send_notebook_state(websocket, session)
        return True

    if action == Action.NOTEBOOK_SELECT_PATTERN:
        pattern = str(payload.get("pattern") or "")
        _notebook_select_pattern(session, pattern)
        await send_notebook_state(websocket, session)
        return True

    if action == Action.NOTEBOOK_SET_WEIGHT_MODE:
        mode = int(payload.get("mode", 0))
        session.notebook_weight_mode = max(0, min(2, mode))
        if session.notebook_pattern:
            _notebook_reset_unseen_boards(session)
            _notebook_next_problem(session)
        await send_notebook_state(websocket, session)
        return True

    if action == Action.NOTEBOOK_NEXT:
        _notebook_next_problem(session)
        await send_notebook_state(websocket, session)
        return True

    if action == Action.NOTEBOOK_ANSWER:
        direction = str(payload.get("direction") or "")
        _notebook_answer(session, direction)
        await send_notebook_state(websocket, session)
        return True

    if action == Action.NOTEBOOK_DELETE:
        _notebook_delete_current(session)
        await send_notebook_state(websocket, session)
        return True

    return False
