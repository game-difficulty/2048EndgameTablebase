from __future__ import annotations

from typing import Any

from Config import SingletonConfig
from fastapi import WebSocket

from ..actions import Action, Message
from ..minigames.service import (
    build_menu_payload,
    build_state_payload,
    cancel_powerup,
    maybe_reward_powerup_from_score,
    request_powerup,
    reset_powerups_for_new_game,
    resolve_powerup_target,
    save_current_game,
    start_minigame,
    trigger_custom_action,
)
from ..session import GameSession


async def _send_menu(websocket: WebSocket, session: GameSession) -> None:
    await websocket.send_json(
        {
            "action": Message.MINIGAME_MENU_DATA,
            "data": build_menu_payload(session.minigame_session),
        }
    )


async def _send_state(websocket: WebSocket, session: GameSession) -> None:
    payload = build_state_payload(session.minigame_session)
    await websocket.send_json(
        {
            "action": Message.MINIGAME_STATE,
            "data": payload,
        }
    )
    if session.minigame_session.engine is not None:
        session.minigame_session.engine.clear_animation()


async def handle_minigame_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> bool:
    if action == Action.MINIGAME_GET_MENU:
        await _send_menu(websocket, session)
        return True

    if action == Action.MINIGAME_SET_DIFFICULTY:
        difficulty = 1 if int(payload.get("difficulty", 0)) else 0
        session.minigame_session.difficulty = difficulty
        config = SingletonConfig().config
        config["minigame_difficulty"] = difficulty
        SingletonConfig().save_config(config)
        await _send_menu(websocket, session)
        if session.minigame_session.engine is not None:
            await _send_state(websocket, session)
        return True

    if action == Action.MINIGAME_START:
        game_id = str(payload.get("gameId") or "")
        start_minigame(session.minigame_session, game_id)
        await _send_menu(websocket, session)
        await _send_state(websocket, session)
        return True

    if action == Action.MINIGAME_NEW_GAME:
        engine = session.minigame_session.engine
        if engine is None:
            raise ValueError("No active minigame")
        engine.setup_new_round()
        reset_powerups_for_new_game(session.minigame_session)
        await _send_state(websocket, session)
        return True

    if action == Action.MINIGAME_MOVE:
        engine = session.minigame_session.engine
        if engine is None:
            raise ValueError("No active minigame")
        previous_score = int(engine.score)
        engine.do_move(str(payload.get("direction") or ""))
        awarded = maybe_reward_powerup_from_score(
            session.minigame_session, int(engine.score) - previous_score
        )
        if awarded:
            engine.queue_message("toast", f"+1 {awarded.capitalize()}")
        await _send_state(websocket, session)
        return True

    if action == Action.MINIGAME_INFO:
        engine = session.minigame_session.engine
        if engine is None:
            raise ValueError("No active minigame")
        engine.request_info()
        await _send_state(websocket, session)
        return True

    if action == Action.MINIGAME_CUSTOM_ACTION:
        engine = session.minigame_session.engine
        if engine is None:
            raise ValueError("No active minigame")
        trigger_custom_action(
            session.minigame_session,
            str(payload.get("key") or ""),
            str(payload.get("phase") or "trigger"),
        )
        await _send_state(websocket, session)
        return True

    if action == Action.MINIGAME_USE_POWERUP:
        engine = session.minigame_session.engine
        if engine is None:
            raise ValueError("No active minigame")
        request_powerup(session.minigame_session, str(payload.get("mode") or ""))
        await _send_state(websocket, session)
        return True

    if action == Action.MINIGAME_TARGET_ACTION:
        engine = session.minigame_session.engine
        if engine is None:
            raise ValueError("No active minigame")
        resolve_powerup_target(session.minigame_session, int(payload.get("index", -1)))
        await _send_state(websocket, session)
        return True

    if action == Action.MINIGAME_CANCEL_INTERACTION:
        cancel_powerup(session.minigame_session)
        await _send_state(websocket, session)
        return True

    if action == Action.MINIGAME_BACK_TO_MENU:
        save_current_game(session.minigame_session)
        session.minigame_session.reset_runtime()
        await _send_menu(websocket, session)
        return True

    if action == Action.MINIGAME_CLOSE:
        save_current_game(session.minigame_session)
        session.minigame_session.reset_runtime()
        return True

    return False
