from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from backend.actions import Action, Message
from backend.quota.service import get_token_balance

from . import repository
from .core import chat
from .service import (
    BattleServiceError,
    current_room,
    handle_mode_action_async,
    record_choice,
    room_snapshot,
    set_broadcast_callback,
)


_room_sockets: dict[str, set[WebSocket]] = defaultdict(set)
_socket_room: dict[WebSocket, str] = {}
_socket_user: dict[WebSocket, int] = {}
_lock = asyncio.Lock()


async def _send(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    try:
        if (
            websocket.application_state != WebSocketState.CONNECTED
            or websocket.client_state != WebSocketState.CONNECTED
        ):
            return False
        await websocket.send_json(payload)
        return True
    except Exception:
        return False


async def _detach(websocket: WebSocket) -> str | None:
    async with _lock:
        room_id = _socket_room.pop(websocket, None)
        _socket_user.pop(websocket, None)
        if room_id is not None:
            sockets = _room_sockets.get(room_id)
            if sockets is not None:
                sockets.discard(websocket)
                if not sockets:
                    _room_sockets.pop(room_id, None)
        return room_id


def _online_user_ids(room_id: str) -> set[int]:
    return {
        int(_socket_user[socket])
        for socket in tuple(_room_sockets.get(room_id, ()))
        if socket in _socket_user
    }


def _with_presence(snapshot: dict[str, Any]) -> dict[str, Any]:
    online = _online_user_ids(str(snapshot["room_id"]))
    for member in snapshot.get("members", []):
        member["online"] = int(member["user_id"]) in online
    for result in snapshot.get("results", []):
        result["online"] = int(result["user_id"]) in online
    return snapshot


async def subscribe(websocket: WebSocket, *, user_id: int, room_ref: str | None) -> None:
    snapshot = (
        room_snapshot(str(room_ref), user_id=user_id)
        if room_ref
        else current_room(user_id=user_id)
    )
    if snapshot is None:
        raise BattleServiceError("ROOM_NOT_FOUND", "No active Battle room was found.", 404)
    previous = await _detach(websocket)
    async with _lock:
        room_id = str(snapshot["room_id"])
        _socket_room[websocket] = room_id
        _socket_user[websocket] = int(user_id)
        _room_sockets[room_id].add(websocket)
    if previous and previous != str(snapshot["room_id"]):
        await broadcast_room(previous)
    await broadcast_room(str(snapshot["room_id"]))
    messages = await asyncio.to_thread(
        chat.recent_messages,
        str(snapshot["room_id"]),
        user_id=int(user_id),
    )
    await _send(
        websocket,
        {
            "action": Message.BATTLE_CHAT_HISTORY,
            "data": {
                "room_id": str(snapshot["room_id"]),
                "messages": messages,
            },
        },
    )


async def broadcast_room(room_id: str) -> None:
    sockets = tuple(_room_sockets.get(str(room_id), ()))
    stale: list[WebSocket] = []
    for websocket in sockets:
        user_id = _socket_user.get(websocket)
        if user_id is None:
            stale.append(websocket)
            continue
        try:
            snapshot = _with_presence(room_snapshot(str(room_id), user_id=user_id))
            sent = await _send(
                websocket,
                {"action": Message.BATTLE_ROOM_STATE, "data": {"room": snapshot}},
            )
        except BattleServiceError:
            try:
                close_code = repository.room_unavailable_reason(
                    str(room_id), user_id=int(user_id)
                )
            except Exception:
                close_code = None
            if close_code is None:
                # Temporary snapshot failures are retried by heartbeat/reconnect.
                continue
            sent = await _send(
                websocket,
                {
                    "action": Message.BATTLE_ROOM_STATE,
                    "data": {
                        "room": None,
                        "closed": True,
                        "room_id": str(room_id),
                        "code": close_code,
                        "token_balance": get_token_balance(int(user_id)),
                    },
                },
            )
        if not sent:
            stale.append(websocket)
    for websocket in stale:
        await _detach(websocket)


async def broadcast_chat_message(room_id: str, message: dict[str, Any]) -> None:
    stale: list[WebSocket] = []
    payload = {
        "action": Message.BATTLE_CHAT_MESSAGE,
        "data": {"message": message},
    }
    for websocket in tuple(_room_sockets.get(str(room_id), ())):
        if not await _send(websocket, payload):
            stale.append(websocket)
    for websocket in stale:
        await _detach(websocket)


async def handle_battle_action(
    action: str | None,
    payload: dict[str, Any],
    session,
    websocket: WebSocket,
) -> bool:
    if action not in {
        Action.BATTLE_SUBSCRIBE,
        Action.BATTLE_ACTION,
        Action.BATTLE_PROGRESS,
        Action.BATTLE_HEARTBEAT,
        Action.BATTLE_CHAT_SEND,
    }:
        return False
    if session.user_id is None:
        raise BattleServiceError("AUTH_REQUIRED", "Authentication required.", 401)
    if action == Action.BATTLE_SUBSCRIBE:
        await subscribe(
            websocket,
            user_id=int(session.user_id),
            room_ref=(str(payload.get("room_code")) if payload.get("room_code") else None),
        )
        return True
    room_id = _socket_room.get(websocket)
    if room_id is None:
        await subscribe(websocket, user_id=int(session.user_id), room_ref=None)
        room_id = _socket_room.get(websocket)
    if action == Action.BATTLE_HEARTBEAT:
        if room_id is not None:
            await broadcast_room(room_id)
        return True
    if action == Action.BATTLE_CHAT_SEND:
        request_id = str(payload.get("request_id") or "")[:160]
        try:
            result = await asyncio.to_thread(
                chat.post_message,
                str(room_id or ""),
                user_id=int(session.user_id),
                request_id=request_id,
                content=payload.get("content"),
            )
        except chat.BattleChatRateLimit as exc:
            await _send(
                websocket,
                {
                    "action": Message.BATTLE_CHAT_RATE_LIMITED,
                    "data": {**exc.detail, "request_id": request_id},
                },
            )
            return True
        except BattleServiceError as exc:
            await _send(
                websocket,
                {
                    "action": Message.BATTLE_CHAT_REJECTED,
                    "data": {**exc.detail, "request_id": request_id},
                },
            )
            return True
        if result.created:
            await broadcast_chat_message(str(result.message["room_id"]), result.message)
        else:
            await _send(
                websocket,
                {
                    "action": Message.BATTLE_CHAT_MESSAGE,
                    "data": {"message": result.message},
                },
            )
        return True
    generic_action = action == Action.BATTLE_ACTION
    request_id = str(payload.get("request_id") or "")[:160]
    mode_payload = payload.get("payload") if generic_action else payload
    if not isinstance(mode_payload, dict):
        mode_payload = {}
    try:
        if generic_action:
            accepted = await handle_mode_action_async(
                str(payload.get("room_code") or room_id or ""),
                user_id=int(session.user_id),
                action=str(payload.get("mode_action") or ""),
                payload={**mode_payload, "request_id": request_id},
            )
        else:
            accepted = await asyncio.to_thread(
                record_choice,
                str(payload.get("room_code") or room_id or ""),
                user_id=int(session.user_id),
                round_id=str(payload.get("round_id") or ""),
                sequence=int(payload.get("sequence")),
                route_index=int(payload.get("route_index")),
                direction=str(payload.get("direction") or ""),
            )
    except BattleServiceError as exc:
        await _send(
            websocket,
            {
                "action": (
                    Message.BATTLE_ACTION_CONFLICT
                    if generic_action
                    else Message.BATTLE_PROGRESS_CONFLICT
                ),
                "data": {**exc.detail, "request_id": request_id},
            },
        )
        return True
    await _send(
        websocket,
        {
            "action": (
                Message.BATTLE_ACTION_ACCEPTED
                if generic_action
                else Message.BATTLE_CHOICE_ACCEPTED
            ),
            "data": {**accepted, "request_id": request_id},
        },
    )
    if room_id is not None:
        await broadcast_room(room_id)
    return True


async def disconnect(websocket: WebSocket) -> None:
    room_id = await _detach(websocket)
    if room_id is not None:
        await broadcast_room(room_id)


set_broadcast_callback(broadcast_room)
