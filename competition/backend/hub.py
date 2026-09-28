from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket

from .domain import Principal


class RoomHub:
    def __init__(self) -> None:
        self._rooms: dict[str, dict[WebSocket, Principal]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def connect(self, room_code: str, websocket: WebSocket, principal: Principal) -> None:
        async with self._lock:
            self._rooms[room_code][websocket] = principal

    async def disconnect(self, room_code: str, websocket: WebSocket) -> None:
        async with self._lock:
            sockets = self._rooms.get(room_code)
            if sockets is None:
                return
            sockets.pop(websocket, None)
            if not sockets:
                self._rooms.pop(room_code, None)

    async def broadcast(
        self,
        room_code: str,
        snapshot_factory: Callable[[Principal], dict[str, Any]],
    ) -> None:
        async with self._lock:
            targets = tuple(self._rooms.get(room_code, {}).items())
        stale: list[WebSocket] = []
        for websocket, principal in targets:
            try:
                snapshot = await asyncio.to_thread(snapshot_factory, principal)
                await websocket.send_json({"type": "room.snapshot", "data": snapshot})
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            await self.disconnect(room_code, websocket)

