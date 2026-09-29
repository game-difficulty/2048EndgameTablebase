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
        async def send(websocket, principal):
            try:
                snapshot = await asyncio.to_thread(snapshot_factory, principal)
                await asyncio.wait_for(websocket.send_json({"type": "room.snapshot", "data": snapshot}), timeout=2)
            except Exception:
                await self.disconnect(room_code, websocket)

        await asyncio.gather(*(send(websocket, principal) for websocket, principal in targets))

    async def broadcast_message(self, room_code: str, message: dict[str, Any]) -> None:
        """Send one shared public packet; no per-viewer snapshot or DB work."""
        async with self._lock:
            targets = tuple(self._rooms.get(room_code, {}))

        async def send(websocket):
            try:
                await asyncio.wait_for(websocket.send_json(message), timeout=2)
            except Exception:
                await self.disconnect(room_code, websocket)

        await asyncio.gather(*(send(websocket) for websocket in targets))
