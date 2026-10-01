from __future__ import annotations

import asyncio
from collections import defaultdict
import json
import logging

from .errors import CompetitionError

log = logging.getLogger(__name__)

class SocketWriter:
    """Exactly one writer per socket. Never leave a silently detached socket."""
    def __init__(self, socket, on_close, byte_limit=4 * 1024 * 1024):
        self.socket, self.on_close, self.byte_limit = socket, on_close, byte_limit
        self.queue = asyncio.Queue(maxsize=128)
        self.bytes = 0
        self.closed = False
        self.task = asyncio.create_task(self.run())

    async def put(self, message):
        if self.closed:
            return
        text = json.dumps(message, separators=(',', ':'), ensure_ascii=False)
        size = len(text.encode())
        if self.queue.full() or self.bytes + size > self.byte_limit:
            log.warning('competition socket backpressure messages=%s queued_bytes=%s incoming_bytes=%s',
                        self.queue.qsize(), self.bytes, size)
            await self.close(1013, 'stream_backpressure')
            return
        self.bytes += size
        self.queue.put_nowait((text, size))

    async def run(self):
        try:
            while True:
                text, size = await self.queue.get()
                self.bytes -= size
                try:
                    await asyncio.wait_for(self.socket.send_text(text), 5)
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as error:
            log.warning('competition socket send failed: %s queued_bytes=%s', type(error).__name__, self.bytes)
            await self.close(1013, 'stream_send_timeout')

    async def finish(self, message, code, reason):
        await self.put(message)
        try:
            await asyncio.wait_for(self.queue.join(), 2)
        except asyncio.TimeoutError:
            pass
        await self.close(code, reason)

    async def close(self, code=1000, reason='stream_closed'):
        if self.closed:
            return
        self.closed = True
        if asyncio.current_task() is not self.task:
            self.task.cancel()
        try:
            await asyncio.wait_for(self.socket.close(code=code, reason=reason), 2)
        except Exception:
            pass
        await self.on_close()

class RoomHub:
    def __init__(self):
        self._rooms = defaultdict(dict)
        self._writers = {}
        self._listeners = defaultdict(set)

    async def connect(self, room_code, websocket, principal):
        self._rooms[room_code][websocket] = principal
        self._writers[websocket] = SocketWriter(websocket, lambda: self.disconnect(room_code, websocket))

    async def send(self, websocket, message):
        writer = self._writers.get(websocket)
        if writer:
            await writer.put(message)

    async def disconnect(self, room_code, websocket):
        self._rooms[room_code].pop(websocket, None)
        writer = self._writers.pop(websocket, None)
        if writer and not writer.closed:
            await writer.close()
        if not self._rooms[room_code]:
            self._rooms.pop(room_code, None)

    def subscribe(self, room_code):
        queue = asyncio.Queue(maxsize=1)
        self._listeners[room_code].add(queue)
        return queue

    def unsubscribe(self, room_code, queue):
        self._listeners[room_code].discard(queue)
        if not self._listeners[room_code]:
            self._listeners.pop(room_code, None)

    def notify(self, room_code):
        # Notification may coalesce: retained frames, not the notification, carry history.
        for queue in tuple(self._listeners.get(room_code, ())):
            if queue.empty():
                queue.put_nowait(True)

    async def broadcast(self, room_code, snapshot_factory):
        self.notify(room_code)
        async def send(socket, principal):
            try:
                snapshot = await asyncio.to_thread(snapshot_factory, principal)
                await self.send(socket, {'type': 'room.snapshot', 'data': snapshot})
            except CompetitionError as error:
                writer = self._writers.get(socket)
                if writer:
                    await writer.close(4403 if error.code == 'REMOVED_FROM_ROOM' else 1013, error.code)
            except Exception:
                log.exception('competition snapshot failed')
                writer = self._writers.get(socket)
                if writer:
                    await writer.close(1013, 'snapshot_failed')
        await asyncio.gather(*(send(s, p) for s, p in tuple(self._rooms.get(room_code, {}).items())))

    async def broadcast_message(self, room_code, message):
        self.notify(room_code)
        await asyncio.gather(*(self.send(socket, message) for socket in tuple(self._rooms.get(room_code, {}))))
