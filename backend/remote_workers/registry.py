from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .config import configured_workers, worker_secret
from .errors import (
    RemoteTablebaseOffline,
    RemoteTablebaseProtocolError,
    RemoteTablebaseTimeout,
)


PROTOCOL_VERSION = 1
HEARTBEAT_TIMEOUT_SECONDS = float(
    os.getenv("REMOTE_TABLEBASE_HEARTBEAT_TIMEOUT_SECONDS", "30")
)
REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("REMOTE_TABLEBASE_REQUEST_TIMEOUT_SECONDS", "30")
)
HELLO_TIMEOUT_SECONDS = float(os.getenv("REMOTE_TABLEBASE_HELLO_TIMEOUT_SECONDS", "10"))
MAX_WORKER_MESSAGE_BYTES = int(
    os.getenv("REMOTE_TABLEBASE_MAX_MESSAGE_BYTES", str(2 * 1024 * 1024))
)
logger = logging.getLogger("2048tables.remote_worker")


@dataclass
class WorkerConnection:
    worker_id: str
    websocket: WebSocket
    tables: frozenset[str]
    connected_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class PendingRequest:
    worker_id: str
    full_pattern: str
    future: asyncio.Future


def _valid_board_hex(value: Any) -> str:
    board = str(value or "").strip().lower()
    if len(board) != 16 or any(char not in "0123456789abcdef" for char in board):
        raise RemoteTablebaseProtocolError("Invalid remote tablebase board.")
    return board


class RemoteWorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, WorkerConnection] = {}
        self._pending: dict[str, PendingRequest] = {}
        self._lock: asyncio.Lock | None = None
        self._monitor_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._availability_epoch = 0
        self._availability_listeners: set[Callable[[int], None]] = set()

    @property
    def availability_epoch(self) -> int:
        return self._availability_epoch

    def add_availability_listener(self, listener: Callable[[int], None]) -> None:
        self._availability_listeners.add(listener)

    def remove_availability_listener(self, listener: Callable[[int], None]) -> None:
        self._availability_listeners.discard(listener)

    def _mark_availability_changed(self) -> None:
        self._availability_epoch += 1
        for listener in tuple(self._availability_listeners):
            try:
                listener(self._availability_epoch)
            except Exception:
                pass

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._ensure_lock()
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor())

    async def close(self) -> None:
        task = self._monitor_task
        self._monitor_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        workers = list(self._workers.values())
        self._workers.clear()
        if workers:
            self._mark_availability_changed()
        self._fail_all(RemoteTablebaseOffline())
        for worker in workers:
            try:
                await worker.websocket.close(code=1001)
            except Exception:
                pass
        self._lock = None
        self._loop = None

    def run_from_worker_thread(self, coroutine, *, timeout: float = 300.0):
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RemoteTablebaseOffline()
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            future.cancel()
            raise RemoteTablebaseTimeout() from exc

    def online_tables(self) -> frozenset[str]:
        now = time.monotonic()
        tables: set[str] = set()
        for worker in list(self._workers.values()):
            if now - worker.last_seen <= HEARTBEAT_TIMEOUT_SECONDS:
                tables.update(worker.tables)
        return frozenset(tables)

    def is_table_online(self, full_pattern: str) -> bool:
        return str(full_pattern or "") in self.online_tables()

    def status(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        result = []
        for worker_id, configured in configured_workers().items():
            worker = self._workers.get(worker_id)
            online = bool(
                worker and now - worker.last_seen <= HEARTBEAT_TIMEOUT_SECONDS
            )
            result.append(
                {
                    "worker_id": worker_id,
                    "online": online,
                    "last_seen_seconds_ago": (
                        max(0.0, now - worker.last_seen) if worker else None
                    ),
                    "tables": sorted(worker.tables if online and worker else ()),
                    "configured_tables": sorted(configured["tables"]),
                }
            )
        return result

    async def handle_connection(self, websocket: WebSocket) -> None:
        await self.start()
        await websocket.accept()
        worker: WorkerConnection | None = None
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(), timeout=HELLO_TIMEOUT_SECONDS
            )
            hello = self._parse_message(raw)
            worker = await self._accept_hello(websocket, hello)
            await self._send(
                worker,
                {
                    "type": "HELLO_ACK",
                    "protocol_version": PROTOCOL_VERSION,
                    "worker_id": worker.worker_id,
                    "heartbeat_timeout_seconds": HEARTBEAT_TIMEOUT_SECONDS,
                    "tables": sorted(worker.tables),
                },
            )
            while True:
                message = self._parse_message(await websocket.receive_text())
                await self._handle_message(worker, message)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as exc:
            logger.warning(
                "Remote tablebase worker connection ended: %s",
                type(exc).__name__,
            )
            try:
                await websocket.close(code=1008)
            except Exception:
                pass
        finally:
            if worker is not None:
                await self._remove_worker(worker, RemoteTablebaseOffline())

    def _parse_message(self, raw: str) -> dict[str, Any]:
        if len(raw.encode("utf-8")) > MAX_WORKER_MESSAGE_BYTES:
            raise RemoteTablebaseProtocolError("Worker message is too large.")
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RemoteTablebaseProtocolError("Invalid worker JSON.") from exc
        if not isinstance(message, dict):
            raise RemoteTablebaseProtocolError("Invalid worker message.")
        return message

    async def _accept_hello(
        self, websocket: WebSocket, hello: dict[str, Any]
    ) -> WorkerConnection:
        if str(hello.get("type") or "").upper() != "HELLO":
            raise RemoteTablebaseProtocolError("Worker HELLO is required.")
        if int(hello.get("protocol_version") or 0) != PROTOCOL_VERSION:
            raise RemoteTablebaseProtocolError("Unsupported worker protocol.")
        configured = configured_workers()
        worker_id = str(hello.get("worker_id") or "").strip()
        if worker_id not in configured:
            raise RemoteTablebaseProtocolError("Unknown worker.")
        expected_secret = worker_secret()
        supplied_secret = str(hello.get("auth_token") or "")
        if not expected_secret or not hmac.compare_digest(
            supplied_secret.encode("utf-8"), expected_secret.encode("utf-8")
        ):
            raise RemoteTablebaseProtocolError("Worker authentication failed.")

        advertised: set[str] = set()
        for item in hello.get("tables", []):
            if isinstance(item, str):
                full_pattern = item
                ready = True
            elif isinstance(item, dict):
                full_pattern = str(item.get("full_pattern") or "")
                ready = bool(item.get("ready", False))
            else:
                continue
            if ready and full_pattern in configured[worker_id]["tables"]:
                advertised.add(full_pattern)

        worker = WorkerConnection(worker_id, websocket, frozenset(advertised))
        lock = self._ensure_lock()
        async with lock:
            previous = self._workers.get(worker_id)
            previous_tables = previous.tables if previous else frozenset()
            self._workers[worker_id] = worker
            availability_changed = previous is None or previous_tables != worker.tables
        if availability_changed:
            self._mark_availability_changed()
        if previous is not None and previous.websocket is not websocket:
            self._fail_worker(worker_id, RemoteTablebaseOffline())
            try:
                await previous.websocket.close(code=1012)
            except Exception:
                pass
        logger.info(
            "Remote tablebase worker connected: %s (%d tables).",
            worker_id,
            len(worker.tables),
        )
        return worker

    async def _handle_message(
        self, worker: WorkerConnection, message: dict[str, Any]
    ) -> None:
        message_type = str(message.get("type") or "").upper()
        worker.last_seen = time.monotonic()
        if message_type in {"HEARTBEAT", "PONG"}:
            if isinstance(message.get("tables"), list):
                configured = configured_workers().get(worker.worker_id, {}).get(
                    "tables", {}
                )
                advertised: set[str] = set()
                for item in message["tables"]:
                    if isinstance(item, str):
                        full_pattern, ready = item, True
                    elif isinstance(item, dict):
                        full_pattern = str(item.get("full_pattern") or "")
                        ready = bool(item.get("ready", False))
                    else:
                        continue
                    if ready and full_pattern in configured:
                        advertised.add(full_pattern)
                next_tables = frozenset(advertised)
                if next_tables != worker.tables:
                    removed_tables = worker.tables - next_tables
                    worker.tables = next_tables
                    self._mark_availability_changed()
                    for full_pattern in removed_tables:
                        self._fail_table(worker.worker_id, full_pattern)
            return
        if message_type not in {
            "LOOKUP_RESULT",
            "LOOKUP_BATCH_RESULT",
            "RANDOM_STATE_RESULT",
            "ERROR",
        }:
            raise RemoteTablebaseProtocolError("Unsupported worker response.")
        request_id = str(message.get("request_id") or "")
        pending = self._pending.pop(request_id, None)
        if pending is None or pending.worker_id != worker.worker_id:
            return
        if message_type == "ERROR":
            logger.warning(
                "Remote tablebase worker request failed: %s (%s).",
                worker.worker_id,
                str(message.get("code") or "WORKER_ERROR")[:64],
            )
            error = RemoteTablebaseOffline()
            if not pending.future.done():
                pending.future.set_exception(error)
            return
        if not pending.future.done():
            pending.future.set_result(message)

    async def _send(self, worker: WorkerConnection, payload: dict[str, Any]) -> None:
        async with worker.send_lock:
            await worker.websocket.send_json(payload)

    def _worker_for_table(self, full_pattern: str) -> WorkerConnection:
        configured = configured_workers()
        target_worker_id = ""
        for worker_id, worker in configured.items():
            if full_pattern in worker["tables"]:
                target_worker_id = worker_id
                break
        worker = self._workers.get(target_worker_id)
        if (
            worker is None
            or full_pattern not in worker.tables
            or time.monotonic() - worker.last_seen > HEARTBEAT_TIMEOUT_SECONDS
        ):
            raise RemoteTablebaseOffline()
        return worker

    async def request(
        self,
        request_type: str,
        full_pattern: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        worker = self._worker_for_table(full_pattern)
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = PendingRequest(
            worker.worker_id, full_pattern, future
        )
        message = {
            "type": request_type,
            "request_id": request_id,
            "full_pattern": full_pattern,
            **payload,
        }
        try:
            await self._send(worker, message)
            return await asyncio.wait_for(
                future,
                timeout=REQUEST_TIMEOUT_SECONDS if timeout is None else timeout,
            )
        except asyncio.TimeoutError as exc:
            try:
                await self._send(
                    worker,
                    {"type": "CANCEL", "request_id": request_id},
                )
            except Exception:
                pass
            raise RemoteTablebaseTimeout() from exc
        except RemoteTablebaseProtocolError:
            raise
        except Exception as exc:
            raise RemoteTablebaseOffline() from exc
        finally:
            self._pending.pop(request_id, None)

    async def lookup(
        self,
        *,
        full_pattern: str,
        pattern: str,
        target: str,
        board: str,
        use_variant: bool,
        board_is_lookup: bool = False,
    ) -> dict[str, Any]:
        response = await self.request(
            "LOOKUP",
            full_pattern,
            {
                "pattern": pattern,
                "target": str(target),
                "board": _valid_board_hex(board),
                "use_variant": bool(use_variant),
                "board_is_lookup": bool(board_is_lookup),
            },
        )
        return response

    async def lookup_batch(
        self,
        *,
        full_pattern: str,
        pattern: str,
        target: str,
        boards: list[str],
        use_variant: bool,
        board_is_lookup: bool = False,
    ) -> dict[str, Any]:
        if not boards or len(boards) > 1024:
            raise RemoteTablebaseProtocolError("Invalid remote lookup batch size.")
        response = await self.request(
            "LOOKUP_BATCH",
            full_pattern,
            {
                "pattern": pattern,
                "target": str(target),
                "boards": [_valid_board_hex(item) for item in boards],
                "use_variant": bool(use_variant),
                "board_is_lookup": bool(board_is_lookup),
            },
            timeout=max(REQUEST_TIMEOUT_SECONDS, min(600.0, len(boards) * 2.0)),
        )
        return response

    async def random_state(
        self,
        *,
        full_pattern: str,
        pattern: str,
        target: str,
    ) -> dict[str, Any]:
        return await self.request(
            "RANDOM_STATE",
            full_pattern,
            {"pattern": pattern, "target": str(target)},
        )

    async def _remove_worker(
        self, worker: WorkerConnection, error: Exception
    ) -> None:
        lock = self._ensure_lock()
        removed = False
        async with lock:
            if self._workers.get(worker.worker_id) is worker:
                self._workers.pop(worker.worker_id, None)
                removed = True
        if removed:
            self._mark_availability_changed()
            self._fail_worker(worker.worker_id, error)
            logger.info("Remote tablebase worker disconnected: %s.", worker.worker_id)

    def _fail_worker(self, worker_id: str, error: Exception) -> None:
        for request_id, pending in list(self._pending.items()):
            if pending.worker_id != worker_id:
                continue
            self._pending.pop(request_id, None)
            if not pending.future.done():
                pending.future.set_exception(error)

    def _fail_table(self, worker_id: str, full_pattern: str) -> None:
        for request_id, pending in list(self._pending.items()):
            if (
                pending.worker_id != worker_id
                or pending.full_pattern != full_pattern
            ):
                continue
            self._pending.pop(request_id, None)
            if not pending.future.done():
                pending.future.set_exception(RemoteTablebaseOffline())

    def _fail_all(self, error: Exception) -> None:
        for request_id, pending in list(self._pending.items()):
            self._pending.pop(request_id, None)
            if not pending.future.done():
                pending.future.set_exception(error)

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            expired = [
                worker
                for worker in list(self._workers.values())
                if now - worker.last_seen > HEARTBEAT_TIMEOUT_SECONDS
            ]
            for worker in expired:
                await self._remove_worker(worker, RemoteTablebaseOffline())
                try:
                    await worker.websocket.close(code=1013)
                except Exception:
                    pass


remote_worker_registry = RemoteWorkerRegistry()
