from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import json
import logging
import math
import os
import struct
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
CAPABILITY_BATTLE_ROUTE_V1 = "battle_route_v1"
CAPABILITY_GAMER_ROUTE_V1 = "gamer_route_v1"
CAPABILITY_GAMER_STREAM_V1 = "gamer_stream_v1"
HEARTBEAT_TIMEOUT_SECONDS = float(
    os.getenv("REMOTE_TABLEBASE_HEARTBEAT_TIMEOUT_SECONDS", "30")
)
REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("REMOTE_TABLEBASE_REQUEST_TIMEOUT_SECONDS", "30")
)
BATTLE_ROUTE_TIMEOUT_SECONDS = float(
    os.getenv("REMOTE_TABLEBASE_BATTLE_ROUTE_TIMEOUT_SECONDS", "600")
)
HELLO_TIMEOUT_SECONDS = float(os.getenv("REMOTE_TABLEBASE_HELLO_TIMEOUT_SECONDS", "10"))
MAX_WORKER_MESSAGE_BYTES = int(
    os.getenv("REMOTE_TABLEBASE_MAX_MESSAGE_BYTES", str(2 * 1024 * 1024))
)
logger = logging.getLogger("2048tables.remote_worker")
TRAINER_ROUTE_RECORD = struct.Struct("<B4I")
TRAINER_ROUTE_RATE_SCALE = 4_000_000_000


@dataclass
class WorkerConnection:
    worker_id: str
    websocket: WebSocket
    tables: frozenset[str]
    capabilities: frozenset[str] = field(default_factory=frozenset)
    connected_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class PendingRequest:
    worker_id: str
    full_pattern: str
    request_type: str
    future: asyncio.Future
    stream: Any = None


class RemoteGamerStream:
    def __init__(self, registry, worker, request_id, future, allowed):
        self.registry, self.worker, self.request_id = registry, worker, request_id
        self.future = future
        self.queue = asyncio.Queue(maxsize=32)
        self.produced = -1
        self.allowed = allowed

    async def credit(self, consumed, allowed):
        if self.future.done():
            return
        self.allowed = max(self.allowed, allowed)
        await self.registry._send(self.worker, {'type': 'GAMER_STREAM_CREDIT',
            'request_id': self.request_id, 'consumed': consumed, 'allow_through': self.allowed})

    async def receive(self):
        if not self.queue.empty():
            return self.queue.get_nowait()
        if self.future.done():
            self.future.result()
            return None
        task = asyncio.create_task(self.queue.get())
        try:
            done, _ = await asyncio.wait([task, self.future], timeout=REQUEST_TIMEOUT_SECONDS,
                                        return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                return task.result()
            if self.future in done:
                self.future.result()
                if not self.queue.empty():
                    return self.queue.get_nowait()
                return None
            raise RemoteTablebaseTimeout()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def close(self):
        self.registry._pending.pop(self.request_id, None)
        if self.future.done() and not self.future.cancelled():
            self.future.exception()
        self.future.cancel()
        try:
            await self.registry._send(self.worker, {'type': 'CANCEL', 'request_id': self.request_id})
        except Exception:
            pass


def _valid_board_hex(value: Any) -> str:
    board = str(value or "").strip().lower()
    if len(board) != 16 or any(char not in "0123456789abcdef" for char in board):
        raise RemoteTablebaseProtocolError("Invalid remote tablebase board.")
    return board


def _valid_route_int(
    value: Any,
    field_name: str,
    *,
    minimum: int,
    allow_none: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RemoteTablebaseProtocolError(f"Invalid {field_name}.")
    parsed = int(value)
    if parsed < minimum or parsed > 9_999:
        raise RemoteTablebaseProtocolError(f"Invalid {field_name}.")
    return parsed


def _validate_trainer_route_blob(route_blob: bytes, step_count: int) -> None:
    if len(route_blob) != (step_count + 1) * TRAINER_ROUTE_RECORD.size:
        raise RemoteTablebaseProtocolError("Battle route length does not match metadata.")
    header = TRAINER_ROUTE_RECORD.unpack_from(route_blob)
    if header[0] != 0 or any(part > 0xFFFF for part in header[1:]):
        raise RemoteTablebaseProtocolError("Invalid battle route header.")
    for offset in range(TRAINER_ROUTE_RECORD.size, len(route_blob), TRAINER_ROUTE_RECORD.size):
        change, *rates = TRAINER_ROUTE_RECORD.unpack_from(route_blob, offset)
        if change & 0x80 or any(rate > TRAINER_ROUTE_RATE_SCALE for rate in rates):
            raise RemoteTablebaseProtocolError("Invalid battle route record.")


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
                    "capabilities": sorted(
                        worker.capabilities if online and worker else ()
                    ),
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

        advertised_capabilities = frozenset(
            capability
            for capability in (
                str(item or "").strip().lower()
                for item in hello.get("capabilities", [])
                if isinstance(item, str)
            )
            if capability in (CAPABILITY_BATTLE_ROUTE_V1, CAPABILITY_GAMER_ROUTE_V1, CAPABILITY_GAMER_STREAM_V1)
        )

        worker = WorkerConnection(
            worker_id,
            websocket,
            frozenset(advertised),
            capabilities=advertised_capabilities,
        )
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
            "BATTLE_ROUTE_RESULT",
            "GAMER_ROUTE_RESULT",
            "GAMER_STREAM_NODE",
            "GAMER_STREAM_END",
            "ERROR",
        }:
            raise RemoteTablebaseProtocolError("Unsupported worker response.")
        request_id = str(message.get("request_id") or "")
        pending = self._pending.get(request_id)
        if pending is not None and pending.stream is not None:
            if pending.worker_id != worker.worker_id:
                return
            stream = pending.stream
            if message_type == 'GAMER_STREAM_NODE':
                seq = message.get('seq')
                if (type(seq) is not int or seq != stream.produced + 1 or seq > stream.allowed
                        or not isinstance(message.get('item'), dict) or stream.queue.full()):
                    if not pending.future.done():
                        pending.future.set_exception(RemoteTablebaseProtocolError('Invalid stream sequence'))
                elif not pending.future.done():
                    stream.produced = seq
                    stream.queue.put_nowait(message['item'])
                return
            self._pending.pop(request_id, None)
            if not pending.future.done():
                if message_type == 'GAMER_STREAM_END':
                    pending.future.set_result(None)
                else:
                    pending.future.set_exception(RemoteTablebaseOffline())
            return
        pending = self._pending.pop(request_id, None)
        if pending is None or pending.worker_id != worker.worker_id:
            return
        if message_type == "ERROR":
            logger.warning(
                "Remote tablebase worker request failed: %s (%s).",
                worker.worker_id,
                str(message.get("code") or "WORKER_ERROR")[:64],
            )
            error = (
                RemoteTablebaseProtocolError(
                    str(message.get("code") or "WORKER_ERROR")[:64]
                )
                if pending.request_type == "GENERATE_BATTLE_ROUTE"
                else RemoteTablebaseOffline()
            )
            if not pending.future.done():
                pending.future.set_exception(error)
            return
        if not pending.future.done():
            pending.future.set_result(message)

    async def _send(self, worker: WorkerConnection, payload: dict[str, Any]) -> None:
        async with worker.send_lock:
            await worker.websocket.send_json(payload)

    def _worker_for_table(
        self,
        full_pattern: str,
        *,
        required_capability: str | None = None,
    ) -> WorkerConnection:
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
        if required_capability and required_capability not in worker.capabilities:
            raise RemoteTablebaseProtocolError(
                "Remote tablebase worker does not support this operation."
            )
        return worker

    async def request(
        self,
        request_type: str,
        full_pattern: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
        required_capability: str | None = None,
    ) -> dict[str, Any]:
        worker = self._worker_for_table(
            full_pattern,
            required_capability=required_capability,
        )
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = PendingRequest(
            worker.worker_id, full_pattern, request_type, future
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

    def supports_gamer_route(self, full_pattern: str) -> bool:
        try:
            return CAPABILITY_GAMER_ROUTE_V1 in self._worker_for_table(full_pattern).capabilities
        except RemoteTablebaseOffline:
            return False

    def supports_gamer_stream(self, full_pattern: str) -> bool:
        try:
            return CAPABILITY_GAMER_STREAM_V1 in self._worker_for_table(full_pattern).capabilities
        except RemoteTablebaseOffline:
            return False

    async def open_gamer_stream(self, *, full_pattern, pattern, target, options, allow_through):
        from backend.gamer_tablebase_route import validate_options
        from backend.gamer_stream_window import StreamWindow
        validate_options(options)
        StreamWindow(allow_through)
        worker = self._worker_for_table(full_pattern, required_capability=CAPABILITY_GAMER_STREAM_V1)
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        stream = RemoteGamerStream(self, worker, request_id, future, allow_through)
        self._pending[request_id] = PendingRequest(worker.worker_id, full_pattern, 'GAMER_STREAM_OPEN', future, stream)
        try:
            await self._send(worker, {'type': 'GAMER_STREAM_OPEN', 'request_id': request_id,
                'full_pattern': full_pattern, 'pattern': pattern, 'target': str(target),
                'options': options, 'allow_through': allow_through})
        except BaseException:
            await stream.close()
            raise
        return stream

    async def generate_gamer_route(self, *, full_pattern, pattern, target, options):
        from backend.gamer_tablebase_route import generate_route, validate_options
        from Config import pattern_32k_tiles_map
        validate_options(options)
        response = await self.request('GENERATE_GAMER_ROUTE', full_pattern,
            {'pattern': pattern, 'target': str(target), 'options': options},
            required_capability=CAPABILITY_GAMER_ROUTE_V1)
        items = response.get('items')
        if not isinstance(items, list) or not 1 <= len(items) <= options['steps']:
            raise RemoteTablebaseProtocolError('Invalid Gamer route length')
        # Reconstruct transitions without additional reads before accepting/cacheing nodes.
        iterator = iter(items)
        def lookup(board):
            item = next(iterator)
            if item.get('lookup_board') != f'{board:016x}' or not isinstance(item.get('results'), dict):
                raise ValueError('Invalid route board')
            return item['results'], str(item['dtype'])
        try:
            expected = generate_route(options, pattern_32k_tiles_map[pattern][0], lookup)
            if expected != items:
                raise ValueError('Invalid route state')
        except (KeyError, ValueError, TypeError, StopIteration) as exc:
            raise RemoteTablebaseProtocolError('Invalid Gamer route') from exc
        return items

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

    async def generate_battle_route(
        self,
        *,
        full_pattern: str,
        pattern: str,
        target: str,
        initial_board: str | None,
        max_steps: int | None,
        min_steps: int,
        spawn_rate: float,
        seed_hex: str,
    ) -> dict[str, Any]:
        parsed_max_steps = _valid_route_int(
            max_steps, "max_steps", minimum=1, allow_none=True
        )
        parsed_min_steps = _valid_route_int(min_steps, "min_steps", minimum=0)
        if parsed_max_steps is not None and parsed_min_steps > parsed_max_steps:
            raise RemoteTablebaseProtocolError("min_steps exceeds max_steps.")
        if isinstance(spawn_rate, bool):
            raise RemoteTablebaseProtocolError("Invalid spawn_rate.")
        try:
            parsed_spawn_rate = float(spawn_rate)
        except (TypeError, ValueError) as exc:
            raise RemoteTablebaseProtocolError("Invalid spawn_rate.") from exc
        if not math.isfinite(parsed_spawn_rate) or not 0 <= parsed_spawn_rate <= 1:
            raise RemoteTablebaseProtocolError("Invalid spawn_rate.")
        normalized_seed = str(seed_hex or "").strip().lower()
        if len(normalized_seed) != 32 or any(
            char not in "0123456789abcdef" for char in normalized_seed
        ):
            raise RemoteTablebaseProtocolError("Invalid seed_hex.")

        response = await self.request(
            "GENERATE_BATTLE_ROUTE",
            full_pattern,
            {
                "pattern": str(pattern),
                "target": str(target),
                "initial_board": (
                    None if initial_board is None else _valid_board_hex(initial_board)
                ),
                "max_steps": parsed_max_steps,
                "min_steps": parsed_min_steps,
                "spawn_rate": parsed_spawn_rate,
                "seed_hex": normalized_seed,
            },
            timeout=BATTLE_ROUTE_TIMEOUT_SECONDS,
            required_capability=CAPABILITY_BATTLE_ROUTE_V1,
        )
        try:
            route_blob = base64.b64decode(
                str(response.get("route_blob_base64") or ""),
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise RemoteTablebaseProtocolError("Invalid battle route payload.") from exc
        step_count = _valid_route_int(
            response.get("step_count"), "step_count", minimum=0
        )
        _validate_trainer_route_blob(route_blob, step_count)
        certainty_step = _valid_route_int(
            response.get("certainty_step"),
            "certainty_step",
            minimum=0,
            allow_none=True,
        )
        if certainty_step is not None and certainty_step > step_count:
            raise RemoteTablebaseProtocolError("Invalid certainty_step.")
        available_layers = _valid_route_int(
            response.get("available_layers"), "available_layers", minimum=0
        )
        termination_reason = str(response.get("termination_reason") or "")
        if termination_reason not in {
            "target_reached",
            "no_legal_move",
            "zero_success",
            "max_steps",
        }:
            raise RemoteTablebaseProtocolError("Invalid battle route termination reason.")
        normalized = dict(response)
        normalized.update(
            {
                "step_count": step_count,
                "certainty_step": certainty_step,
                "termination_reason": termination_reason,
                "initial_board": _valid_board_hex(response.get("initial_board")),
                "available_layers": available_layers,
                "route_blob": route_blob,
            }
        )
        return normalized

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
