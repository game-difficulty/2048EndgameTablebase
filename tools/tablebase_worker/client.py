from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import random
import time
from typing import Any

from .config import WorkerConfig
from .protocol import (
    PROTOCOL_VERSION,
    WORKER_CAPABILITIES,
    ProtocolError,
    Request,
    decode_message,
    encode_message,
    error_message,
    sanitize_results,
    validate_hello_ack,
    validate_request,
)
from .reader_pool import BattleRouteGenerationError, ReaderPool, TableUnavailable


logger = logging.getLogger("tablebase_worker.client")


def connection_error_summary(exc: Exception) -> str:
    """Return a compact handshake error without logging response headers."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is None:
        return type(exc).__name__
    return f"{type(exc).__name__} (HTTP {status_code})"


class WorkerClient:
    def __init__(self, config: WorkerConfig, reader_pool: ReaderPool):
        self.config = config
        self.reader_pool = reader_pool
        self._stop = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._request_tasks: dict[str, asyncio.Task] = {}

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        delay = self.config.reconnect_initial_seconds
        while not self._stop.is_set():
            connected_seconds = 0.0
            try:
                connected_seconds = await self._run_connection()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Worker connection failed: %s", connection_error_summary(exc)
                )
            if self._stop.is_set():
                break
            if connected_seconds >= self.config.heartbeat_seconds * 3:
                delay = self.config.reconnect_initial_seconds
            sleep_for = min(
                self.config.reconnect_max_seconds,
                delay * random.uniform(0.85, 1.15),
            )
            logger.info("Reconnecting in %.1f seconds", sleep_for)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2, self.config.reconnect_max_seconds)
        await self._cancel_requests(wait=True)

    async def _connect(self):
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "The 'websockets' package is required; install project requirements."
            ) from exc
        connection = websockets.connect(
            self.config.server_url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=2 * 1024 * 1024,
        )
        return await asyncio.wait_for(
            connection, timeout=self.config.connect_timeout_seconds
        )

    async def _run_connection(self) -> float:
        websocket = await self._connect()
        connected_at: float | None = None
        try:
            await self._send(
                websocket,
                encode_message(
                    "HELLO",
                    protocol_version=PROTOCOL_VERSION,
                    worker_id=self.config.worker_id,
                    auth_token=self.config.auth_token,
                    tables=self.reader_pool.hello_tables(),
                    capabilities=list(WORKER_CAPABILITIES),
                ),
            )
            raw_ack = await asyncio.wait_for(
                websocket.recv(), timeout=self.config.hello_timeout_seconds
            )
            ack = validate_hello_ack(raw_ack, worker_id=self.config.worker_id)
            connected_at = time.monotonic()
            logger.info(
                "Worker accepted by server with %d table(s)", len(ack.get("tables", []))
            )
            receiver = asyncio.create_task(self._receive_loop(websocket))
            heartbeat = asyncio.create_task(self._heartbeat_loop(websocket))
            stopper = asyncio.create_task(self._stop.wait())
            done, pending = await asyncio.wait(
                {receiver, heartbeat, stopper}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task is not stopper:
                    task.result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Worker connection failed: %s", connection_error_summary(exc)
            )
        finally:
            await self._cancel_requests(wait=False)
            with contextlib.suppress(Exception):
                await websocket.close()
        return 0.0 if connected_at is None else time.monotonic() - connected_at

    async def _heartbeat_loop(self, websocket) -> None:
        while True:
            await asyncio.sleep(self.config.heartbeat_seconds)
            await self._send(
                websocket,
                encode_message(
                    "HEARTBEAT", tables=self.reader_pool.refresh_readiness()
                ),
            )

    async def _receive_loop(self, websocket) -> None:
        async for raw in websocket:
            request_type: str | None = None
            request_id: str | None = None
            try:
                message = decode_message(raw)
                request_type = str(message.get("type") or "").upper()
                raw_request_id = message.get("request_id")
                request_id = str(raw_request_id) if raw_request_id is not None else None
                request = validate_request(
                    message,
                    allowed_tables=self.reader_pool.allowed_tables,
                    table_metadata=self.reader_pool.table_metadata,
                    max_batch_size=self.config.max_batch_size,
                )
                if request.message_type == "CANCEL":
                    self._cancel_request(request.request_id)
                    continue
                if request.request_id in self._request_tasks:
                    raise ProtocolError(
                        "DUPLICATE_REQUEST", "request_id is already active", request.request_id
                    )
                task = asyncio.create_task(self._execute_request(websocket, request))
                self._request_tasks[request.request_id] = task
                task.add_done_callback(
                    lambda completed, rid=request.request_id: self._request_done(rid, completed)
                )
            except ProtocolError as exc:
                logger.warning("Rejected worker request: %s", exc.code)
                await self._send(websocket, error_message(exc, request_id))

    def _request_done(self, request_id: str, task: asyncio.Task) -> None:
        if self._request_tasks.get(request_id) is task:
            self._request_tasks.pop(request_id, None)
        if not task.cancelled():
            with contextlib.suppress(Exception):
                task.exception()

    def _cancel_request(self, request_id: str) -> None:
        task = self._request_tasks.get(request_id)
        if task is not None:
            task.cancel()

    async def _cancel_requests(self, *, wait: bool) -> None:
        tasks = list(self._request_tasks.values())
        for task in tasks:
            task.cancel()
        if wait and tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_request(self, websocket, request: Request) -> None:
        started = time.monotonic()
        try:
            if request.message_type == "LOOKUP":
                raw_results, dtype = await self.reader_pool.lookup(
                    request.full_pattern or "",
                    request.boards[0],
                    use_variant=request.use_variant,
                    board_is_lookup=request.board_is_lookup,
                )
                payload = encode_message(
                    "LOOKUP_RESULT",
                    request_id=request.request_id,
                    results=sanitize_results(raw_results),
                    dtype=str(dtype or "?"),
                    board=f"{request.boards[0]:016x}",
                )
            elif request.message_type == "LOOKUP_BATCH":
                raw_items = await self.reader_pool.lookup_batch(
                    request.full_pattern or "",
                    request.boards,
                    use_variant=request.use_variant,
                    board_is_lookup=request.board_is_lookup,
                )
                items = [
                    {
                        "board": f"{board:016x}",
                        "results": sanitize_results(results),
                        "dtype": str(dtype or "?"),
                    }
                    for board, (results, dtype) in zip(request.boards, raw_items)
                ]
                payload = encode_message(
                    "LOOKUP_BATCH_RESULT",
                    request_id=request.request_id,
                    items=items,
                )
            elif request.message_type == "GENERATE_GAMER_ROUTE":
                items = await self.reader_pool.generate_gamer_route(
                    request.full_pattern or '', request.gamer_options)
                payload = encode_message('GAMER_ROUTE_RESULT', request_id=request.request_id, items=items)
            elif request.message_type == "RANDOM_STATE":
                board = await self.reader_pool.random_state(request.full_pattern or "")
                payload = encode_message(
                    "RANDOM_STATE_RESULT",
                    request_id=request.request_id,
                    board=f"{board:016x}",
                )
            else:
                route = await self.reader_pool.generate_battle_route(
                    request.full_pattern or "",
                    initial_board=request.initial_board,
                    max_steps=request.max_steps,
                    min_steps=request.min_steps,
                    spawn_rate=request.spawn_rate,
                    seed_hex=request.seed_hex,
                )
                payload = encode_message(
                    "BATTLE_ROUTE_RESULT",
                    request_id=request.request_id,
                    route_blob_base64=base64.b64encode(route.route_blob).decode("ascii"),
                    step_count=route.step_count,
                    certainty_step=route.certainty_step,
                    termination_reason=route.termination_reason,
                    initial_board=f"{route.initial_board:016x}",
                    available_layers=route.available_layers,
                )
            await self._send(websocket, payload)
            logger.debug(
                "%s %s completed in %.1f ms",
                request.message_type,
                request.request_id,
                (time.monotonic() - started) * 1000,
            )
        except asyncio.CancelledError:
            logger.debug("Request cancelled: %s", request.request_id)
            raise
        except TableUnavailable as exc:
            error = ProtocolError("TABLE_UNAVAILABLE", str(exc), request.request_id)
            await self._send(websocket, error_message(error))
        except BattleRouteGenerationError as exc:
            error = ProtocolError(exc.code, str(exc), request.request_id)
            await self._send(websocket, error_message(error))
        except Exception as exc:
            logger.exception(
                "Tablebase request failed: %s (%s)",
                request.request_id,
                type(exc).__name__,
            )
            error = ProtocolError(
                "LOOKUP_FAILED", "Local tablebase query failed", request.request_id
            )
            await self._send(websocket, error_message(error))

    async def _send(self, websocket, payload: str) -> None:
        async with self._send_lock:
            await websocket.send(payload)
