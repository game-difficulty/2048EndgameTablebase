from __future__ import annotations
from engine_core.GoalSpec import available_target_tokens

import asyncio
import threading
from typing import Any

from Config import SingletonConfig, category_info, logger
from fastapi import WebSocket

from ..actions import Action, EventType
from ..analysis import normalize_target_value, resolve_analysis_inputs, run_batch_analysis
from ..session import GameSession
from ..webview_api import Api


class _AnalysisTask:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket: WebSocket | None = websocket
        self.message: dict[str, Any] | None = None
        self.running = True
        self.lock = asyncio.Lock()

    async def _send(self) -> None:
        if self.websocket is None or self.message is None:
            return
        try:
            await asyncio.wait_for(self.websocket.send_json(self.message), timeout=5.0)
        except Exception:
            logger.warning("Analysis connection lost; retaining task state for reconnect", exc_info=True)
            self.websocket = None

    async def publish(self, message_type: str, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.message = {"type": message_type, "payload": payload}
            self.running = message_type in (
                EventType.ANALYSIS_STARTED, EventType.ANALYSIS_PROGRESS
            )
            await self._send()

    async def attach(self, websocket: WebSocket) -> None:
        async with self.lock:
            self.websocket = websocket
            await self._send()


# The client ID survives WebSocket reconnects; GameSession and WebSocket do not.
_analysis_tasks: dict[str, _AnalysisTask] = {}


async def handle_analysis_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> bool:
    if action == Action.ANALYSIS_GET_INIT:
        await websocket.send_json(
            {
                "type": EventType.ANALYSIS_BOOTSTRAP,
                "payload": {
                    "categories": category_info,
                    "target_tiles": available_target_tokens(),
                    "available_tables": SingletonConfig.get_available_pattern_targets(),
                },
            }
        )
        task = _analysis_tasks.get(session.client_id)
        if task is not None:
            await task.attach(websocket)
        return True

    if action == Action.ANALYSIS_TRIGGER_SELECT_FILES:
        selected_paths = await asyncio.to_thread(Api().select_analysis_files)
        await websocket.send_json(
            {
                "type": EventType.ANALYSIS_FILES_SELECTED,
                "payload": {"paths": selected_paths or []},
            }
        )
        return True

    if action == Action.ANALYSIS_START:
        task = _analysis_tasks.get(session.client_id)
        if task is not None and task.running:
            await task.attach(websocket)
            return True

        task = _AnalysisTask(websocket)
        _analysis_tasks[session.client_id] = task
        try:
            await _start_analysis(payload, task)
        except Exception as exc:
            logger.exception("Failed to start replay analysis")
            await task.publish(EventType.ANALYSIS_FAILED, {"message": str(exc)})
        return True

    return False


async def _start_analysis(payload: dict[str, Any], task: _AnalysisTask) -> None:
    pattern = str(payload.get("pattern") or "").strip()
    target = payload.get("target")
    raw_paths = payload.get("paths") or []
    if not pattern:
        raise ValueError("Missing analysis pattern")
    if not isinstance(raw_paths, list):
        raise ValueError("Invalid analysis input paths")

    target_tile, target_value, numeric_target = normalize_target_value(target)
    full_pattern = f"{pattern}_{numeric_target}"
    file_list = resolve_analysis_inputs(raw_paths)
    if not file_list:
        raise ValueError("No valid .txt or .vrs file found")

    loop = asyncio.get_running_loop()

    def publish_progress(progress_payload: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(
            task.publish(EventType.ANALYSIS_PROGRESS, progress_payload), loop
        )

    def run_batch() -> None:
        try:
            entries = run_batch_analysis(
                file_list=file_list,
                pattern=pattern,
                target_value=target_value,
                full_pattern=full_pattern,
                on_progress=publish_progress,
            )
            asyncio.run_coroutine_threadsafe(
                task.publish(
                    EventType.ANALYSIS_FINISHED,
                    {
                        "pattern": pattern,
                        "target": target_tile,
                        "total": len(file_list),
                        "entries": entries[-8:],
                        "done": sum(item["status"] == "done" for item in entries),
                        "failed": sum(item["status"] == "failed" for item in entries),
                    },
                ),
                loop,
            )
        except Exception as exc:
            logger.exception("Replay analysis batch failed (%s)", full_pattern)
            asyncio.run_coroutine_threadsafe(
                task.publish(EventType.ANALYSIS_FAILED, {"message": str(exc)}), loop
            )

    await task.publish(
        EventType.ANALYSIS_STARTED,
        {
            "pattern": pattern,
            "target": target_tile,
            "total": len(file_list),
            "done": 0,
            "failed": 0,
            "entries": [],
        },
    )
    threading.Thread(target=run_batch, daemon=True).start()
