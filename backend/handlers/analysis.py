from __future__ import annotations

import asyncio
import threading
from typing import Any

from Config import category_info
from fastapi import WebSocket

from ..actions import Action, EventType
from ..analysis import normalize_target_value, resolve_analysis_inputs, run_batch_analysis
from ..session import GameSession


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
                    "target_tiles": [str(2**i) for i in range(6, 14)],
                },
            }
        )
        return True

    if action == Action.ANALYSIS_START:
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

        async def send_analysis_message(
            message_type: str, message_payload: dict[str, Any]
        ) -> None:
            try:
                await websocket.send_json(
                    {"type": message_type, "payload": message_payload}
                )
            except Exception:
                pass

        def publish_progress(progress_payload: dict[str, Any]) -> None:
            asyncio.run_coroutine_threadsafe(
                send_analysis_message(
                    EventType.ANALYSIS_PROGRESS,
                    progress_payload,
                ),
                loop,
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
                    send_analysis_message(
                        EventType.ANALYSIS_FINISHED,
                        {
                            "pattern": pattern,
                            "target": target_tile,
                            "total": len(file_list),
                            "entries": entries[-8:],
                            "done": sum(
                                1 for item in entries if item["status"] == "done"
                            ),
                            "failed": sum(
                                1 for item in entries if item["status"] == "failed"
                            ),
                        },
                    ),
                    loop,
                )
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    send_analysis_message(
                        EventType.ANALYSIS_FAILED,
                        {"message": str(exc)},
                    ),
                    loop,
                )

        threading.Thread(target=run_batch, daemon=True).start()
        await websocket.send_json(
            {
                "type": EventType.ANALYSIS_STARTED,
                "payload": {
                    "pattern": pattern,
                    "target": target_tile,
                    "total": len(file_list),
                    "done": 0,
                    "failed": 0,
                    "entries": [],
                },
            }
        )
        return True

    return False
