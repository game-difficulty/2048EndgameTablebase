from __future__ import annotations

import asyncio
import html
import json
import math
import os
import re
import threading
import time

import markdown
from Config import SingletonConfig, category_info, theme_map
from egtb_core import BookBuilder
from fastapi import WebSocket
from SignalHub import progress_signal
from typing import Any

from ..actions import Action, EventType
from ..serialization import sanitize_config
from ..session import GameSession
from ..state import ConnectionManager


MAX_DELETION_THRESHOLD = 0.999999
MIN_BUILD_PROGRESS_INTERVAL = 1.2
MAX_BUILD_PROGRESS_UPDATES = 120
NATIVE_BUILD_PROGRESS_POLL_INTERVAL = 0.25
BUILD_STATE_LOCK = threading.Lock()
BUILD_STATE = {
    "is_building": False,
    "current": 0,
    "total": 0,
    "pattern": "",
    "target_tile": "",
    "folder_path": "",
    "error": "",
}


def normalize_deletion_threshold(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return min(MAX_DELETION_THRESHOLD, max(0.0, parsed))


def get_build_state_snapshot() -> dict[str, Any]:
    with BUILD_STATE_LOCK:
        return dict(BUILD_STATE)


def update_build_state(**kwargs: Any) -> None:
    with BUILD_STATE_LOCK:
        BUILD_STATE.update(kwargs)


async def handle_settings_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
    manager: ConnectionManager,
) -> bool:
    if action == Action.GET_SETTINGS:
        config = SingletonConfig().config.copy()
        config["ui_scale"] = config.get("ui_scale", 100)
        config["deletion_threshold"] = normalize_deletion_threshold(
            config.get("deletion_threshold", 0.0)
        )
        await websocket.send_json(
            {
                "type": EventType.SETTINGS_DATA,
                "payload": {
                    "config": sanitize_config(config),
                    "categories": sanitize_config(category_info),
                    "theme_map": sanitize_config(theme_map),
                    "target_tiles": [2**i for i in range(6, 15)],
                    "build_state": sanitize_config(get_build_state_snapshot()),
                },
            }
        )
        return True

    if action == Action.UPDATE_SETTING:
        key = payload.get("key")
        value = payload.get("value")
        config = SingletonConfig().config
        config["ui_scale"] = config.get("ui_scale", 100)
        config["deletion_threshold"] = normalize_deletion_threshold(
            config.get("deletion_threshold", 0.0)
        )

        if key == "theme":
            config["theme"] = value
            config["use_custom_theme"] = False
            if value in theme_map:
                config["colors"] = list(theme_map[value]) + ["#000000"] * 20
                SingletonConfig.tile_font_colors()
        elif key == "colors":
            config["colors"] = value
            config["custom_colors"] = value
            config["use_custom_theme"] = True
            SingletonConfig.tile_font_colors()
        elif key == "use_custom_theme":
            config["use_custom_theme"] = value
            if value:
                config["colors"] = config.get("custom_colors", config["colors"])
            else:
                theme = config.get("theme", "Default")
                if theme in theme_map:
                    config["colors"] = list(theme_map[theme]) + ["#000000"] * 20
            SingletonConfig.tile_font_colors()
        elif key == "deletion_threshold":
            value = normalize_deletion_threshold(value)
            config[key] = value
        else:
            config[key] = value

        SingletonConfig().save_config(config)

        if key in ["theme", "colors", "use_custom_theme"]:
            await manager.broadcast(
                json.dumps(
                    {
                        "type": EventType.SETTINGS_DATA,
                        "payload": {
                            "config": sanitize_config(config),
                            "categories": sanitize_config(category_info),
                            "theme_map": sanitize_config(theme_map),
                            "target_tiles": [2**i for i in range(6, 15)],
                        },
                    }
                )
            )
        else:
            await manager.broadcast(
                json.dumps(
                    {
                        "type": EventType.SETTING_UPDATED,
                        "payload": {
                            "key": key,
                            "value": value,
                            "status": "success",
                        },
                    }
                )
            )
        return True

    if action == Action.START_BUILD:
        pattern = payload.get("pattern")
        target = payload.get("target")
        target_tile = payload.get("target_tile")
        folder_path = payload.get("folder_path")
        pathname = payload.get("pathname")
        try:
            target = int(target)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid build target: {target}")

        if target > 63 and target > 0 and (target & (target - 1)) == 0:
            target = int(math.log2(target))
        if target_tile is None and target > 0:
            target_tile = str(2**target)
        if not folder_path:
            folder_path = os.path.dirname(pathname.rstrip("\\/"))
        if not folder_path or not os.path.isdir(folder_path):
            raise ValueError(f"Invalid build folder: {folder_path}")

        config = SingletonConfig().config
        spawn_rate4 = config["4_spawn_rate"]
        success_rate_dtype = config.get("success_rate_dtype", "uint32")
        pattern_key = f"{pattern}_{target_tile}"
        config["filepath_map"][(pattern_key, spawn_rate4)] = [
            (folder_path, success_rate_dtype)
        ]
        SingletonConfig().save_config(config)
        loop = asyncio.get_running_loop()
        progress_state = {"current": 0, "total": 0}
        progress_lock = threading.Lock()
        progress_meta = {
            "last_sent_at": 0.0,
            "last_sent_current": -1,
            "last_sent_total": -1,
        }
        native_progress_module = getattr(BookBuilder, "formation_core", None)
        supports_native_progress = bool(
            native_progress_module is not None
            and hasattr(native_progress_module, "get_build_progress")
            and hasattr(native_progress_module, "reset_build_progress")
        )

        async def send_build_message(
            message_type: str, message_payload: dict[str, Any]
        ) -> None:
            try:
                await websocket.send_json(
                    {"type": message_type, "payload": message_payload}
                )
            except Exception:
                pass

        def publish_build_progress(current: Any, total: Any) -> None:
            try:
                current = int(current)
                total = int(total)
            except (TypeError, ValueError):
                return

            current = max(0, current)
            total = max(current, total)
            update_build_state(
                current=current,
                total=total,
                is_building=not (total > 0 and current >= total),
            )
            with progress_lock:
                progress_state["current"] = current
                progress_state["total"] = total

            now = time.monotonic()
            with progress_lock:
                last_sent_at = progress_meta["last_sent_at"]
                last_sent_current = progress_meta["last_sent_current"]
                last_sent_total = progress_meta["last_sent_total"]

            step_delta = max(2, total // MAX_BUILD_PROGRESS_UPDATES)
            should_emit = (
                current >= total
                or last_sent_current < 0
                or total != last_sent_total
                or (current - last_sent_current) >= step_delta
                or (now - last_sent_at) >= MIN_BUILD_PROGRESS_INTERVAL
            )
            if not should_emit:
                return

            with progress_lock:
                progress_meta["last_sent_at"] = now
                progress_meta["last_sent_current"] = current
                progress_meta["last_sent_total"] = total

            asyncio.run_coroutine_threadsafe(
                send_build_message(
                    EventType.BUILD_PROGRESS,
                    {"current": current, "total": total},
                ),
                loop,
            )

        def notify_build_failed(message: Any) -> None:
            asyncio.run_coroutine_threadsafe(
                send_build_message(EventType.BUILD_FAILED, {"message": str(message)}),
                loop,
            )

        def read_native_progress_snapshot() -> tuple[int, int] | None:
            if not supports_native_progress:
                return None

            try:
                current, total = native_progress_module.get_build_progress()
            except Exception:
                return None

            try:
                current = int(current)
                total = int(total)
            except (TypeError, ValueError):
                return None

            current = max(0, current)
            total = max(current, total)
            return current, total

        def start_native_progress_polling() -> tuple[threading.Event | None, threading.Thread | None]:
            if not supports_native_progress:
                return None, None

            try:
                native_progress_module.reset_build_progress()
            except Exception:
                return None, None

            stop_event = threading.Event()

            def poll() -> None:
                last_snapshot: tuple[int, int] | None = None
                while not stop_event.is_set():
                    snapshot = read_native_progress_snapshot()
                    if (
                        snapshot is not None
                        and snapshot[1] > 0
                        and snapshot != last_snapshot
                    ):
                        publish_build_progress(*snapshot)
                        last_snapshot = snapshot

                    if stop_event.wait(NATIVE_BUILD_PROGRESS_POLL_INTERVAL):
                        break

            thread = threading.Thread(target=poll, daemon=True)
            thread.start()
            return stop_event, thread

        def stop_native_progress_polling(
            stop_event: threading.Event | None,
            thread: threading.Thread | None,
        ) -> None:
            if stop_event is None:
                return
            stop_event.set()
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.5)

        def run_build() -> None:
            progress_signal.progress_updated.connect(publish_build_progress)
            stop_event, poll_thread = start_native_progress_polling()
            try:
                if pattern in category_info.get("variant", []):
                    BookBuilder.v_start_build(pattern, target, pathname)
                else:
                    BookBuilder.start_build(pattern, target, pathname)

                native_snapshot = read_native_progress_snapshot()
                if native_snapshot is not None and native_snapshot[1] > 0:
                    publish_build_progress(*native_snapshot)

                final_total = (
                    progress_state["total"]
                    if progress_state["total"] > 0
                    else max(1, progress_state["current"])
                )
                publish_build_progress(final_total, final_total)
                update_build_state(
                    is_building=False,
                    current=final_total,
                    total=final_total,
                    error="",
                )
            except Exception as e:
                print(f"Build error: {e}")
                update_build_state(is_building=False, error=str(e))
                notify_build_failed(e)
            finally:
                stop_native_progress_polling(stop_event, poll_thread)
                progress_signal.progress_updated.disconnect(publish_build_progress)

        update_build_state(
            is_building=True,
            current=0,
            total=0,
            pattern=str(pattern or ""),
            target_tile=str(target_tile or ""),
            folder_path=str(folder_path or ""),
            error="",
        )
        threading.Thread(target=run_build, daemon=True).start()
        await websocket.send_json(
            {"type": EventType.BUILD_STARTED, "payload": {"status": "running"}}
        )
        return True

    if action == Action.SELECT_FOLDER_CMD:
        await websocket.send_json({"type": EventType.TRIGGER_FOLDER_DIALOG})
        return True

    if action == Action.GET_HELP:
        lang = SingletonConfig().config.get("language", "en")
        md_file = "helpZH.md" if lang == "zh" else "help.md"
        docs_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "docs_and_configs",
            "help",
        )
        md_path = os.path.join(docs_root, md_file)

        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as file:
                md_content = file.read()

            extensions = ["extra", "sane_lists", "nl2br", "toc"]
            html_body = markdown.markdown(md_content, extensions=extensions)

            header_pattern = re.compile(r'<h([1-6]) id="(.*?)">(.*?)</h\1>')
            headers = header_pattern.findall(html_body)
            toc = []
            for level, anchor_id, text in headers:
                clean_text = re.sub(r"<.*?>", "", text).strip()
                toc.append(
                    {
                        "level": int(level),
                        "id": anchor_id,
                        "text": html.unescape(clean_text),
                    }
                )

            await websocket.send_json(
                {"type": EventType.HELP_DATA, "payload": {"html": html_body, "toc": toc}}
            )
        else:
            await websocket.send_json(
                {
                    "type": EventType.HELP_DATA,
                    "payload": {
                        "html": f"<h1>Help file not found: {md_file}</h1>",
                        "toc": [],
                    },
                }
            )
        return True

    if action == Action.UPDATE_SETTINGS:
        difficulty = payload.get("difficulty")
        speed = payload.get("speed")
        if difficulty is not None:
            session.difficulty = float(difficulty) / 100.0
        if speed is not None:
            exponent = (100.0 - float(speed)) / 100.0
            ratio = 10.0**exponent
            session.speed = float(speed)
            if session.ai_dispatcher is not None:
                session.ai_dispatcher.time_limit_ratio = ratio
        return True

    return False
