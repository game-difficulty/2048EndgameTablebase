from __future__ import annotations

import asyncio
import html
import json
import math
import os
import re
import threading

import markdown
from Config import SingletonConfig, category_info, theme_map
from egtb_core.BookBuilder import start_build, v_start_build
from fastapi import WebSocket
from SignalHub import progress_signal
from typing import Any

from ..actions import Action, EventType
from ..serialization import sanitize_config
from ..session import GameSession
from ..state import ConnectionManager


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
        await websocket.send_json(
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
        return True

    if action == Action.UPDATE_SETTING:
        key = payload.get("key")
        value = payload.get("value")
        config = SingletonConfig().config
        config["ui_scale"] = config.get("ui_scale", 100)

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
            progress_state["current"] = current
            progress_state["total"] = total
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

        def run_build() -> None:
            progress_signal.progress_updated.connect(publish_build_progress)
            try:
                if pattern in category_info.get("variant", []):
                    v_start_build(pattern, target, pathname)
                else:
                    start_build(pattern, target, pathname)

                final_total = (
                    progress_state["total"]
                    if progress_state["total"] > 0
                    else max(1, progress_state["current"])
                )
                publish_build_progress(final_total, final_total)
            except Exception as e:
                print(f"Build error: {e}")
                notify_build_failed(e)
            finally:
                progress_signal.progress_updated.disconnect(publish_build_progress)

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
            session.ai_dispatcher.time_limit_ratio = ratio
            session.speed = float(speed)
        return True

    return False
