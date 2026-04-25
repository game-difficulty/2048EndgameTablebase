from __future__ import annotations

import atexit
from contextlib import asynccontextmanager
import json
import os

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.actions import Action, Message
from backend.handlers.analysis import handle_analysis_action
from backend.handlers.game import handle_game_action
from backend.handlers.minigames import handle_minigame_action
from backend.handlers.notebook import handle_notebook_action
from backend.handlers.replay import handle_replay_action
from backend.handlers.settings import handle_settings_action
from backend.handlers.tester import handle_tester_action
from backend.handlers.trainer import handle_trainer_action
from backend.preload import start_preload_thread
from backend.resource_paths import get_resource_path
from backend.state import ConnectionManager, save_game_state
from Config import SingletonConfig
from error_bridge import publish_frontend_exception


manager = ConnectionManager()
mathjax_path = get_resource_path("mathjax")
minigame_assets_path = get_resource_path(os.path.join("assets", "minigames"))
pic_path = get_resource_path("pic")
frontend_dist_path = get_resource_path(os.path.join("frontend", "dist"))


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    SingletonConfig()
    start_preload_thread()
    yield


app = FastAPI(lifespan=app_lifespan)


async def _send_ws_error(websocket: WebSocket, message: str):
    try:
        await websocket.send_json(
            {"action": Message.ERROR, "data": {"message": message}}
        )
    except Exception as send_error:
        print(f"Send error message failed: {send_error}")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):  # type: ignore
    await manager.connect(websocket, client_id)
    session = manager.active_connections[websocket]

    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                action = message.get("action") or message.get("type")
                payload = (
                    message.get("data")
                    if isinstance(message.get("data"), dict)
                    else message
                )

                if action == Action.GET_STATE:
                    await manager.send_state(websocket)

                elif await handle_tester_action(action, payload, session, websocket):
                    continue

                elif await handle_replay_action(action, payload, session, websocket):
                    continue

                elif await handle_notebook_action(action, payload, session, websocket):
                    continue

                elif await handle_analysis_action(action, payload, session, websocket):
                    continue

                elif await handle_minigame_action(action, payload, session, websocket):
                    continue

                elif await handle_settings_action(
                    action, payload, session, websocket, manager
                ):
                    continue

                elif await handle_game_action(
                    action, payload, session, websocket, manager
                ):
                    continue

                elif await handle_trainer_action(
                    action, payload, session, websocket, manager
                ):
                    continue

            except WebSocketDisconnect:
                break
            except Exception as exc:
                print(f"WebSocket action error: {exc}")
                publish_frontend_exception("WebSocket Action Error", exc)
                await _send_ws_error(websocket, str(exc))
    except Exception as exc:
        print(f"Connection error: {exc}")
        publish_frontend_exception("WebSocket Connection Error", exc)
    finally:
        save_game_state(session)
        manager.disconnect(websocket)


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_index():
    path = os.path.join(frontend_dist_path, "index.html")
    if os.path.exists(path):
        return FileResponse(path)

    return {
        "ERROR": "FRONTEND_NOT_FOUND",
        "ATTEMPTED_ABS_PATH": os.path.abspath(path),
        "CWD": os.getcwd(),
        "SYS_MEIPASS": getattr(os.sys, "_MEIPASS", "NOT_BUNDLE"),
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon_path = get_resource_path("favicon.ico")
    if os.path.exists(icon_path):
        return FileResponse(icon_path)

    alt_icon_path = get_resource_path(os.path.join("pic", "2048_2.ico"))
    if os.path.exists(alt_icon_path):
        return FileResponse(alt_icon_path)

    return None


if os.path.exists(mathjax_path):
    app.mount("/mathjax", StaticFiles(directory=mathjax_path), name="mathjax")

if os.path.exists(minigame_assets_path):
    app.mount(
        "/minigames-assets",
        StaticFiles(directory=minigame_assets_path),
        name="minigames-assets",
    )

if os.path.exists(pic_path):
    app.mount(
        "/shared-assets",
        StaticFiles(directory=pic_path),
        name="shared-assets",
    )

if os.path.exists(frontend_dist_path):
    app.mount(
        "/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend"
    )


def persist_runtime_config() -> None:
    try:
        SingletonConfig().save_config(SingletonConfig().config)
    except Exception as exc:
        print(f"Failed to persist config on exit: {exc}")


def run_backend_server(port: int) -> None:
    atexit.register(persist_runtime_config)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
