import webview
import threading
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import json
import os
import sys
import socket
from backend.actions import Action, Message
from backend.handlers.analysis import handle_analysis_action
from backend.handlers.game import handle_game_action
from backend.handlers.minigames import handle_minigame_action
from backend.handlers.notebook import handle_notebook_action
from backend.handlers.replay import handle_replay_action
from backend.handlers.settings import handle_settings_action
from backend.handlers.tester import handle_tester_action
from backend.handlers.trainer import handle_trainer_action
from backend.state import ConnectionManager, save_game_state
from backend.webview_api import Api
from error_bridge import register_frontend_error_dispatcher, publish_frontend_exception

from fastapi.staticfiles import StaticFiles

app = FastAPI()


def get_resource_path(relative_path):
    """获取资源的绝对路径，兼容开发环境和 PyInstaller 打包环境"""
    if getattr(sys, "frozen", False):
        # PyInstaller 打包环境
        base_path = sys._MEIPASS  # type: ignore
    else:
        # 开发环境
        base_path = os.path.dirname(os.path.abspath(__file__))

    # 优先尝试在基准目录下寻找
    path = os.path.join(base_path, relative_path)
    if os.path.exists(path):
        return os.path.abspath(path)

    # 备选：尝试直接匹配相对路径（兼容性处理）
    if os.path.exists(relative_path):
        return os.path.abspath(relative_path)

    return os.path.abspath(path)


def find_available_port(start_port=8000):
    """检查指定端口是否可用，不可用则寻找随机空闲端口"""
    try:
        # 尝试绑定到 start_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", start_port))
            return start_port
    except socket.error:
        # 端口已被占用，寻找动态可用端口
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


# 在模块加载时确定端口
SERVER_PORT = find_available_port(8000)


# 静态资源路径准备（挂载动作移至底部）
mathjax_path = get_resource_path("mathjax")
minigame_pic_path = get_resource_path(os.path.join("minigames", "pic"))
pic_path = get_resource_path("pic")
frontend_dist_path = get_resource_path(os.path.join("frontend", "dist"))


# --- 路由定义开始 ---


manager = ConnectionManager()
window: webview.Window | None = None


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
            except Exception as e:
                print(f"WebSocket action error: {e}")
                publish_frontend_exception("WebSocket Action Error", e)
                await _send_ws_error(websocket, str(e))
    except Exception as e:
        print(f"Connection error: {e}")
        publish_frontend_exception("WebSocket Connection Error", e)
    finally:
        save_game_state(session)
        manager.disconnect(websocket)


# --- 路由结束，开始挂载静态资源 ---


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_index():
    """显式处理首页，包含详细的路径诊断信息"""
    path = os.path.join(frontend_dist_path, "index.html")
    if os.path.exists(path):
        return FileResponse(path)

    # 诊断：如果找不到，告诉用户程序尝试寻找的绝对路径
    return {
        "ERROR": "FRONTEND_NOT_FOUND",
        "ATTEMPTED_ABS_PATH": os.path.abspath(path),
        "CWD": os.getcwd(),
        "SYS_MEIPASS": getattr(sys, "_MEIPASS", "NOT_BUNDLE"),
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """为浏览器/Webview 提供图标，避免 404"""
    icon_path = get_resource_path("favicon.ico")
    if os.path.exists(icon_path):
        return FileResponse(icon_path)
    # 如果根目录没有，尝试去 pic 目录找备选
    alt_icon_path = get_resource_path(os.path.join("pic", "2048.ico"))
    if os.path.exists(alt_icon_path):
        return FileResponse(alt_icon_path)
    return None


# 挂载特殊资源
if os.path.exists(mathjax_path):
    app.mount("/mathjax", StaticFiles(directory=mathjax_path), name="mathjax")

if os.path.exists(minigame_pic_path):
    app.mount(
        "/minigames-assets",
        StaticFiles(directory=minigame_pic_path),
        name="minigames-assets",
    )

if os.path.exists(pic_path):
    app.mount(
        "/shared-assets",
        StaticFiles(directory=pic_path),
        name="shared-assets",
    )

# 最后挂载主前端根目录
if os.path.exists(frontend_dist_path):
    app.mount(
        "/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend"
    )


def start_server():
    # 使用动态确定的端口
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT, log_level="error")


def _initialize_frontend_error_bridge():
    if window is None:
        return

    def _dispatch(payload: dict[str, str]):
        if window is None:
            return
        payload_json = json.dumps(payload, ensure_ascii=False)
        window.evaluate_js(
            "window.__appGlobalErrors = window.__appGlobalErrors || [];"
            f"window.__appGlobalErrors.push({payload_json});"
            "window.dispatchEvent(new CustomEvent('app-global-error', { detail: "
            f"{payload_json} }}));"
        )

    register_frontend_error_dispatcher(_dispatch)


if __name__ == "__main__":
    is_frozen = getattr(sys, "frozen", False)

    # Step 1: Start FastAPI server in a daemon thread
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    # Step 2: Start the Pywebview App
    url = "http://localhost:5173"
    if os.path.exists(frontend_dist_path):
        url = f"http://localhost:{SERVER_PORT}/"  # 使用动态端口

    window = webview.create_window(
        "2048 Endgame TableBase",
        url,
        js_api=Api(),
        width=1200,
        height=940,
        min_size=(800, 600),
    )

    # 只有在开发环境下开启调试控制台
    webview.start(_initialize_frontend_error_bridge, debug=not is_frozen)

    os._exit(0)
