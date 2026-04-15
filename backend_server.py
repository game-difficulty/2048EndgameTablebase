import webview
import atexit
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


_MOTW_EXTENSIONS = {".dll", ".pyd", ".exe", ".ocx"}


def _runtime_search_roots() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass).resolve())
    else:
        roots.append(Path(__file__).resolve().parent)

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _clear_motw_stream(path: Path) -> None:
    try:
        os.remove(f"{path}:Zone.Identifier")
    except FileNotFoundError:
        return
    except OSError:
        return


def _unblock_distribution_files() -> None:
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return

    for root in _runtime_search_roots():
        sentinel = root / ".motw_unblocked"
        if sentinel.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if name.lower() not in {".git", "__pycache__", "node_modules"}
            ]
            for filename in filenames:
                path = Path(dirpath) / filename
                if path.suffix.lower() in _MOTW_EXTENSIONS:
                    _clear_motw_stream(path)

        try:
            sentinel.write_text("ok", encoding="utf-8")
        except OSError:
            pass


_unblock_distribution_files()

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
from backend.state import ConnectionManager, save_game_state
from backend.webview_api import Api
from Config import SingletonConfig
from error_bridge import register_frontend_error_dispatcher, publish_frontend_exception

app = FastAPI()
manager = ConnectionManager()
window: webview.Window | None = None
_server_process: subprocess.Popen | None = None
_server_job_handle = None
_cleanup_started = False


if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


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


def _ensure_server_job_object():
    global _server_job_handle
    if os.name != "nt":
        return None
    if _server_job_handle is not None:
        return _server_job_handle

    handle = _kernel32.CreateJobObjectW(None, None)
    if not handle:
        return None

    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    success = _kernel32.SetInformationJobObject(
        handle,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not success:
        _kernel32.CloseHandle(handle)
        return None

    _server_job_handle = handle
    return handle


def _attach_process_to_server_job(pid: int) -> None:
    if os.name != "nt":
        return

    job_handle = _ensure_server_job_object()
    if job_handle is None:
        return

    process_handle = _kernel32.OpenProcess(
        _PROCESS_SET_QUOTA
        | _PROCESS_TERMINATE
        | _PROCESS_QUERY_LIMITED_INFORMATION,
        False,
        pid,
    )
    if not process_handle:
        return

    try:
        _kernel32.AssignProcessToJobObject(job_handle, process_handle)
    finally:
        _kernel32.CloseHandle(process_handle)


def _close_server_job_object() -> None:
    global _server_job_handle
    if os.name != "nt" or _server_job_handle is None:
        return
    _kernel32.CloseHandle(_server_job_handle)
    _server_job_handle = None


# 在模块加载时确定端口
SERVER_PORT = find_available_port(8000)


# 静态资源路径准备（挂载动作移至底部）
mathjax_path = get_resource_path("mathjax")
minigame_pic_path = get_resource_path(os.path.join("minigames", "pic"))
pic_path = get_resource_path("pic")
frontend_dist_path = get_resource_path(os.path.join("frontend", "dist"))


# --- 路由定义开始 ---


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


def _run_server_process(port: int):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


def _server_subprocess_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--backend-server-child", str(SERVER_PORT)]
    return [
        sys.executable,
        os.path.abspath(__file__),
        "--backend-server-child",
        str(SERVER_PORT),
    ]


def _server_subprocess_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kwargs


def _wait_for_server_ready(timeout_seconds: float = 5.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _server_process is not None and _server_process.poll() is not None:
            raise RuntimeError("Backend server process exited during startup.")
        try:
            with socket.create_connection(("127.0.0.1", SERVER_PORT), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)

    raise RuntimeError(f"Backend server did not become ready on port {SERVER_PORT}.")


def start_server_process() -> None:
    global _server_process
    if _server_process is not None and _server_process.poll() is None:
        return

    process = subprocess.Popen(
        _server_subprocess_command(),
        cwd=os.path.dirname(os.path.abspath(__file__)),
        **_server_subprocess_kwargs(),
    )
    _server_process = process
    _attach_process_to_server_job(process.pid)
    _wait_for_server_ready()


def stop_server_process() -> None:
    global _cleanup_started, _server_process
    if _cleanup_started:
        return
    _cleanup_started = True

    process = _server_process
    _server_process = None
    if process is not None:
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2.0)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1.0)
        except Exception:
            pass

    _close_server_job_object()


def persist_runtime_config() -> None:
    try:
        SingletonConfig().save_config(SingletonConfig().config)
    except Exception as exc:
        print(f"Failed to persist config on exit: {exc}")


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


def _handle_exit_signal(signum, frame):
    stop_server_process()
    raise SystemExit(0)


if __name__ == "__main__":
    if "--backend-server-child" in sys.argv:
        child_port = SERVER_PORT
        try:
            child_port = int(sys.argv[-1])
        except (TypeError, ValueError):
            pass
        atexit.register(persist_runtime_config)
        _run_server_process(child_port)
        raise SystemExit(0)

    is_frozen = getattr(sys, "frozen", False)
    multiprocessing.freeze_support()
    atexit.register(stop_server_process)

    for sig in (
        signal.SIGINT,
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGBREAK", None),
    ):
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_exit_signal)
        except (ValueError, OSError):
            pass

    start_server_process()

    url = "http://localhost:5173"
    if os.path.exists(frontend_dist_path):
        url = f"http://localhost:{SERVER_PORT}/"

    window = webview.create_window(
        "2048 Endgame TableBase",
        url,
        js_api=Api(),
        width=1200,
        height=940,
        min_size=(800, 600),
    )

    try:
        webview.start(_initialize_frontend_error_bridge, debug=not is_frozen)
    finally:
        stop_server_process()
