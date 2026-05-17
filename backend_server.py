from __future__ import annotations

import atexit
import ctypes
import ipaddress
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import urlencode

import webview

from backend.launcher.platform_linux import (
    is_linux_platform as _is_linux_platform,
    probe_linux_webview_backends as _probe_linux_webview_backends,
    show_linux_backend_notice as _launcher_show_linux_backend_notice,
)
from backend.launcher.platform_windows import (
    has_webview2_runtime as _has_webview2_runtime,
    is_pythonnet_loader_failure as _is_pythonnet_loader_failure,
    probe_windows_pythonnet_runtime as _probe_windows_pythonnet_runtime,
    show_webview2_runtime_required_notice as _launcher_show_webview2_runtime_required_notice,
    show_windows_pythonnet_runtime_notice as _launcher_show_windows_pythonnet_runtime_notice,
)
from backend.launcher.startup_config import (
    startup_language as _startup_language,
    startup_uses_dark_mode as _startup_uses_dark_mode,
)
from backend.launcher.startup_support import (
    open_startup_error_page as _launcher_open_startup_error_page,
    show_external_text_dialog as _launcher_show_external_text_dialog,
    write_startup_error_log as _launcher_write_startup_error_log,
)
from backend.launcher.startup_ui import (
    build_startup_error_page_html as _launcher_build_startup_error_page_html,
    build_startup_page_html as _launcher_build_startup_page_html,
)
from backend.resource_paths import get_resource_path
from backend.webview_api import Api
from error_bridge import register_frontend_error_dispatcher

APP_TITLE = "2048 Endgame TableBase"
APP_WINDOW_SIZE = (1200, 940)
APP_WINDOW_MIN_SIZE = (480, 320)

window: webview.Window | None = None
_server_process: subprocess.Popen | None = None
_server_job_handle = None
_cleanup_started = False
_frontend_error_queue: queue.Queue = queue.Queue()
_frontend_error_bridge_started = False
_frontend_error_bridge_lock = threading.Lock()
SERVER_BIND_HOST = "0.0.0.0"
DESKTOP_ACCESS_HOST = "127.0.0.1"


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


STARTUP_TRANSLATIONS = {
    "en": {
        "loading_badge": "Loading",
        "error_badge": "Error",
        "setup_badge": "Setup",
        "starting_title": "Starting application",
        "startup_failed_title": "Startup failed",
        "initial_loading_message": "Preparing the backend service. Please wait...",
        "backend_start_failed_template": (
            "The backend service failed to start.\n\n"
            "{details}\n\n"
            "Check the terminal output for the full traceback."
        ),
        "webview2_missing_title": "Microsoft Edge WebView2 Runtime required",
        "webview2_missing_message": (
            "This app requires Microsoft Edge WebView2 Runtime, but it is missing or outdated on this computer.\n\n"
            "Install the Evergreen WebView2 Runtime from Microsoft's official download page, then restart the app.\n\n"
            "Having Google Chrome installed does not satisfy this requirement."
        ),
        "webview2_download_action": "Download WebView2 Runtime",
        "webview2_manual_link_hint": "If the button doesn't open, paste this link into your browser:",
    },
    "zh": {
        "loading_badge": "加载中",
        "error_badge": "错误",
        "setup_badge": "安装提示",
        "starting_title": "正在启动应用",
        "startup_failed_title": "启动失败",
        "initial_loading_message": "正在准备后端服务，请稍候...",
        "backend_start_failed_template": (
            "后端服务启动失败。\n\n"
            "{details}\n\n"
            "请查看终端输出中的完整错误信息。"
        ),
        "webview2_missing_title": "缺少 Microsoft Edge WebView2 Runtime",
        "webview2_missing_message": (
            "此应用需要 Microsoft Edge WebView2 Runtime，但当前电脑上的该运行时缺失或版本过旧。\n\n"
            "请从微软官方下载页安装 Evergreen WebView2 Runtime，然后重新启动本应用。\n\n"
            "仅安装 Google Chrome 不能替代这个运行时。"
        ),
        "webview2_download_action": "下载 WebView2 Runtime",
        "webview2_manual_link_hint": "如果按钮没有打开页面，请将下面的链接粘贴到浏览器中：",
    },
}

def find_available_port(start_port: int = 8000) -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((SERVER_BIND_HOST, start_port))
            return start_port
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((SERVER_BIND_HOST, 0))
            return sock.getsockname()[1]


def _lan_ipv4_address_rank(address: str) -> int | None:
    if not address:
        return None
    try:
        ip = ipaddress.IPv4Address(address)
    except ipaddress.AddressValueError:
        return None
    if (
        ip.is_unspecified
        or ip.is_loopback
        or ip.is_link_local
    ):
        return None

    first_octet, second_octet, *_ = address.split(".")
    first = int(first_octet)
    second = int(second_octet)
    if first == 10:
        return 0
    if first == 172 and 16 <= second <= 31:
        return 0
    if first == 192 and second == 168:
        return 0
    return None


def _discover_lan_access_host() -> str:
    candidates: list[str] = []

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidates.append(sock.getsockname()[0])
    except OSError:
        pass

    try:
        hostname = socket.gethostname()
        for address in socket.gethostbyname_ex(hostname)[2]:
            candidates.append(address)
    except OSError:
        pass

    seen: set[str] = set()
    fallback_address: str | None = None
    for address in candidates:
        if address in seen:
            continue
        seen.add(address)
        rank = _lan_ipv4_address_rank(address)
        if rank is None:
            continue
        if rank == 0:
            return address
        if fallback_address is None:
            fallback_address = address

    return fallback_address or "127.0.0.1"


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
        _PROCESS_SET_QUOTA | _PROCESS_TERMINATE | _PROCESS_QUERY_LIMITED_INFORMATION,
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
    if os.name != "nt":
        return

    handle = _server_job_handle
    _server_job_handle = None
    if handle is None:
        return

    _kernel32.CloseHandle(handle)


def _frontend_url() -> str:
    startup_theme = "dark" if _startup_uses_dark_mode() else "light"
    query = urlencode(
        {
            "startup_theme": startup_theme,
            "backend_port": SERVER_PORT,
            "lan_host": SERVER_ACCESS_HOST,
        }
    )
    if os.path.exists(frontend_dist_path):
        return f"http://{DESKTOP_ACCESS_HOST}:{SERVER_PORT}/?{query}"
    return f"http://localhost:5173/?{query}"


def _startup_text(key: str) -> str:
    language = _startup_language()
    localized_strings = STARTUP_TRANSLATIONS.get(language, STARTUP_TRANSLATIONS["en"])
    return localized_strings.get(key, STARTUP_TRANSLATIONS["en"].get(key, key))


def _show_webview2_runtime_required_notice() -> None:
    _launcher_show_webview2_runtime_required_notice(
        window,
        _build_startup_page_html,
        _startup_text,
    )


def _build_startup_error_page_html(title: str, message: str) -> str:
    return _launcher_build_startup_error_page_html(
        APP_TITLE,
        _startup_language(),
        title,
        message,
    )


def _write_startup_error_log(title: str, message: str) -> Path | None:
    return _launcher_write_startup_error_log(
        APP_TITLE,
        title,
        message,
        is_frozen=getattr(sys, "frozen", False),
        executable_path=sys.executable,
        module_file=__file__,
    )


def _open_startup_error_page(title: str, message: str) -> bool:
    return _launcher_open_startup_error_page(
        _build_startup_error_page_html,
        title,
        message,
    )


def _show_external_text_dialog(title: str, message: str) -> bool:
    return _launcher_show_external_text_dialog(APP_TITLE, title, message)


def _show_linux_backend_notice(
    probe: dict[str, object] | None = None,
    *,
    startup_error: str | None = None,
) -> None:
    _launcher_show_linux_backend_notice(
        APP_TITLE,
        _startup_language,
        _open_startup_error_page,
        _show_external_text_dialog,
        _write_startup_error_log,
        probe,
        startup_error=startup_error,
    )


def _build_startup_page_html(
    message: str,
    is_error: bool = False,
    *,
    title: str | None = None,
    badge: str | None = None,
    show_spinner: bool | None = None,
    action_label: str | None = None,
    action_url: str | None = None,
    manual_link_hint: str | None = None,
) -> str:
    resolved_title = title or (
        _startup_text("startup_failed_title")
        if is_error
        else _startup_text("starting_title")
    )
    resolved_badge = badge or (
        _startup_text("error_badge") if is_error else _startup_text("loading_badge")
    )
    return _launcher_build_startup_page_html(
        APP_TITLE,
        _startup_language(),
        _startup_uses_dark_mode(),
        message,
        is_error=is_error,
        title=resolved_title,
        badge=resolved_badge,
        show_spinner=show_spinner,
        action_label=action_label,
        action_url=action_url,
        manual_link_hint=manual_link_hint,
    )


def _show_windows_pythonnet_runtime_notice(error: str) -> None:
    _launcher_show_windows_pythonnet_runtime_notice(
        error,
        _startup_language,
        _write_startup_error_log,
        APP_TITLE,
    )


def _server_subprocess_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--backend-server-child", str(SERVER_PORT)]
    return [
        sys.executable,
        os.path.abspath(__file__),
        "--backend-server-child",
        str(SERVER_PORT),
    ]


def _stream_is_writable(stream) -> bool:
    return stream is not None and not getattr(stream, "closed", False)


def _parent_log_targets():
    stdout_target = sys.stdout if _stream_is_writable(sys.stdout) else None
    stderr_target = sys.stderr if _stream_is_writable(sys.stderr) else stdout_target
    if stdout_target is None:
        stdout_target = stderr_target
    return stdout_target, stderr_target


def _relay_server_output(pipe, target) -> None:
    if pipe is None or target is None:
        return

    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            try:
                target.write(line)
                target.flush()
            except Exception:
                break
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _start_server_log_forwarders(process: subprocess.Popen) -> None:
    stdout_target, stderr_target = _parent_log_targets()
    if process.stdout is not None and stdout_target is not None:
        threading.Thread(
            target=_relay_server_output,
            args=(process.stdout, stdout_target),
            daemon=True,
        ).start()
    if process.stderr is not None and stderr_target is not None:
        threading.Thread(
            target=_relay_server_output,
            args=(process.stderr, stderr_target),
            daemon=True,
        ).start()


def _server_subprocess_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    stdout_target, stderr_target = _parent_log_targets()
    if stdout_target is not None or stderr_target is not None:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        kwargs["env"] = env
        kwargs["stdin"] = subprocess.DEVNULL
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"] = True
        kwargs["encoding"] = "utf-8"
        kwargs["errors"] = "replace"
        kwargs["bufsize"] = 1
    return kwargs


def _wait_for_server_ready(timeout_seconds: float = 20.0) -> None:
    deadline = time.perf_counter() + timeout_seconds
    while time.perf_counter() < deadline:
        if _server_process is not None and _server_process.poll() is not None:
            raise RuntimeError("Backend server process exited during startup.")
        try:
            with socket.create_connection((SERVER_PROBE_HOST, SERVER_PORT), timeout=0.2):
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
        **_server_subprocess_kwargs(),  # type: ignore
    )  # type: ignore
    _server_process = process
    _start_server_log_forwarders(process)
    _attach_process_to_server_job(process.pid)
    _wait_for_server_ready()


def _terminate_server_process(close_job: bool) -> None:
    global _server_process
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

    if close_job:
        _close_server_job_object()


def stop_server_process() -> None:
    global _cleanup_started
    if _cleanup_started:
        return
    _cleanup_started = True
    _terminate_server_process(close_job=True)


def _dispatch_frontend_error_payload(payload: dict[str, str]) -> None:
    if window is None:
        return

    payload_json = json.dumps(payload, ensure_ascii=False)
    window.evaluate_js(
        "window.__appGlobalErrors = window.__appGlobalErrors || [];"
        f"window.__appGlobalErrors.push({payload_json});"
        "window.dispatchEvent(new CustomEvent('app-global-error', { detail: "
        f"{payload_json} }}));"
    )


def _drain_frontend_error_queue() -> None:
    while True:
        if _cleanup_started:
            return

        try:
            payload = _frontend_error_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if window is None:
            continue

        try:
            if not window.events.loaded.wait(20):
                _frontend_error_queue.put(payload)
                time.sleep(0.2)
                continue
            _dispatch_frontend_error_payload(payload)
        except Exception:
            if _cleanup_started:
                return
            _frontend_error_queue.put(payload)
            time.sleep(0.2)


def _initialize_frontend_error_bridge() -> None:
    global _frontend_error_bridge_started
    if window is None:
        return

    def _dispatch(payload: dict[str, str]) -> None:
        _frontend_error_queue.put(payload)

    with _frontend_error_bridge_lock:
        if not _frontend_error_bridge_started:
            threading.Thread(
                target=_drain_frontend_error_queue,
                name="frontend-error-bridge",
                daemon=True,
            ).start()
            _frontend_error_bridge_started = True
    register_frontend_error_dispatcher(_dispatch)


def _show_startup_error(message: str) -> None:
    if window is None:
        return
    try:
        window.load_html(_build_startup_page_html(message, is_error=True))
    except Exception:
        pass


def _launch_backend_and_frontend() -> None:
    try:
        start_server_process()
        if _cleanup_started or window is None:
            return
        window.load_url(_frontend_url())
        _initialize_frontend_error_bridge()
    except Exception as exc:
        _terminate_server_process(close_job=True)
        traceback.print_exc()
        if _cleanup_started:
            return
        _show_startup_error(
            _startup_text("backend_start_failed_template").format(details=str(exc))
        )


def _start_launcher_runtime() -> None:
    if os.name == "nt" and not _has_webview2_runtime():
        _show_webview2_runtime_required_notice()
        return
    _launch_backend_and_frontend()


def _handle_exit_signal(signum, frame) -> None:
    stop_server_process()
    raise SystemExit(0)


SERVER_PORT = find_available_port(8000)
SERVER_ACCESS_HOST = _discover_lan_access_host()
SERVER_PROBE_HOST = DESKTOP_ACCESS_HOST
frontend_dist_path = get_resource_path(os.path.join("frontend", "dist"))


if __name__ == "__main__":
    if "--backend-server-child" in sys.argv:
        child_port = SERVER_PORT
        try:
            child_port = int(sys.argv[-1])
        except (TypeError, ValueError):
            pass
        from backend.app import run_backend_server

        run_backend_server(child_port, host=SERVER_BIND_HOST)
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

    try:
        if _is_linux_platform():
            linux_probe = _probe_linux_webview_backends()
            if not bool(linux_probe.get("available")):
                _show_linux_backend_notice(linux_probe)
                raise SystemExit(1)
        if os.name == "nt":
            pythonnet_runtime_error = _probe_windows_pythonnet_runtime()
            if pythonnet_runtime_error is not None:
                _show_windows_pythonnet_runtime_notice(pythonnet_runtime_error)
                raise SystemExit(1)

        window = webview.create_window(
            APP_TITLE,
            html=_build_startup_page_html(_startup_text("initial_loading_message")),
            js_api=Api(),
            width=APP_WINDOW_SIZE[0],
            height=APP_WINDOW_SIZE[1],
            min_size=APP_WINDOW_MIN_SIZE,
        )

        try:
            webview.start(_start_launcher_runtime, debug=not is_frozen)
        except Exception as exc:
            if os.name == "nt" and _is_pythonnet_loader_failure(str(exc)):
                _show_windows_pythonnet_runtime_notice(str(exc))
                raise SystemExit(1)
            if isinstance(exc, webview.WebViewException) and _is_linux_platform():
                _show_linux_backend_notice(startup_error=str(exc))
                raise SystemExit(1)
            raise
    finally:
        stop_server_process()
