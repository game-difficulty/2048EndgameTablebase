from __future__ import annotations

import atexit
import ctypes
import html
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

import webview

from backend.resource_paths import get_resource_path
from backend.webview_api import Api
from error_bridge import register_frontend_error_dispatcher


_MOTW_EXTENSIONS = {".dll", ".pyd", ".exe", ".ocx"}
_MOTW_SENTINEL = ".motw_unblocked"
_MOTW_SKIP_DIRS = {".git", "__pycache__", "node_modules"}
_MOTW_ALWAYS_SCAN_DIRS = {
    "_internal",
    "ai_and_sort",
    "bin",
    "dlls",
    "lib",
    "libs",
    "pywin32_system32",
}

APP_TITLE = "2048 Endgame TableBase"
APP_WINDOW_SIZE = (1200, 940)
APP_WINDOW_MIN_SIZE = (800, 600)

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


def _runtime_search_roots() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass).resolve())
    else:
        roots.append(Path(__file__).resolve().parent)

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)

    deduped: list[Path] = []
    for root in sorted(unique_roots, key=lambda item: (len(item.parts), str(item).lower())):
        if any(existing in root.parents for existing in deduped):
            continue
        deduped.append(root)

    return deduped


def _motw_global_sentinel(roots: list[Path]) -> Path | None:
    if not roots or not getattr(sys, "frozen", False):
        return None

    executable_root = Path(sys.executable).resolve().parent
    if all(root == executable_root or executable_root in root.parents for root in roots):
        return executable_root / _MOTW_SENTINEL

    return None


def _clear_motw_stream(path: Path) -> None:
    try:
        os.remove(f"{path}:Zone.Identifier")
    except (FileNotFoundError, OSError):
        return


def _has_motw_stream(path: Path) -> bool:
    try:
        return os.path.exists(f"{path}:Zone.Identifier")
    except OSError:
        return False


def _is_motw_candidate_file(path: Path) -> bool:
    return path.suffix.lower() in _MOTW_EXTENSIONS


def _should_scan_dir(path: Path) -> bool:
    if path.name.lower() in _MOTW_ALWAYS_SCAN_DIRS:
        return True

    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if (
                    entry.is_file(follow_symlinks=False)
                    and Path(entry.name).suffix.lower() in _MOTW_EXTENSIONS
                ):
                    return True
    except OSError:
        return False

    return False


def _iter_root_candidate_dirs(root: Path):
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                path = Path(entry.path)
                if path.name.lower() in _MOTW_SKIP_DIRS:
                    continue
                if _should_scan_dir(path):
                    yield path
    except OSError:
        return


def _iter_motw_candidate_files(root: Path):
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False):
                    continue
                path = Path(entry.path)
                if _is_motw_candidate_file(path):
                    yield path
    except OSError:
        return

    for scan_dir in _iter_root_candidate_dirs(root):
        for dirpath, dirnames, filenames in os.walk(scan_dir):
            dirnames[:] = [
                name
                for name in dirnames
                if name.lower() not in _MOTW_SKIP_DIRS
            ]
            for filename in filenames:
                path = Path(dirpath) / filename
                if _is_motw_candidate_file(path):
                    yield path


def _iter_motw_probe_files(root: Path):
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False):
                    path = Path(entry.path)
                    if _is_motw_candidate_file(path):
                        yield path
    except OSError:
        return

    for scan_dir in _iter_root_candidate_dirs(root):
        try:
            with os.scandir(scan_dir) as entries:
                for entry in entries:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    path = Path(entry.path)
                    if _is_motw_candidate_file(path):
                        yield path
                        break
        except OSError:
            continue


def _bundle_appears_blocked(roots: list[Path]) -> bool:
    executable_path = Path(sys.executable).resolve()
    if _has_motw_stream(executable_path):
        return True

    for root in roots:
        for path in _iter_motw_probe_files(root):
            if _has_motw_stream(path):
                return True

    return False


def _unblock_distribution_files() -> None:
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return

    roots = _runtime_search_roots()
    global_sentinel = _motw_global_sentinel(roots)
    if global_sentinel is not None and global_sentinel.exists():
        return

    if not _bundle_appears_blocked(roots):
        if global_sentinel is not None:
            try:
                global_sentinel.write_text("ok", encoding="utf-8")
            except OSError:
                pass
        else:
            for root in roots:
                try:
                    (root / _MOTW_SENTINEL).write_text("ok", encoding="utf-8")
                except OSError:
                    pass
        return

    for root in roots:
        sentinel = root / _MOTW_SENTINEL
        if global_sentinel is None and sentinel.exists():
            continue

        for path in _iter_motw_candidate_files(root):
            _clear_motw_stream(path)

        if global_sentinel is None:
            try:
                sentinel.write_text("ok", encoding="utf-8")
            except OSError:
                pass

    if global_sentinel is not None:
        try:
            global_sentinel.write_text("ok", encoding="utf-8")
        except OSError:
            pass


def find_available_port(start_port: int = 8000) -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", start_port))
            return start_port
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]


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
    if os.path.exists(frontend_dist_path):
        return f"http://localhost:{SERVER_PORT}/"
    return "http://localhost:5173"


def _build_startup_page_html(message: str, is_error: bool = False) -> str:
    escaped_message = html.escape(message).replace("\n", "<br>")
    title = "Startup failed" if is_error else "Starting application"
    accent = "#b93818" if is_error else "#1f6f5f"
    badge = "Error" if is_error else "Loading"
    background = (
        "radial-gradient(circle at top, #fff4ec 0%, #f4eee5 42%, #e9e2d8 100%)"
        if is_error
        else "radial-gradient(circle at top, #f5fbf7 0%, #ece9dd 42%, #e0d8cb 100%)"
    )
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(APP_TITLE)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: \"Segoe UI\", \"PingFang SC\", \"Microsoft YaHei\", sans-serif;
      --accent: {accent};
      --surface: rgba(255, 255, 255, 0.84);
      --text: #1d1d1f;
      --muted: #5f6368;
      --border: rgba(0, 0, 0, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: {background};
      color: var(--text);
    }}
    .panel {{
      width: min(540px, calc(100vw - 48px));
      padding: 32px 30px;
      border-radius: 24px;
      background: var(--surface);
      border: 1px solid var(--border);
      box-shadow: 0 28px 80px rgba(45, 35, 24, 0.14);
      backdrop-filter: blur(18px);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(0, 0, 0, 0.06);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 18px 0 8px;
      font-size: 34px;
      line-height: 1.1;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 19px;
      line-height: 1.35;
      font-weight: 700;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
      word-break: break-word;
    }}
    .spinner {{
      width: 18px;
      height: 18px;
      margin-top: 22px;
      border-radius: 50%;
      border: 2px solid rgba(0, 0, 0, 0.08);
      border-top-color: var(--accent);
      animation: spin 0.9s linear infinite;
      display: {("none" if is_error else "block")};
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
  </style>
</head>
<body>
  <main class=\"panel\">
    <div class=\"badge\">{badge}</div>
    <h1>{html.escape(APP_TITLE)}</h1>
    <h2>{html.escape(title)}</h2>
    <p>{escaped_message}</p>
    <div class=\"spinner\" aria-hidden=\"true\"></div>
  </main>
</body>
</html>
"""


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


def _wait_for_server_ready(timeout_seconds: float = 10.0) -> None:
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


def _initialize_frontend_error_bridge() -> None:
    if window is None:
        return

    def _dispatch(payload: dict[str, str]) -> None:
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
    except Exception as exc:
        _terminate_server_process(close_job=True)
        traceback.print_exc()
        if _cleanup_started:
            return
        _show_startup_error(
            "The backend service failed to start.\n\n"
            f"{exc}\n\n"
            "Check the terminal output for the full traceback."
        )


def _start_launcher_runtime() -> None:
    _initialize_frontend_error_bridge()
    threading.Thread(target=_launch_backend_and_frontend, daemon=True).start()


def _handle_exit_signal(signum, frame) -> None:
    stop_server_process()
    raise SystemExit(0)


_unblock_distribution_files()
SERVER_PORT = find_available_port(8000)
frontend_dist_path = get_resource_path(os.path.join("frontend", "dist"))


if __name__ == "__main__":
    if "--backend-server-child" in sys.argv:
        child_port = SERVER_PORT
        try:
            child_port = int(sys.argv[-1])
        except (TypeError, ValueError):
            pass
        from backend.app import run_backend_server

        run_backend_server(child_port)
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

    window = webview.create_window(
        APP_TITLE,
        html=_build_startup_page_html("Preparing the backend service. Please wait..."),
        js_api=Api(),
        width=APP_WINDOW_SIZE[0],
        height=APP_WINDOW_SIZE[1],
        min_size=APP_WINDOW_MIN_SIZE,
    )

    try:
        webview.start(_start_launcher_runtime, debug=not is_frozen)
    finally:
        stop_server_process()
