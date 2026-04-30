from __future__ import annotations

import atexit
import ctypes
import html
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
from platform import machine, system
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from urllib.parse import urlencode
import webbrowser

import webview

from backend.resource_paths import get_resource_path
from backend.webview_api import Api
from error_bridge import register_frontend_error_dispatcher

APP_TITLE = "2048 Endgame TableBase"
APP_WINDOW_SIZE = (1200, 940)
APP_WINDOW_MIN_SIZE = (800, 600)

window: webview.Window | None = None
_server_process: subprocess.Popen | None = None
_server_job_handle = None
_cleanup_started = False
_linux_backend_probe_cache: dict[str, object] | None = None


if os.name == "nt":
    import winreg

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _WEBVIEW2_DOWNLOAD_URL = "https://developer.microsoft.com/en-us/microsoft-edge/webview2/#download-section"
    _WEBVIEW2_MIN_VERSION = (86, 0, 622, 0)
    _WEBVIEW2_MIN_DOTNET_RELEASE = 394802
    _WEBVIEW2_RUNTIME_CLIENT_IDS = (
        "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        "{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}",
        "{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}",
        "{65C35B14-6C1D-4122-AC46-7148CC9D6497}",
    )

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
    startup_theme = "dark" if _startup_uses_dark_mode() else "light"
    query = urlencode(
        {
            "startup_theme": startup_theme,
            "backend_port": SERVER_PORT,
        }
    )
    if os.path.exists(frontend_dist_path):
        return f"http://localhost:{SERVER_PORT}/?{query}"
    return f"http://localhost:5173/?{query}"


def _load_startup_config() -> dict[str, object]:
    config_path = Path(get_resource_path(os.path.join("docs_and_configs", "config")))
    try:
        with config_path.open("rb") as config_file:
            config = pickle.load(config_file)
    except Exception:
        return {}

    return config if isinstance(config, dict) else {}


def _startup_uses_dark_mode() -> bool:
    return bool(_load_startup_config().get("dark_mode", False))


def _startup_language() -> str:
    language = str(_load_startup_config().get("language") or "").strip().lower()
    if language.startswith("zh"):
        return "zh"
    return "en"


def _startup_text(key: str) -> str:
    language = _startup_language()
    localized_strings = STARTUP_TRANSLATIONS.get(language, STARTUP_TRANSLATIONS["en"])
    return localized_strings.get(key, STARTUP_TRANSLATIONS["en"].get(key, key))


def _is_linux_platform() -> bool:
    return system() == "Linux"


def _preferred_linux_backends() -> list[str]:
    forced_gui = str(os.environ.get("PYWEBVIEW_GUI") or "").strip().lower()
    if forced_gui == "qt":
        return ["qt", "gtk"]
    return ["gtk", "qt"]


def _format_exception_summary(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _probe_linux_backend_module(backend: str) -> tuple[bool, str | None]:
    module_name = "webview.platforms.gtk" if backend == "gtk" else "webview.platforms.qt"
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:
        return False, _format_exception_summary(exc)


def _probe_linux_webview_backends() -> dict[str, object]:
    global _linux_backend_probe_cache
    if _linux_backend_probe_cache is not None:
        return _linux_backend_probe_cache

    if not _is_linux_platform():
        _linux_backend_probe_cache = {
            "available": True,
            "has_display": True,
            "results": [],
        }
        return _linux_backend_probe_cache

    results: list[dict[str, object]] = []
    for backend in _preferred_linux_backends():
        available, error = _probe_linux_backend_module(backend)
        results.append(
            {
                "name": backend,
                "available": available,
                "error": error,
            }
        )
        if available:
            break

    _linux_backend_probe_cache = {
        "available": any(bool(item["available"]) for item in results),
        "has_display": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
        "results": results,
    }
    return _linux_backend_probe_cache


def _linux_backend_notice_texts() -> dict[str, str]:
    if _startup_language() == "zh":
        return {
            "missing_title": "\u7f3a\u5c11 Linux Webview \u540e\u7aef\u4f9d\u8d56",
            "missing_intro": (
                "\u5f53\u524d Python/\u8fd0\u884c\u65f6\u73af\u5883\u65e0\u6cd5\u521d\u59cb\u5316 "
                "GTK/WebKit \u6216 Qt\uff0cpywebview \u65e0\u6cd5\u5728 Linux \u4e0a\u542f\u52a8\u7a97\u53e3\u3002"
            ),
            "install_intro": "\u8bf7\u5b89\u88c5\u4e0b\u5217\u4efb\u4e00\u5957\u53ef\u7528\u540e\u7aef\u540e\u91cd\u8bd5\uff1a",
            "gtk_option": "GTK/WebKit \u65b9\u6848\uff08\u5927\u591a\u6570 Linux \u684c\u9762\u73af\u5883\u63a8\u8350\uff09\uff1a",
            "gtk_debian": (
                "Ubuntu/Debian: sudo apt install python3-gi gir1.2-gtk-3.0 "
                "gir1.2-webkit2-4.1"
            ),
            "gtk_debian_legacy": (
                "\u5982\u679c\u6ca1\u6709 WebKit 4.1\uff1a"
                " sudo apt install gir1.2-webkit2-4.0"
            ),
            "gtk_fedora": "Fedora: sudo dnf install python3-gobject gtk3 webkit2gtk4.1",
            "gtk_arch": "Arch: sudo pacman -S python-gobject gtk3 webkit2gtk",
            "qt_option": "Qt \u65b9\u6848\uff1a",
            "qt_example": (
                "\u5b89\u88c5\u5e26 WebEngine \u7684 Qt Python \u7ed1\u5b9a\uff0c"
                "\u4f8b\u5982\uff1apip install qtpy PySide6\uff0c"
                "\u6216\u4f7f\u7528\u5df2\u6253\u5305 Qt WebEngine \u7684\u53d1\u884c\u7248\u3002"
            ),
            "start_failed_title": "Linux Webview \u542f\u52a8\u5931\u8d25",
            "start_failed_intro": (
                "\u5df2\u68c0\u6d4b\u5230 Linux Webview \u540e\u7aef\u6a21\u5757\uff0c"
                "\u4f46 pywebview \u4ecd\u7136\u6ca1\u6709\u6210\u529f\u6253\u5f00\u7a97\u53e3\u3002"
            ),
            "display_hint": (
                "\u672a\u68c0\u6d4b\u5230\u56fe\u5f62\u4f1a\u8bdd\uff0c"
                "\u8bf7\u68c0\u67e5 DISPLAY \u6216 WAYLAND_DISPLAY \u73af\u5883\u53d8\u91cf\u3002"
            ),
            "details_header": "\u6280\u672f\u7ec6\u8282\uff1a",
            "gtk_label": "GTK/WebKit",
            "qt_label": "Qt",
            "pywebview_label": "pywebview",
            "close_action": "\u5173\u95ed",
        }

    return {
        "missing_title": "Linux webview backend missing",
        "missing_intro": (
            "This Python/runtime environment could not initialize either GTK/WebKit or Qt, "
            "so pywebview could not open a Linux desktop window."
        ),
        "install_intro": "Install one of the supported backend stacks and try again:",
        "gtk_option": "GTK/WebKit option (recommended on most Linux desktops):",
        "gtk_debian": (
            "Ubuntu/Debian: sudo apt install python3-gi gir1.2-gtk-3.0 "
            "gir1.2-webkit2-4.1"
        ),
        "gtk_debian_legacy": "If WebKit 4.1 is unavailable: sudo apt install gir1.2-webkit2-4.0",
        "gtk_fedora": "Fedora: sudo dnf install python3-gobject gtk3 webkit2gtk4.1",
        "gtk_arch": "Arch: sudo pacman -S python-gobject gtk3 webkit2gtk",
        "qt_option": "Qt option:",
        "qt_example": (
            "Install Qt bindings with WebEngine support, for example: pip install qtpy PySide6, "
            "or use a build that already bundles Qt WebEngine."
        ),
        "start_failed_title": "Linux webview startup failed",
        "start_failed_intro": (
            "A Linux webview backend module was detected, but pywebview still failed to open the window."
        ),
        "display_hint": "No graphical session was detected. Check DISPLAY or WAYLAND_DISPLAY before launching the app.",
        "details_header": "Technical details:",
        "gtk_label": "GTK/WebKit",
        "qt_label": "Qt",
        "pywebview_label": "pywebview",
        "close_action": "Close",
    }


def _build_linux_backend_notice(
    probe: dict[str, object],
    *,
    startup_error: str | None = None,
) -> tuple[str, str]:
    texts = _linux_backend_notice_texts()
    title = texts["missing_title"]
    intro = texts["missing_intro"]
    lines = [intro]

    if probe.get("available"):
        title = texts["start_failed_title"]
        lines = [texts["start_failed_intro"]]
    else:
        lines.extend(
            [
                "",
                texts["install_intro"],
                "",
                texts["gtk_option"],
                texts["gtk_debian"],
                texts["gtk_debian_legacy"],
                texts["gtk_fedora"],
                texts["gtk_arch"],
                "",
                texts["qt_option"],
                texts["qt_example"],
            ]
        )

    if not bool(probe.get("has_display", True)):
        lines.extend(["", texts["display_hint"]])

    details: list[str] = []
    for item in probe.get("results", []):
        backend_name = str(item.get("name") or "")
        error = str(item.get("error") or "").strip()
        if not error:
            continue
        label = texts["gtk_label"] if backend_name == "gtk" else texts["qt_label"]
        details.append(f"{label}: {error}")

    if startup_error:
        details.append(f'{texts["pywebview_label"]}: {startup_error}')

    if details:
        lines.extend(["", texts["details_header"], *details])

    return title, "\n".join(lines)


def _startup_output_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _startup_error_log_path() -> Path:
    return _startup_output_root() / "startup_error.txt"


def _write_startup_error_log(title: str, message: str) -> Path | None:
    log_path = _startup_error_log_path()
    try:
        log_path.write_text(
            f"{APP_TITLE}\n{title}\n\n{message}\n",
            encoding="utf-8",
        )
        return log_path
    except Exception:
        return None


def _build_startup_error_page_html(title: str, message: str) -> str:
    escaped_title = html.escape(title)
    escaped_message = html.escape(message)
    language = html.escape(_startup_language(), quote=True)
    return f"""<!DOCTYPE html>
<html lang="{language}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(APP_TITLE)} - {escaped_title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(180deg, #f5f7fb 0%, #e9edf5 100%);
      color: #111827;
      display: grid;
      place-items: center;
      padding: 24px;
    }}
    main {{
      width: min(820px, 100%);
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 20px;
      box-shadow: 0 24px 60px rgba(15, 23, 42, 0.14);
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    h2 {{
      margin: 0 0 18px;
      font-size: 18px;
      color: #b91c1c;
    }}
    pre {{
      margin: 0;
      padding: 18px;
      border-radius: 14px;
      background: #0f172a;
      color: #e5e7eb;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font: 14px/1.65 "Cascadia Mono", "Consolas", monospace;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(APP_TITLE)}</h1>
    <h2>{escaped_title}</h2>
    <pre>{escaped_message}</pre>
  </main>
</body>
</html>
"""


def _write_startup_error_page(title: str, message: str) -> Path | None:
    filename = f"2048_endgame_tablebase_startup_error_{os.getpid()}.html"
    page_path = Path(tempfile.gettempdir()) / filename
    try:
        page_path.write_text(
            _build_startup_error_page_html(title, message),
            encoding="utf-8",
        )
        return page_path
    except Exception:
        return None


def _open_startup_error_page(title: str, message: str) -> bool:
    page_path = _write_startup_error_page(title, message)
    if page_path is None:
        return False

    if shutil.which("xdg-open") is not None:
        try:
            result = subprocess.run(
                ["xdg-open", str(page_path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    try:
        return bool(webbrowser.open(page_path.as_uri(), new=2))
    except Exception:
        return False


def _show_external_text_dialog(title: str, message: str) -> bool:
    dialog_message = f"{title}\n\n{message}"
    commands = [
        [
            "zenity",
            "--error",
            "--width=700",
            "--height=460",
            "--title",
            APP_TITLE,
            "--text",
            dialog_message,
        ],
        [
            "kdialog",
            "--title",
            APP_TITLE,
            "--msgbox",
            dialog_message,
        ],
        [
            "xmessage",
            "-center",
            dialog_message,
        ],
    ]

    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
            if result.returncode == 0:
                return True
        except Exception:
            continue

    return False


def _show_linux_backend_notice(
    probe: dict[str, object] | None = None,
    *,
    startup_error: str | None = None,
) -> None:
    probe = probe or _probe_linux_webview_backends()
    title, message = _build_linux_backend_notice(probe, startup_error=startup_error)

    if _open_startup_error_page(title, message):
        return
    if _show_external_text_dialog(title, message):
        return

    log_path = _write_startup_error_log(title, message)
    try:
        sys.stderr.write(f"{APP_TITLE}\n{title}\n\n{message}\n")
        if log_path is not None:
            sys.stderr.write(f"\nstartup_error.txt: {log_path}\n")
        sys.stderr.flush()
    except Exception:
        pass


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
    escaped_message = html.escape(message).replace("\n", "<br>")
    is_dark_mode = _startup_uses_dark_mode()
    title = title or (
        _startup_text("startup_failed_title")
        if is_error
        else _startup_text("starting_title")
    )
    accent = (
        "#ff8e72"
        if is_error and is_dark_mode
        else "#b93818"
        if is_error
        else "#6fe0c2"
        if is_dark_mode
        else "#1f6f5f"
    )
    badge = badge or (
        _startup_text("error_badge") if is_error else _startup_text("loading_badge")
    )
    color_scheme = "dark" if is_dark_mode else "light"
    show_spinner = (not is_error) if show_spinner is None else show_spinner
    if is_dark_mode:
        background = (
            "radial-gradient(circle at top, #402018 0%, #251918 44%, #111315 100%)"
            if is_error
            else "radial-gradient(circle at top, #14332f 0%, #1b2025 42%, #101214 100%)"
        )
        surface = "rgba(20, 23, 27, 0.84)"
        text = "#f3f4f6"
        muted = "#aab3bd"
        border = "rgba(255, 255, 255, 0.10)"
        panel_shadow = "0 28px 80px rgba(0, 0, 0, 0.40)"
        badge_background = "rgba(255, 255, 255, 0.08)"
        badge_border = "rgba(255, 255, 255, 0.10)"
        spinner_border = "rgba(255, 255, 255, 0.14)"
        action_background = "rgba(111, 224, 194, 0.16)"
        action_text = "#d9fff6"
        action_border = "rgba(111, 224, 194, 0.28)"
        link_color = "#8ff6d8"
        link_note_color = "#cfd6df"
    else:
        background = (
            "radial-gradient(circle at top, #fff4ec 0%, #f4eee5 42%, #e9e2d8 100%)"
            if is_error
            else "radial-gradient(circle at top, #f5fbf7 0%, #ece9dd 42%, #e0d8cb 100%)"
        )
        surface = "rgba(255, 255, 255, 0.84)"
        text = "#1d1d1f"
        muted = "#5f6368"
        border = "rgba(0, 0, 0, 0.08)"
        panel_shadow = "0 28px 80px rgba(45, 35, 24, 0.14)"
        badge_background = "rgba(255, 255, 255, 0.72)"
        badge_border = "rgba(0, 0, 0, 0.06)"
        spinner_border = "rgba(0, 0, 0, 0.08)"
        action_background = "rgba(31, 111, 95, 0.12)"
        action_text = "#12453b"
        action_border = "rgba(31, 111, 95, 0.16)"
        link_color = "#0f766e"
        link_note_color = "#4b5563"

    link_markup = ""
    if action_label and action_url:
        escaped_url = html.escape(action_url, quote=True)
        escaped_label = html.escape(action_label)
        hint_markup = ""
        if manual_link_hint:
            escaped_hint = html.escape(manual_link_hint).replace("\n", "<br>")
            hint_markup = (
                f"<p class=\"link-note\">{escaped_hint}<br>"
                f"<a class=\"inline-link\" href=\"{escaped_url}\" target=\"_blank\" "
                f"rel=\"noreferrer noopener\" onclick='return openExternalLink({json.dumps(action_url)})'>"
                f"{escaped_url}</a></p>"
            )
        link_markup = (
            "<div class=\"actions\">"
            f"<a class=\"action-link\" href=\"{escaped_url}\" target=\"_blank\" "
            f"rel=\"noreferrer noopener\" onclick='return openExternalLink({json.dumps(action_url)})'>"
            f"{escaped_label}</a>"
            f"{hint_markup}"
            "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang=\"{html.escape(_startup_language(), quote=True)}\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(APP_TITLE)}</title>
  <script>
    function openExternalLink(url) {{
      try {{
        if (
          window.pywebview &&
          window.pywebview.api &&
          typeof window.pywebview.api.open_external_url === 'function'
        ) {{
          window.pywebview.api.open_external_url(url);
          return false;
        }}
      }} catch (error) {{
      }}
      return true;
    }}
  </script>
  <style>
    :root {{
      color-scheme: {color_scheme};
      font-family: \"Segoe UI\", \"PingFang SC\", \"Microsoft YaHei\", sans-serif;
      --accent: {accent};
      --surface: {surface};
      --text: {text};
      --muted: {muted};
      --border: {border};
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
      width: min(560px, calc(100vw - 48px));
      padding: 32px 30px;
      border-radius: 24px;
      background: var(--surface);
      border: 1px solid var(--border);
      box-shadow: {panel_shadow};
      backdrop-filter: blur(18px);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 6px 12px;
      border-radius: 999px;
      background: {badge_background};
      border: 1px solid {badge_border};
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
    .actions {{
      display: grid;
      gap: 14px;
      margin-top: 24px;
    }}
    .action-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 46px;
      padding: 0 18px;
      border-radius: 14px;
      background: {action_background};
      border: 1px solid {action_border};
      color: {action_text};
      text-decoration: none;
      font-size: 14px;
      font-weight: 700;
    }}
    .inline-link {{
      color: {link_color};
      text-decoration: none;
      word-break: break-all;
      font-weight: 600;
    }}
    .link-note {{
      color: {link_note_color};
      font-size: 13px;
      line-height: 1.6;
    }}
    .spinner {{
      width: 18px;
      height: 18px;
      margin-top: 22px;
      border-radius: 50%;
      border: 2px solid {spinner_border};
      border-top-color: var(--accent);
      animation: spin 0.9s linear infinite;
      display: {("block" if show_spinner else "none")};
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
    {link_markup}
    <div class=\"spinner\" aria-hidden=\"true\"></div>
  </main>
</body>
</html>
"""


def _version_at_least(version: str, minimum: tuple[int, ...]) -> bool:
    try:
        version_parts = [int(part) for part in str(version).split(".")]
    except (TypeError, ValueError):
        return False

    padded_version = version_parts + [0] * max(0, len(minimum) - len(version_parts))
    return tuple(padded_version[: len(minimum)]) >= minimum


def _read_webview2_client_version(root_key, client_id: str) -> str | None:
    if os.name != "nt":
        return None

    if machine() == "x86" or root_key == winreg.HKEY_CURRENT_USER:
        key_path = rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{client_id}"
    else:
        key_path = rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{client_id}"

    try:
        with winreg.OpenKey(root_key, key_path) as registry_key:
            version, _ = winreg.QueryValueEx(registry_key, "pv")
            return str(version)
    except OSError:
        return None


def _has_webview2_runtime() -> bool:
    if os.name != "nt":
        return True

    webview_settings = getattr(webview, "settings", None)
    if hasattr(webview_settings, "get") and webview_settings.get("WEBVIEW2_RUNTIME_PATH"):
        return True

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full",
        ) as net_key:
            dotnet_release, _ = winreg.QueryValueEx(net_key, "Release")
    except (OSError, TypeError, ValueError):
        return False

    try:
        if int(dotnet_release) < _WEBVIEW2_MIN_DOTNET_RELEASE:
            return False
    except (TypeError, ValueError):
        return False

    for client_id in _WEBVIEW2_RUNTIME_CLIENT_IDS:
        for root_key in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            version = _read_webview2_client_version(root_key, client_id)
            if version and _version_at_least(version, _WEBVIEW2_MIN_VERSION):
                return True

    return False


def _show_webview2_runtime_required_notice() -> None:
    if window is None:
        return

    try:
        window.load_html(
            _build_startup_page_html(
                _startup_text("webview2_missing_message"),
                is_error=True,
                title=_startup_text("webview2_missing_title"),
                badge=_startup_text("setup_badge"),
                show_spinner=False,
                action_label=_startup_text("webview2_download_action"),
                action_url=_WEBVIEW2_DOWNLOAD_URL,
                manual_link_hint=_startup_text("webview2_manual_link_hint"),
            )
        )
    except Exception:
        pass


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
            _startup_text("backend_start_failed_template").format(details=str(exc))
        )


def _start_launcher_runtime() -> None:
    if os.name == "nt" and not _has_webview2_runtime():
        _show_webview2_runtime_required_notice()
        return
    _initialize_frontend_error_bridge()
    threading.Thread(target=_launch_backend_and_frontend, daemon=True).start()


def _handle_exit_signal(signum, frame) -> None:
    stop_server_process()
    raise SystemExit(0)


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

    try:
        if _is_linux_platform():
            linux_probe = _probe_linux_webview_backends()
            if not bool(linux_probe.get("available")):
                _show_linux_backend_notice(linux_probe)
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
        except webview.WebViewException as exc:
            if _is_linux_platform():
                _show_linux_backend_notice(startup_error=str(exc))
                raise SystemExit(1)
            raise
    finally:
        stop_server_process()
