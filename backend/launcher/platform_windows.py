from __future__ import annotations

from collections.abc import Callable
import ctypes
import importlib
import os
from pathlib import Path
from platform import machine
import sys

import webview


if os.name == "nt":
    import winreg

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _WEBVIEW2_DOWNLOAD_URL = "https://developer.microsoft.com/en-us/microsoft-edge/webview2/#download-section"
    _WEBVIEW2_MIN_VERSION = (86, 0, 622, 0)
    _WEBVIEW2_MIN_DOTNET_RELEASE = 394802
    _WEBVIEW2_RUNTIME_CLIENT_IDS = (
        "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        "{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}",
        "{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}",
        "{65C35B14-6C1D-4122-AC46-7148CC9D6497}",
    )

    _MB_OK = 0x00000000
    _MB_ICONERROR = 0x00000010
    _MB_SYSTEMMODAL = 0x00001000

    _user32.MessageBoxW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint,
    ]
    _user32.MessageBoxW.restype = ctypes.c_int


def is_pythonnet_loader_failure(message: str) -> bool:
    text = str(message or "").strip()
    return (
        "Python.Runtime.Loader.Initialize" in text
        and "Python.Runtime.dll" in text
    )


def extract_pythonnet_runtime_dll_path(message: str) -> str | None:
    if not is_pythonnet_loader_failure(message):
        return None

    _, separator, tail = str(message).rpartition(" from ")
    if not separator:
        return None

    candidate = tail.strip().strip('"')
    if not candidate.lower().endswith("python.runtime.dll"):
        return None
    return candidate


def file_has_mark_of_the_web(path: str | os.PathLike[str]) -> bool:
    if os.name != "nt":
        return False

    try:
        with open(
            f"{Path(path)}:Zone.Identifier",
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as zone_file:
            return "[ZoneTransfer]" in zone_file.read(4096)
    except OSError:
        return False


def build_windows_pythonnet_runtime_notice(
    error: str,
    startup_language: Callable[[], str],
) -> tuple[str, str]:
    dll_path = extract_pythonnet_runtime_dll_path(error)
    internal_root: Path | None = None
    has_motw = False

    if dll_path:
        try:
            runtime_path = Path(dll_path)
            internal_root = runtime_path.parents[2]
        except (IndexError, RuntimeError, OSError):
            internal_root = None
        has_motw = file_has_mark_of_the_web(dll_path)

    if startup_language() == "zh":
        title = "\u65e0\u6cd5\u52a0\u8f7d Python.Runtime.dll"
        lines = [
            "\u7a0b\u5e8f\u5728\u521d\u59cb\u5316 Windows WebView \u540e\u7aef\u65f6\uff0c"
            "\u65e0\u6cd5\u52a0\u8f7d Python.Runtime.dll\u3002",
        ]
        if dll_path:
            lines.extend(["", f"DLL: {dll_path}"])
        if has_motw:
            lines.extend(
                [
                    "",
                    "\u68c0\u6d4b\u5230 Windows \u53ef\u80fd\u5df2\u5c06\u8be5\u6587\u4ef6"
                    "\u6807\u8bb0\u4e3a\u6765\u81ea Internet\uff08Mark of the Web\uff09\uff0c"
                    "\u5bfc\u81f4 pythonnet \u88ab\u963b\u6b62\u52a0\u8f7d\u3002",
                    "\u5904\u7406\u65b9\u5f0f\uff1a",
                    "1. \u627e\u5230\u539f\u59cb\u4e0b\u8f7d\u7684 7z \u538b\u7f29\u5305\u3002",
                    "2. \u53f3\u952e -> \u5c5e\u6027\u3002",
                    "3. \u5728\u7a97\u53e3\u4e0b\u65b9\u70b9\u51fb "
                    "\u201cUnblock/\u89e3\u9664\u9501\u5b9a\u201d\uff0c\u7136\u540e\u5e94\u7528\u3002",
                    "4. \u91cd\u65b0\u89e3\u538b\u6574\u4e2a\u538b\u7f29\u5305\u540e\uff0c"
                    "\u518d\u6b21\u542f\u52a8\u7a0b\u5e8f\u3002",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "\u5e38\u89c1\u539f\u56e0\u662f Windows \u5c06\u4ece\u4e0b\u8f7d\u538b\u7f29\u5305"
                    "\u89e3\u538b\u51fa\u6765\u7684 DLL \u6807\u8bb0\u4e3a\u4e0d\u5b89\u5168"
                    "\uff08Mark of the Web\uff09\u3002",
                    "\u53ef\u6309\u4ee5\u4e0b\u65b9\u5f0f\u5904\u7406\uff1a",
                    "1. \u627e\u5230\u539f\u59cb\u4e0b\u8f7d\u7684 7z \u538b\u7f29\u5305\u3002",
                    "2. \u53f3\u952e -> \u5c5e\u6027\u3002",
                    "3. \u5982\u679c\u7a97\u53e3\u4e0b\u65b9\u6709 "
                    "\u201cUnblock/\u89e3\u9664\u9501\u5b9a\u201d\uff0c"
                    "\u8bf7\u52fe\u9009\u6216\u70b9\u51fb\u540e\u5e94\u7528\u3002",
                    "4. \u91cd\u65b0\u89e3\u538b\u6574\u4e2a\u538b\u7f29\u5305\u540e\uff0c"
                    "\u518d\u6b21\u542f\u52a8\u7a0b\u5e8f\u3002",
                ]
            )
        if internal_root is not None:
            lines.extend(
                [
                    "",
                    "PowerShell \u53ef\u53c2\u8003\uff1a",
                    f"Get-ChildItem '{internal_root}' -Recurse | Unblock-File",
                ]
            )
    else:
        title = "Failed to load Python.Runtime.dll"
        lines = [
            "The app could not initialize the Windows webview backend because "
            "Python.Runtime.dll could not be loaded.",
        ]
        if dll_path:
            lines.extend(["", f"DLL: {dll_path}"])
        if has_motw:
            lines.extend(
                [
                    "",
                    "Windows appears to have marked this file as downloaded from the "
                    "Internet (Mark of the Web), which can block pythonnet from loading it.",
                    "Fix:",
                    "1. Locate the original downloaded 7z archive.",
                    "2. Right-click it and open Properties.",
                    "3. Click Unblock at the bottom of the dialog, then Apply.",
                    "4. Extract the archive again, then restart the app.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "A common cause is Windows blocking DLLs extracted from a downloaded "
                    "ZIP package (Mark of the Web).",
                    "Try this fix:",
                    "1. Locate the original downloaded 7z archive.",
                    "2. Right-click it and open Properties.",
                    "3. If there is an Unblock option at the bottom, click it and Apply.",
                    "4. Extract the archive again, then restart the app.",
                ]
            )
        if internal_root is not None:
            lines.extend(
                [
                    "",
                    "PowerShell example:",
                    f"Get-ChildItem '{internal_root}' -Recurse | Unblock-File",
                ]
            )

    return title, "\n".join(lines)


def show_windows_native_message_box(app_title: str, title: str, message: str) -> bool:
    if os.name != "nt":
        return False

    try:
        _user32.MessageBoxW(
            None,
            message,
            f"{app_title} - {title}",
            _MB_OK | _MB_ICONERROR | _MB_SYSTEMMODAL,
        )
        return True
    except Exception:
        return False


def show_windows_pythonnet_runtime_notice(
    error: str,
    startup_language: Callable[[], str],
    write_startup_error_log: Callable[[str, str], Path | None],
    app_title: str,
) -> None:
    title, message = build_windows_pythonnet_runtime_notice(error, startup_language)
    log_path = write_startup_error_log(title, f"{message}\n\nRaw error:\n{error}")

    if log_path is not None:
        if startup_language() == "zh":
            message = (
                f"{message}\n\n"
                "\u8be6\u7ec6\u4fe1\u606f\u5df2\u5199\u5165\uff1a\n"
                f"{log_path}"
            )
        else:
            message = f"{message}\n\nDetails were written to:\n{log_path}"

    if show_windows_native_message_box(app_title, title, message):
        return

    try:
        sys.stderr.write(f"{app_title}\n{title}\n\n{message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def probe_windows_pythonnet_runtime() -> str | None:
    if os.name != "nt":
        return None

    try:
        importlib.import_module("webview.platforms.winforms")
    except Exception as exc:
        if is_pythonnet_loader_failure(str(exc)):
            return str(exc)

    return None


def version_at_least(version: str, minimum: tuple[int, ...]) -> bool:
    try:
        version_parts = [int(part) for part in str(version).split(".")]
    except (TypeError, ValueError):
        return False

    padded_version = version_parts + [0] * max(0, len(minimum) - len(version_parts))
    return tuple(padded_version[: len(minimum)]) >= minimum


def read_webview2_client_version(root_key, client_id: str) -> str | None:
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


def has_webview2_runtime() -> bool:
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
            version = read_webview2_client_version(root_key, client_id)
            if version and version_at_least(version, _WEBVIEW2_MIN_VERSION):
                return True

    return False


def show_webview2_runtime_required_notice(
    window,
    build_startup_page_html: Callable[..., str],
    startup_text: Callable[[str], str],
) -> None:
    if window is None:
        return

    try:
        window.load_html(
            build_startup_page_html(
                startup_text("webview2_missing_message"),
                is_error=True,
                title=startup_text("webview2_missing_title"),
                badge=startup_text("setup_badge"),
                show_spinner=False,
                action_label=startup_text("webview2_download_action"),
                action_url=_WEBVIEW2_DOWNLOAD_URL,
                manual_link_hint=startup_text("webview2_manual_link_hint"),
            )
        )
    except Exception:
        pass
