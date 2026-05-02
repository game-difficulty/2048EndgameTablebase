from __future__ import annotations

from collections.abc import Callable
import importlib
import os
from platform import system
import sys


_linux_backend_probe_cache: dict[str, object] | None = None


def is_linux_platform() -> bool:
    return system() == "Linux"


def preferred_linux_backends() -> list[str]:
    forced_gui = str(os.environ.get("PYWEBVIEW_GUI") or "").strip().lower()
    if forced_gui == "qt":
        return ["qt", "gtk"]
    return ["gtk", "qt"]


def format_exception_summary(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def probe_linux_backend_module(backend: str) -> tuple[bool, str | None]:
    module_name = "webview.platforms.gtk" if backend == "gtk" else "webview.platforms.qt"
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:
        return False, format_exception_summary(exc)


def probe_linux_webview_backends() -> dict[str, object]:
    global _linux_backend_probe_cache
    if _linux_backend_probe_cache is not None:
        return _linux_backend_probe_cache

    if not is_linux_platform():
        _linux_backend_probe_cache = {
            "available": True,
            "has_display": True,
            "results": [],
        }
        return _linux_backend_probe_cache

    results: list[dict[str, object]] = []
    for backend in preferred_linux_backends():
        available, error = probe_linux_backend_module(backend)
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


def linux_backend_notice_texts(startup_language: Callable[[], str]) -> dict[str, str]:
    if startup_language() == "zh":
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


def build_linux_backend_notice(
    probe: dict[str, object],
    startup_language: Callable[[], str],
    *,
    startup_error: str | None = None,
) -> tuple[str, str]:
    texts = linux_backend_notice_texts(startup_language)
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


def show_linux_backend_notice(
    app_title: str,
    startup_language: Callable[[], str],
    open_startup_error_page: Callable[[str, str], bool],
    show_external_text_dialog: Callable[[str, str], bool],
    write_startup_error_log: Callable[[str, str], object | None],
    probe: dict[str, object] | None = None,
    *,
    startup_error: str | None = None,
) -> None:
    probe = probe or probe_linux_webview_backends()
    title, message = build_linux_backend_notice(
        probe,
        startup_language,
        startup_error=startup_error,
    )

    if open_startup_error_page(title, message):
        return
    if show_external_text_dialog(title, message):
        return

    log_path = write_startup_error_log(title, message)
    try:
        sys.stderr.write(f"{app_title}\n{title}\n\n{message}\n")
        if log_path is not None:
            sys.stderr.write(f"\nstartup_error.txt: {log_path}\n")
        sys.stderr.flush()
    except Exception:
        pass

