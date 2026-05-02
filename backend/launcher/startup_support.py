from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import webbrowser


def startup_output_root(
    *,
    is_frozen: bool,
    executable_path: str,
    module_file: str,
) -> Path:
    if is_frozen:
        return Path(executable_path).resolve().parent
    return Path(module_file).resolve().parent


def startup_error_log_path(
    *,
    is_frozen: bool,
    executable_path: str,
    module_file: str,
) -> Path:
    return startup_output_root(
        is_frozen=is_frozen,
        executable_path=executable_path,
        module_file=module_file,
    ) / "startup_error.txt"


def write_startup_error_log(
    app_title: str,
    title: str,
    message: str,
    *,
    is_frozen: bool,
    executable_path: str,
    module_file: str,
) -> Path | None:
    log_path = startup_error_log_path(
        is_frozen=is_frozen,
        executable_path=executable_path,
        module_file=module_file,
    )
    try:
        log_path.write_text(
            f"{app_title}\n{title}\n\n{message}\n",
            encoding="utf-8",
        )
        return log_path
    except Exception:
        return None


def write_startup_error_page(
    page_builder: Callable[[str, str], str],
    title: str,
    message: str,
) -> Path | None:
    filename = f"2048_endgame_tablebase_startup_error_{os.getpid()}.html"
    page_path = Path(tempfile.gettempdir()) / filename
    try:
        page_path.write_text(page_builder(title, message), encoding="utf-8")
        return page_path
    except Exception:
        return None


def open_startup_error_page(
    page_builder: Callable[[str, str], str],
    title: str,
    message: str,
) -> bool:
    page_path = write_startup_error_page(page_builder, title, message)
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


def show_external_text_dialog(app_title: str, title: str, message: str) -> bool:
    dialog_message = f"{title}\n\n{message}"
    commands = [
        [
            "zenity",
            "--error",
            "--width=700",
            "--height=460",
            "--title",
            app_title,
            "--text",
            dialog_message,
        ],
        [
            "kdialog",
            "--title",
            app_title,
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

