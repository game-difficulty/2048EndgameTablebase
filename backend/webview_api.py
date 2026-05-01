from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
import webbrowser

import webview

if os.name == "nt":
    import ctypes
    from ctypes import wintypes


_FILTER_PATTERN = re.compile(r"^\s*(.*?)\s*\((.*?)\)\s*$")
_WINDOWS_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_GTK_DIALOG_LOCK = threading.Lock()

if os.name == "nt":
    _HRESULT = ctypes.c_long
    _ULONG = ctypes.c_ulong
    _DWORD = wintypes.DWORD
    _UINT = wintypes.UINT
    _LPCWSTR = wintypes.LPCWSTR
    _LPWSTR = wintypes.LPWSTR
    _LPVOID = wintypes.LPVOID
    _HWND = wintypes.HWND

    _COINIT_APARTMENTTHREADED = 0x2
    _COINIT_DISABLE_OLE1DDE = 0x4
    _CLSCTX_INPROC_SERVER = 0x1

    _FOS_OVERWRITEPROMPT = 0x00000002
    _FOS_PICKFOLDERS = 0x00000020
    _FOS_FORCEFILESYSTEM = 0x00000040
    _FOS_PATHMUSTEXIST = 0x00000800
    _FOS_FILEMUSTEXIST = 0x00001000
    _FOS_ALLOWMULTISELECT = 0x00000200

    _SIGDN_FILESYSPATH = 0x80058000
    _ERROR_CANCELLED = 0x800704C7

    class _GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_uint32),
            ("Data2", ctypes.c_uint16),
            ("Data3", ctypes.c_uint16),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class _COMDLG_FILTERSPEC(ctypes.Structure):
        _fields_ = [("pszName", _LPCWSTR), ("pszSpec", _LPCWSTR)]

    def _make_guid(text: str) -> _GUID:
        value = uuid.UUID(text)
        raw = value.bytes_le
        return _GUID.from_buffer_copy(raw)

    _CLSID_FILE_OPEN_DIALOG = _make_guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")
    _CLSID_FILE_SAVE_DIALOG = _make_guid("C0B4E2F3-BA21-4773-8DBA-335EC946EB8B")
    _IID_IFILE_OPEN_DIALOG = _make_guid("D57C7288-D4AD-4768-BE02-9D969532D960")
    _IID_IFILE_SAVE_DIALOG = _make_guid("84BCCD23-5FDE-4CDB-AEA4-AF64B83D78AB")

    _ole32 = ctypes.OleDLL("ole32")
    _user32 = ctypes.WinDLL("user32", use_last_error=True)

    _ole32.CoInitializeEx.argtypes = [_LPVOID, _DWORD]
    _ole32.CoInitializeEx.restype = _HRESULT
    _ole32.CoUninitialize.argtypes = []
    _ole32.CoUninitialize.restype = None
    _ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(_GUID),
        _LPVOID,
        _DWORD,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(_LPVOID),
    ]
    _ole32.CoCreateInstance.restype = _HRESULT
    _ole32.CoTaskMemFree.argtypes = [_LPVOID]
    _ole32.CoTaskMemFree.restype = None
    _user32.GetForegroundWindow.argtypes = []
    _user32.GetForegroundWindow.restype = _HWND

    def _hresult_code(value: int) -> int:
        return ctypes.c_ulong(value).value

    def _check_hresult(value: int) -> None:
        if value < 0:
            raise OSError(f"HRESULT 0x{_hresult_code(value):08X}")

    class _ComObject:
        def __init__(self, ptr: _LPVOID):
            self.ptr = ptr

        def call(self, index: int, restype, *argtypes_and_values):
            if not self.ptr:
                raise OSError("COM pointer is null")
            vtbl = ctypes.cast(self.ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            prototype = ctypes.WINFUNCTYPE(
                restype,
                _LPVOID,
                *(argtype for argtype, _ in argtypes_and_values),
            )
            method = prototype(vtbl[index])
            return method(self.ptr, *(value for _, value in argtypes_and_values))

        def release(self) -> None:
            if self.ptr:
                self.call(2, _ULONG)
                self.ptr = _LPVOID()


class Api:
    @staticmethod
    def _has_webview_window() -> bool:
        return bool(getattr(webview, "windows", None))

    @classmethod
    def _parse_file_types(
        cls, file_types: tuple[str, ...] | list[str] | str | None
    ) -> list[tuple[str, list[str]]]:
        if file_types is None:
            return [("All Files", ["*.*"])]

        raw_entries: list[str]
        if isinstance(file_types, str):
            raw_entries = [file_types]
        else:
            raw_entries = [str(entry or "").strip() for entry in file_types]

        parsed: list[tuple[str, list[str]]] = []
        for entry in raw_entries:
            if not entry:
                continue
            match = _FILTER_PATTERN.match(entry)
            if match:
                label = match.group(1).strip() or "Files"
                patterns = [
                    part.strip()
                    for part in re.split(r"[;,]", match.group(2))
                    if part.strip()
                ]
            else:
                label = entry
                patterns = [entry]

            if patterns:
                parsed.append((label, patterns))

        return parsed or [("All Files", ["*.*"])]

    @classmethod
    def _kdialog_filter_string(
        cls, file_types: tuple[str, ...] | list[str] | str | None
    ) -> str:
        return "\n".join(
            f"{' '.join(patterns)}|{label}"
            for label, patterns in cls._parse_file_types(file_types)
        )

    @staticmethod
    def _run_native_command(args: list[str]) -> str | None:
        run_kwargs = {
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": 300,
            "check": False,
        }
        if os.name == "nt":
            run_kwargs["creationflags"] = _WINDOWS_NO_WINDOW

        try:
            completed = subprocess.run(args, **run_kwargs)
        except Exception as exc:
            print("Native dialog err:", exc)
            return None

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode != 0 and stderr:
            print("Native dialog err:", stderr)
        return stdout or None

    @classmethod
    def _pick_via_windows_common_dialog(
        cls,
        kind: str,
        *,
        allow_multiple: bool = False,
        file_types: tuple[str, ...] | list[str] | str | None = None,
    ) -> str | list[str] | None:
        result_holder: dict[str, str | list[str] | None] = {}
        error_holder: dict[str, Exception] = {}

        def runner() -> None:
            initialized = False
            try:
                hr = _ole32.CoInitializeEx(
                    None,
                    _COINIT_APARTMENTTHREADED | _COINIT_DISABLE_OLE1DDE,
                )
                if hr not in (0, 1):
                    _check_hresult(hr)
                initialized = True
                result_holder["value"] = cls._pick_via_windows_common_dialog_sta(
                    kind,
                    allow_multiple=allow_multiple,
                    file_types=file_types,
                )
            except Exception as exc:
                error_holder["value"] = exc
            finally:
                if initialized:
                    _ole32.CoUninitialize()

        thread = threading.Thread(target=runner, name="native-file-dialog")
        thread.start()
        thread.join()

        if "value" in error_holder:
            raise error_holder["value"]
        return result_holder.get("value", [] if allow_multiple else None)

    @classmethod
    def _pick_via_windows_common_dialog_sta(
        cls,
        kind: str,
        *,
        allow_multiple: bool = False,
        file_types: tuple[str, ...] | list[str] | str | None = None,
    ) -> str | list[str] | None:
        dialog_ptr = _LPVOID()
        clsid = _CLSID_FILE_SAVE_DIALOG if kind == "save" else _CLSID_FILE_OPEN_DIALOG
        iid = _IID_IFILE_SAVE_DIALOG if kind == "save" else _IID_IFILE_OPEN_DIALOG
        _check_hresult(
            _ole32.CoCreateInstance(
                ctypes.byref(clsid),
                None,
                _CLSCTX_INPROC_SERVER,
                ctypes.byref(iid),
                ctypes.byref(dialog_ptr),
            )
        )

        dialog = _ComObject(dialog_ptr)
        try:
            options = _DWORD()
            _check_hresult(
                dialog.call(
                    10,
                    _HRESULT,
                    (ctypes.POINTER(_DWORD), ctypes.byref(options)),
                )
            )

            updated_options = options.value | _FOS_FORCEFILESYSTEM | _FOS_PATHMUSTEXIST
            if kind == "folder":
                updated_options |= _FOS_PICKFOLDERS
            elif kind == "save":
                updated_options |= _FOS_OVERWRITEPROMPT
            else:
                updated_options |= _FOS_FILEMUSTEXIST
                if allow_multiple:
                    updated_options |= _FOS_ALLOWMULTISELECT

            _check_hresult(
                dialog.call(
                    9,
                    _HRESULT,
                    (_DWORD, _DWORD(updated_options)),
                )
            )

            filter_specs: list[tuple[str, str]] = []
            if kind != "folder":
                for label, patterns in cls._parse_file_types(file_types):
                    filter_specs.append((label, ";".join(patterns)))

            if filter_specs:
                filters = (_COMDLG_FILTERSPEC * len(filter_specs))()
                filter_names = [ctypes.c_wchar_p(label) for label, _ in filter_specs]
                filter_patterns = [ctypes.c_wchar_p(spec) for _, spec in filter_specs]
                for index, (label, spec) in enumerate(filter_specs):
                    filters[index].pszName = filter_names[index]
                    filters[index].pszSpec = filter_patterns[index]

                _check_hresult(
                    dialog.call(
                        4,
                        _HRESULT,
                        (_UINT, _UINT(len(filter_specs))),
                        (ctypes.POINTER(_COMDLG_FILTERSPEC), filters),
                    )
                )
                _check_hresult(
                    dialog.call(
                        5,
                        _HRESULT,
                        (_UINT, _UINT(1)),
                    )
                )

            owner_handle = _user32.GetForegroundWindow()
            result = dialog.call(
                3,
                _HRESULT,
                (_HWND, owner_handle),
            )
            if _hresult_code(result) == _ERROR_CANCELLED:
                return [] if allow_multiple else None
            _check_hresult(result)

            if kind != "save" and allow_multiple:
                return cls._windows_shell_item_array_paths(dialog)

            return cls._windows_shell_item_result(dialog)
        finally:
            dialog.release()

    @staticmethod
    def _windows_shell_item_result(dialog: _ComObject) -> str | None:
        item_ptr = _LPVOID()
        _check_hresult(
            dialog.call(
                20,
                _HRESULT,
                (ctypes.POINTER(_LPVOID), ctypes.byref(item_ptr)),
            )
        )
        item = _ComObject(item_ptr)
        try:
            return Api._windows_shell_item_path(item)
        finally:
            item.release()

    @staticmethod
    def _windows_shell_item_array_paths(dialog: _ComObject) -> list[str]:
        array_ptr = _LPVOID()
        _check_hresult(
            dialog.call(
                27,
                _HRESULT,
                (ctypes.POINTER(_LPVOID), ctypes.byref(array_ptr)),
            )
        )
        shell_items = _ComObject(array_ptr)
        try:
            count = _DWORD()
            _check_hresult(
                shell_items.call(
                    7,
                    _HRESULT,
                    (ctypes.POINTER(_DWORD), ctypes.byref(count)),
                )
            )
            paths: list[str] = []
            for index in range(count.value):
                item_ptr = _LPVOID()
                _check_hresult(
                    shell_items.call(
                        8,
                        _HRESULT,
                        (_DWORD, _DWORD(index)),
                        (ctypes.POINTER(_LPVOID), ctypes.byref(item_ptr)),
                    )
                )
                item = _ComObject(item_ptr)
                try:
                    path = Api._windows_shell_item_path(item)
                    if path:
                        paths.append(path)
                finally:
                    item.release()
            return paths
        finally:
            shell_items.release()

    @staticmethod
    def _windows_shell_item_path(item: _ComObject) -> str | None:
        name_ptr = _LPWSTR()
        _check_hresult(
            item.call(
                5,
                _HRESULT,
                (_DWORD, _DWORD(_SIGDN_FILESYSPATH)),
                (ctypes.POINTER(_LPWSTR), ctypes.byref(name_ptr)),
            )
        )
        try:
            return str(name_ptr.value) if name_ptr.value else None
        finally:
            _ole32.CoTaskMemFree(ctypes.cast(name_ptr, _LPVOID))

    @classmethod
    def _pick_via_zenity_family(
        cls,
        executable: str,
        kind: str,
        *,
        allow_multiple: bool = False,
        file_types: tuple[str, ...] | list[str] | str | None = None,
    ) -> str | list[str] | None:
        executable_name = os.path.basename(executable).lower()
        args = [executable, "--file-selection", "--modal"]
        if executable_name in {"yad", "qarma"}:
            args.append("--on-top")
        if kind == "folder":
            args.append("--directory")
        elif kind == "save":
            args.extend(["--save", "--confirm-overwrite"])
        elif allow_multiple:
            args.extend(["--multiple", "--separator=\n"])

        if kind != "folder":
            for label, patterns in cls._parse_file_types(file_types):
                args.append(f"--file-filter={label} | {' '.join(patterns)}")

        output = cls._run_native_command(args)
        if not output:
            return [] if allow_multiple else None
        if allow_multiple:
            return [line for line in output.splitlines() if line.strip()]
        return output

    @classmethod
    def _pick_via_gtk_dialog(
        cls,
        kind: str,
        *,
        allow_multiple: bool = False,
        file_types: tuple[str, ...] | list[str] | str | None = None,
    ) -> str | list[str] | None:
        result_holder: dict[str, str | list[str] | None] = {}
        error_holder: dict[str, Exception] = {}

        def runner() -> None:
            try:
                import gi

                gi.require_version("Gtk", "3.0")
                from gi.repository import Gdk, Gtk

                init_result = Gtk.init_check()
                if isinstance(init_result, tuple):
                    init_ok = bool(init_result[0])
                else:
                    init_ok = bool(init_result)
                if not init_ok:
                    raise RuntimeError("GTK display initialization failed")

                action_map = {
                    "folder": Gtk.FileChooserAction.SELECT_FOLDER,
                    "save": Gtk.FileChooserAction.SAVE,
                    "open": Gtk.FileChooserAction.OPEN,
                }
                action = action_map[kind]
                accept_label = {
                    "folder": Gtk.STOCK_OK,
                    "save": Gtk.STOCK_SAVE,
                    "open": Gtk.STOCK_OPEN,
                }[kind]

                dialog = Gtk.FileChooserDialog(
                    title=None,
                    parent=None,
                    action=action,
                )
                try:
                    dialog.add_buttons(
                        Gtk.STOCK_CANCEL,
                        Gtk.ResponseType.CANCEL,
                        accept_label,
                        Gtk.ResponseType.ACCEPT,
                    )
                    dialog.set_modal(True)
                    dialog.set_keep_above(True)
                    dialog.set_skip_taskbar_hint(True)
                    dialog.set_skip_pager_hint(True)
                    dialog.set_urgency_hint(True)
                    dialog.set_type_hint(Gdk.WindowTypeHint.DIALOG)
                    dialog.set_position(Gtk.WindowPosition.CENTER_ALWAYS)
                    dialog.set_resizable(True)
                    dialog.set_local_only(True)

                    if kind == "save":
                        dialog.set_do_overwrite_confirmation(True)
                        dialog.set_create_folders(True)
                    elif kind == "open":
                        dialog.set_select_multiple(allow_multiple)

                    if kind != "folder":
                        for label, patterns in cls._parse_file_types(file_types):
                            file_filter = Gtk.FileFilter()
                            file_filter.set_name(label)
                            normalized_patterns: list[str] = []
                            for pattern in patterns:
                                normalized_patterns.append(pattern)
                                if pattern == "*.*":
                                    normalized_patterns.append("*")
                            for pattern in dict.fromkeys(normalized_patterns):
                                file_filter.add_pattern(pattern)
                            dialog.add_filter(file_filter)

                    dialog.present()
                    response = dialog.run()
                    if response != Gtk.ResponseType.ACCEPT:
                        result_holder["value"] = [] if allow_multiple else None
                        return

                    if kind == "open" and allow_multiple:
                        result_holder["value"] = list(dialog.get_filenames())
                    else:
                        result_holder["value"] = dialog.get_filename()
                finally:
                    dialog.destroy()
                    while Gtk.events_pending():
                        Gtk.main_iteration()
            except Exception as exc:
                error_holder["value"] = exc

        with _GTK_DIALOG_LOCK:
            thread = threading.Thread(target=runner, name="linux-gtk-file-dialog")
            thread.start()
            thread.join()

        if "value" in error_holder:
            raise error_holder["value"]
        return result_holder.get("value", [] if allow_multiple else None)

    @classmethod
    def _pick_via_kdialog(
        cls,
        executable: str,
        kind: str,
        *,
        allow_multiple: bool = False,
        file_types: tuple[str, ...] | list[str] | str | None = None,
    ) -> str | list[str] | None:
        home_dir = os.path.expanduser("~")
        filter_string = cls._kdialog_filter_string(file_types)
        if kind == "folder":
            args = [executable, "--title", "Select Folder", "--getexistingdirectory", home_dir]
        elif kind == "save":
            args = [executable, "--title", "Save File", "--getsavefilename", home_dir, filter_string]
        else:
            args = [executable, "--title", "Open File", "--getopenfilename", home_dir, filter_string]
            if allow_multiple:
                args.extend(["--multiple", "--separate-output"])

        output = cls._run_native_command(args)
        if not output:
            return [] if allow_multiple else None
        if allow_multiple:
            return [line for line in output.splitlines() if line.strip()]
        return output

    @classmethod
    def _pick_via_osascript(
        cls,
        kind: str,
        *,
        allow_multiple: bool = False,
    ) -> str | list[str] | None:
        if kind == "folder":
            script = 'POSIX path of (choose folder)'
        elif kind == "save":
            script = 'POSIX path of (choose file name)'
        elif allow_multiple:
            script = (
                "set selectedItems to choose file with multiple selections allowed true\n"
                "set outputLines to {}\n"
                "repeat with selectedItem in selectedItems\n"
                "set end of outputLines to POSIX path of selectedItem\n"
                "end repeat\n"
                "set text item delimiters to linefeed\n"
                "outputLines as text"
            )
        else:
            script = 'POSIX path of (choose file)'

        output = cls._run_native_command(["osascript", "-e", script])
        if not output:
            return [] if allow_multiple else None
        if allow_multiple:
            return [line for line in output.splitlines() if line.strip()]
        return output

    @classmethod
    def _pick_via_native(
        cls,
        kind: str,
        *,
        allow_multiple: bool = False,
        file_types: tuple[str, ...] | list[str] | str | None = None,
    ) -> str | list[str] | None:
        if os.name == "nt":
            return cls._pick_via_windows_common_dialog(
                kind, allow_multiple=allow_multiple, file_types=file_types
            )

        if sys.platform == "darwin":
            return cls._pick_via_osascript(kind, allow_multiple=allow_multiple)

        if os.name == "posix":
            try:
                return cls._pick_via_gtk_dialog(
                    kind,
                    allow_multiple=allow_multiple,
                    file_types=file_types,
                )
            except Exception as exc:
                print("GTK dialog err:", exc)

            for executable in ("zenity", "qarma", "yad"):
                resolved = shutil.which(executable)
                if resolved:
                    return cls._pick_via_zenity_family(
                        resolved,
                        kind,
                        allow_multiple=allow_multiple,
                        file_types=file_types,
                    )

            kdialog = shutil.which("kdialog")
            if kdialog:
                return cls._pick_via_kdialog(
                    kdialog,
                    kind,
                    allow_multiple=allow_multiple,
                    file_types=file_types,
                )

        return [] if allow_multiple else None

    @classmethod
    def _pick_via_webview(
        cls,
        kind: str,
        *,
        allow_multiple: bool = False,
        file_types: tuple[str, ...] | list[str] | str | None = None,
    ) -> str | list[str] | None:
        if not cls._has_webview_window():
            return [] if allow_multiple else None

        if kind == "folder":
            dialog_type = cls._dialog_type("FOLDER", webview.FOLDER_DIALOG)
            dialog_kwargs = {}
        elif kind == "save":
            dialog_type = cls._dialog_type("SAVE", webview.SAVE_DIALOG)
            dialog_kwargs = {"file_types": file_types}
        else:
            dialog_type = cls._dialog_type("OPEN", webview.OPEN_DIALOG)
            dialog_kwargs = {
                "allow_multiple": allow_multiple,
                "file_types": file_types,
            }

        result = webview.windows[0].create_file_dialog(
            dialog_type,
            **dialog_kwargs,  # type: ignore[arg-type]
        )
        if not result:
            return [] if allow_multiple else None
        if allow_multiple:
            return list(result) if isinstance(result, (tuple, list)) else [result]
        if isinstance(result, (tuple, list)):
            return result[0] if result else None
        return result

    @classmethod
    def _pick(
        cls,
        kind: str,
        *,
        allow_multiple: bool = False,
        file_types: tuple[str, ...] | list[str] | str | None = None,
    ) -> str | list[str] | None:
        if cls._has_webview_window():
            return cls._pick_via_webview(
                kind,
                allow_multiple=allow_multiple,
                file_types=file_types,
            )
        return cls._pick_via_native(
            kind,
            allow_multiple=allow_multiple,
            file_types=file_types,
        )

    @staticmethod
    def _dialog_type(name, fallback):
        dialog_api = getattr(webview, "FileDialog", None)
        return getattr(dialog_api, name, fallback) if dialog_api is not None else fallback

    def select_folder(self):
        try:
            return self._pick("folder")
        except Exception as exc:
            print("Select folder err:", exc)
            return None

    def select_open_record(self):
        try:
            return self._pick(
                "open",
                file_types=("Record Files (*.rec)", "All Files (*.*)"),
            )
        except Exception as exc:
            print("Select open record err:", exc)
            return None

    def select_open_replay_file(self):
        try:
            return self._pick(
                "open",
                file_types=("Replay Files (*.rpl)", "All Files (*.*)"),
            )
        except Exception as exc:
            print("Select open replay err:", exc)
            return None

    def select_analysis_files(self):
        try:
            result = self._pick(
                "open",
                allow_multiple=True,
                file_types=(
                    "Supported Files (*.txt;*.vrs)",
                    "Text Files (*.txt)",
                    "VRS Files (*.vrs)",
                    "All Files (*.*)",
                ),
            )
            return result if isinstance(result, list) else []
        except Exception as exc:
            print("Select analysis files err:", exc)
            return []

    def select_save_record(self):
        try:
            return self._pick(
                "save",
                file_types=("Record Files (*.rec)", "All Files (*.*)"),
            )
        except Exception as exc:
            print("Select save record err:", exc)
            return None

    def select_save_tester_log(self):
        try:
            return self._pick(
                "save",
                file_types=("Text Files (*.txt)", "All Files (*.*)"),
            )
        except Exception as exc:
            print("Select save tester log err:", exc)
            return None

    def select_save_tester_replay(self):
        try:
            return self._pick(
                "save",
                file_types=("Replay Files (*.rpl)", "All Files (*.*)"),
            )
        except Exception as exc:
            print("Select save tester replay err:", exc)
            return None

    def open_external_url(self, url):
        try:
            if not url:
                return False
            webbrowser.open(str(url), new=2)
            return True
        except Exception as exc:
            print("Open external url err:", exc)
            return False
