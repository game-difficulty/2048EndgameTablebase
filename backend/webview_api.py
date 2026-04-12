import webview
import webbrowser


class Api:
    @staticmethod
    def _pick_dialog(dialog_type, *args, **kwargs):
        if not webview.windows:
            return None
        result = webview.windows[0].create_file_dialog(dialog_type, *args, **kwargs)  # type: ignore
        if not result:
            return None
        if isinstance(result, (tuple, list)):
            return result[0] if result else None
        return result

    @staticmethod
    def _pick_dialog_many(dialog_type, *args, **kwargs):
        if not webview.windows:
            return []
        result = webview.windows[0].create_file_dialog(dialog_type, *args, **kwargs)  # type: ignore
        if not result:
            return []
        if isinstance(result, (tuple, list)):
            return list(result)
        return [result]

    @staticmethod
    def _dialog_type(name, fallback):
        dialog_api = getattr(webview, "FileDialog", None)
        return getattr(dialog_api, name, fallback) if dialog_api is not None else fallback

    def select_folder(self):
        try:
            return self._pick_dialog(self._dialog_type("FOLDER", webview.FOLDER_DIALOG))
        except Exception as e:
            print("Select folder err:", e)
            return None

    def select_open_record(self):
        try:
            return self._pick_dialog(
                self._dialog_type("OPEN", webview.OPEN_DIALOG),
                allow_multiple=False,
                file_types=("Record Files (*.rec)", "All Files (*.*)"),
            )
        except Exception as e:
            print("Select open record err:", e)
            return None

    def select_open_replay_file(self):
        try:
            return self._pick_dialog(
                self._dialog_type("OPEN", webview.OPEN_DIALOG),
                allow_multiple=False,
                file_types=("Replay Files (*.rpl)", "All Files (*.*)"),
            )
        except Exception as e:
            print("Select open replay err:", e)
            return None

    def select_analysis_files(self):
        try:
            return self._pick_dialog_many(
                self._dialog_type("OPEN", webview.OPEN_DIALOG),
                allow_multiple=True,
                file_types=(
                    "Supported Files (*.txt;*.vrs)",
                    "Text Files (*.txt)",
                    "VRS Files (*.vrs)",
                    "All Files (*.*)",
                ),
            )
        except Exception as e:
            print("Select analysis files err:", e)
            return []

    def select_save_record(self):
        try:
            return self._pick_dialog(
                self._dialog_type("SAVE", webview.SAVE_DIALOG),
                file_types=("Record Files (*.rec)", "All Files (*.*)"),
            )
        except Exception as e:
            print("Select save record err:", e)
            return None

    def select_save_tester_log(self):
        try:
            return self._pick_dialog(
                self._dialog_type("SAVE", webview.SAVE_DIALOG),
                file_types=("Text Files (*.txt)", "All Files (*.*)"),
            )
        except Exception as e:
            print("Select save tester log err:", e)
            return None

    def select_save_tester_replay(self):
        try:
            return self._pick_dialog(
                self._dialog_type("SAVE", webview.SAVE_DIALOG),
                file_types=("Replay Files (*.rpl)", "All Files (*.*)"),
            )
        except Exception as e:
            print("Select save tester replay err:", e)
            return None

    def open_external_url(self, url):
        try:
            if not url:
                return False
            webbrowser.open(str(url), new=2)
            return True
        except Exception as e:
            print("Open external url err:", e)
            return False
