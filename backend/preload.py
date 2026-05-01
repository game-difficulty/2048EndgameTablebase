from __future__ import annotations

import threading
import traceback

import numpy as np


_preload_lock = threading.Lock()
_preload_thread: threading.Thread | None = None
_preload_finished = False


def _run_preload() -> None:
    global _preload_finished

    try:
        from Config import SingletonConfig
        import engine_core.BoardMover as bm
        import engine_core.VBoardMover as vbm

        SingletonConfig()

        regular_board = np.uint64(0x10120342216902AC)
        variant_board = np.uint64(0x10120342216FF2AC)

        bm.move_all_dir(regular_board)
        for direction in (1, 2, 3, 4):
            bm.decode_board(bm.s_move_board(regular_board, direction)[0])

        vbm.move_all_dir(variant_board)
        for direction in (1, 2, 3, 4):
            vbm.decode_board(vbm.s_move_board(variant_board, direction)[0])

        print("Preloading complete.")
    except Exception as exc:
        print(f"Preloading failed: {exc}")
        traceback.print_exc()
    finally:
        with _preload_lock:
            _preload_finished = True


def start_preload_thread() -> threading.Thread | None:
    global _preload_thread

    with _preload_lock:
        if _preload_finished:
            return _preload_thread
        if _preload_thread is not None and _preload_thread.is_alive():
            return _preload_thread

        _preload_thread = threading.Thread(
            target=_run_preload,
            name="runtime-preload",
            daemon=True,
        )
        _preload_thread.start()
        return _preload_thread
