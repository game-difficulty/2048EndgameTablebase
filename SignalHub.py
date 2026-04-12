"""Cross-module signal hub."""

from __future__ import annotations

import threading
from typing import Callable, List


class _Signal:
    def __init__(self) -> None:
        self._callbacks: List[Callable[..., None]] = []
        self._lock = threading.RLock()

    def connect(self, callback: Callable[..., None]) -> Callable[..., None]:
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
        return callback

    def disconnect(self, callback: Callable[..., None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def emit(self, *args, **kwargs) -> None:
        with self._lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            callback(*args, **kwargs)


class ProgressSignal:
    def __init__(self) -> None:
        self.progress_updated = _Signal()


class PracticeSignal:
    def __init__(self) -> None:
        self.board_update = _Signal()


progress_signal = ProgressSignal()
practice_signal = PracticeSignal()

