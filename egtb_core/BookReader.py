from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from Config import SingletonConfig, category_info, pattern_catalog
from egtb_core.BookReaderAD import BookReaderAD

try:
    from ai_and_sort import formation_core
except Exception:
    formation_core = None


def _require_native_reader():
    if formation_core is None:
        raise RuntimeError("formation_core is unavailable")


_SYMM_MODE_BY_NAME = {
    "identity": formation_core.SymmMode.Identity if formation_core else 0,
    "full": formation_core.SymmMode.Full if formation_core else 1,
    "diagonal": formation_core.SymmMode.Diagonal if formation_core else 2,
    "horizontal": formation_core.SymmMode.Horizontal if formation_core else 3,
    "min33": formation_core.SymmMode.Min33 if formation_core else 4,
    "min24": formation_core.SymmMode.Min24 if formation_core else 5,
    "min34": formation_core.SymmMode.Min34 if formation_core else 6,
}


def _symm_mode_value(name: str) -> int:
    mode = _SYMM_MODE_BY_NAME.get(name, 0)
    return int(mode.value if hasattr(mode, "value") else mode)


class BookReader:
    _native_readers: dict[str, Any] = {}

    @classmethod
    def _get_native_reader(cls, pattern: str):
        _require_native_reader()
        reader = cls._native_readers.get(pattern)
        if reader is not None:
            return reader

        meta = pattern_catalog.get(pattern)
        if meta is None:
            return None

        pattern_spec = formation_core.PatternSpec()
        pattern_spec.name = pattern
        pattern_spec.pattern_masks = list(meta.get("pattern_masks", ()))
        pattern_spec.success_shifts = list(meta.get("success_shifts", ()))
        pattern_spec.symm_mode = _symm_mode_value(meta.get("canonical_mode", "identity"))
        reader = formation_core.ClassicBookReader(
            pattern_spec,
            pattern in category_info.get("variant", []),
        )
        cls._native_readers[pattern] = reader
        return reader

    @classmethod
    def move_on_dic(
        cls,
        board: NDArray,
        pattern: str,
        target: str,
        pattern_full: str,
    ) -> tuple[dict[str, str | float | int | None], str]:
        del target
        meta = pattern_catalog.get(pattern)
        reader = cls._get_native_reader(pattern)
        spawn_rate4 = SingletonConfig().config["4_spawn_rate"]
        path_list = SingletonConfig().config["filepath_map"].get((pattern_full, spawn_rate4), [])
        if meta is None or reader is None:
            return {"?": "?"}, ""
        return reader.move_on_dic(
            board.tolist(),
            path_list,
            pattern_full,
            int(meta.get("nums_adjust", 0)),
        )

    @classmethod
    def get_random_state(cls, path_list: list, pattern_full: str) -> np.uint64:
        pattern = pattern_full.split("_", 1)[0]
        reader = cls._get_native_reader(pattern)
        if reader is None:
            return np.uint64(0)
        return np.uint64(
            reader.get_random_state(
                path_list,
                pattern_full,
                float(SingletonConfig().config["4_spawn_rate"]),
            )
        )


class BookReaderDispatcher:
    _book_reader = BookReader

    def __init__(self):
        self.book_reader_ad: BookReaderAD | None = None
        self.use_ad = False

    def set_book_reader_ad(self, pattern: str, target: int):
        if self.book_reader_ad is not None:
            if pattern == self.book_reader_ad.pattern and target == self.book_reader_ad.target:
                return
        self.book_reader_ad = BookReaderAD(pattern, target)

    def move_on_dic(
        self,
        board: NDArray,
        pattern: str,
        target: str,
        pattern_full: str,
    ) -> tuple[dict[str, str | float | int | None], str]:
        if self.use_ad and self.book_reader_ad is not None:
            return self.book_reader_ad.move_on_dic(board, pattern_full)
        return self._book_reader.move_on_dic(board, pattern, target, pattern_full)

    def get_random_state(self, path_list: list, pattern_full: str):
        if self.use_ad and self.book_reader_ad is not None:
            return self.book_reader_ad.get_random_state(path_list, pattern_full)
        return self._book_reader.get_random_state(path_list, pattern_full)

    def dispatch(self, path_list: list, pattern: str, target: str | int):
        try:
            target = int(target)
            if target >= 128:
                target = int(np.log2(target))
        except ValueError:
            return
        if not pattern or not target:
            return

        found = False
        for path, _success_rate_dtype in path_list:
            if not os.path.exists(path):
                continue

            with os.scandir(path) as entries:
                for entry in entries:
                    for rank in (1, 0.75, 0.5, 0.25):
                        if entry.name.endswith(f"_{int(2 ** target * rank)}b"):
                            found = True
                            break
                    if found:
                        break
            if found:
                break

        if not found:
            self.use_ad = False
            return

        self.set_book_reader_ad(pattern, target)
        self.use_ad = self.book_reader_ad is not None