from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from Config import SingletonConfig, category_info, pattern_catalog
from engine_core.BookReaderAD import BookReaderAD
from engine_core.BookReaderBC import BookReaderBC
from engine_core.BookReaderEX import BookReaderEX
from engine_core.BookReaderEXAD import BookReaderEXAD

try:
    from native_core import formation_core
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


def _target_rank(target: str | int) -> int:
    value = int(target)
    if value >= 32 and (value & (value - 1)) == 0:
        return int(np.log2(value))
    return value


class BookReader:
    _native_readers: dict[str, Any] = {}

    @staticmethod
    def gen_all_mirror(pattern: str) -> list[tuple[str, str, Callable[[np.ndarray], np.ndarray]]]:
        # Keep the legacy tester randomization contract: variant patterns are not
        # symmetrically remapped, and LL only rotates.
        if pattern in category_info.get("variant", []):
            return [("none", "none", lambda board: board)]

        operations = [
            ("none", "none", lambda board: board),
            ("rotate_90", "none", lambda board: np.rot90(board)),
            ("rotate_180", "none", lambda board: np.rot90(board, k=2)),
            ("rotate_270", "none", lambda board: np.rot90(board, k=3)),
            ("none", "horizontal", lambda board: np.flip(board, axis=1)),
            ("rotate_90", "horizontal", lambda board: np.flip(np.rot90(board), axis=1)),
            ("rotate_180", "horizontal", lambda board: np.flip(np.rot90(board, k=2), axis=1)),
            ("rotate_270", "horizontal", lambda board: np.flip(np.rot90(board, k=3), axis=1)),
        ]
        return operations if pattern != "LL" else operations[:4]

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
        pattern_key = SingletonConfig.get_pattern_key(pattern_full, spawn_rate4)
        path_list = SingletonConfig().config["filepath_map"].get(pattern_key, [])
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
        self.book_reader_ex: BookReaderEX | None = None
        self.book_reader_exad: BookReaderEXAD | None = None
        self.book_reader_bc: BookReaderBC | None = None
        self.use_ad = False
        self.use_ex = False
        self.use_exad = False
        self.use_bc = False

    def set_book_reader_ad(self, pattern: str, target: int):
        if self.book_reader_ad is not None:
            if pattern == self.book_reader_ad.pattern and target == self.book_reader_ad.target:
                return
        self.book_reader_ad = BookReaderAD(pattern, target)

    def set_book_reader_ex(self, pattern: str, target: int):
        if self.book_reader_ex is not None and pattern == self.book_reader_ex.pattern and target == self.book_reader_ex.target:
            return
        self.book_reader_ex = BookReaderEX(pattern, target)

    def set_book_reader_exad(self, pattern: str, target: int):
        if self.book_reader_exad is not None:
            if pattern == self.book_reader_exad.pattern and target == self.book_reader_exad.target:
                return
        self.book_reader_exad = BookReaderEXAD(pattern, target)

    def set_book_reader_bc(self, pattern: str, target: int):
        if self.book_reader_bc is not None:
            if pattern == self.book_reader_bc.pattern and target == self.book_reader_bc.target:
                return
        self.book_reader_bc = BookReaderBC(pattern, target)

    def move_on_dic(
        self,
        board: NDArray,
        pattern: str,
        target: str,
        pattern_full: str,
    ) -> tuple[dict[str, str | float | int | None], str]:
        if self.use_exad and self.book_reader_exad is not None:
            return self.book_reader_exad.move_on_dic(board, pattern_full)
        if self.use_ad and self.book_reader_ad is not None:
            return self.book_reader_ad.move_on_dic(board, pattern_full)
        if self.use_ex and self.book_reader_ex is not None:
            return self.book_reader_ex.move_on_dic(board, pattern_full)
        if self.use_bc and self.book_reader_bc is not None:
            return self.book_reader_bc.move_on_dic(board, pattern_full)
        return self._book_reader.move_on_dic(board, pattern, target, pattern_full)

    def get_random_state(self, path_list: list, pattern_full: str):
        if self.use_exad and self.book_reader_exad is not None:
            return self.book_reader_exad.get_random_state(path_list, pattern_full)
        if self.use_ad and self.book_reader_ad is not None:
            return self.book_reader_ad.get_random_state(path_list, pattern_full)
        if self.use_ex and self.book_reader_ex is not None:
            return self.book_reader_ex.get_random_state(path_list, pattern_full)
        if self.use_bc and self.book_reader_bc is not None:
            return self.book_reader_bc.get_random_state(path_list, pattern_full)
        return self._book_reader.get_random_state(path_list, pattern_full)

    def dispatch(self, path_list: list, pattern: str, target: str | int):
        try:
            target = _target_rank(target)
        except ValueError:
            return
        if not pattern or not target:
            return

        prefer_bc = str(SingletonConfig().config.get("algorithm_mode", "")).lower() == "bc"
        found_ad = False
        found_ex = False
        found_exad = False
        found_bc = False
        has_ex_lut = False
        has_ex_layer = False
        has_exad_lut = False
        has_exad_layer = False
        bc_positions: set[str] = set()
        bc_successes: set[str] = set()
        ex_prefix = f"{pattern}_{2 ** target}_"
        for path, _success_rate_dtype in path_list:
            if not os.path.exists(path):
                continue

            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.name == f"{ex_prefix}.zlut":
                        has_ex_lut = True
                    elif entry.name == f"{ex_prefix}.exadlut":
                        has_exad_lut = True
                    elif entry.name.startswith(ex_prefix) and (
                        entry.name.endswith(".zbook") or entry.name.endswith(".exzbook")
                    ):
                        has_ex_layer = True
                    elif entry.name.startswith(ex_prefix) and (
                        entry.name.endswith(".exadbook") or entry.name.endswith(".exadzbook")
                    ):
                        has_exad_layer = True
                    elif entry.name.startswith(ex_prefix) and entry.name.endswith(".bccmp"):
                        found_bc = True
                    elif entry.name.startswith(ex_prefix) and entry.name.endswith(".bcpos"):
                        bc_positions.add(entry.name[:-len(".bcpos")])
                    elif entry.name.startswith(ex_prefix) and entry.name.endswith(".bcsuc"):
                        bc_successes.add(entry.name[:-len(".bcsuc")])
                    for rank in (1, 0.75, 0.5, 0.25):
                        if entry.name.endswith(f"_{int(2 ** target * rank)}b"):
                            found_ad = True
                            break
                    if bc_positions.intersection(bc_successes):
                        found_bc = True
                    if has_exad_lut and has_exad_layer:
                        found_exad = True
                        break
                    if has_ex_lut and has_ex_layer:
                        found_ex = True
                        break
            if prefer_bc and found_bc:
                break
            if found_exad and not prefer_bc:
                break
            if found_ex and not prefer_bc:
                break

        if prefer_bc and found_bc:
            self.use_ad = False
            self.use_ex = False
            self.use_exad = False
            self.book_reader_ad = None
            self.book_reader_ex = None
            self.book_reader_exad = None
            self.set_book_reader_bc(pattern, target)
            self.use_bc = self.book_reader_bc is not None
            return

        if found_exad:
            self.use_ad = False
            self.use_ex = False
            self.use_bc = False
            self.book_reader_ad = None
            self.book_reader_ex = None
            self.book_reader_bc = None
            self.set_book_reader_exad(pattern, target)
            self.use_exad = self.book_reader_exad is not None
            return

        if found_ex:
            self.use_ad = False
            self.use_exad = False
            self.use_bc = False
            self.book_reader_ad = None
            self.book_reader_exad = None
            self.book_reader_bc = None
            self.set_book_reader_ex(pattern, target)
            self.use_ex = self.book_reader_ex is not None
            return

        self.use_ex = False
        self.use_exad = False
        self.book_reader_ex = None
        self.book_reader_exad = None
        if not found_ad:
            self.use_ad = False
            self.book_reader_ad = None
            if found_bc:
                self.set_book_reader_bc(pattern, target)
                self.use_bc = self.book_reader_bc is not None
                return
            self.use_bc = False
            self.book_reader_bc = None
            return

        self.use_bc = False
        self.book_reader_exad = None
        self.book_reader_bc = None
        self.set_book_reader_ad(pattern, target)
        self.use_ad = self.book_reader_ad is not None
