from __future__ import annotations

import os
import random
import re

import numpy as np

from Config import DTYPE_CONFIG, SingletonConfig, pattern_catalog
from engine_core import mover_runtime

try:
    from native_core import formation_core
except Exception:
    formation_core = None


class BookReaderBC:
    def __new__(cls, pattern: str, target: int):
        if str(pattern).startswith("free"):
            return super().__new__(cls)
        return None

    def __init__(self, pattern: str, target: int):
        if formation_core is None:
            raise RuntimeError("formation_core is unavailable")
        if not hasattr(formation_core, "sample_bc_compressed_random_state"):
            raise RuntimeError("formation_core does not expose BC compressed sampling")
        if not hasattr(formation_core, "lookup_bc_compressed_result_cold"):
            raise RuntimeError("formation_core does not expose BC compressed lookup")
        if not hasattr(formation_core, "sample_bc_exact_random_state"):
            raise RuntimeError("formation_core does not expose BC exact sampling")
        if not hasattr(formation_core, "lookup_bc_exact_result_cold"):
            raise RuntimeError("formation_core does not expose BC exact lookup")

        self.pattern = pattern
        self.target = int(np.log2(target)) if int(target) >= 128 else int(target)
        meta = pattern_catalog.get(pattern, {})
        self._seed_sum = self._pattern_seed_sum(pattern)
        self._canonical_mode = str(meta.get("canonical_mode", "identity"))
        self._pattern_masks = tuple(int(mask) for mask in meta.get("pattern_masks", ()))
        self._last_operation_index = 0

    @staticmethod
    def _encoded_board_sum(board: int) -> int:
        total = 0
        raw = int(board)
        for cell in range(16):
            tile = (raw >> (cell * 4)) & 0xF
            if tile:
                total += 1 << tile
        return total

    @classmethod
    def _pattern_seed_sum(cls, pattern: str) -> int:
        meta = pattern_catalog.get(pattern, {})
        nums_adjust = meta.get("nums_adjust")
        if nums_adjust is not None:
            return -int(nums_adjust)
        seeds = meta.get("seed_boards", ())
        if seeds:
            return cls._encoded_board_sum(int(seeds[0]))
        raise KeyError(f"Unknown BC pattern seed: {pattern}")

    @staticmethod
    def _dtype_name(dtype: int) -> str:
        return {
            1: "uint32",
            2: "uint64",
            3: "float32",
            4: "float64",
            5: "1-float32",
            6: "1-float64",
        }.get(int(dtype), "uint32")

    @staticmethod
    def _maybe_round_value(value: float, dtype_name: str) -> float:
        if "32" not in dtype_name or abs(value) <= 1e-7:
            return value
        return round(value, 9)

    @classmethod
    def _normalize_lookup_value(cls, lookup: dict) -> tuple[float | None, str]:
        dtype_name = cls._dtype_name(int(lookup.get("dtype", 1)))
        if not lookup.get("found"):
            return None, dtype_name
        raw_bits = int(lookup.get("raw_value_bits", 0))
        if dtype_name == "uint64":
            value = float(raw_bits) / float(DTYPE_CONFIG["uint64"][2])
        elif dtype_name == "uint32":
            value = float(raw_bits & 0xFFFFFFFF) / float(DTYPE_CONFIG["uint32"][2])
        else:
            value = float(lookup.get("numeric_value", 0.0))
        return cls._maybe_round_value(value, dtype_name), dtype_name

    @staticmethod
    def _compressed_candidates(path_list: list, pattern_full: str) -> list[tuple[int, str]]:
        prefix = f"{pattern_full}_"
        suffix = ".bccmp"
        candidates: list[tuple[int, str]] = []
        for path_entry in path_list:
            path = path_entry[0] if isinstance(path_entry, (list, tuple)) else path_entry
            if not path or not os.path.isdir(path):
                continue
            try:
                for name in os.listdir(path):
                    if not name.startswith(prefix) or not name.endswith(suffix):
                        continue
                    ordinal_text = name[len(prefix):-len(suffix)]
                    if not re.fullmatch(r"\d+", ordinal_text):
                        continue
                    candidates.append((int(ordinal_text), os.path.join(path, name)))
            except OSError:
                continue
        candidates.sort()
        return candidates

    @staticmethod
    def _compressed_path(path_list: list, pattern_full: str, ordinal: int) -> str | None:
        filename = f"{pattern_full}_{int(ordinal)}.bccmp"
        for path_entry in path_list:
            path = path_entry[0] if isinstance(path_entry, (list, tuple)) else path_entry
            if not path:
                continue
            candidate = os.path.join(path, filename)
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _exact_candidates(path_list: list, pattern_full: str) -> list[tuple[int, str]]:
        prefix = f"{pattern_full}_"
        suffix = ".bcpos"
        candidates: list[tuple[int, str]] = []
        for path_entry in path_list:
            path = path_entry[0] if isinstance(path_entry, (list, tuple)) else path_entry
            if not path or not os.path.isdir(path):
                continue
            try:
                for name in os.listdir(path):
                    if not name.startswith(prefix) or not name.endswith(suffix):
                        continue
                    ordinal_text = name[len(prefix):-len(suffix)]
                    if not re.fullmatch(r"\d+", ordinal_text):
                        continue
                    success_path = os.path.join(path, f"{prefix}{ordinal_text}.bcsuc")
                    if os.path.exists(success_path):
                        candidates.append((int(ordinal_text), os.path.join(path, name)))
            except OSError:
                continue
        candidates.sort()
        return candidates

    @staticmethod
    def _exact_paths(path_list: list, pattern_full: str, ordinal: int) -> tuple[str, str] | None:
        position_name = f"{pattern_full}_{int(ordinal)}.bcpos"
        success_name = f"{pattern_full}_{int(ordinal)}.bcsuc"
        for path_entry in path_list:
            path = path_entry[0] if isinstance(path_entry, (list, tuple)) else path_entry
            if not path:
                continue
            position_path = os.path.join(path, position_name)
            success_path = os.path.join(path, success_name)
            if os.path.exists(position_path) and os.path.exists(success_path):
                return position_path, success_path
        return None

    def _layer_ordinal(self, board: int) -> int | None:
        board_sum = self._encoded_board_sum(board)
        delta = board_sum - self._seed_sum
        if delta < 0 or (delta & 1):
            return None
        return delta // 2

    def _is_pattern(self, board: int) -> bool:
        if not self._pattern_masks:
            return True
        raw = int(board)
        return any((raw & mask) == mask for mask in self._pattern_masks)

    def _canonical_board(self, board: int) -> int:
        value = np.uint64(board)
        if self._canonical_mode == "full":
            return int(mover_runtime.canonical_full(value))
        if self._canonical_mode == "diagonal":
            return int(mover_runtime.canonical_diagonal(value))
        if self._canonical_mode == "horizontal":
            return int(mover_runtime.canonical_horizontal(value))
        if self._canonical_mode == "min33":
            return int(mover_runtime.canonical_min33(value))
        if self._canonical_mode == "min24":
            return int(mover_runtime.canonical_min24(value))
        if self._canonical_mode == "min34":
            return int(mover_runtime.canonical_min34(value))
        if self._canonical_mode == "min34top":
            return int(mover_runtime.canonical_min34top(value))
        return int(mover_runtime.canonical_identity(value))

    @staticmethod
    def _apply_operation(board: np.typing.NDArray, operation_index: int) -> np.typing.NDArray:
        source = np.asarray(board, dtype=np.int64)
        if operation_index == 1:
            return np.rot90(source).copy()
        if operation_index == 2:
            return np.rot90(source, k=2).copy()
        if operation_index == 3:
            return np.rot90(source, k=3).copy()
        if operation_index == 4:
            return np.flip(source, axis=1).copy()
        if operation_index == 5:
            return np.flip(np.rot90(source), axis=1).copy()
        if operation_index == 6:
            return np.flip(np.rot90(source, k=2), axis=1).copy()
        if operation_index == 7:
            return np.flip(np.rot90(source, k=3), axis=1).copy()
        return source.copy()

    @staticmethod
    def _adjust_direction(operation_index: int, direction: str) -> str:
        adjusted = direction
        if operation_index >= 4:
            if adjusted == "left":
                adjusted = "right"
            elif adjusted == "right":
                adjusted = "left"

        direction_index = 0
        if adjusted == "right":
            direction_index = 1
        elif adjusted == "down":
            direction_index = 2
        elif adjusted == "left":
            direction_index = 3
        direction_names = ("up", "right", "down", "left")
        return direction_names[(direction_index + (operation_index % 4)) % 4]

    def _operation_sequence(self) -> list[int]:
        operations = [self._last_operation_index]
        operations.extend(range(8))
        return operations

    def _lookup_value(
        self,
        path_list: list,
        pattern_full: str,
        board: int,
    ) -> tuple[float | None, str]:
        ordinal = self._layer_ordinal(board)
        if ordinal is None:
            return None, "uint32"
        compressed_path = self._compressed_path(path_list, pattern_full, ordinal)
        if compressed_path is not None:
            lookup = formation_core.lookup_bc_compressed_result_cold(
                compressed_path,
                int(self.target),
                int(board),
                0,
            )
        else:
            exact_paths = self._exact_paths(path_list, pattern_full, ordinal)
            if exact_paths is None:
                return None, "uint32"
            lookup = formation_core.lookup_bc_exact_result_cold(
                exact_paths[0],
                exact_paths[1],
                int(self.target),
                int(board),
                0,
            )
        return self._normalize_lookup_value(lookup)

    def _evaluate_operation(
        self,
        board: np.typing.NDArray,
        path_list: list,
        pattern_full: str,
        operation_index: int,
    ) -> tuple[list[tuple[str, str | float]], str, bool]:
        transformed_board = self._apply_operation(board, operation_index)
        encoded = int(mover_runtime.encode_board(transformed_board))
        if not self._is_pattern(encoded):
            return [], "uint32", False

        moved_boards = tuple(int(value) for value in mover_runtime.std.move_all_dir(np.uint64(encoded)))
        ordered_names = ("down", "right", "left", "up")
        result_values: list[tuple[str | float | None, str]] = [
            (None, "uint32"),
            (None, "uint32"),
            (None, "uint32"),
            (None, "uint32"),
        ]
        dtype_name = "uint32"

        for index, moved in enumerate(moved_boards):
            if moved == encoded or not self._is_pattern(moved):
                continue
            canonical = self._canonical_board(moved)
            value, value_dtype = self._lookup_value(path_list, pattern_full, canonical)
            dtype_name = value_dtype or dtype_name
            if index == 0:
                result_values[2] = (value, value_dtype)
            elif index == 1:
                result_values[1] = (value, value_dtype)
            elif index == 2:
                result_values[3] = (value, value_dtype)
            else:
                result_values[0] = (value, value_dtype)

        adjusted_entries: list[tuple[str, str | float]] = []
        for ordered_index, direction in enumerate(ordered_names):
            value, value_dtype = result_values[ordered_index]
            dtype_name = value_dtype or dtype_name
            adjusted_entries.append((
                self._adjust_direction(operation_index, direction),
                float(value) if value is not None else "",
            ))

        numeric_entries = [
            entry for entry in adjusted_entries
            if isinstance(entry[1], (int, float, np.integer, np.floating))
        ]
        other_entries = [
            entry for entry in adjusted_entries
            if not isinstance(entry[1], (int, float, np.integer, np.floating))
        ]
        numeric_entries.sort(key=lambda item: -float(item[1]))
        sorted_entries = numeric_entries + other_entries
        return sorted_entries, dtype_name, bool(numeric_entries)

    def move_on_dic(
        self,
        board: np.typing.NDArray,
        pattern_full: str,
    ) -> tuple[dict[str, str | float | int | None], str]:
        spawn_rate4 = SingletonConfig().config["4_spawn_rate"]
        pattern_key = SingletonConfig.get_pattern_key(pattern_full, spawn_rate4)
        path_list = SingletonConfig().config["filepath_map"].get(pattern_key, [])
        if not path_list:
            return {"?": "?"}, "uint32"

        for operation_index in self._operation_sequence():
            sorted_entries, dtype_name, has_numeric = self._evaluate_operation(
                board,
                path_list,
                pattern_full,
                operation_index,
            )
            if has_numeric:
                self._last_operation_index = operation_index
                return dict(sorted_entries), dtype_name

        return {"down": "", "right": "", "left": "", "up": ""}, "uint32"

    def _has_numeric_move(self, path_list: list, pattern_full: str, board: int) -> bool:
        board_matrix = mover_runtime.decode_board(np.uint64(board))
        for operation_index in self._operation_sequence():
            _, _, has_numeric = self._evaluate_operation(
                board_matrix,
                path_list,
                pattern_full,
                operation_index,
            )
            if has_numeric:
                return True
        return False

    def get_random_state(self, path_list: list, pattern_full: str) -> np.uint64:
        candidates = self._compressed_candidates(path_list, pattern_full)
        spawn_rate4 = float(SingletonConfig().config["4_spawn_rate"])
        preferred = [item for item in candidates if item[0] < 10]
        fallback = [item for item in candidates if item[0] >= 10]
        ordered_candidates = random.sample(preferred, len(preferred))
        ordered_candidates.extend(random.sample(fallback, len(fallback)))
        for _, path in ordered_candidates:
            for _ in range(8):
                state = formation_core.sample_bc_compressed_random_state(
                    path,
                    int(self.target),
                    spawn_rate4,
                )
                if not state:
                    continue
                state_int = int(state)
                if self._has_numeric_move(path_list, pattern_full, state_int):
                    return np.uint64(state_int)
        exact_candidates = self._exact_candidates(path_list, pattern_full)
        preferred = [item for item in exact_candidates if item[0] < 10]
        fallback = [item for item in exact_candidates if item[0] >= 10]
        ordered_exact = random.sample(preferred, len(preferred))
        ordered_exact.extend(random.sample(fallback, len(fallback)))
        for _, position_path in ordered_exact:
            for _ in range(8):
                state = formation_core.sample_bc_exact_random_state(
                    position_path,
                    int(self.target),
                    spawn_rate4,
                )
                if not state:
                    continue
                state_int = int(state)
                if self._has_numeric_move(path_list, pattern_full, state_int):
                    return np.uint64(state_int)
        return np.uint64(0)
