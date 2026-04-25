from __future__ import annotations

import numpy as np

from Config import SingletonConfig, category_info, pattern_32k_tiles_map, pattern_catalog

try:
    from ai_and_sort import formation_core
except Exception:
    formation_core = None


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


class BookReaderAD:
    def __new__(cls, pattern: str, target: int):
        if pattern in pattern_32k_tiles_map:
            return super().__new__(cls)
        return None

    def __init__(self, pattern: str, target: int):
        if formation_core is None:
            raise RuntimeError("formation_core is unavailable")

        self.pattern = pattern
        self.target = target
        meta = pattern_catalog.get(pattern, {})
        _, num_free_32k, pos_fixed_32k = pattern_32k_tiles_map[pattern]

        pattern_spec = formation_core.AdvancedPatternSpec()
        pattern_spec.name = pattern
        pattern_spec.pattern_masks = list(meta.get("pattern_masks", ()))
        pattern_spec.symm_mode = _symm_mode_value(meta.get("canonical_mode", "identity"))
        pattern_spec.num_free_32k = int(num_free_32k)
        pattern_spec.fixed_32k_shifts = list(np.asarray(pos_fixed_32k, dtype=np.uint8))
        pattern_spec.small_tile_sum_limit = int(SingletonConfig().config.get("SmallTileSumLimit", 96))
        pattern_spec.target = int(target)

        self._nums_adjust = int(meta.get("nums_adjust", 0))
        self._native_reader = formation_core.AdvancedBookReader(
            pattern_spec,
            pattern in category_info.get("variant", []),
        )

    def move_on_dic(
        self,
        board: np.typing.NDArray,
        pattern_full: str,
    ) -> tuple[dict[str, str | float | int | None], str]:
        spawn_rate4 = SingletonConfig().config["4_spawn_rate"]
        path_list = SingletonConfig().config["filepath_map"].get((pattern_full, spawn_rate4), [])
        return self._native_reader.move_on_dic(
            board.tolist(),
            path_list,
            pattern_full,
            self._nums_adjust,
        )

    def get_random_state(self, path_list: list, pattern_full: str) -> np.uint64:
        return np.uint64(
            self._native_reader.get_random_state(
                path_list,
                pattern_full,
                float(SingletonConfig().config["4_spawn_rate"]),
            )
        )