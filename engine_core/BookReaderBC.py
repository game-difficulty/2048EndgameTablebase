from __future__ import annotations

import numpy as np

from Config import SingletonConfig, category_info, pattern_catalog
from engine_core.EXPhysicalPattern import resolve_ex_physical_pattern

try:
    from native_core import formation_core
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
    "min34top": formation_core.SymmMode.Min34Top if formation_core else 7,
}


def _symm_mode_value(name: str) -> int:
    mode = _SYMM_MODE_BY_NAME.get(str(name).lower(), 0)
    return int(mode.value if hasattr(mode, "value") else mode)


class BookReaderBC:
    def __new__(cls, pattern: str, target: int):
        return super().__new__(cls)

    def __init__(self, pattern: str, target: int):
        if formation_core is None:
            raise RuntimeError("formation_core is unavailable")
        if not hasattr(formation_core, "BCBookReader"):
            raise RuntimeError("formation_core does not expose BCBookReader")

        self.pattern = pattern
        self.target = int(np.log2(target)) if int(target) >= 128 else int(target)
        meta = pattern_catalog.get(pattern)
        if meta is None:
            raise KeyError(f"Unknown pattern: {pattern}")

        resolution = None
        if not pattern.startswith("free"):
            resolution = resolve_ex_physical_pattern(
                pattern,
                meta.get("seed_boards", ()),
                self.target,
                int(SingletonConfig().config.get("SmallTileSumLimit", 96)),
                advanced=False,
            )

        pattern_spec = formation_core.PatternSpec()
        pattern_spec.name = pattern
        if resolution is not None:
            pattern_spec.pattern_masks = list(resolution.pattern_masks)
            pattern_spec.success_shifts = list(resolution.success_shifts)
            pattern_spec.symm_mode = _symm_mode_value(resolution.physical_canonical_mode)
            pattern_spec.physical_transform = int(resolution.transform_id)
            pattern_spec.inverse_physical_transform = int(resolution.inverse_transform_id)
            pattern_spec.logical_pattern_signature = int(resolution.logical_pattern_signature)
            pattern_spec.physical_pattern_signature = int(resolution.physical_pattern_signature)
        else:
            pattern_spec.pattern_masks = list(meta.get("pattern_masks", ()))
            pattern_spec.success_shifts = list(meta.get("success_shifts", ()))
            pattern_spec.symm_mode = _symm_mode_value(meta.get("canonical_mode", "identity"))

        self._nums_adjust = int(meta.get("nums_adjust", 0))
        self._native_reader = formation_core.BCBookReader(
            pattern_spec,
            int(self.target),
            pattern in category_info.get("variant", []),
        )

    def move_on_dic(
        self,
        board: np.typing.NDArray,
        pattern_full: str,
    ) -> tuple[dict[str, str | float | int | None], str]:
        spawn_rate4 = SingletonConfig().config["4_spawn_rate"]
        pattern_key = SingletonConfig.get_pattern_key(pattern_full, spawn_rate4)
        path_list = SingletonConfig().config["filepath_map"].get(pattern_key, [])
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
                self._nums_adjust,
            )
        )
