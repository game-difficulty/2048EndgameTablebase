from __future__ import annotations

import random
from typing import Any

import numpy as np

from Config import SingletonConfig

from ..engine.base import BaseMinigameEngine


class IsolatedIslandEngine(BaseMinigameEngine):
    def gen_new_num(self) -> None:
        spawn_chance = 0.04 - float(np.sum(self.board == -3)) * 0.02 + int(self.difficulty) * 0.01
        zero_positions = np.argwhere(self.board == 0)
        if zero_positions.size and random.random() < spawn_chance:
            row, col = random.choice(zero_positions.tolist())
            self.board[row, col] = -3
            self.newtile_pos = int(row * self.cols + col)
            self.newtile = -3
            return
        board, _, pos, val = self.mover.gen_new_num(
            np.array(self.board, copy=True),
            SingletonConfig().config.get("4_spawn_rate", 0.1),
        )
        self.board = np.array(board, dtype=np.int32)
        self.newtile_pos = int(pos)
        self.newtile = int(val)

    def build_view_state(self) -> dict[str, Any]:
        view = super().build_view_state()
        cover_sprites: dict[str, list[str]] = {}
        tile_style_variant: dict[str, Any] = {}
        for index, value in enumerate(self.board.flatten().tolist()):
            if int(value) == -3:
                cover_sprites[str(index)] = ["portal.png"]
                tile_style_variant[str(index)] = {"kind": "portal"}
        view["coverSprites"] = cover_sprites
        view["tileStyleVariant"] = tile_style_variant
        return view

    def get_info_text(self) -> str:
        return "Discovered a tile that merges with none but itself."
