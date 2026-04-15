from __future__ import annotations

import numpy as np

from Config import SingletonConfig
from egtb_core.BoardMover import encode_board

from ..engine.base import BaseMinigameEngine


class TrickyTilesEngine(BaseMinigameEngine):
    def __init__(self, definition, difficulty: int) -> None:
        self.evil_gen_prob = 0.33 + np.random.rand() / 12 + int(difficulty) * 0.1
        self.evil_gen = None
        super().__init__(definition, difficulty)
        self.evil_gen = self._create_evil_gen()

    def _create_evil_gen(self):
        try:
            from ai_and_sort import ai_core
        except Exception as exc:
            raise RuntimeError(
                "Tricky Tiles requires ai_and_sort.ai_core, but it is unavailable."
            ) from exc
        return ai_core.EvilGen(encode_board(self.original_board()))

    def setup_new_game(self) -> None:
        super().setup_new_game()
        self.evil_gen_prob = 0.33 + np.random.rand() / 12 + int(self.difficulty) * 0.1

    def original_board(self) -> np.ndarray:
        board = np.zeros_like(self.board, dtype=np.int32)
        for row in range(self.rows):
            for col in range(self.cols):
                board[row][col] = 2 ** int(self.board[row][col]) if int(self.board[row][col]) > 0 else 0
        return board

    def gen_new_num(self) -> None:
        if np.random.rand() > self.evil_gen_prob:
            board, _, new_tile_pos, val = self.mover.gen_new_num(
                np.array(self.board, copy=True),
                SingletonConfig().config.get("4_spawn_rate", 0.1),
            )
            self.board = np.array(board, dtype=np.int32)
            self.newtile_pos, self.newtile = int(new_tile_pos), int(val)
            return

        board = self.original_board()
        self.evil_gen.reset_board(encode_board(board))
        depth = 5 if int(np.sum(board == 0)) < 6 else 4
        _board_encoded, new_tile_pos, val = self.evil_gen.gen_new_num(depth)
        self.board[new_tile_pos // self.cols][new_tile_pos % self.cols] = int(val)
        self.newtile_pos, self.newtile = int(new_tile_pos), int(val)

    def get_info_text(self) -> str:
        return "Brace yourself! New numbers may appear in the most challenging spots!"
