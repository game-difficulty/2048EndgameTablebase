from __future__ import annotations

import random
from typing import Any

import numpy as np

from Config import SingletonConfig

from ..engine.base import BaseMinigameEngine
from ..engine.mover import create_mover


class ShapeShifterEngine(BaseMinigameEngine):
    n = 12

    def get_initial_shape(self) -> tuple[int, int]:
        config = SingletonConfig().config
        saved = config.get("minigame_state", [dict(), dict()])[int(self.difficulty)].get(self.legacy_name)
        if saved:
            try:
                shape = np.array(saved[0][0], dtype=np.int32).shape
                if len(shape) == 2:
                    return int(shape[0]), int(shape[1])
            except Exception:
                pass
        return self.n, self.n

    def setup_new_game(self) -> None:
        rectangle_area = 10000
        while rectangle_area > 13 - int(self.difficulty) * 2 or rectangle_area < 8 - int(self.difficulty) * 2:
            self.board = self._select_connected_cells(self.n, 18)
            rectangle_area = self._max_rectangle_area(self.board)

        self.board = self._min_bounding_rectangle(self.board)
        if self.board.shape[1] < self.board.shape[0]:
            self.board = np.array(self.board.T, dtype=np.int32)
        self.rows, self.cols = self.board.shape
        self.mover = create_mover((self.rows, self.cols))
        spawn_rate = SingletonConfig().config.get("4_spawn_rate", 0.1)
        board, _, _, _ = self.mover.gen_new_num(np.array(self.board, copy=True), spawn_rate)
        board, _, pos, val = self.mover.gen_new_num(np.array(board, copy=True), spawn_rate)
        self.board = np.array(board, dtype=np.int32)
        self.newtile_pos = int(pos)
        self.newtile = int(val)
        self.score = 0
        self.is_over = False
        self.current_max_num = self._positive_max(self.board)
        self._pending_messages.clear()
        self._animation = self._animation.__class__(
            appear_index=self.newtile_pos,
            appear_value=self.newtile,
            valid_move=True,
        )
        self.save_to_config()

    @staticmethod
    def _is_valid(x: int, y: int, visited: set[tuple[int, int]], n: int) -> bool:
        return 0 <= x < n and 0 <= y < n and (x, y) not in visited

    @staticmethod
    def _get_neighbors(x: int, y: int) -> list[tuple[int, int]]:
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        random.shuffle(neighbors)
        return neighbors

    @classmethod
    def _select_connected_cells(cls, n: int, m: int) -> np.ndarray:
        array = np.full((n, n), -1, dtype=np.int32)
        visited: set[tuple[int, int]] = set()
        remaining: set[tuple[int, int]] = set()
        start_x, start_y = random.randint(0, n - 1), random.randint(0, n - 1)
        stack = [(start_x, start_y)]

        while len(visited) < m:
            if not stack:
                stack = [remaining.pop()]

            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            array[x, y] = 0

            for nx, ny in cls._get_neighbors(x, y):
                if cls._is_valid(nx, ny, visited, n):
                    if random.random() > 0.5:
                        stack.append((nx, ny))
                    else:
                        remaining.add((nx, ny))
        return array

    @staticmethod
    def _largest_rectangle_area_in_histogram(heights: list[int]) -> int:
        stack: list[int] = []
        max_area = 0
        extended = heights + [0]
        for index, height in enumerate(extended):
            while stack and height < extended[stack[-1]]:
                h = extended[stack.pop()]
                width = index if not stack else index - stack[-1] - 1
                max_area = max(max_area, h * width)
            stack.append(index)
        return max_area

    @classmethod
    def _max_rectangle_area(cls, matrix: np.ndarray) -> int:
        if not np.any(matrix == 0):
            return 0
        rows, cols = matrix.shape
        heights = [0] * cols
        max_area = 0
        for row in range(rows):
            for col in range(cols):
                heights[col] = heights[col] + 1 if matrix[row][col] == 0 else 0
            max_area = max(max_area, cls._largest_rectangle_area_in_histogram(heights))
        return max_area

    @staticmethod
    def _min_bounding_rectangle(matrix: np.ndarray) -> np.ndarray:
        rows, cols = matrix.shape
        min_x, max_x, min_y, max_y = rows, 0, cols, 0
        for row in range(rows):
            for col in range(cols):
                if matrix[row][col] == 0:
                    min_x = min(min_x, row)
                    max_x = max(max_x, row)
                    min_y = min(min_y, col)
                    max_y = max(max_y, col)
        return np.array(matrix[min_x : max_x + 1, min_y : max_y + 1], dtype=np.int32)

    def build_view_state(self) -> dict[str, Any]:
        view = super().build_view_state()
        view["blockedMask"] = (self.board.flatten() == -1).tolist()
        return view

    def get_info_text(self) -> str:
        return "Every game begins with a unique and unexpected board shape!"
