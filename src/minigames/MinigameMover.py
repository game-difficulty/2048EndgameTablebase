import random
from typing import Tuple

import numpy as np


class MinigameBoardMover_mxn:
    def __init__(self, shape):
        self.rows, self.cols = shape

    def move_left(self, board: np.ndarray) -> Tuple[np.ndarray, np.uint32]:
        total_score = 0
        new_board: np.ndarray = np.zeros_like(board)
        for i in range(self.rows):
            new_board[i], new_score = self.merge_line(board[i])
            total_score += new_score
        return new_board, total_score

    def move_right(self, board: np.ndarray) -> Tuple[np.ndarray, np.uint32]:
        total_score = 0
        new_board: np.ndarray = np.zeros_like(board)
        for i in range(self.rows):
            new_board[i], new_score = self.merge_line(board[i], True)
            total_score += new_score
        return new_board, total_score

    def move_up(self, board: np.ndarray) -> Tuple[np.ndarray, np.uint32]:
        total_score = 0
        new_board: np.ndarray = np.zeros_like(board)
        for i in range(self.cols):
            new_board[:, i], new_score = self.merge_line(board[:, i])
            total_score += new_score
        return new_board, total_score

    def move_down(self, board: np.ndarray) -> Tuple[np.ndarray, np.uint32]:
        total_score = 0
        new_board: np.ndarray = np.zeros_like(board)
        for i in range(self.cols):
            new_board[:, i], new_score = self.merge_line(board[:, i], True)
            total_score += new_score
        return new_board, total_score

    def move_board(self, board: np.ndarray, direction: np.uint8) -> Tuple[np.ndarray, np.uint32]:
        if direction == 1:
            return self.move_left(board)
        elif direction == 2:
            return self.move_right(board)
        elif direction == 3:
            return self.move_up(board)
        elif direction == 4:
            return self.move_down(board)
        else:
            print(f'bad direction input:{direction}')
            return board, 0

    @staticmethod
    def merge_line(line: np.ndarray, reverse: bool = False) -> Tuple[np.ndarray, np.uint32]:
        if reverse:
            line = line[::-1]

        merged = []
        score = 0
        skip = False

        segments = []
        current_segment = []

        for value in line:
            if value == -1:
                if current_segment:
                    segments.append(current_segment)
                segments.append([-1])
                current_segment = []
            else:
                current_segment.append(value)

        if current_segment:
            segments.append(current_segment)

        for segment in segments:
            if segment == [-1]:
                merged.append(-1)
            else:
                non_zero = [i for i in segment if i != 0]  # 去掉所有的0
                temp_merged = []
                for i in range(len(non_zero)):
                    if skip:
                        skip = False
                        continue
                    if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1] and non_zero[i] != -1:
                        merged_value = non_zero[i] + 1
                        score += 2 ** merged_value
                        temp_merged.append(merged_value)
                        skip = True
                    else:
                        temp_merged.append(non_zero[i])
                temp_merged += [0] * (len(segment) - len(temp_merged))
                merged.extend(temp_merged)

        if reverse:
            merged = merged[::-1]

        return np.array(merged), np.uint32(score)

    def gen_new_num(self, board: np.ndarray, p: float = 0.1) -> Tuple[np.ndarray, int, int, int]:
        empty_positions = [(i, j) for i in range(self.rows) for j in range(self.cols) if board[i, j] == 0]
        if not empty_positions:
            return board, 0, -1, -1  # No empty positions available
        i, j = random.choice(empty_positions)
        new_num = 2 if random.random() < p else 1
        board[i, j] = new_num
        return board, len(empty_positions), i * self.cols + j, new_num


class MinigameBoardMover:
    def __init__(self):
        self.move_results = np.zeros(12 ** 4, dtype=np.uint32)
        self.move_scores = np.zeros(12 ** 4, dtype=np.uint32)
        self.calculate_all_moves()

    @staticmethod
    def encode_line(line: np.ndarray) -> int:
        code = 0
        base = 12
        for i in range(len(line)):
            code = code * base + line[i]
        return code

    @staticmethod
    def decode_line(code: int) -> np.ndarray:
        base = 12
        line = []
        for _ in range(4):
            line.append(code % base)
            code //= base
        return np.array(line[::-1])

    def move_left(self, board: np.ndarray) -> Tuple[np.ndarray, np.uint32]:
        total_score = 0
        new_board: np.ndarray = np.zeros_like(board)
        for i in range(4):
            if max(board[i]) < 11 and min(board[i]) >= 0:
                code = self.encode_line(board[i])
                new_board[i] = self.decode_line(self.move_results[code])
                total_score += self.move_scores[code]
            else:
                new_board[i], new_score = self.merge_line(board[i])
                total_score += new_score
        return new_board, total_score

    def move_right(self, board: np.ndarray) -> Tuple[np.ndarray, np.uint32]:
        total_score = 0
        new_board: np.ndarray = np.zeros_like(board)
        for i in range(4):
            if max(board[i]) < 11 and min(board[i]) >= 0:
                code = self.encode_line(board[i][::-1])
                new_board[i] = self.decode_line(self.move_results[code])[::-1]
                total_score += self.move_scores[code]
            else:
                new_board[i], new_score = self.merge_line(board[i], True)
                total_score += new_score
        return new_board, total_score

    def move_up(self, board: np.ndarray) -> Tuple[np.ndarray, np.uint32]:
        total_score = 0
        new_board: np.ndarray = np.zeros_like(board)
        for i in range(4):
            if max(board[:, i]) < 11 and min(board[:, i]) >= 0:
                code = self.encode_line(board[:, i])
                new_board[:, i] = self.decode_line(self.move_results[code])
                total_score += self.move_scores[code]
            else:
                new_board[:, i], new_score = self.merge_line(board[:, i])
                total_score += new_score
        return new_board, total_score

    def move_down(self, board: np.ndarray) -> Tuple[np.ndarray, np.uint32]:
        total_score = 0
        new_board: np.ndarray = np.zeros_like(board)
        for i in range(4):
            if max(board[:, i]) < 11 and min(board[:, i]) >= 0:
                code = self.encode_line(board[:, i][::-1])
                new_board[:, i] = self.decode_line(self.move_results[code])[::-1]
                total_score += self.move_scores[code]
            else:
                new_board[:, i], new_score = self.merge_line(board[:, i], True)
                total_score += new_score
        return new_board, total_score

    def move_board(self, board: np.ndarray, direction: np.uint8) -> Tuple[np.ndarray, np.uint32]:
        if direction == 1:
            return self.move_left(board)
        elif direction == 2:
            return self.move_right(board)
        elif direction == 3:
            return self.move_up(board)
        elif direction == 4:
            return self.move_down(board)
        else:
            print(f'bad direction input:{direction}')
            return board, 0

    @staticmethod
    def merge_line(line: np.ndarray, reverse: bool = False) -> Tuple[np.ndarray, np.uint32]:
        """-1不可移动与合并，-2暂时用于礼盒无尽，-3无限合并自身不加分"""
        if reverse:
            line = line[::-1]

        merged = []
        score = 0
        skip = False
        skip2 = False

        segments = []
        current_segment = []

        for value in line:
            if value == -1:
                if current_segment:
                    segments.append(current_segment)
                segments.append([-1])
                current_segment = []
            else:
                current_segment.append(value)

        if current_segment:
            segments.append(current_segment)

        for segment in segments:
            if segment == [-1]:
                merged.append(-1)
            else:
                non_zero = [i for i in segment if i != 0]  # 去掉所有的0
                temp_merged = []
                for i in range(len(non_zero)):
                    if skip:
                        skip = False
                        continue
                    if skip2:
                        if non_zero[i] == -3:
                            continue
                        else:
                            skip2 = False
                    if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                        if non_zero[i] >= 0:
                            merged_value = non_zero[i] + 1
                            score += 2 ** merged_value
                            temp_merged.append(merged_value)
                            skip = True
                        elif non_zero[i] == -3:
                            merged_value = non_zero[i]
                            temp_merged.append(merged_value)
                            skip2 = True
                        else:
                            temp_merged.append(non_zero[i])
                    else:
                        temp_merged.append(non_zero[i])
                temp_merged += [0] * (len(segment) - len(temp_merged))
                merged.extend(temp_merged)

        if reverse:
            merged = merged[::-1]

        return np.array(merged), np.uint32(score)

    def calculate_all_moves(self) -> None:
        # 存储所有可能的行移动结果
        for code in range(12 ** 4):
            line = self.decode_line(code)
            merged_line, score = self.merge_line(line)
            self.move_results[code] = self.encode_line(merged_line)
            self.move_scores[code] = score

    @staticmethod
    def gen_new_num(board: np.ndarray, p: float = 0.1) -> Tuple[np.ndarray, int, int, int]:
        empty_positions = [(i, j) for i in range(4) for j in range(4) if board[i, j] == 0]
        if not empty_positions:
            return board, 0, -1, -1  # No empty positions available
        i, j = random.choice(empty_positions)
        new_num = 2 if random.random() < p else 1
        board[i, j] = new_num
        return board, len(empty_positions), i * 4 + j, new_num


if __name__ == "__main__":
    mover = MinigameBoardMover()

    # 测试打表范围内的移动
    board_test = np.array([
        [1, 1, 1, 1],
        [3, 2, 2, 2],
        [1, 0, 1, 1],
        [0, 1, 0, 0]
    ])
    board_new, score_new = mover.move_left(board_test)
    assert np.array_equal(board_new, np.array([
        [2, 2, 0, 0],
        [3, 3, 2, 0],
        [2, 1, 0, 0],
        [1, 0, 0, 0]
    ])), f"Expected: \n[[2, 2, 0, 0],\n [3, 3, 2, 0],\n [2, 1, 0, 0],\n [1, 0, 0, 0]], but got: \n{board_new}"
    assert score_new == 20, f"Expected score: 20, but got: {score_new}"

    # 测试超出打表范围的移动
    board_test = np.array([
        [10, 10, 10, 12],
        [10, 11, 11, 2],
        [10, 0, 4, 0],
        [10, 11, 0, 15]
    ])
    board_new, score_new = mover.move_down(board_test)
    assert np.array_equal(board_new, np.array([
        [0, 0, 0, 0],
        [0, 0, 10, 12],
        [11, 10, 11, 2],
        [11, 12, 4, 15]
    ])), f"Expected: \n[[0, 0, 0, 0],\n [0, 0, 10, 12],\n [11, 10, 11, 2],\n [11, 12, 4, 15]], but got: \n{board_new}"
    assert score_new == 8192, f"Expected score: 8192, but got: {score_new}"

    # 测试打表范围边界的移动
    print(mover.move_board(np.array([[0, 0, 0, 3],
                                     [0, 1, 1, 1],
                                     [3, 4, 3, 2],
                                     [5, 9, 9, 8]]), 2))
