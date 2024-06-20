import numpy as np
from numba import uint64
from numba.experimental import jitclass

spec = {
    'movel': uint64[:],  # 表示一个uint64类型的一维数组
    'mover': uint64[:],
    'moveu': uint64[:],
    'moved': uint64[:],
}


@jitclass(spec)
class BoardMover:
    def __init__(self):
        self.movel, self.mover, self.moveu, self.moved = self.calculate_all_moves()
        print('BoardMover init')

    @staticmethod
    def encode_board(board):
        encoded_board = np.uint64(0)
        tile_log2 = {0: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11,
                     4096: 12, 8192: 13, 16384: 14, 32768: 15, 65536: 16}
        for i, row in enumerate(board):
            for j, num in enumerate(row):
                encoded_board |= np.uint64(tile_log2[int(num)]) << np.uint64(4 * ((3 - i) * 4 + (3 - j)))
        return encoded_board

    @staticmethod
    def decode_board(encoded_board):
        encoded_board = np.uint64(encoded_board)
        board = np.zeros((4, 4), dtype=np.int32)
        for i in range(3,-1,-1):
            for j in range(3,-1,-1):
                encoded_num = (encoded_board >> (4 * ((3 - i) * 4 + (3 - j)))) & 0xF
                if encoded_num > 0:
                    board[i, j] = 2 ** encoded_num
                else:
                    board[i, j] = 0
        return board

    @staticmethod
    def encode_row(row):
        encoded = np.uint64(0)
        tile_log2 = {0: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11,
                     4096: 12, 8192: 13, 16384: 14, 32768: 15, 65536: 16}
        for i, num in enumerate(row):
            encoded |= np.uint64(tile_log2[num]) << np.uint64(4 * (3 - i))
        return encoded

    @staticmethod
    def decode_row(encoded):
        row = np.empty(4, dtype=np.uint32)
        for i in range(4):
            num = (np.uint64(encoded) >> np.uint64(4 * (3 - i))) & np.uint64(0xF)
            if num > 0:
                row[i] = (2 ** num)
            else:
                row[i] = np.uint64(0)
        return row

    @staticmethod
    def reverse(board):
        board = (board & np.uint64(0xFF00FF0000FF00FF)) | ((board & np.uint64(0x00FF00FF00000000)) >> np.uint64(24)) | (
                (board & np.uint64(0x00000000FF00FF00)) << np.uint64(24))
        board = (board & np.uint64(0xF0F00F0FF0F00F0F)) | ((board & np.uint64(0x0F0F00000F0F0000)) >> np.uint64(12)) | (
                (board & np.uint64(0x0000F0F00000F0F0)) << np.uint64(12))
        return board

    def move_left(self, board):
        board ^= self.movel[board & np.uint64(0xffff)]
        board ^= self.movel[board >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(16)
        board ^= self.movel[board >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(32)
        board ^= self.movel[board >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(48)
        return board

    def move_right(self, board):
        board ^= self.mover[board & np.uint64(0xffff)]
        board ^= self.mover[board >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(16)
        board ^= self.mover[board >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(32)
        board ^= self.mover[board >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(48)
        return board

    def move_up(self, board, board2):
        board ^= self.moveu[board2 & np.uint64(0xffff)]
        board ^= self.moveu[board2 >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(4)
        board ^= self.moveu[board2 >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(8)
        board ^= self.moveu[board2 >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(12)
        return board

    def move_down(self, board, board2):
        board ^= self.moved[board2 & np.uint64(0xffff)]
        board ^= self.moved[board2 >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(4)
        board ^= self.moved[board2 >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(8)
        board ^= self.moved[board2 >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(12)
        return board

    def move_board(self, board, direction):
        if direction == 1:
            return self.move_left(board)
        elif direction == 2:
            return self.move_right(board)
        elif direction == 3:
            board2 = self.reverse(board)
            return self.move_up(board, board2)
        elif direction == 4:
            board2 = self.reverse(board)
            return self.move_down(board, board2)
        else:
            print(f'bad direction input:{direction}')
            return board

    def move_all_dir(self, board):
        board = np.uint64(board)
        board2 = self.reverse(board)
        return (
            self.move_down(board, board2), self.move_right(board), self.move_left(board), self.move_up(board, board2))

    @staticmethod
    def merge_line(line, reverse=False):
        if reverse:
            line = line[::-1]
        non_zero = [i for i in line if i != 0]  # 去掉所有的0
        merged = []
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1] and non_zero[i] != 32768:
                merged_value = 2 * non_zero[i]
                merged.append(merged_value)
                skip = True
            else:
                merged.append(non_zero[i])

        # 补齐剩下的 0
        merged += [0] * (len(line) - len(merged))
        if reverse:
            merged = merged[::-1]
        return np.array(merged)

    def calculate_all_moves(self):
        # 初始化存储所有可能的行及其移动后结果差值的字典
        movel = np.empty(65536, dtype=np.uint64)
        mover = np.empty(65536, dtype=np.uint64)
        moveu = np.empty(65536, dtype=np.uint64)
        moved = np.empty(65536, dtype=np.uint64)

        # 生成所有可能的行
        for i in range(16 ** 4):
            line = [(i // (16 ** j)) % 16 for j in range(4)]
            line = np.array([2 ** k if k else 0 for k in line])  # 把游戏中的数字转换成2的幂
            original_line = self.encode_row(line)  # 编码原始行为整数

            # 向左移动
            merged_linel = self.merge_line(line, reverse=False)
            movel[original_line] = self.encode_row(merged_linel) ^ original_line

            # 向右移动
            merged_liner = self.merge_line(line, reverse=True)
            mover[original_line] = self.encode_row(merged_liner) ^ original_line

        # 使用reverse函数计算向上和向下的移动差值
        for i in range(16 ** 4):
            moveu[i] = self.reverse(movel[i])
            moved[i] = self.reverse(mover[i])

        return movel, mover, moveu, moved

    @staticmethod
    def gen_new_num(t, p=0.9):
        empty_slots = [i for i in range(16) if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0]  # 找到所有空位
        if not empty_slots:
            return t, 0  # 如果没有空位，返回原面板
        i = int(np.random.choice(np.array(empty_slots)))  # 随机选择一个空位
        val = 1 if np.random.random() < p else 2  # 生成2或4，其中2的概率为0.9
        t |= np.uint64(val) << np.uint64(4 * i)  # 在选中的位置放置新值
        return t, len(empty_slots)


spec2 = {
    'movel': uint64[:],  # 表示一个uint64类型的一维数组
    'mover': uint64[:],
    'moveu': uint64[:],
    'moved': uint64[:],
    'score': uint64[:],
}


@jitclass(spec2)
class BoardMoverWithScore:
    def __init__(self):
        self.movel, self.mover, self.moveu, self.moved, self.score = self.calculate_all_moves()
        print('BoardMover init')

    @staticmethod
    def encode_board(board):
        encoded_board = np.uint64(0)
        tile_log2 = {0: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11,
                     4096: 12, 8192: 13, 16384: 14, 32768: 15, 65536: 16}
        for i, row in enumerate(board):
            for j, num in enumerate(row):
                encoded_board |= np.uint64(tile_log2[int(num)]) << np.uint64(4 * ((3 - i) * 4 + (3 - j)))
        return encoded_board

    @staticmethod
    def decode_board(encoded_board):
        encoded_board = np.uint64(encoded_board)
        board = np.zeros((4, 4), dtype=np.int32)
        for i in range(3,-1,-1):
            for j in range(3,-1,-1):
                encoded_num = (encoded_board >> (4 * ((3 - i) * 4 + (3 - j)))) & 0xF
                if encoded_num > 0:
                    board[i, j] = 2 ** encoded_num
                else:
                    board[i, j] = 0
        return board

    @staticmethod
    def encode_row(row):
        encoded = np.uint64(0)
        tile_log2 = {0: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11,
                     4096: 12, 8192: 13, 16384: 14, 32768: 15, 65536: 16}
        for i, num in enumerate(row):
            encoded |= np.uint64(tile_log2[num]) << np.uint64(4 * (3 - i))
        return encoded

    @staticmethod
    def decode_row(encoded):
        row = np.empty(4, dtype=np.uint32)
        for i in range(4):
            num = (np.uint64(encoded) >> np.uint64(4 * (3 - i))) & np.uint64(0xF)
            if num > 0:
                row[i] = (2 ** num)
            else:
                row[i] = np.uint64(0)
        return row

    @staticmethod
    def reverse(board):
        board = (board & np.uint64(0xFF00FF0000FF00FF)) | ((board & np.uint64(0x00FF00FF00000000)) >> np.uint64(24)) | (
                (board & np.uint64(0x00000000FF00FF00)) << np.uint64(24))
        board = (board & np.uint64(0xF0F00F0FF0F00F0F)) | ((board & np.uint64(0x0F0F00000F0F0000)) >> np.uint64(12)) | (
                (board & np.uint64(0x0000F0F00000F0F0)) << np.uint64(12))
        return board

    def move_left(self, board):
        total_score = 0
        for i in range(4):
            line = (board >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            total_score += self.score[line]
            board ^= self.movel[line] << np.uint64(16 * i)
        return board, total_score

    def move_right(self, board):
        total_score = 0
        for i in range(4):
            line = (board >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            total_score += self.score[line]
            board ^= self.mover[line] << np.uint64(16 * i)
        return board, total_score

    def move_up(self, board, board2):
        total_score = 0
        for i in range(4):
            line = (board2 >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            total_score += self.score[line]
            board ^= self.moveu[line] << np.uint64(4 * i)
        return board, total_score

    def move_down(self, board, board2):
        total_score = 0
        for i in range(4):
            line = (board2 >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            total_score += self.score[line]
            board ^= self.moved[line] << np.uint64(4 * i)
        return board, total_score

    def move_board(self, board: np.uint64, direction: np.uint8) -> tuple[np.uint64, np.uint32]:
        if direction == 1:
            return self.move_left(board)
        elif direction == 2:
            return self.move_right(board)
        elif direction == 3:
            board2 = self.reverse(board)
            return self.move_up(board, board2)
        elif direction == 4:
            board2 = self.reverse(board)
            return self.move_down(board, board2)
        else:
            print(f'bad direction input:{direction}')
            return board, 0

    def move_all_dir(self, board):
        board = np.uint64(board)
        board2 = self.reverse(board)
        return (
            self.move_down(board, board2)[0], self.move_right(board)[0],
            self.move_left(board)[0], self.move_up(board, board2)[0])

    @staticmethod
    def merge_line(line, reverse=False):
        if reverse:
            line = line[::-1]
        non_zero = [i for i in line if i != 0]  # 去掉所有的0
        merged = []
        score = 0
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1] and non_zero[i] != 32768:
                merged_value = 2 * non_zero[i]
                score += merged_value
                merged.append(merged_value)
                skip = True
            else:
                merged.append(non_zero[i])

        # 补齐剩下的 0
        merged += [0] * (len(line) - len(merged))
        if reverse:
            merged = merged[::-1]
        return np.array(merged), score

    def calculate_all_moves(self):
        # 初始化存储所有可能的行及其移动后结果差值的字典
        movel = np.empty(65536, dtype=np.uint64)
        mover = np.empty(65536, dtype=np.uint64)
        moveu = np.empty(65536, dtype=np.uint64)
        moved = np.empty(65536, dtype=np.uint64)
        score = np.empty(65536, dtype=np.uint64)
        # 生成所有可能的行
        for i in range(16 ** 4):
            line = [(i // (16 ** j)) % 16 for j in range(4)]
            line = np.array([2 ** k if k else 0 for k in line])  # 把游戏中的数字转换成2的幂
            original_line = self.encode_row(line)  # 编码原始行为整数

            # 向左移动
            merged_linel, s = self.merge_line(line, reverse=False)
            movel[original_line] = self.encode_row(merged_linel) ^ original_line

            # 向右移动
            merged_liner, s = self.merge_line(line, reverse=True)
            mover[original_line] = self.encode_row(merged_liner) ^ original_line

            score[original_line] = s
        # 使用reverse函数计算向上和向下的移动差值
        for i in range(16 ** 4):
            moveu[i] = self.reverse(movel[i])
            moved[i] = self.reverse(mover[i])

        return movel, mover, moveu, moved, score

    @staticmethod
    def gen_new_num(t, p=0.9):
        empty_slots = [i for i in range(16) if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0]  # 找到所有空位
        if not empty_slots:
            return t, 0, 0, 0  # 如果没有空位，返回原面板
        i = int(np.random.choice(np.array(empty_slots)))  # 随机选择一个空位
        val = 1 if np.random.random() < p else 2  # 生成2或4，其中2的概率为0.9
        t |= np.uint64(val) << np.uint64(4 * i)  # 在选中的位置放置新值
        return t, len(empty_slots), 15 - i, val
