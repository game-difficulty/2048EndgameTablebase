import numpy as np
from numba import int32, types, int64, boolean
from numba.experimental import jitclass
from BoardMover import BoardMover

spec = {
    'max_d': int32,
    'best_operation': types.Optional(types.unicode_type),
    'board': types.Array(int64, 2, 'C'),  # board是二维整数数组
    'diffs2': int64[:],
    'diffs2_r': int64[:],
    'bm': BoardMover.class_type.instance_type,
    'node': int32,
    'interact': boolean,
}


# 使用64位整数编码

@jitclass(spec)
class AutoplayS:
    def __init__(self, board):
        # 如果修改 init 需要同步修改 reset_board
        self.max_d = 3
        self.best_operation = ''
        self.board = board
        self.bm = BoardMover()
        self.diffs2, self.diffs2_r = self.calculates_for_estimate()
        self.node = 0
        self.interact = True
        print('AI init')

    def reset_board(self, board):
        self.best_operation = ''
        self.board = board
        self.node = 0

    def calculates_for_estimate(self):
        diffs2 = np.empty(65536, dtype=np.int64)
        diffs2_r = np.empty(65536, dtype=np.int64)
        # 生成所有可能的行
        for i in range(16 ** 4):
            line = [(i // (16 ** j)) % 16 for j in range(4)][::-1]
            diffs2[i] = self.diffs2_evaluation_func(line)
            diffs2_r[i] = self.diffs2_evaluation_func(line[::-1])
        return diffs2, diffs2_r

    @staticmethod
    def tile_score(r):
        return r << r

    def diffs2_evaluation_func(self, line):
        score = self.tile_score(line[0])
        for x in range(3):
            a = self.tile_score(line[x])
            b = self.tile_score(line[x + 1])
            if a >= b:
                score += a + b
            else:
                score += (a - b) * 12
            if a == b:
                score += a
        return score

    def estimate(self, s: np.uint64) -> np.float64:
        self.node += 1
        s = np.uint64(s)
        s_reverse = self.bm.reverse(s)
        if self.interact:
            diff_u, diff_d, diff_l, diff_r = 0, 0, 0, 0
            for i in range(4):
                l1 = (s >> np.uint64(16 * i)) & np.uint64(0xFFFF)
                l2 = (s_reverse >> np.uint64(16 * i)) & np.uint64(0xFFFF)
                diff_l += self.diffs2[l1]
                diff_u += self.diffs2[l2]
                diff_r += self.diffs2_r[l1]
                diff_d += self.diffs2_r[l2]
            diffs = max(diff_l, diff_r) + max(diff_u, diff_d)
        else:
            diff_u, diff_l = 0, 0
            for i in range(4):
                l1 = (s >> np.uint64(16 * i)) & np.uint64(0xFFFF)
                l2 = (s_reverse >> np.uint64(16 * i)) & np.uint64(0xFFFF)
                diff_l += self.diffs2[l1]
                diff_u += self.diffs2[l2]
            diffs = diff_l + diff_u
        return np.float64(diffs)

    def search0(self, b):
        best = -4444444.4
        for d in ['Left', 'Up', 'Down', 'Right']:
            t = self.bm.move_board(b, d)
            if t == b:
                continue
            temp = self.estimate(t)
            if temp > best:
                best = temp
        return best

    def search1(self, b):
        best = -4444444.4
        for d in ['Left', 'Up', 'Down', 'Right']:
            t = self.bm.move_board(b, d)
            if t == b:
                continue
            temp = np.float64(0.0)

            empty_slots_count = 0
            # 每个空位分别尝试放入2和4
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                    empty_slots_count += 1
                    for val in (1, 2):  # 编码后的2和4
                        one_step_prob = 0.9 if val == 1 else 0.1
                        temp += self.search0(np.uint64(t | (val << (4 * i)))) * one_step_prob
            temp = temp / empty_slots_count  # + empty_slots_count * (1)
            if temp > best:
                best = temp
                if self.max_d == 1:
                    self.best_operation = d
        return best

    def search2(self, b):
        best = -4444444.4
        for d in ['Left', 'Up', 'Down', 'Right']:
            t = self.bm.move_board(b, d)
            if t == b:
                continue
            temp = 0.0

            empty_slots_count = 0
            # 每个空位分别尝试放入2和4
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                    empty_slots_count += 1
                    for val in (1, 2):  # 编码后的2和4
                        one_step_prob = 0.9 if val == 1 else 0.1
                        temp += self.search1(np.uint64(t | (val << (4 * i)))) * one_step_prob
            temp = temp / empty_slots_count  # + empty_slots_count * (1)
            if temp > best:
                best = temp
                if self.max_d == 2:
                    self.best_operation = d
        return best

    def search3(self, b):
        best = -4444444.4
        for d in ['Left', 'Up', 'Down', 'Right']:
            t = self.bm.move_board(b, d)
            if t == b:
                continue
            temp = 0.0

            empty_slots_count = 0
            # 每个空位分别尝试放入2和4
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                    empty_slots_count += 1
                    for val in (1, 2):  # 编码后的2和4
                        one_step_prob = 0.9 if val == 1 else 0.1
                        temp += self.search2(np.uint64(t | (val << (4 * i)))) * one_step_prob
            temp = temp / empty_slots_count  # + empty_slots_count * (1)
            if temp > best:
                best = temp
                if self.max_d == 3:
                    self.best_operation = d
        return best

    def search4(self, b):
        best = -4444444.4
        for d in ['Left', 'Up', 'Down', 'Right']:
            t = self.bm.move_board(b, d)
            if t == b:
                continue
            temp = 0.0

            empty_slots_count = 0
            # 每个空位分别尝试放入2和4
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                    empty_slots_count += 1
                    for val in (1, 2):  # 编码后的2和4
                        one_step_prob = 0.9 if val == 1 else 0.1
                        temp += self.search3(np.uint64(t | (val << (4 * i)))) * one_step_prob
            temp = temp / empty_slots_count  # + empty_slots_count * (1)
            if temp > best:
                best = temp
                if self.max_d == 4:
                    self.best_operation = d
        return best

    def search5(self, b):
        best = -4444444.4
        for d in ['Left', 'Up', 'Down', 'Right']:
            t = self.bm.move_board(b, d)
            if t == b:
                continue
            temp = 0.0

            empty_slots_count = 0
            # 每个空位分别尝试放入2和4
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                    empty_slots_count += 1
                    for val in (1, 2):  # 编码后的2和4
                        one_step_prob = 0.9 if val == 1 else 0.1
                        temp += self.search4(np.uint64(t | (val << (4 * i)))) * one_step_prob
            temp = temp / empty_slots_count  # + empty_slots_count * (1)
            if temp > best:
                best = temp
                if self.max_d == 5:
                    self.best_operation = d
        return best

    def search6(self, b):
        best = -4444444.4
        for d in ['Left', 'Up', 'Down', 'Right']:
            t = self.bm.move_board(b, d)
            if t == b:
                continue
            temp = 0.0

            empty_slots_count = 0
            # 每个空位分别尝试放入2和4
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                    empty_slots_count += 1
                    for val in (1, 2):  # 编码后的2和4
                        one_step_prob = 0.9 if val == 1 else 0.1
                        temp += self.search5(np.uint64(t | (val << (4 * i)))) * one_step_prob
            temp = temp / empty_slots_count  # + empty_slots_count * (1)
            if temp > best:
                best = temp
                if self.max_d == 6:
                    self.best_operation = d
        return best

    def search7(self, b):
        best = -4444444.4
        for d in ['Left', 'Up', 'Down', 'Right']:
            t = self.bm.move_board(b, d)
            if t == b:
                continue
            temp = 0.0

            empty_slots_count = 0
            # 每个空位分别尝试放入2和4
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                    empty_slots_count += 1
                    for val in (1, 2):  # 编码后的2和4
                        one_step_prob = 0.9 if val == 1 else 0.1
                        temp += self.search6(np.uint64(t | (val << (4 * i)))) * one_step_prob
            temp = temp / empty_slots_count  # + empty_slots_count * (1)
            if temp > best:
                best = temp
                if self.max_d == 7:
                    self.best_operation = d
        return best

    def start_search(self, depth=3):
        self.best_operation = ''
        self.max_d = depth
        self.node = 0
        self.dispatcher(self.bm.encode_board(self.board))

    def dispatcher(self, board):
        depth = self.max_d
        if depth == 1:
            return self.search1(board)
        elif depth == 2:
            return self.search2(board)
        elif depth == 3:
            return self.search3(board)
        elif depth == 4:
            return self.search4(board)
        elif depth == 5:
            return self.search5(board)
        elif depth == 6:
            return self.search6(board)
        elif depth == 7:
            return self.search7(board)
        else:
            self.max_d = 2
            return self.search2(board)

    def play(self, depth=2, max_step=1e6):
        board = np.uint64(17)
        step = 0
        empty_slots = 14
        self.interact = False
        while step < max_step:
            if empty_slots > 10:
                self.max_d = min(depth, 1)
                self.dispatcher(board)
            elif empty_slots > 4:
                self.max_d = min(depth, 2)
                self.dispatcher(board)
            elif empty_slots > 2:
                self.max_d = min(depth, 3)
                self.dispatcher(board)
            else:
                self.max_d = 4
                self.dispatcher(board)
                while self.node < 400000 and self.max_d <= depth:
                    self.max_d += 1
                    self.node = 0
                    self.dispatcher(board)

            self.node = 0
            board = self.bm.move_board(board, self.best_operation)
            step += 1
            board, empty_slots = self.bm.gen_new_num(board)
            if step % 1 == 0:
                print(self.bm.decode_board(board))
                print('')
            if empty_slots == 0:
                break
        self.interact = True
        return self.bm.decode_board(board)
