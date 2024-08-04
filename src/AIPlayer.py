import time

import numpy as np
from numba import int32, types, uint64, from_dtype, float32, uint8
from numba.experimental import jitclass

from BoardMover import BoardMoverWithScore
from BookReader import BookReader
from Config import SingletonConfig


spec_c = {
    'length': uint64,
    'cache_board': from_dtype(np.dtype('uint64,int32'))[:],
    'lookup_count': uint64,
}


@jitclass(spec_c)
class Cache:
    def __init__(self):
        self.length = 8388607  # 2的幂-1
        self.cache_board = np.zeros(self.length + 1, dtype='uint64,int32')
        self.lookup_count = 0

    def clear(self):
        self.lookup_count = 0
        self.cache_board = np.zeros(self.length + 1, dtype='uint64,int32')

    def lookup(self, board, depth):
        self.lookup_count += 1
        index = self.hash(board, depth)
        return self.cache_board[index]['f1'] if self.cache_board[index]['f0'] == board else None

    def update(self, board, depth, score):
        index = self.hash(board, depth)
        self.cache_board[index]['f0'], self.cache_board[index]['f1'] = board, score

    def hash(self, board, depth):
        board = (board ^ (board >> 27)) * 0x1A85EC53 + board >> 23
        return np.uint64(board - (depth << 5)) & self.length


spec = {
    'max_d': int32,
    'best_operation': uint8,
    'board': types.Array(int32, 2, 'C'),  # board是二维整数数组
    'diffs': int32[:],
    'diffs2': int32[:],
    'bm': BoardMoverWithScore.class_type.instance_type,
    'cache': Cache.class_type.instance_type,
    'node': uint64,
    'min_prob': float32,
    'spawn_rate4': float32,
}


# numba.jitclass的类型推断系统不支持复杂递归，搜索部分必须拆开
@jitclass(spec)
class AutoplayS:
    def __init__(self, board):
        # 如果修改 init 需要同步修改 reset_board
        self.max_d = 3
        self.best_operation = 0
        self.board = board
        self.bm = BoardMoverWithScore()
        self.cache = Cache()
        self.diffs, self.diffs2 = self.calculates_for_evaluate()
        self.node = 0
        self.min_prob = 0.0001
        self.spawn_rate4 = 0.1
        print('AI init')

    def reset_board(self, board):
        self.best_operation = 0
        self.board = board
        self.node = 0

    def calculates_for_evaluate(self):
        diffs = np.empty(65536, dtype=np.int32)
        diffs2 = np.empty(65536, dtype=np.int32)
        # 生成所有可能的行
        for i in range(65536):
            line = [(i // (16 ** j)) % 16 for j in range(4)]
            line = [1 << k if k else 0 for k in line]
            diffs[i] = self.diffs_evaluation_func(line)
            diffs2[i] = self.diffs_evaluation_func(line[::-1])
        return diffs, diffs2

    @staticmethod
    def diffs_evaluation_func(line):
        mask_num = [0, 128, 168, 180, 184, 188]
        line_masked = [min(1024 + mask_num[int(np.log2(max(k >> 10, 1)))], k) for k in line]
        score = line_masked[0]
        for x in range(3):
            if line_masked[x + 1] > line_masked[x]:
                if line_masked[x] < 1080:
                    score -= (line_masked[x + 1] - line_masked[x]) << 3
                else:
                    score += min(line_masked[x + 1], line_masked[x]) << 1
            elif x < 2:
                score += line_masked[x + 1] + line_masked[x]
            else:
                score += int((line_masked[x + 1] + line_masked[x]) * 0.75)
        return int32(score / 4)

    def evaluate(self, s):
        self.node += 1
        s_reverse = self.bm.reverse(s)
        diffv1, diffv2, diffh1, diffh2 = np.int32(0), np.int32(0), np.int32(0), np.int32(0)
        for i in range(4):
            l1 = (s >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            l2 = (s_reverse >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            diffh1 += self.diffs[l1]
            diffh2 += self.diffs2[l1]
            diffv1 += self.diffs[l2]
            diffv2 += self.diffs2[l2]
        return max(diffv1, diffv2) + max(diffh1, diffh2)

    def search0(self, b):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, score - 32) if score < 4000 else 32768
            temp = self.cache.lookup(t, 0)
            if temp is None:
                temp = int32(self.evaluate(t))
                self.cache.update(t, 0, temp)
            if temp + score >= best:
                best = temp + score
        return best

    def search1(self, b, prob=1.0):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, score - 32) if score < 4000 else 32768
            temp = self.cache.lookup(t, 1)
            if temp is None:
                temp = np.float64(0.0)
                empty_slots_count = 0
                # 每个空位分别尝试放入2和4
                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                        empty_slots_count += 1
                        temp_t = self.search0(np.uint64(t | (1 << (4 * i))))
                        if prob * self.spawn_rate4 > self.min_prob:
                            temp_t = temp_t * (1-self.spawn_rate4) + self.search0(
                                np.uint64(t | (2 << (4 * i)))) * self.spawn_rate4
                        temp += temp_t
                temp = int32(temp / empty_slots_count)
                self.cache.update(t, 1, temp)
            if temp + score >= best:
                best = temp + score
                if self.max_d == 1:
                    self.best_operation = d
        return best

    def search2(self, b, prob=1.0):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, score - 32) if score < 4000 else 32768
            temp = self.cache.lookup(t, 2)
            if temp is None:
                temp = np.float64(0.0)
                empty_slots_count = 0
                # 每个空位分别尝试放入2和4
                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                        empty_slots_count += 1
                        temp_t = self.search1(np.uint64(t | (1 << (4 * i))), prob * (1-self.spawn_rate4))
                        if prob * self.spawn_rate4 > self.min_prob:
                            temp_t = temp_t * (1-self.spawn_rate4) + self.search1(
                                np.uint64(t | (2 << (4 * i))), prob * self.spawn_rate4) * self.spawn_rate4
                        temp += temp_t
                temp = int32(temp / empty_slots_count)
                self.cache.update(t, 2, temp)
            if temp + score >= best:
                best = temp + score
                if self.max_d == 2:
                    self.best_operation = d
        return best

    def search3(self, b, prob=1.0):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, score - 32) if score < 4000 else 32768
            temp = self.cache.lookup(t, 3)
            if temp is None:
                temp = np.float64(0.0)
                empty_slots_count = 0
                # 每个空位分别尝试放入2和4
                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                        empty_slots_count += 1
                        temp_t = self.search2(np.uint64(t | (1 << (4 * i))), prob * (1-self.spawn_rate4))
                        if prob * self.spawn_rate4 > self.min_prob:
                            temp_t = temp_t * (1-self.spawn_rate4) + self.search2(
                                np.uint64(t | (2 << (4 * i))), prob * self.spawn_rate4) * self.spawn_rate4
                        temp += temp_t
                temp = int32(temp / empty_slots_count)
                self.cache.update(t, 3, temp)
            if temp + score >= best:
                best = temp + score
                if self.max_d == 3:
                    self.best_operation = d
        return best

    def search4(self, b, prob=1.0):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, score - 32) if score < 4000 else 32768
            temp = self.cache.lookup(t, 4)
            if temp is None:
                temp = np.float64(0.0)
                empty_slots_count = 0
                # 每个空位分别尝试放入2和4
                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                        empty_slots_count += 1
                        temp_t = self.search3(np.uint64(t | (1 << (4 * i))), prob * (1-self.spawn_rate4))
                        if prob * self.spawn_rate4 > self.min_prob:
                            temp_t = temp_t * (1-self.spawn_rate4) + self.search3(
                                np.uint64(t | (2 << (4 * i))), prob * self.spawn_rate4) * self.spawn_rate4
                        temp += temp_t
                temp = int32(temp / empty_slots_count)
                self.cache.update(t, 4, temp)
            if temp + score >= best:
                best = temp + score
                if self.max_d == 4:
                    self.best_operation = d
        return best

    def search5(self, b, prob=1.0):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, score - 32) if score < 4000 else 32768
            temp = self.cache.lookup(t, 5)
            if temp is None:
                temp = np.float64(0.0)
                empty_slots_count = 0
                # 每个空位分别尝试放入2和4
                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                        empty_slots_count += 1
                        temp_t = self.search4(np.uint64(t | (1 << (4 * i))), prob * (1-self.spawn_rate4))
                        if prob * self.spawn_rate4 > self.min_prob:
                            temp_t = temp_t * (1-self.spawn_rate4) + self.search4(
                                np.uint64(t | (2 << (4 * i))), prob * self.spawn_rate4) * self.spawn_rate4
                        temp += temp_t
                temp = int32(temp / empty_slots_count)
                self.cache.update(t, 5, temp)
            if temp + score >= best:
                best = temp + score
                if self.max_d == 5:
                    self.best_operation = d
        return best

    def search6(self, b, prob=1.0):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, score - 32) if score < 4000 else 32768
            temp = self.cache.lookup(t, 6)
            if temp is None:
                temp = np.float64(0.0)
                empty_slots_count = 0
                # 每个空位分别尝试放入2和4
                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                        empty_slots_count += 1
                        temp_t = self.search5(np.uint64(t | (1 << (4 * i))), prob * (1-self.spawn_rate4))
                        if prob * self.spawn_rate4 > self.min_prob:
                            temp_t = temp_t * (1-self.spawn_rate4) + self.search5(
                                np.uint64(t | (2 << (4 * i))), prob * self.spawn_rate4) * self.spawn_rate4
                        temp += temp_t
                temp = int32(temp / empty_slots_count)
                self.cache.update(t, 6, temp)
            if temp + score >= best:
                best = temp + score
                if self.max_d == 6:
                    self.best_operation = d
        return best

    def search7(self, b, prob=1.0):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, score - 32) if score < 4000 else 32768
            temp = self.cache.lookup(t, 7)
            if temp is None:
                temp = np.float64(0.0)
                empty_slots_count = 0
                # 每个空位分别尝试放入2和4
                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                        empty_slots_count += 1
                        temp_t = self.search6(np.uint64(t | (1 << (4 * i))), prob * (1-self.spawn_rate4))
                        if prob * self.spawn_rate4 > self.min_prob:
                            temp_t = temp_t * (1-self.spawn_rate4) + self.search6(
                                np.uint64(t | (2 << (4 * i))), prob * self.spawn_rate4) * self.spawn_rate4
                        temp += temp_t
                temp = int32(temp / empty_slots_count)
                self.cache.update(t, 7, temp)
            if temp + score >= best:
                best = temp + score
                if self.max_d == 7:
                    self.best_operation = d
        return best

    def search8(self, b, prob=1.0):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, score - 32) if score < 4000 else 32768
            temp = self.cache.lookup(t, 8)
            if temp is None:
                temp = np.float64(0.0)
                empty_slots_count = 0
                # 每个空位分别尝试放入2和4
                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                        empty_slots_count += 1
                        temp_t = self.search7(np.uint64(t | (1 << (4 * i))), prob * (1-self.spawn_rate4))
                        if prob * self.spawn_rate4 > self.min_prob:
                            temp_t = temp_t * (1-self.spawn_rate4) + self.search7(
                                np.uint64(t | (2 << (4 * i))), prob * self.spawn_rate4) * self.spawn_rate4
                        temp += temp_t
                temp = int32(temp / empty_slots_count)
                self.cache.update(t, 8, temp)
            if temp + score >= best:
                best = temp + score
                if self.max_d == 8:
                    self.best_operation = d
        return best

    def search9(self, b, prob=1.0):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, score - 32) if score < 4000 else 32768
            temp = self.cache.lookup(t, 9)
            if temp is None:
                temp = np.float64(0.0)
                empty_slots_count = 0
                # 每个空位分别尝试放入2和4
                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                        empty_slots_count += 1
                        temp_t = self.search8(np.uint64(t | (1 << (4 * i))), prob * (1-self.spawn_rate4))
                        if prob * self.spawn_rate4 > self.min_prob:
                            temp_t = temp_t * (1-self.spawn_rate4) + self.search8(
                                np.uint64(t | (2 << (4 * i))), prob * self.spawn_rate4) * self.spawn_rate4
                        temp += temp_t
                temp = int32(temp / empty_slots_count)
                self.cache.update(t, 9, temp)
            if temp + score >= best:
                best = temp + score
                if self.max_d == 9:
                    self.best_operation = d
        return best

    def start_search(self, depth=3):
        self.cache.clear()
        self.best_operation = 0
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
        elif depth == 8:
            return self.search8(board)
        elif depth == 9:
            return self.search9(board)
        else:
            self.max_d = 6
            return self.search6(board)

    def play(self, depth=2, max_step=1e6):
        board = np.uint64(17)
        step = 0
        empty_slots = 14
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
            board, empty_slots, _, __ = self.bm.gen_new_num(board)
            if step % 1 == 0:
                print(self.bm.decode_board(board))
                print('')
            if empty_slots == 0:
                break
        return self.bm.decode_board(board)


class Dispatcher:
    def __init__(self, board, board_encoded):
        self.board_encoded = board_encoded
        self.counts = self.frequency_count()
        self.board = board
        self.last_operator = 0  # 0:AI 1:LL 2:4431 3:4441 4:4432 5:free10 6:free9
        self.transfer_control_count = [0, 0, 0, 0, 0]  # 4441, 4432, LL, free, AI

    def reset(self, board, board_encoded):
        self.board_encoded = board_encoded
        self.counts = self.frequency_count()
        self.board = board

    # 统计各个数字数量
    def frequency_count(self):
        counts = np.zeros(16, dtype=np.uint32)
        for i in range(16):
            num_code = (self.board_encoded >> np.uint64((4 * i))) & np.uint64(0xF)
            counts[num_code] += 1
        return counts

    # mask最大的n个格子
    def mask(self, n):
        flat_arr = self.board.flatten()
        max_indices = np.argpartition(flat_arr, -n)[-n:]
        flat_arr[max_indices] = 32768
        masked_board = flat_arr.reshape(4, 4)
        if masked_board.sum() - n * 32768 < 24:
            masked_board = np.array([[32768, 32768, 32768, 32768],
                                     [0, 32768, 32768, 0],
                                     [0, 32768, 32768, 0],
                                     [32768, 32768, 32768, 32768]], dtype=np.int32)  # 其他数字太小让AI玩
        return masked_board

    # mask最大的4个格子, 5大数特殊处理适配final2k的非常规排列。非32k数字和太小返回0
    def mask_LL(self):
        flat_arr = self.board.flatten()
        max_indices = np.argpartition(flat_arr, -4)[-4:]
        nth_largest = flat_arr[max_indices[0]]
        nth_largest_pos = (max_indices[0] // 4, max_indices[0] % 4)
        flat_arr[max_indices] = 32768
        masked_board = flat_arr.reshape(4, 4)
        if np.all(self.counts[-5:] == 1):
            max_indices = np.argpartition(flat_arr, -5)[-5:]
            corners = [
                {0, 1, 4, 5},  # 左上角
                {2, 3, 6, 7},  # 右上角
                {8, 9, 12, 13},  # 左下角
                {10, 11, 14, 15}  # 右下角
            ]
            # 检查每个角落
            for corner in corners:
                # 计算交集，看是否有四个索引在同一个角落
                if len(corner.intersection(max_indices)) == 4:
                    # 找到剩下的那个索引
                    remaining_index = list(set(max_indices) - corner)[0]
                    # 将角落的四个索引对应的值设为 32768
                    for idx in corner:
                        flat_arr[idx] = 32768
                    # 将剩下的索引对应的值设为 2048
                    flat_arr[remaining_index] = 2048
                    masked_board = flat_arr.reshape(4, 4)
                    break
        if masked_board.sum() - 4 * 32768 < 80:
            masked_board = np.array([[32768, 32768, 32768, 32768],
                                     [0, 32768, 32768, 0],
                                     [0, 32768, 32768, 0],
                                     [32768, 32768, 32768, 32768]], dtype=np.int32)  # 其他数字太小让AI玩
        return masked_board, nth_largest, nth_largest_pos

    def check_LL_4431(self):
        masked_board, target, nth_largest_pos = self.mask_LL()
        # LL
        pos = 1 if (0 in nth_largest_pos or 3 in nth_largest_pos) else 0
        full_pattern = 'LL_' + str(target) + '_' + str(pos)
        if full_pattern in SingletonConfig().config['filepath_map'].keys():
            if pos == 1 and nth_largest_pos in ((2, 3), (1, 0), (0, 2), (3, 1)):
                r1 = BookReader.move_on_dic(masked_board.T, 'LL', str(target), full_pattern, str(pos))
            else:
                r1 = BookReader.move_on_dic(masked_board, 'LL', str(target), full_pattern, str(pos))
            move = list(r1.keys())[0]
            success_rate_threshold = {4096: 0.02, 2048: 0.1, 1024: 0.4, 512: 0.6}.get(target, 0.05)
            if isinstance(r1[move], float) and success_rate_threshold < r1[move]:
                prob = r1[move]
                if pos == 1 and nth_largest_pos in ((2, 3), (1, 0), (0, 2), (3, 1)):
                    move = {'up': 'left', 'left': 'up', 'right': 'down', 'down': 'right'}[move]
                self.last_operator = 1
                return (self.transfer_control(prob, 2) or move) if target == 512 else move
        # 没有对应定式的话 'LL_4096_0' 给LL兜底
        elif 'LL_4096_0' in SingletonConfig().config['filepath_map'].keys():
            r1 = BookReader.move_on_dic(masked_board, 'LL', str(4096), 'LL_4096_0', str(0))
            move = list(r1.keys())[0]
            if isinstance(r1[move], float) and 0.02 < r1[move]:
                self.last_operator = 1
                return move
        # 4431
        if '4431_2048' in SingletonConfig().config['filepath_map'].keys():
            r1 = BookReader.move_on_dic(masked_board, '4431', '2048', '4431_2048')
            move = list(r1.keys())[0]
            if isinstance(r1[move], float) and 0.2 < r1[move] < 1:
                self.last_operator = 2
                return move
        self.last_operator = 0
        return 'AI'

    def check_free10w_1024(self):
        if 'free10w_1024' not in SingletonConfig().config['filepath_map'].keys():
            return self.check_LL_4431()
        masked_board = self.mask(6)
        r1 = BookReader.move_on_dic(masked_board, 'free10w', '1024', 'free10w_1024')
        move = list(r1.keys())[0]
        if isinstance(r1[move], float) and r1[move] < 1:
            self.last_operator = 5
            return move
        else:
            self.last_operator = 0
            return 'AI'

    def check_free9w_256(self):
        if 'free9w_256' not in SingletonConfig().config['filepath_map'].keys():
            return self.check_LL_4431()
        masked_board = self.mask(7)
        r1 = BookReader.move_on_dic(masked_board, 'free9w', '256', 'free9w_256')
        move = list(r1.keys())[0]
        if isinstance(r1[move], float) and r1[move] < 1:
            self.last_operator = 6
            return move
        else:
            return self.check_LL_4431()

    # 衔接方式：如果定式成功率连续为1达n次，换ai走至少m步。ai走m步后清空全部计数，若定式成功率仍为1，继续AI走；若m步后直到定式成功率不为1，定式走
    # 定式多个方向成功率均为1时，会出现随机移动，可能造成最后成功后AI接手时立刻暴毙。
    # 但是如果定式成功率为1就立刻让AI接手，对于4441-512,4432-512等小定式，在成功之前很多步成功率已经为1，此时AI仍有小概率走向死亡
    def transfer_control(self, prob, sender):
        n = 5 if sender > 1 else 7
        m = 3
        if prob == 1:
            self.transfer_control_count[sender] += 1
            if self.transfer_control_count[sender] >= n:
                self.transfer_control_count[-1] += 1
                if self.transfer_control_count[-1] >= m:
                    self.transfer_control_count = [0, 0, 0, -1]
                self.last_operator = 0
                return 'AI'
            elif self.transfer_control_count[-1] == -1:
                self.last_operator = 0
                return 'AI'
            elif self.counts[9] > 1 or self.counts[8] > 1:
                self.last_operator = 0
                return 'AI'
        else:
            if self.transfer_control_count[-1] <= 0:
                self.transfer_control_count = [0, 0, 0, 0]
            else:
                self.transfer_control_count[-1] += 1
                if self.transfer_control_count[-1] >= m:
                    self.transfer_control_count = [0, 0, 0, 0]
                self.last_operator = 0
                return 'AI'

    def check_4432_512_4441_512(self):
        masked_board = self.mask(3)
        if '4432_512' in SingletonConfig().config['filepath_map'].keys():
            r1 = BookReader.move_on_dic(masked_board, '4432', '512', '4432_512')
            move = list(r1.keys())[0]
            if isinstance(r1[move], float) and 0.9 < r1[move]:
                self.last_operator = 4
                return self.transfer_control(r1[move], 1) or move
        if '4441_512' in SingletonConfig().config['filepath_map'].keys():
            r1 = BookReader.move_on_dic(masked_board, '4441', '512', '4441_512')
            move = list(r1.keys())[0]
            if isinstance(r1[move], float) and 0.9 < r1[move]:
                self.last_operator = 3
                return self.transfer_control(r1[move], 0) or move
        self.last_operator = 0
        return 'AI'

    # 前n大数有相同值
    def with_duplicate(self, n):
        i = -1
        c = 0
        while c < n and i > -12:
            if self.counts[i] > 1:
                return True
            c += self.counts[i]
            i -= 1
        return False

    def is_endgame_65k(self):
        """判断是否是残局，如果是就使用free定式.
        1、是残局 2、不即将成功(大数无重复小数无连续合并) 3、其余数字不是太小（free定式初始状态局面不全），至少有一个16"""
        return np.all(self.counts[-6:] == 1) and not self.is_sequential_merging(4, False) and \
            np.sum(self.counts[-12:-6]) > 0  # and not self.with_duplicate(6)

    # free9 256
    def is_endgame_32k(self):
        return np.sum(self.counts[-8:]) == 7 and np.sum(self.counts[-4:]) < 4 and \
            not self.is_sequential_merging(4, False) and np.sum(self.counts[-12:-8]) > 0 and not self.with_duplicate(7)

    # 判断是否可以连续合成(默认要求可合并数字相连），如果是就使用AI（衔接）
    def is_sequential_merging(self, k=3, check_neighboring=True):
        for i in (9, 10, 11):
            if np.sum(self.counts[:i]) == 12 and self.counts[i] == 1:
                for j in range(i - 1, i - k, -1):
                    if self.counts[j] == 2:
                        return self.check_neighboring(j) if check_neighboring else True
                    if self.counts[j] == 0:
                        return False
        return False

    def check_neighboring(self, j):
        if not self.counts[j] == 2:
            return False
        pos_h, pos_v = np.where(self.board == 2 ** j)
        return (pos_h[0] == pos_h[1] and abs(pos_v[0] - pos_v[1]) == 1) or \
            (pos_v[0] == pos_v[1] and abs(pos_h[0] - pos_h[1]) == 1)

    def dispatcher(self):
        large_tiles = np.sum(self.counts[-7:])  # 大于等于512的格子数
        if large_tiles < 2:
            self.last_operator = 0
            return 'AI'
        if large_tiles == 2:
            return self.check_4432_512_4441_512()
        if (large_tiles < 6 and self.with_duplicate(large_tiles)) or self.is_sequential_merging():
            self.last_operator = 0
            return 'AI'
        if large_tiles == 3:
            return self.check_4432_512_4441_512()
        if large_tiles == 4:
            return self.check_LL_4431()
        if large_tiles > 4:
            if self.is_endgame_65k():
                return self.check_free10w_1024()
            if self.is_endgame_32k():
                return self.check_free9w_256()
            return self.check_LL_4431()
        self.last_operator = 0
        return 'AI'


spec = {
    'max_d': int32,
    'hardest_pos': uint8,
    'hardest_num': uint8,
    'board': types.Array(int32, 2, 'C'),
    'diffs': int32[:],
    'diffs2': int32[:],
    'bm': BoardMoverWithScore.class_type.instance_type,
    'cache': Cache.class_type.instance_type,
    'node': uint64,
}


@jitclass(spec)
class EvilGen:
    def __init__(self, board):
        # 如果修改 init 需要同步修改 reset_board
        self.max_d = 3
        self.hardest_pos = 0
        self.hardest_num = 1
        self.board = board
        self.bm = BoardMoverWithScore()
        self.cache = Cache()
        self.diffs, self.diffs2 = self.calculates_for_evaluate()
        self.node = 0
        print('Evil gen init')

    def reset_board(self, board):
        self.hardest_pos = 0
        self.hardest_num = 1
        self.board = board
        self.node = 0

    def calculates_for_evaluate(self):
        diffs = np.empty(65536, dtype=np.int32)
        diffs2 = np.empty(65536, dtype=np.int32)
        # 生成所有可能的行
        for i in range(65536):
            line = [(i // (16 ** j)) % 16 for j in range(4)]
            line = [1 << k if k else 0 for k in line]
            diffs[i] = self.diffs_evaluation_func(line)
            diffs2[i] = self.diffs_evaluation_func(line[::-1])
        return diffs, diffs2

    @staticmethod
    def diffs_evaluation_func(line):
        mask_num = [0, 128, 168, 180, 184, 188]
        line_masked = [min(1024 + mask_num[int(np.log2(max(k >> 10, 1)))], k) for k in line]
        score = line_masked[0]
        for x in range(3):
            if line_masked[x + 1] > line_masked[x]:
                if line_masked[x] < 1080:
                    score -= (line_masked[x + 1] - line_masked[x]) << 3
                else:
                    score += min(line_masked[x + 1], line_masked[x]) << 1
            elif x < 2:
                score += line_masked[x + 1] + line_masked[x]
            else:
                score += int((line_masked[x + 1] + line_masked[x]) * 0.75)
        return int32(score / 4)

    def evaluate(self, s):
        self.node += 1
        s_reverse = self.bm.reverse(s)
        diffv1, diffv2, diffh1, diffh2 = np.int32(0), np.int32(0), np.int32(0), np.int32(0)
        for i in range(4):
            l1 = (s >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            l2 = (s_reverse >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            diffh1 += self.diffs[l1]
            diffh2 += self.diffs2[l1]
            diffv1 += self.diffs[l2]
            diffv2 += self.diffs2[l2]
        return max(diffv1, diffv2) + max(diffh1, diffh2)

    def search1(self, b):
        evil = 131072
        for i in range(16):  # 遍历所有位置
            if ((b >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                for num in (1, 2):
                    t = np.uint64(b | (num << (4 * i)))
                    best = -131072
                    for d in [1, 3, 4, 2]:
                        t1, score = self.bm.move_board(t, d)
                        if t1 == t:
                            continue
                        temp = self.cache.lookup(t1, 1)
                        if temp is None:
                            temp = self.evaluate(t1)
                            self.cache.update(t1, 1, temp)
                        best = max(best, temp + score)  # 玩家通过操作最大化得分
                        if best >= evil:
                            break  # Beta剪枝
                    if best <= evil:
                        evil = best
                        if self.max_d == 1:
                            self.hardest_pos = i
                            self.hardest_num = num
        return evil

    def search2(self, b):
        evil = 131072
        for i in range(16):  # 遍历所有位置
            if ((b >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                for num in (1, 2):
                    t = np.uint64(b | (num << (4 * i)))
                    best = -131072
                    for d in [1, 3, 4, 2]:
                        t1, score = self.bm.move_board(t, d)
                        if t1 == t:
                            continue
                        temp = self.cache.lookup(t1, 2)
                        if temp is None:
                            temp = self.search1(t1)
                            self.cache.update(t1, 2, temp)
                        best = max(best, temp + score)  # 玩家通过操作最大化得分
                        if best >= evil:
                            break  # Beta剪枝
                    if best <= evil:
                        evil = best
                        if self.max_d == 2:
                            self.hardest_pos = i
                            self.hardest_num = num
        return evil

    def search3(self, b):
        evil = 131072
        for i in range(16):  # 遍历所有位置
            if ((b >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                for num in (1, 2):
                    t = np.uint64(b | (num << (4 * i)))
                    best = -131072
                    for d in [1, 3, 4, 2]:
                        t1, score = self.bm.move_board(t, d)
                        if t1 == t:
                            continue
                        temp = self.cache.lookup(t1, 3)
                        if temp is None:
                            temp = self.search2(t1)
                            self.cache.update(t1, 3, temp)
                        best = max(best, temp + score)  # 玩家通过操作最大化得分
                        if best >= evil:
                            break  # Beta剪枝
                    if best <= evil:
                        evil = best
                        if self.max_d == 3:
                            self.hardest_pos = i
                            self.hardest_num = num
        return evil

    def search4(self, b):
        evil = 131072
        for i in range(16):  # 遍历所有位置
            if ((b >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                for num in (1, 2):
                    t = np.uint64(b | (num << (4 * i)))
                    best = -131072
                    for d in [1, 3, 4, 2]:
                        t1, score = self.bm.move_board(t, d)
                        if t1 == t:
                            continue
                        temp = self.cache.lookup(t1, 4)
                        if temp is None:
                            temp = self.search3(t1)
                            self.cache.update(t1, 4, temp)
                        best = max(best, temp + score)  # 玩家通过操作最大化得分
                        if best >= evil:
                            break  # Beta剪枝
                    if best <= evil:
                        evil = best
                        if self.max_d == 4:
                            self.hardest_pos = i
                            self.hardest_num = num
        return evil

    def search5(self, b):
        evil = 131072
        for i in range(16):  # 遍历所有位置
            if ((b >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                for num in (1, 2):
                    t = np.uint64(b | (num << (4 * i)))
                    best = -131072
                    for d in [1, 3, 4, 2]:
                        t1, score = self.bm.move_board(t, d)
                        if t1 == t:
                            continue
                        temp = self.cache.lookup(t1, 5)
                        if temp is None:
                            temp = self.search4(t1)
                            self.cache.update(t1, 5, temp)
                        best = max(best, temp + score)  # 玩家通过操作最大化得分
                        if best >= evil:
                            break  # Beta剪枝
                    if best <= evil:
                        evil = best  # 数字生成机制减小得分
                        if self.max_d == 5:
                            self.hardest_pos = i
                            self.hardest_num = num
        return evil

    def search6(self, b):
        evil = 131072
        for i in range(16):  # 遍历所有位置
            if ((b >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                for num in (1, 2):
                    t = np.uint64(b | (num << (4 * i)))
                    best = -131072
                    for d in [1, 3, 4, 2]:
                        t1, score = self.bm.move_board(t, d)
                        if t1 == t:
                            continue
                        temp = self.cache.lookup(t1, 6)
                        if temp is None:
                            temp = self.search5(t1)
                            self.cache.update(t1, 6, temp)
                        best = max(best, temp + score)  # 玩家通过操作最大化得分
                        if best >= evil:
                            break  # Beta剪枝
                    if best <= evil:
                        evil = best  # 数字生成机制减小得分
                        if self.max_d == 6:
                            self.hardest_pos = i
                            self.hardest_num = num
        return evil

    def search7(self, b):
        evil = 131072
        for i in range(16):  # 遍历所有位置
            if ((b >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                for num in (1, 2):
                    t = np.uint64(b | (num << (4 * i)))
                    best = -131072
                    for d in [1, 3, 4, 2]:
                        t1, score = self.bm.move_board(t, d)
                        if t1 == t:
                            continue
                        temp = self.cache.lookup(t1, 7)
                        if temp is None:
                            temp = self.search6(t1)
                            self.cache.update(t1, 7, temp)
                        best = max(best, temp + score)  # 玩家通过操作最大化得分
                        if best >= evil:
                            break  # Beta剪枝
                    if best <= evil:
                        evil = best  # 数字生成机制减小得分
                        if self.max_d == 7:
                            self.hardest_pos = i
                            self.hardest_num = num
        return evil

    def start_search(self, depth=3):
        self.cache.clear()
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
            self.max_d = 5
            return self.search5(board)

    def gen_new_num(self, depth=5):
        self.start_search(depth)
        board_encoded = self.bm.encode_board(self.board)
        return board_encoded | (self.hardest_num << (4 * self.hardest_pos)), 15 - self.hardest_pos, self.hardest_num


if __name__ == "__main__":
    b1 = np.array([
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 2]], dtype=np.int32)
    print(b1)
    s1 = AutoplayS(b1)
    g1 = EvilGen(b1)
    s1.start_search(2)
    s1.start_search(3)
    g1.gen_new_num(2)
    g1.gen_new_num(3)
    b1 = np.uint64(s1.bm.encode_board(b1))
    t_start = time.time()
    for steps in range(500):
        s1.reset_board(s1.bm.decode_board(b1))
        t0 = time.time()
        s1.start_search(7)
        # print(s1.cache.lookup_count / (time.time() - t0), s1.node / (time.time() - t0), time.time() - t0)
        print({0:None, 1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}[s1.best_operation])
        if not s1.best_operation:
            break
        b1 = np.uint64(s1.bm.move_board(b1, s1.best_operation)[0])
        g1.reset_board(g1.bm.decode_board(b1))
        b1 = np.uint64(g1.gen_new_num(7)[0])
        print(s1.bm.decode_board(b1))
    print(time.time() - t_start)
