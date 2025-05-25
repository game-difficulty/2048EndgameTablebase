import time

import numpy as np
from numba import int32, types, uint64, from_dtype, float32, uint8, njit
from numba.experimental import jitclass

from BoardMover import BoardMoverWithScore
from BookReader import BookReaderDispatcher
from Config import SingletonConfig


@njit()
def diffs_evaluation_func(line_masked):
    if ((line_masked[0] == 598 and line_masked[1] == 598) or (line_masked[1] == 598 and line_masked[2] == 598)
            or (line_masked[2] == 598 and line_masked[3] == 598)):
        return 16384

    # dpdf
    score_dpdf = line_masked[0]
    for x in range(3):
        if line_masked[x + 1] > line_masked[x]:
            if line_masked[x] > 400:
                score_dpdf += (line_masked[x] << 1) + (line_masked[x + 1] - line_masked[x]) * x
            elif line_masked[x] > 300 and x == 1 and line_masked[0] > line_masked[1]:
                score_dpdf += (line_masked[x] << 1)
            else:
                score_dpdf -= (line_masked[x + 1] - line_masked[x]) << 3
        elif x < 2:
            score_dpdf += line_masked[x + 1] + line_masked[x]
        else:
            score_dpdf += int((line_masked[x + 1] + line_masked[x]) * 0.5)

    # t
    if min(line_masked[0], line_masked[3]) < 16:
        score_t = -16384
    elif (line_masked[0] < line_masked[1] and line_masked[0] < 400) or (
            line_masked[3] < line_masked[2] and line_masked[3] < 400):
        score_t = -(max(line_masked[1], line_masked[2]) << 3)
    else:
        score_t = (int(line_masked[0] * 3.75 + line_masked[1] + line_masked[3] * 3.75 + line_masked[2]) >> 1) + max(
            line_masked[1], line_masked[2])

    return int32(max(score_dpdf, score_t) / 4)


@njit()
def calculates_for_evaluate():
    diffs = np.empty(65536, dtype=np.int32)
    diffs2 = np.empty(65536, dtype=np.int32)
    mask_num = [0, 112, 240, 360, 420, 440, 450, 460, 470, 480, 480]
    # 生成所有可能的行
    for i in range(65536):
        line = [(i // (16 ** j)) % 16 for j in range(4)]
        line = [1 << k if k else 0 for k in line]
        line_masked = [min(128 + mask_num[int(np.log2(max(k >> 7, 1)))], k) for k in line]
        diffs[i] = diffs_evaluation_func(line_masked)
        diffs2[i] = diffs_evaluation_func(line_masked[::-1])
    return diffs, diffs2


_diffs, _diffs2 = calculates_for_evaluate()


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
        board = (board ^ (board >> 27)) * 0x1A85EC53 + (board >> 23)
        return np.uint64(board - (depth << 5)) & self.length


spec = {
    'max_d': int32,
    'best_operation': uint8,
    'board': types.Array(int32, 2, 'C'),  # board是二维整数数组
    'diffs': int32[:],
    'diffs2': int32[:],
    'bm': BoardMoverWithScore.class_type.instance_type,  # type: ignore
    'cache': Cache.class_type.instance_type,  # type: ignore
    'node': uint64,
    'min_prob': float32,
    'spawn_rate4': float32,
}


# numba.jitclass的类型推断系统不支持复杂递归，搜索部分必须拆开
@jitclass(spec)
class AIPlayer:
    def __init__(self, board):
        # 如果修改 init 需要同步修改 reset_board
        self.max_d = 3
        self.best_operation = 0
        self.board: np.ndarray = board
        self.bm = BoardMoverWithScore()
        self.cache = Cache()
        self.diffs, self.diffs2 = _diffs, _diffs2
        self.node = 0
        self.min_prob = 0.00005
        self.spawn_rate4 = 0.1
        print('AI init')

    def reset_board(self, board):
        self.best_operation = 0
        self.board = board
        self.node = 0

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
        result = max(diffv1, diffv2) + max(diffh1, diffh2)
        if 1700 > result > 1400 and self.check_corner(s):
            result -= 3600
        return result

    @staticmethod
    def check_corner(s):
        c = 0
        tile0 = (s >> np.uint64(0)) & np.uint64(0xF)
        c += (tile0 > 0x6)
        tile12 = (s >> np.uint64(12)) & np.uint64(0xF)
        c += (tile12 > 0x6)
        tile48 = (s >> np.uint64(48)) & np.uint64(0xF)
        c += (tile48 > 0x6)
        tile60 = (s >> np.uint64(60)) & np.uint64(0xF)
        c += (tile60 > 0x6)
        return c > 2

    def search0(self, b):
        best = -131072
        for d in [1, 3, 4, 2]:
            t, score = self.bm.move_board(b, d)
            if t == b:
                continue
            score = max(0, (score >> 1) - 8) if score < 1000 else (1000 if score < 2000 else score * 2)
            temp = self.cache.lookup(t, 0)
            if temp is None:
                temp = int32(self.evaluate(t))
                self.cache.update(t, 0, temp)
            if temp + score >= best:
                best = temp + score
        return best

    def start_search(self, depth=3):
        self.cache.clear()
        self.best_operation = 0
        self.max_d = depth
        self.dispatcher(self.bm.encode_board(self.board))

    def dispatcher(self, board: np.uint64):
        self.node = 0
        return search_ai_player(self, board, 1, self.max_d)

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
                    self.dispatcher(board)

            board, score = self.bm.move_board(board, self.best_operation)
            step += 1
            board, empty_slots, _, __ = self.bm.gen_new_num(board)
            if step % 100 == 0:
                # print(self.bm.decode_board(board))
                # print('')
                pass
            if empty_slots == 0:
                break
        return self.bm.decode_board(board)


@njit()
def search_ai_player(player, b, prob, depth):
    if depth == 0:
        return player.search0(b)
    best = -131072
    for d in [1, 2, 3, 4]:
        t, score = player.bm.move_board(b, d)
        if t == b:
            continue
        # 若更改需同步修改 player.search0
        score = max(0, (score >> 1) - 8) if score < 1000 else (1000 if score < 2000 else score * 2)
        temp = player.cache.lookup(t, depth)
        if temp is None:
            temp = np.float64(0.0)
            empty_slots_count = 0
            # 每个空位分别尝试放入2和4
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
                    empty_slots_count += 1
                    temp_t = search_ai_player(player, np.uint64(t | (1 << (4 * i))), prob * 0.9, depth - 1)
                    if prob * player.spawn_rate4 > player.min_prob:
                        temp_t = temp_t * (1 - player.spawn_rate4) + search_ai_player(player,
                            np.uint64(t | (2 << (4 * i))), prob * 0.1, depth - 1) * player.spawn_rate4
                    temp += temp_t
            temp = int32(temp / empty_slots_count)
            player.cache.update(t, depth, temp)
        if temp + score >= best:
            best = temp + score
            if player.max_d == depth:
                player.best_operation = d

        # if player.max_d == depth:
        #     print({1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}[d])
        #     print(temp + score)
    return best


class Dispatcher:
    def __init__(self, board, board_encoded):
        self.board_encoded = board_encoded
        self.counts = self.frequency_count()
        self.board = board
        self.last_operator = 0  # 0:AI 1:free12w-2k 2:free11w-2k 3:4442f-2k 4:free11w-512
        self.book_reader: BookReaderDispatcher = BookReaderDispatcher()
        self.ad_readers = dict()
        self.init_bookreader()
        self.success_count = 0

    def init_bookreader(self):
        if 'free12w_2048' in SingletonConfig().config['filepath_map'].keys():
            self.book_reader.dispatch(SingletonConfig().config['filepath_map']['free12w_2048'], 'free12w', '2048')
            self.ad_readers['free12w_2048'] = (self.book_reader.use_ad, self.book_reader.book_reader_ad)
        if 'free11w_2048' in SingletonConfig().config['filepath_map'].keys():
            self.book_reader.dispatch(SingletonConfig().config['filepath_map']['free11w_2048'], 'free11w', '2048')
            self.ad_readers['free11w_2048'] = (self.book_reader.use_ad, self.book_reader.book_reader_ad)
        if '4442f_2048' in SingletonConfig().config['filepath_map'].keys():
            self.book_reader.dispatch(SingletonConfig().config['filepath_map']['4442f_2048'], '4442f', '2048')
            self.ad_readers['4442f_2048'] = (self.book_reader.use_ad, self.book_reader.book_reader_ad)
        if 'free11w_512' in SingletonConfig().config['filepath_map'].keys():
            self.book_reader.dispatch(SingletonConfig().config['filepath_map']['free11w_512'], 'free11w', '512')
            self.ad_readers['free11w_512'] = (self.book_reader.use_ad, self.book_reader.book_reader_ad)

    def reset_bookreader(self, key):
        self.book_reader.use_ad, self.book_reader.book_reader_ad = self.ad_readers[key]

    def reset(self, board, board_encoded):
        self.board_encoded = board_encoded
        self.counts = self.frequency_count()
        self.board = board
        # 不要reset success_count

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
            # 小数字和太小让AI玩(返回一个任何定式都查不到的局面)
            masked_board = np.array([[32768, 32768, 32768, 32768],
                                     [0, 32768, 32768, 0],
                                     [0, 32768, 32768, 0],
                                     [32768, 32768, 32768, 32768]], dtype=np.int32)  # 其他数字太小让AI玩
        return masked_board

    def check_free12w_2k(self):
        if 'free12w_2048' not in SingletonConfig().config['filepath_map'].keys():
            self.last_operator = 0
            return 'AI'
        masked_board = self.mask(4)
        self.reset_bookreader('free12w_2048')
        r1 = self.book_reader.move_on_dic(masked_board, 'free12w', '2048', 'free12w_2048')
        move = list(r1.keys())[0]
        if isinstance(r1[move], float) and r1[move] > 0.6:
            if r1[move] > 0.9999999:
                self.success_count += 1
            else:
                self.success_count = max(0, self.success_count-2)
            if self.success_count < 4:
                self.last_operator = 1
                return move
            elif r1[move] == 1 and self.is_final_2k_merge():
                return self.check_free11w_2k()
        elif ((r1[move] in (0, '') or (isinstance(r1[move], float) and r1[move] <= 0.6))
              and np.max(self.counts[8:]) == 1):  # 删除的局面比较多，可能查不到
            self.free11w2k_to_free11w512()
            return self.check_free11w_512()
        self.last_operator = 0
        return 'AI'

    def check_free11w_2k(self):
        if 'free11w_2048' not in SingletonConfig().config['filepath_map'].keys():
            self.last_operator = 0
            return 'AI'
        masked_board = self.mask(5)
        self.reset_bookreader('free11w_2048')
        r1 = self.book_reader.move_on_dic(masked_board, 'free11w', '2048', 'free11w_2048')
        move = list(r1.keys())[0]
        if isinstance(r1[move], float) and r1[move] > 0.06:
            if r1[move] > 0.9999999:
                self.success_count += 1
            else:
                self.success_count = max(0, self.success_count-2)
            if self.success_count < 5:
                self.last_operator = 2
                return move
        elif ((r1[move] in (0, '') or (isinstance(r1[move], float) and r1[move] <= 0.06))
              and np.max(self.counts[8:]) == 1):  # 删除的局面比较多，可能查不到
            self.free11w2k_to_free11w512()
            return self.check_free11w_512()

        self.last_operator = 0
        return 'AI'

    def check_4442f_2k(self):
        if '4442f_2048' not in SingletonConfig().config['filepath_map'].keys():
            self.last_operator = 0
            return 'AI'
        if ((self.counts[10] == 2 and np.sum(self.counts[11:]) == 2 and self.counts[11] == 1 and np.sum(self.board[self.board < 512]) < 120) or
            (self.counts[9] == 2 and np.sum(self.counts[10:]) == 2 and self.counts[10] == 1 and np.sum(self.board[self.board < 512]) < 120) or
            (self.counts[9] == 2 and np.sum(self.counts[10:]) == 3 and self.counts[10] == 1 and self.counts[11] == 1)):
            self.last_operator = 0
            return 'AI'
        if ((np.sum(self.board[self.board < 1024]) < 100 and self.counts[10] == 2 and np.sum(self.counts[11:]) == 2) or
            (np.sum(self.board[self.board < 512]) < 80 and self.counts[9] == 2 and np.sum(self.counts[10:]) == 2)):
            self.last_operator = 0
            return 'AI'

        masked_board = self.mask(3)

        if self.counts[9] == 2 and np.sum(self.counts[10:]) == 2 and self.counts[10] == 0:
            # 把两个512改为1024，让4442f变阵
            masked_board[masked_board == 512] = 1024

        self.reset_bookreader('4442f_2048')
        r1 = self.book_reader.move_on_dic(masked_board, '4442f', '2048', '4442f_2048')
        move = list(r1.keys())[0]
        if isinstance(r1[move], float) and r1[move] > 0.95:
            if r1[move] > 0.9999999:
                self.success_count += 1
            else:
                self.success_count = max(0, self.success_count-2)
            if self.success_count < 4:
                self.last_operator = 3
                return move
        elif ((r1[move] in (0, '') or (isinstance(r1[move], float) and r1[move] <= 0.95)) and
              np.max(self.counts[7:]) == 1) and np.sum(self.counts[7:]) > 4:  # 4442f删除的局面比较多，可能查不到
            return self.check_free11w_512()

        self.last_operator = 0
        return 'AI'

    def check_free11w_512(self):
        if 'free11w_512' not in SingletonConfig().config['filepath_map'].keys():
            self.last_operator = 0
            return 'AI'
        masked_board = self.mask(5)
        self.reset_bookreader('free11w_512')
        r1 = self.book_reader.move_on_dic(masked_board, 'free11w', '512', 'free11w_512')
        move = list(r1.keys())[0]
        if isinstance(r1[move], float) and r1[move] > 0.1:
            if r1[move] > 0.9998:
                self.success_count += 1
            else:
                self.success_count = max(0, self.success_count-2)
            if self.success_count < 3:
                self.last_operator = 4
                return move
        self.last_operator = 0
        return 'AI'

    def is_endgame_65k(self):
        """1、是残局 2、不即将成功(大数无重复小数无连续合并)"""
        return np.all(self.counts[-5:] == 1)

    def is_endgame_32k(self):
        return np.sum(self.counts[-5:]) == 4 and np.max(self.counts[-6:]) == 1

    def is_final_2k_merge(self):
        # 让free11w-2k做衔接，尽可能以2432进final 2k
        if np.all(self.counts[-4:] == 1) and self.counts[11] == 0 and self.counts[10] == 1 and self.counts[9] >= 1:
            board_flatten = self.board.flatten()
            top4_pos = np.argpartition(board_flatten, -4)[-4:]
            top4_pos = tuple(sorted(top4_pos))
            return top4_pos in ((2, 3, 14, 15), (0, 1, 12, 13), (8, 11, 12, 15), (0, 3, 4, 7))
        return False

    # 前n大数(>512)有相同值
    def with_duplicate(self, n, tile_th=-7):
        i = -1
        c = 0
        while c < (n - 1) and i > tile_th:
            if self.counts[i] > 1:
                return True
            c += self.counts[i]
            i -= 1
        return False

    def free11w2k_to_free11w512(self):
        if (self.counts[10] + self.counts[9] == 1) and self.counts[8] == 0 and self.counts[7] < 2:
            self.board[self.board == 512] = 256
            self.board[self.board == 1024] = 256
        elif self.counts[9] == 1 and np.sum(self.counts[9:]) == 6 and self.counts[8] == 0 and self.counts[7] < 2:
            self.board[self.board == 512] = 256

    def dispatcher(self):
        large_tiles = np.sum(self.counts[-7:])  # 大于等于512的格子数
        if large_tiles < 3 or self.with_duplicate(3, -8) or self.with_duplicate(4, -6):  #or np.sum(self.counts[-8:]) < 3
            self.last_operator = 0
            return 'AI'
        if self.is_endgame_65k():
            return self.check_free11w_2k()
        if self.is_endgame_32k():
            return self.check_free12w_2k()
        if large_tiles == 5 and np.max(self.counts[8:]) == 1:
            return self.check_free11w_512()
        return self.check_4442f_2k()


spec = {
    'max_d': int32,
    'hardest_pos': uint8,
    'hardest_num': uint8,
    'board': types.Array(int32, 2, 'C'),
    'diffs': int32[:],
    'diffs2': int32[:],
    'bm': BoardMoverWithScore.class_type.instance_type,  # type: ignore
    'cache': Cache.class_type.instance_type,  # type: ignore
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
        self.diffs, self.diffs2 = _diffs, _diffs2
        self.node = 0
        print('Evil gen init')

    def reset_board(self, board):
        self.hardest_pos = 0
        self.hardest_num = 1
        self.board = board
        self.node = 0

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

    def start_search(self, depth=3):
        self.cache.clear()
        self.max_d = depth
        self.dispatcher(self.bm.encode_board(self.board))

    def dispatcher(self, board):
        self.node = 0
        return search_evil_gen(self, board, self.max_d)

    def gen_new_num(self, depth=5):
        self.start_search(depth)
        board_encoded = self.bm.encode_board(self.board)
        return board_encoded | (self.hardest_num << (4 * self.hardest_pos)), 15 - self.hardest_pos, self.hardest_num


@njit()
def search_evil_gen(evil_gen, b, depth):
    evil = 131072
    for i in range(16):  # 遍历所有位置
        if ((b >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:  # 如果当前位置为空
            for num in (1, 2):
                t = np.uint64(b | (num << (4 * i)))
                best = -131072
                for d in [1, 2, 3, 4]:
                    t1, score = evil_gen.bm.move_board(t, d)
                    if t1 == t:
                        continue
                    temp = evil_gen.cache.lookup(t1, depth)
                    if temp is None:
                        temp = search_evil_gen(evil_gen, t1, depth-1) if depth > 1 else evil_gen.evaluate(t1)
                        evil_gen.cache.update(t1, depth, temp)
                    best = max(best, temp + score)  # 玩家通过操作最大化得分
                    if best >= evil:
                        break  # Beta剪枝
                if best <= evil:
                    evil = best  # 数字生成机制减小得分
                    if evil_gen.max_d == depth:
                        evil_gen.hardest_pos = i
                        evil_gen.hardest_num = num
    return evil


if __name__ == "__main__":
    b1 = np.array([[   4 ,   8,  128, 1024],
 [   4  ,  8 ,  64,  512],
 [   2  ,  0 ,  32,  256],
 [   0  ,  0 ,   8,   64]], dtype=np.int32)
    print(b1)
    s1 = AIPlayer(b1)
    #g1 = EvilGen(b1)
    s1.start_search(2)
    #g1.gen_new_num(2)
    b1 = np.uint64(s1.bm.encode_board(b1))
    print(s1.evaluate(b1))
    t_start = time.time()
    for steps in range(5000):
        s1.reset_board(s1.bm.decode_board(b1))
        t0 = time.time()
        s1.start_search(5)
        print(round(s1.cache.lookup_count / (time.time() - t0) / 1e6, 1), round(s1.node / (time.time() - t0) / 1e6, 1),
              s1.node, round(time.time() - t0, 4))
        print({0:None, 1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}[s1.best_operation])
        if not s1.best_operation:
            print((time.time() - t_start) / steps)
            break
        b1 = np.uint64(s1.bm.move_board(b1, s1.best_operation)[0])
        #g1.reset_board(g1.bm.decode_board(b1))
        #b1 = np.uint64(g1.gen_new_num(5)[0]) if np.random.rand() > 0.99 else np.uint64(s1.bm.gen_new_num(b1)[0])
        b1 = np.uint64(s1.bm.gen_new_num(b1)[0])
        print(s1.bm.decode_board(b1))

    be = np.uint64(s1.bm.encode_board(b1))
    ai_dispatcher = Dispatcher(b1, be)
    ai_dispatcher.reset(b1, be)
    best_move = ai_dispatcher.dispatcher()
    print(best_move)

    s1.start_search(5)
    print({0:None, 1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}[s1.best_operation])
    print(s1.node)


