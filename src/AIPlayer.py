import time

import numpy as np
from numba import int32, types, uint64, from_dtype, float32, uint8, njit
from numba.experimental import jitclass

from BoardMover import reverse, s_move_board, encode_board, s_gen_new_num, decode_board
from BookReader import BookReaderDispatcher
from Config import SingletonConfig, pattern_32k_tiles_map, DTYPE_CONFIG


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

    zero_count = sum([k == 0 for k in line_masked])
    sum_123 = line_masked[1] + line_masked[2] + line_masked[3]
    if line_masked[0] > 100 and ((zero_count > 1 and sum_123 < 100) or sum_123 < 12):
        penalty = 8
    else:
        penalty = 0
    return int32(max(score_dpdf, score_t) / 4 - zero_count - penalty)


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
        board = (board ^ (board >> 27)) * 0x1A85EC53 + board >> 23
        return np.uint64(board - (depth << 5)) & self.length


spec = {
    'max_d': int32,
    'best_operation': uint8,
    'board': types.Array(int32, 2, 'C'),  # board是二维整数数组
    'diffs': int32[:],
    'diffs2': int32[:],
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
        s_reverse = reverse(s)
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
            t, score = s_move_board(b, d)
            if t == b:
                continue
            score = max(0, (score >> 2) - 10) if score < 200 else ((score >> 1) - 20 if score < 800 else (score if score < 2000 else score * 2))
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
        self.dispatcher(encode_board(self.board))

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

            board, score = s_move_board(board, self.best_operation)
            step += 1
            board, empty_slots, _, __ = s_gen_new_num(board)
            if step % 100 == 0:
                # print(decode_board(board))
                # print('')
                pass
            if empty_slots == 0:
                break
        return decode_board(board)


@njit()
def search_ai_player(player, b, prob, depth):
    if depth == 0:
        return player.search0(b)
    best = -131072
    for d in [1, 2, 3, 4]:
        t, score = s_move_board(b, d)
        if t == b:
            continue
        # 若更改需同步修改 player.search0
        score = max(0, (score >> 2) - 10) if score < 200 else ((score >> 1) - 20 if score < 800 else (score if score < 2000 else score * 2))
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


class BaseDispatcher:
    def __init__(self, board, board_encoded):
        self.board_encoded = board_encoded
        self.counts = self.frequency_count()
        self.board = board
        self.last_operator = 0
        self.current_table = 'AI'
        self.book_reader: BookReaderDispatcher = BookReaderDispatcher()

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
            # 小数字和太小让AI玩(返回一个任何定式都查不到的局面)
            masked_board = np.array([[32768, 32768, 32768, 32768],
                                     [0, 32768, 32768, 0],
                                     [0, 32768, 32768, 0],
                                     [32768, 32768, 32768, 32768]], dtype=np.int32)  # 其他数字太小让AI玩
        return masked_board

    def dispatcher(self):
        raise NotImplementedError('Subclasses must implement the dispatcher method')


class DispatcherCommon(BaseDispatcher):
    def __init__(self, board, board_encoded):
        super().__init__(board, board_encoded)
        self.ad_readers = dict()
        self.init_bookreader()

    def init_bookreader(self):
        current_spawn_rate4 = SingletonConfig().config['4_spawn_rate']
        for i, (table, spawn_rate4) in enumerate(SingletonConfig().config['filepath_map'].keys()):
            spawn_rate4 = float(spawn_rate4)
            if abs(current_spawn_rate4 - spawn_rate4) >= 0.01:
                continue
            if SingletonConfig().check_pattern_file(table):
                pattern_param = table.split('_')

                if len(pattern_param) > 2:
                    # 可能是自定义定式名中存在 _ 符号
                    continue

                pattern = pattern_param[0]
                target_str = pattern_param[1]
                target = int(np.log2(int(target_str)))
                self.book_reader.dispatch(SingletonConfig().config['filepath_map'][(table, spawn_rate4)], pattern, target_str)
                _32k, _free32k, _fix32k_pos = pattern_32k_tiles_map[pattern]
                lvl = _32k + target

                if (lvl, _32k) not in self.ad_readers:
                    self.ad_readers[(lvl, _32k)] = []
                self.ad_readers[(lvl, _32k)].append((lvl, _32k, _free32k, pattern, target, target_str, table, i + 1,
                                             self.book_reader.use_ad, self.book_reader.book_reader_ad))
        for key in self.ad_readers:
            # 按_free32k 降序排序
            self.ad_readers[key].sort(key=lambda x: x[2], reverse=True)

    def check_table(self, table_param:list, table_type:int):
        (lvl, _32k, _free32k, pattern, target, target_str, table, i,
        self.book_reader.use_ad, self.book_reader.book_reader_ad) = table_param
        masked_board = self.mask(_32k)

        r1, success_rate_dtype = self.book_reader.move_on_dic(masked_board, pattern, target_str, table)
        _, _, _, zero_val = DTYPE_CONFIG.get(success_rate_dtype, DTYPE_CONFIG['uint32'])
        r1 = {key: (value + zero_val if isinstance(value, (int, float, np.integer, np.floating)) else value) 
              for key, value in r1.items()}

        move = list(r1.keys())[0]
        success_rate = r1[move]

        if isinstance(success_rate, (float, np.floating)):
            remainder = np.sum(self.board) % (1 << target)
            if (table_type == 1 and success_rate > 0.9999999 and remainder < 24) or (
                table_type == 2 and remainder < 32) or (
                table_type == 3 and success_rate > 0.9999999 and (remainder > ((1 << target) - 4) or remainder < 24)):
                self.last_operator = 0
                self.current_table = 'AI'
                return 'AI'

            if success_rate > 0:
                self.last_operator = i
                self.current_table = table
                return move

        return None

    def get_endgame_lvls(self):
        endgame_lvls1, endgame_lvls2, endgame_lvls3 = [], [], []
        large_tile_count = 0
        for i in range(15,7,-1):
            if self.counts[i] > 1 and i != 15:
                break
            large_tile_count += self.counts[i]
            lvl = large_tile_count + i
            if lvl < 12:
                continue
            if self.counts[i] > 0:
                if i <= 12 and self.is_unfree_endgame(i):
                    readers = self.ad_readers.get((lvl, large_tile_count), [])
                    endgame_lvls3.extend([reader for reader in readers if reader[2] > 4])
                    endgame_lvls1.extend([reader for reader in readers if reader[2] <= 4])
                else:
                    endgame_lvls1.extend(self.ad_readers.get((lvl, large_tile_count), []))
                endgame_lvls2.extend(self.ad_readers.get((lvl + 1, large_tile_count), []))
                endgame_lvls2.extend(self.ad_readers.get((lvl + 2, large_tile_count), []))
            elif self.counts[i] == 0:
                endgame_lvls3.extend(self.ad_readers.get((lvl, large_tile_count), []))

        endgame_lvls1.sort(key=lambda x: x[2], reverse=True)

        return endgame_lvls1, endgame_lvls2, endgame_lvls3

    def is_unfree_endgame(self, i):
        for j in range(i + 1, 14):
            if self.counts[j] == 0:
                return sum(self.counts[j:]) >= 4
        return False

    def dispatcher(self):
        for tables_list, table_type in zip(self.get_endgame_lvls(), (1, 2, 3)):
            for table_param in tables_list:
                result = self.check_table(table_param, table_type)
                if result == 'AI':
                    return 'AI'
                elif result:
                    return result

        self.last_operator = 0
        self.current_table = 'AI'
        return 'AI'


class Dispatcher:
    def __init__(self, board, board_encoded):
        self._current_strategy = None
        self._strategies = {}
        self._init_strategies(board, board_encoded)
        strategy_type = self.check_tables()
        self._current_strategy_type = strategy_type
        self.switch_strategy(strategy_type)

    def _init_strategies(self, board, board_encoded):
        self._strategies['common'] = DispatcherCommon(board, board_encoded)

    @staticmethod
    def check_tables():
        return 'common'

    def switch_strategy(self, strategy_type):
        """切换当前使用的策略"""
        if strategy_type in self._strategies:
            self._current_strategy = self._strategies[strategy_type]
            self._current_strategy_type = strategy_type
        else:
            raise ValueError(f"Unsupported Strategy Type: {strategy_type}")

    def __getattr__(self, name):
        """
        魔法方法：将未定义的属性调用委托给当前策略实例
        """
        if hasattr(self._current_strategy, name):
            return getattr(self._current_strategy, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


spec = {
    'max_d': int32,
    'hardest_pos': uint8,
    'hardest_num': uint8,
    'board': types.Array(int32, 2, 'C'),
    'diffs': int32[:],
    'diffs2': int32[:],
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
        s_reverse = reverse(s)
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
        self.dispatcher(encode_board(self.board))

    def dispatcher(self, board):
        self.node = 0
        return search_evil_gen(self, board, self.max_d)

    def gen_new_num(self, depth=5):
        self.start_search(depth)
        board_encoded = encode_board(self.board)
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
                    t1, score = s_move_board(t, d)
                    if t1 == t:
                        continue
                    temp = evil_gen.cache.lookup(t1, depth)
                    if temp is None:
                        temp = search_evil_gen(evil_gen, t1, depth-1) if depth > 1 else evil_gen.evaluate(t1)
                        evil_gen.cache.update(t1, depth, temp)
                    best = max(best, temp + (score << 1))  # 玩家通过操作最大化得分
                    if best >= evil:
                        break  # Beta剪枝
                if best <= evil:
                    evil = best  # 数字生成机制减小得分
                    if evil_gen.max_d == depth:
                        evil_gen.hardest_pos = i
                        evil_gen.hardest_num = num
    return evil


if __name__ == "__main__":
    pass
 #    history = np.empty(1000, dtype='uint64,uint32')
 #    score_sum = 0
 #
 #    b1 = np.array([[   0 ,   0,  0,  8],
 # [   0,   2 ,  2,  4],
 # [   2  ,  8 ,  4096,  8],
 # [   8192,   2,   1024,   512]], dtype=np.int32)
 #    print(b1)
 #    s1 = AIPlayer(b1)
 #    g1 = EvilGen(b1)
 #    s1.start_search(2)
 #    g1.gen_new_num(2)
 #    b1 = np.uint64(encode_board(b1))
 #    print(s1.evaluate(b1))
 #
 #    t_start = time.time()
 #    for steps in range(240):
 #        history[steps] = b1, score_sum
 #        s1.reset_board(decode_board(b1))
 #        t0 = time.time()
 #        s1.start_search(6)
 #        # print(round(s1.cache.lookup_count / (time.time() - t0) / 1e6, 1), round(s1.node / (time.time() - t0) / 1e6, 1),
 #        #     s1.node, round(time.time() - t0, 4))
 #        print({0:None, 1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}[s1.best_operation])
 #        if not s1.best_operation:
 #            print((time.time() - t_start) / steps)
 #            break
 #        b1, score = s_move_board(b1, s1.best_operation)
 #        score_sum += score
 #        b1 = np.uint64(b1)
 #        g1.reset_board(decode_board(b1))
 #        b1 = np.uint64(g1.gen_new_num(5)[0]) if (steps < 0 or (np.random.rand(1)[0] < 0.10) and steps < 0) else np.uint64(s_gen_new_num(b1)[0])
 #
 #        print(decode_board(b1))
 #        print()

    # be = np.uint64(encode_board(b1))
    # ai_dispatcher = Dispatcher(b1, be)
    # ai_dispatcher.reset(b1, be)
    # best_move = ai_dispatcher.dispatcher()
    # b1 = np.uint64(encode_board(b1))
    # for steps in range(240):
    #     history[steps] = b1, score_sum
    #     ai_dispatcher.reset(decode_board(b1), b1)
    #     t0 = time.time()
    #     best_move = ai_dispatcher.dispatcher().capitalize()
    #     b1, score = s_move_board(b1, {'Left': 1, 'Right': 2, 'Up': 3, 'Down': 4}[best_move])
    #     score_sum += score
    #     b1 = np.uint64(b1)
    #     b1 = np.uint64(s_gen_new_num(b1)[0])
    #
    #     print(decode_board(b1))
    #     print()
    #
    #
    # history[:steps + 1].tofile(fr'C:\Users\Administrator\Desktop\record\0')
