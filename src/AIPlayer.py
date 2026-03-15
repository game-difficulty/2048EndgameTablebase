import threading
import time
from typing import Tuple, List

import numpy as np

from BookReader import BookReaderDispatcher
from Calculator import (
    ReverseLR, ReverseUD, ReverseUL, ReverseUR,
    Rotate180, RotateL, RotateR
)
from Config import SingletonConfig, pattern_32k_tiles_map, DTYPE_CONFIG
from BoardMover import decode_board
from ai_and_sort import ai_core
from BoardMover import move_board, encode_board


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
                if pattern not in pattern_32k_tiles_map:
                    continue

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
        for i in range(15,6,-1):
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
                if self.counts[i - 1] < 2:
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


class L3Manager:
    def __init__(self):
        # 移动方向逆映射表: map_move[对称变换索引][变换后查到的方向] = 原始盘面应走的方向
        # 0: 占位, 1: Left, 2: Right, 3: Up, 4: Down
        self.map_move = [
            [0, 1, 2, 3, 4],  # 0: Identity
            [0, 3, 4, 2, 1],  # 1: RotateL
            [0, 2, 1, 4, 3],  # 2: Rotate180
            [0, 4, 3, 1, 2],  # 3: RotateR
            [0, 2, 1, 3, 4],  # 4: ReverseLR
            [0, 1, 2, 4, 3],  # 5: ReverseUD
            [0, 3, 4, 1, 2],  # 6: ReverseUL
            [0, 4, 3, 2, 1],  # 7: ReverseUR
        ]

    def probe(self, board: np.uint64, counts: np.ndarray, board_sum: int) -> Tuple[int, bool, int, List[float], int]:
        threshold = 0

        # 残局特征识别与阈值确定
        if np.sum(counts[10:]) == 6 and np.max(counts[10: 15]) == 1:
            if not ((board_sum % 1024) < 480 and counts[9] == 1):
                if board_sum % 1024 > 96:
                    threshold = 10
                elif board_sum % 1024 > 60:
                    threshold = 8
        elif np.sum(counts[9:]) == 6 and np.max(counts[9: 15]) == 1:
            if not ((board_sum % 512) < 240 and counts[8] == 1):
                threshold = 9
        elif np.sum(counts[8:]) == 6 and np.max(counts[8: 15]) == 1 and (board_sum % 256) < 240:
            if not ((board_sum % 256) < 120 and counts[7] == 1) and not (
                    np.sum(counts[10:]) == 5 and counts[9] == 0 and (board_sum % 256) < 64):
                threshold = 8
        elif np.sum(counts[7:]) == 6 and np.max(counts[7: 15]) == 1 and 20 < (board_sum % 128) < 120 and counts[7] == 1:
            if not ((board_sum % 128) < 60 and counts[6] == 1):
                threshold = 7

        if threshold == 0:
            return 0, False, 0, [0.0, 0.0, 0.0, 0.0], threshold

        board = np.uint64(board & 0xFFFFFFFFFFFFFFFF)
        masked_board = self.mask_large_tiles(board, threshold)

        if threshold == 10 and (board_sum % 1024) < 128:
            table_types = (256, 512)
        elif threshold > 8 and ((board_sum % 256) > 128 or (board_sum % 512) < 72):
            table_types = (512, )
        elif threshold > 8:
            table_types = (512, 256)
        elif (threshold == 8 and (board_sum % 256) > 60) or (threshold == 7 and (board_sum % 128) > 60):
            table_types = (256, 512)
        else:
            table_types = (256, )

        best_move, win_rates, table_type = self.probe_L3(masked_board, table_types, board_sum)

        # 广义层叠识别
        if best_move == 0 and np.sum(counts[8:]) == 7 and np.max(counts[8: 15]) == 1 and (board_sum % 256) < 240:
            threshold = 9
            best_move, win_rates = self.probe_441(board, threshold - 1, board_sum)
            table_type = 512
        if best_move == 0 and np.sum(counts[7:]) == 7 and np.max(counts[7: 15]) == 1 and (board_sum % 128) < 120:
            threshold = 8
            best_move, win_rates = self.probe_441(board, threshold - 1, board_sum)
            table_type = 512

        if threshold <= 8 and 0 < max(win_rates) < 0.625:
            # 16k残局应该更高
            return 0, True, 0, win_rates, threshold

        return best_move, False, table_type, win_rates, threshold

    @staticmethod
    def get_syms(board: np.uint64):
        # 生成 8 个对称型
        syms = [
            board,
            RotateL(board),
            Rotate180(board),
            RotateR(board),
            ReverseLR(board),
            ReverseUD(board),
            ReverseUL(board),
            ReverseUR(board)
        ]
        return syms

    def probe_L3(self, masked_board: np.uint64, table_types: Tuple[int], board_sum: int) -> Tuple[int, List[float], int]:
        syms = self.get_syms(masked_board)

        for i, b in enumerate(syms):
            b = np.uint64(b & 0xFFFFFFFFFFFFFFFF)
            if not b & np.uint64(0xfff0fff) == np.uint64(0xfff0fff):
                continue
            for table_type in table_types:
                if table_type == 512:
                    best_move, original_win_rates = self.probe_44_128(b, i, board_sum)
                    if best_move != 0:
                        return best_move, original_win_rates, table_type

                win_rates = ai_core.find_best_egtb_move(b, table_type)

                if max(win_rates) > 0:
                    best_move, original_win_rates = self.handle_result(win_rates, i)
                    return best_move, original_win_rates, table_type

        return 0, [0.0, 0.0, 0.0, 0.0], 0

    def probe_441(self, board: np.uint64, threshold: int, board_sum: int) -> Tuple[int, List[float]]:
        masked = self.mask_large_tiles(board, threshold, threshold)
        syms = self.get_syms(masked)

        for i, b in enumerate(syms):
            b = np.uint64(b & 0xFFFFFFFFFFFFFFFF)

            if b & np.uint64(0xfff0fff) == np.uint64(0x1110111) * threshold:
                b |= np.uint64(0xfff0fff)
            else:
                continue

            best_move, original_win_rates = self.probe_44_128(b, i, board_sum)
            if best_move != 0:
                return best_move, original_win_rates

            win_rates = ai_core.find_best_egtb_move(b, 512)

            if max(win_rates) > 0:
                best_move, original_win_rates = self.handle_result(win_rates, i)
                return best_move, original_win_rates

        return 0, [0.0, 0.0, 0.0, 0.0]

    def probe_44_128(self, masked_board: np.uint64, i: int, board_sum: int):
        if 390 < board_sum % 512 < 480 and board_sum > 63000:
            remask_44 = self.mask_large_tiles(masked_board, 7)

            if not remask_44 & np.uint64(0xffffffff) == np.uint64(0xffffffff):
                return 0, [0.0, 0.0, 0.0, 0.0]

            remask_44 = np.uint64((ReverseLR(remask_44) & 0xFFFFFFFF00000000) + 0x7fff8fff)
            win_rates = ai_core.find_best_egtb_move(remask_44, 512)

            if max(win_rates) > 0:
                win_rates[0], win_rates[1] = win_rates[1], win_rates[0]
                best_move, original_win_rates = self.handle_result(win_rates, i)
                return best_move, original_win_rates

        return 0, [0.0, 0.0, 0.0, 0.0]

    def handle_result(self, win_rates, i):
        # 1. 还原最佳走法
        found_dir = int(np.argmax(win_rates)) + 1
        best_move = self.map_move[i][found_dir]

        # 2. 还原完整的胜率数组 (映射回原图视角的 左=0, 右=1, 上=2, 下=3)
        original_win_rates = [0.0, 0.0, 0.0, 0.0]
        for d in range(1, 5):
            # d 是克隆体视角的移动方向 (1到4)
            # orig_dir 是对应到原图的实际移动方向 (1到4)
            orig_dir = self.map_move[i][d]
            # 存入新数组，注意列表索引是方向减 1
            original_win_rates[orig_dir - 1] = win_rates[d - 1]
        return best_move, original_win_rates

    @staticmethod
    def mask_large_tiles(board: np.uint64, threshold: int, mask: int = 0xF) -> np.uint64:
        """
        遍历 64位盘面的 16 个槽位，将所有 >= threshold 的数字替换为 0xF
        """
        res = np.uint64(0)
        for i in range(16):
            shift = np.uint64(4 * i)
            val = (board >> shift) & np.uint64(0xF)
            if val >= threshold:
                val = np.uint64(mask)
            res |= (val << shift)
        return res

    def probe_after_move(self, board: np.uint64, threshold: int, table_types: Tuple[int], board_sum: int) -> float:
        masked_board = self.mask_large_tiles(board, threshold)

        win_rate = 0.0
        empty_slots = 0
        for i in range(16):
            if (masked_board >> (4 * i)) & np.uint64(0xF) == np.uint64(0):
                empty_slots += 1
                t1 = masked_board | (np.uint64(1) << np.uint64(4 * i))
                t2 = masked_board | (np.uint64(2) << np.uint64(4 * i))
                best_move, win_rates, table_type = self.probe_L3(t1, table_types, board_sum + 2)
                win_rate += max(win_rates) * 0.9
                best_move, win_rates, table_type = self.probe_L3(t2, table_types, board_sum + 4)
                win_rate += max(win_rates) * 0.1
        win_rate /= empty_slots
        return win_rate


class CoreAILogic:
    """提取出的核心 AI 逻辑，供测试脚本和 GUI 线程共用"""
    SCORE_CRITICAL = -5000
    SCORE_HOPELESS = -30000
    FALLBACK_DEPTH = 7

    def __init__(self):
        self.manager = L3Manager()
        self.last_depth = 4
        self.last_sum = 0
        self.last_prune = np.uint8(0)
        self.last_move = ''
        self.time_ratio = 4.0
        self.time_limit_ratio = 1.0

    @staticmethod
    def is_mess(board):
        """检查是否乱阵 (代码与原版完全一致)"""
        if np.sum(board) % 512 < 12:
            return False

        large_tiles = (board > 128).sum()
        board_flatten = board.flatten()
        if large_tiles < 3:
            return False
        elif large_tiles == 4:
            top4_pos = np.argpartition(board_flatten, -4)[-4:]
            if len(np.unique(board_flatten[top4_pos])) < 4:
                return False
            top4_pos = tuple(sorted(top4_pos))
            return top4_pos not in (
                (0, 1, 2, 3), (0, 4, 8, 12), (12, 13, 14, 15), (3, 7, 11, 15),
                (0, 1, 2, 4), (4, 8, 12, 13), (11, 13, 14, 15), (2, 3, 7, 11),
                (0, 1, 4, 8), (8, 12, 13, 14), (7, 11, 14, 15), (1, 2, 3, 7),
                (0, 1, 4, 5), (8, 9, 12, 13), (10, 11, 14, 15), (2, 3, 6, 7),
                (0, 1, 3, 4), (0, 1, 4, 12), (0, 2, 3, 7), (2, 3, 7, 15),
                (0, 8, 12, 13), (8, 12, 13, 15), (3, 11, 14, 15), (11, 12, 14, 15)
            )
        elif large_tiles == 3:
            top3_pos = np.argpartition(board_flatten, -3)[-3:]
            if len(np.unique(board_flatten[top3_pos])) < 3:
                return False
            top3_pos = tuple(sorted(top3_pos))
            return top3_pos not in (
                (0, 1, 2), (1, 2, 3), (3, 7, 11), (7, 11, 15), (13, 14, 15), (12, 13, 14), (4, 8, 12), (0, 4, 8),
                (0, 1, 3), (0, 2, 3), (3, 7, 15), (3, 11, 15), (12, 14, 15), (12, 13, 15), (0, 8, 12), (0, 4, 12),
                (0, 1, 4), (2, 3, 7), (11, 14, 15), (8, 12, 13)
            )
        else:
            top_n_pos = np.argpartition(board_flatten, -large_tiles)[-large_tiles:]
            top_n_pos_set = set(top_n_pos)
            corners_l_shapes = [
                {0, 1, 4}, {3, 2, 7}, {12, 8, 13}, {15, 11, 14}
            ]
            for corner in corners_l_shapes:
                if corner.issubset(top_n_pos_set):
                    return False
            return True

    def calculate_step(self, ai_player, board, counts) -> int:
        """执行 AI 核心深度计算并返回最佳操作的数字代号"""
        empty_slots = counts[0]
        board_sum = np.sum(board)
        big_nums = np.sum(counts[8:])

        move, is_evil, table_type, win_rates, threshold = self.manager.probe(ai_player.board, counts, board_sum)
        if move:
            # 查到了，进行二次安全校验
            if self.validate_egtb_move(board, ai_player, move, table_type, win_rates, board_sum, threshold):
                self.last_move = 'L3'
                return move

        is_5tiler = (board_sum > 62000 or (np.sum(counts[11:]) == 4 and counts[10] == 0)) and (
                board_sum % 1024 < 24 or board_sum % 1024 > 996)

        is_not_merging = (np.max(counts[8: 15]) == 1) and not (counts[7] > 1 and counts[8] == 1) and not (
            counts[6] > 1 and counts[7] == 1 and counts[8] == 1 and board_sum % 1024 < 96)
        is_mess = self.is_mess(board) if is_not_merging else False

        ai_player.do_check = np.uint8(big_nums) if is_mess and big_nums in (3, 4) else np.uint8(0)
        ai_player.prune = np.uint8(1) if is_not_merging and not (
              (not 40 < board_sum % 512 < 500 and max(counts[7:9]) > 1 and big_nums > 2) or
              (not 32 < board_sum % 256 < 250 and max(counts[6:8]) > 1 and big_nums > 4) or
              (not 24 < board_sum % 128 < 126 and max(counts[5:7]) > 1 and big_nums > 4) or
              is_mess) else np.uint8(0)
        if is_mess or is_5tiler or is_evil or self.tiles_all_set(counts) or (
                max(counts[6:]) == 1 and np.sum(counts[6:]) >= 9):
            ai_player.prune = np.uint8(0)


        if is_mess or is_5tiler:
            big_nums2 = np.sum(counts[9:])
            initial_depth = 5
            max_depth = 24
            time_limit = 1.2 * big_nums2 ** 0.25

            best_op, final_depth, scores = self.perform_iterative_search(
                ai_player, initial_depth, max_depth, time_limit
            )

        elif empty_slots > 4 and big_nums < 2 and is_not_merging:
            initial_depth = 3
            # 单次搜索
            best_op, final_depth, scores = self.perform_iterative_search(
                ai_player, initial_depth, initial_depth, 0.1
            )

        elif big_nums <= 3 and 32 < board_sum % 256 < 240 and is_not_merging and counts[6] < 2:
            initial_depth = 4 if counts[7] == 0 else 5
            best_op, final_depth, scores = self.perform_iterative_search(
                ai_player, initial_depth, initial_depth, 0.1
            )

        else:
            if counts[7] > 1 or (board_sum % 512 < 20 and np.sum(counts[8:]) > 2):
                initial_depth, max_depth, time_limit = 5, 32, 0.15 * big_nums ** 0.25
            elif is_not_merging and np.sum(counts[7:]) > 5:
                initial_depth, max_depth, time_limit = 5, 48, 0.32 * big_nums ** 0.25
            else:
                initial_depth, max_depth, time_limit = 5, 24, 0.16 * big_nums ** 0.25

            initial_depth += ai_player.prune
            if not is_mess and np.sum(counts[9:]) <= 3:
                max_depth = 10

            last_sum = self.last_sum
            last_depth = self.last_depth

            if ai_player.prune and abs(board_sum - last_sum) < 6:
                min_initial = min(last_depth - 1, round(last_depth * 0.9))
                initial_depth = max(initial_depth, min_initial)

            best_op, final_depth, scores = self.perform_iterative_search(
                ai_player, initial_depth, max_depth, time_limit
            )

        # 记录状态
        self.last_sum = board_sum
        self.last_depth = final_depth
        self.last_prune = ai_player.prune
        self.last_move = 'search'
        # print(scores)

        return best_op

    def perform_iterative_search(self, ai_player, initial_depth: int, max_depth: int, time_limit: float) -> tuple[
        int, int, list[float]]:
        """
        基于指数时间预测的迭代加深搜索框架
        """
        best_op_so_far = -1
        final_depth = 0
        valid_scores = []

        start_time = time.time()
        local_limit = time_limit * self.time_limit_ratio
        last_depth_time = None
        depth = initial_depth
        fallback_attempts = 0

        while depth <= max_depth and best_op_so_far != 0:
            elapsed = time.time() - start_time
            remaining_time = local_limit - elapsed

            # 1. 终止条件检查 (基于剩余时间与预测时间)
            if self._should_stop_search(best_op_so_far, last_depth_time, remaining_time):
                break

            # 2. 准备当前层搜索环境
            timer = self._start_timeout_timer(ai_player, best_op_so_far, remaining_time, time_limit)
            depth_start = time.time()

            # 3. 执行搜索
            try:
                ai_player.start_search(depth)
            finally:
                timer.cancel()

            depth_elapsed = time.time() - depth_start

            # 4. 处理搜索结果
            if getattr(ai_player, 'stop_search', False):
                # 被定时器截断：尝试抢救或直接退出
                depth, fallback_attempts = self._handle_timeout(
                    best_op_so_far, depth, fallback_attempts
                )
                if depth is None:  # 抢救失败或已有保底解
                    break
                continue

            # 5. 完整跑完当前层，更新可靠数据
            best_op_so_far = ai_player.best_operation
            final_depth = depth
            valid_scores = list(ai_player.scores)

            # 6. 动态更新策略 (时间限制、最大深度、时间预测倍数)
            local_limit, max_depth = self._update_search_strategy(
                ai_player, best_op_so_far, valid_scores, depth, initial_depth, local_limit, time_limit, max_depth
            )
            self._update_time_ratio(last_depth_time, depth_elapsed)

            # 准备进入下一层
            last_depth_time = depth_elapsed
            depth += 1

        # 7. 绝对兜底逻辑 (首层死活跑不出来)
        if best_op_so_far == -1:
            best_op_so_far, final_depth, last_depth_time = self._force_fallback_search(ai_player)

        # if last_depth_time is not None:
        #     print(f"--- Depth: {final_depth} | Time: {last_depth_time:.4f}s ---")

        ai_player.stop_search = False
        return best_op_so_far, final_depth, valid_scores

    def _should_stop_search(self, best_op: int, last_time: float, remaining_time: float) -> bool:
        if best_op == -1:
            return False
        if remaining_time <= 0.0001:
            return True
        if last_time is not None and (last_time * self.time_ratio) > remaining_time * 0.9:
            return True
        return False

    @staticmethod
    def _start_timeout_timer(ai_player, best_op: int, remaining_time: float, time_limit: float
                             ) -> threading.Timer:
        ai_player.stop_search = False
        # 首层放宽限制，非首层严格限制
        timeout = max(0.005, max(remaining_time * 2.5, time_limit * 0.2) if best_op == -1 else remaining_time - 0.001)
        timer = threading.Timer(timeout, lambda: setattr(ai_player, 'stop_search', True))
        timer.start()
        return timer

    def _handle_timeout(self, best_op: int, depth: int, attempts: int) -> tuple[int | None, int]:
        self.time_ratio *= 1.03  # 惩罚预测倍数
        if best_op == -1 and attempts < 2:
            # 首层超时抢救：退层
            new_depth = int(min(max(depth - 2, 1), depth * 0.8))
            return new_depth, attempts + 1
        return None, attempts

    def _update_search_strategy(self, ai_player, best_op, scores, depth, init_depth, local_limit, time_limit,
                                max_depth):
        if not scores: return local_limit, max_depth

        # 索引越界保护
        best_score = scores[best_op - 1] if 0 < best_op <= len(scores) else 0

        if best_score < self.SCORE_CRITICAL:
            local_limit = time_limit * self.time_limit_ratio + 1 + 0.1 * (depth - init_depth)
        if max(scores) < self.SCORE_HOPELESS:
            max_depth = min(max_depth, 16)

        return local_limit, max_depth

    def _update_time_ratio(self, last_time: float, current_time: float):
        if last_time is not None and last_time > 0.001:
            current_ratio = max(1.2, min(12.0, current_time / last_time))
            # 使用对数空间的 EMA 平滑
            self.time_ratio = np.exp2(0.75 * np.log2(self.time_ratio) + 0.25 * np.log2(current_ratio))

    def _force_fallback_search(self, ai_player) -> tuple[int, int, float]:
        fallback_start = time.time()
        ai_player.stop_search = False
        ai_player.prune = np.uint64(0)
        ai_player.start_search(self.FALLBACK_DEPTH)
        return ai_player.best_operation, self.FALLBACK_DEPTH, time.time() - fallback_start

    def validate_egtb_move(self, board, ai_player, move: int, table_type: int,
                           win_rates: List[float], board_sum: int, threshold: int) -> bool:
        """
        对残局表返回的走法进行浅层验证，过滤哈希碰撞产生的致命假阳性。
        """
        move_map = {1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}

        ai_player.stop_search = False
        ai_player.prune = np.uint64(1) if 48 < board_sum % 256 < 234 else np.uint64(0)
        ai_player.do_check = np.uint8(0)
        depth = 8 if ai_player.prune else 6
        ai_player.start_search(depth)

        need_further_check = False
        scores = list(ai_player.scores)
        if (win_rates[np.argmax(scores)] == 0.0) and (max(scores) - scores[np.argmax(win_rates)] > 5):
            win_rate = max(win_rates)
            if (table_type == 256 and win_rate > 0.993) or (
                    table_type == 512 and board_sum % 512 < 64 and win_rate > 0.84):
                need_further_check = False
            else:
                need_further_check = True

        if not need_further_check:
            target_score = scores[move - 1]
            sorted_scores = sorted(scores, reverse=True)

            # 合理性校验
            if (table_type == 512) and ((
                    target_score >= sorted_scores[1] - 18 and sorted_scores[2] > 2400) or (
                    target_score >= sorted_scores[0] - 24 and sorted_scores[2] > 1800) or (
                    target_score >= sorted_scores[0] - 16)):
                return True
            if (table_type == 256) and ((
                    target_score >= sorted_scores[1] - 24 and sorted_scores[2] > 2400) or (
                    target_score >= sorted_scores[0] - 30 and sorted_scores[2] > 1800) or (
                    target_score >= sorted_scores[0] - 20)):
                return True

        best_op, final_depth, scores = self.perform_iterative_search(
            ai_player, 7, 10, 0.5
        )

        target_score = scores[move - 1]
        sorted_scores = sorted(scores, reverse=True)
        if target_score < -ai_player.dead_score // 2 and sorted_scores[0] > 0:
            return False

        if (table_type == 512) and ((
                target_score >= sorted_scores[1] - 10 and sorted_scores[2] > 2400) or (
                target_score >= sorted_scores[0] - 12 and sorted_scores[2] > 1800) or (
                target_score >= sorted_scores[0] - 8)):
            return True
        if (table_type == 256) and ((
                target_score >= sorted_scores[1] - 15 and sorted_scores[2] > 2400) or (
                target_score >= sorted_scores[0] - 16 and sorted_scores[2] > 1800) or (
                target_score >= sorted_scores[0] - 12)):
            return True

        if not (threshold == 8 and table_type == 512 and board_sum % 256 < 96):
            board_encoded = np.uint64(encode_board(board) & 0xffffffffffffffff)
            after_board1 = move_board(board_encoded, best_op)
            after_board2 = move_board(board_encoded, move)
            win_rate1 = self.manager.probe_after_move(after_board1, threshold, (table_type, ), board_sum)
            win_rate2 = self.manager.probe_after_move(after_board2, threshold, (table_type, ), board_sum)
            if max(win_rate2, win_rate1) < 0.2:
                return False
            if win_rate2 > win_rate1 and ((
                    win_rate1 > 0 or target_score > 2000) or (
                    win_rate1 == 0 and target_score < -3000 and 60 < board_sum % 256 < 200)):
                # print(f"valid move {move_str} on board {hex(ai_player.board)} with "
                #       f"{win_rate2} vs {win_rate1}({move_map[best_op]})")
                return True

        # print(f"Invalid move {move_str} on board {hex(ai_player.board)} with scores {ai_player.scores}")

        return False

    @staticmethod
    def tiles_all_set(counts):
        last_dup = 0
        i = 0
        for i in range(5, 15):
            if counts[i] > 1:
                last_dup = i
        if last_dup == 0:
            return False
        for i in range(last_dup + 1, 15):
            if counts[i] == 0:
                break
        final_big_tiles = np.sum(counts[i:]) + 1
        return final_big_tiles < 5 and i > 9 and not (final_big_tiles + i < 14 and last_dup < 6)


if __name__ == '__main__':
    print(['Left', 'Right', 'Up', 'Down'])
    print(ai_core.find_best_egtb_move(np.uint64(0x100111252fff3fff), 512))

    l3mng = L3Manager()
    l3mng.probe(np.uint64(0x3201323259bd6ace), np.array([0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,0,]),32390)
    print(l3mng.probe_after_move(np.uint64(0x200022502fff3fff), 9, (512,),np.sum(decode_board(0x200022502fff3fff))))
