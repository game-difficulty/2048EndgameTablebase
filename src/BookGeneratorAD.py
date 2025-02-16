import itertools
import math
import os
import time
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange, uint8
from numba.typed import Dict
from numba import types
from numba.experimental import jitclass

from BoardMoverAD import MaskedBoardMover
from BookGeneratorUtils import merge_inplace, hash_, parallel_unique, sort_array, concatenate, merge_deduplicate_all, \
    update_seg
from BookGenerator import predict_next_length_factor_quadratic, \
    update_hashmap_length, validate_length_and_balance, log_performance
import Config
from Config import SingletonConfig
from LzmaCompressor import compress_with_7z

logger = Config.logger

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]

KeyType = types.UniTuple(types.uint8, 2)
ValueType1 = types.uint8[:, :]
ValueType2 = types.uint8[:]

SmallTileSumLimit = 56

spec = {
    "target" : uint8,
    "num_free_32k": uint8,  # 可移动32k格子的数量
    "pos_fixed_32k": uint8[:],  # 不可移动32k格子的位置,右下角是0
    "num_fixed_32k": uint8,  # 不可移动32k格子的数量
    "permutation_dict1": types.DictType(KeyType, ValueType1),  # A(m,n)排列字典
    "permutation_dict2": types.DictType(KeyType, ValueType1),  # A(m,n)排列字典,但只有一半
    "tiles_combinations_dict": types.DictType(KeyType, ValueType2),  # 将数字和拆分为数字组成的字典
}


@jitclass(spec)
class BoardMasker:
    def __init__(self, target, permutation_dict1, permutation_dict2, num_free_32k, pos_fixed_32k):
        self.target = target
        self.num_free_32k = num_free_32k
        self.pos_fixed_32k = pos_fixed_32k
        self.num_fixed_32k = len(pos_fixed_32k)
        self.permutation_dict1 = permutation_dict1
        self.permutation_dict2 = permutation_dict2
        self.tiles_combinations_dict = self.generate_tiles_combinations_dict()

    @staticmethod
    def mask_board(board: np.uint64, th: int = 6) -> np.uint64:
        """
        mask盘面上所有大于等于阈值的格子
        """
        masked_board = board
        for k in range(16):
            encoded_num = (board >> np.uint64(4 * k)) & np.uint64(0xF)
            if encoded_num >= th:
                masked_board |= (np.uint64(0xF) << np.uint64(4 * k))  # 将对应位置的数设置为 0xF
        return masked_board

    def tile_sum_and_32k_count(self, masked_board: np.uint64) -> Tuple[np.uint32, np.uint8, NDArray[np.uint64]]:
        """
        统计一个masked board中非32k的数字和、可移动32k数量和位置
        """
        total_sum = 0
        count_32k = 0
        pos_32k = np.empty(16, dtype=np.uint64)
        pos_ind = 0
        for i in range(60, -4, -4):
            tile_value = (masked_board >> i) & 0xf
            if tile_value == 0xf:
                if i not in self.pos_fixed_32k:
                    count_32k += 1
                    pos_32k[pos_ind] = i
                    pos_ind += 1
            elif tile_value > 0:
                total_sum += 2 ** tile_value
        return total_sum, count_32k, pos_32k[:pos_ind]

    def tile_sum_and_32k_count2(self, masked_board: np.uint64) -> Tuple[np.uint32, np.uint8, NDArray[np.uint64]]:
        """
        统计一个masked board中小数数字和、可移动大数数量和位置
        """
        total_sum = 0
        count_32k = 0
        pos_32k = np.empty(16, dtype=np.uint64)
        pos_ind = 0
        for i in range(60, -4, -4):
            tile_value = (masked_board >> i) & 0xf
            if tile_value > 5:
                if i not in self.pos_fixed_32k:
                    count_32k += 1
                    pos_32k[pos_ind] = i
                    pos_ind += 1
            elif tile_value > 0:
                total_sum += 2 ** tile_value
        return total_sum, count_32k, pos_32k[:pos_ind]

    @staticmethod
    def _masked_tiles_combinations(remaining_sum: np.uint64, remaining_count: int
                                   ) -> NDArray[np.uint8] | None:
        """
        输入masked board中非32k的数字和和32k数量，输出被masked的格子的数字组成
        """
        tiles = np.array([8192, 4096, 2048, 1024, 512, 256, 128, 64])
        result = np.empty(9, dtype=np.uint8)
        index = 0
        for tile in tiles:
            if tile > remaining_sum:
                continue
            if tile == remaining_sum:
                if remaining_count == 1:
                    result[index] = np.log2(tile)
                    return result[:index + 1].copy()
            elif tile < remaining_sum:
                if (tile << 1) == remaining_sum and remaining_count == 2:
                    result[index] = np.log2(tile)
                    result[index + 1] = np.log2(tile)
                    return result[:index + 2].copy()
                else:  # 分配一个格子
                    remaining_sum -= tile
                    remaining_count -= 1
                    result[index] = np.log2(tile)
                    index += 1
        return None

    def generate_tiles_combinations_dict(self) -> Dict[KeyType, ValueType2]:
        """
        combinations_dict: {(数字和//64, 数字个数): 数字组成array}
        确定数字和和个数的情况下，组成唯一
        """
        tiles_combinations_dict = Dict.empty(KeyType, ValueType2)
        for i in range(1, 129):
            for j in range(1, 9):
                tile = self._masked_tiles_combinations(np.uint64(i << 6), j)
                if tile is not None:
                    # 从小到大
                    tiles_combinations_dict[(np.uint8(i), np.uint8(j))] = tile[::-1].astype(np.uint8)
        return tiles_combinations_dict

    def derive(self, board: np.uint64, original_board_sum: np.uint32
               ) -> Tuple[bool, bool, NDArray[np.uint64]]:
        """
        在tiles_combinations最小两个数字相等的前提下，暴露这两个tiles，返回全部可能组合
        第一个返回值是board是否合法，第二个返回值是是否derive，第三个返回值是全部可能组合（如有，否则为空数组）
        """
        total_sum, count_32k, pos_32k = self.tile_sum_and_32k_count(board)
        if total_sum >= SmallTileSumLimit + 64:
            return False, False, np.empty(0, dtype=np.uint64)
        large_tiles_sum = original_board_sum - total_sum - ((self.num_free_32k + self.num_fixed_32k) << 15)

        # assert large_tiles_sum % 64 == 0
        tiles_combinations = self.tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6),
                                                               np.uint8(count_32k - self.num_free_32k)), None)
        if tiles_combinations is None:
            return False, False, np.empty(0, dtype=np.uint64)
        if tiles_combinations[-1] == self.target:
            return False, False, np.empty(0, dtype=np.uint64)
        if count_32k < 2:
            return True, False, np.empty(0, dtype=np.uint64)
        if tiles_combinations[0] == tiles_combinations[1]:
            if total_sum >= SmallTileSumLimit:
                return False, False, np.empty(0, dtype=np.uint64)
            result = np.empty(count_32k * (count_32k - 1) // 2, dtype=np.uint64)
            ind = 0
            for pos1 in range(count_32k - 1):
                for pos2 in range(pos1 + 1, count_32k):
                    derived = board & ~(0xf << pos_32k[pos1])
                    derived |= tiles_combinations[0] << pos_32k[pos1]
                    derived &= ~(0xf << pos_32k[pos2])
                    derived |= tiles_combinations[0] << pos_32k[pos2]
                    result[ind] = derived
                    ind += 1
            return True, True, result
        return True, False, np.empty(0, dtype=np.uint64)

    def validate(self, board: np.uint64, original_board_sum: np.uint32) -> bool:
        """
        检验一个masked_board是否符合条件：
        1 小数字和不大于x
        2 大数组成满足--最小大数不超过2个，其余大数不超过1个
        3 如最小大数为2个，小数字和不大于y
        """
        total_sum, count_32k, pos_32k = self.tile_sum_and_32k_count2(board)
        if total_sum >= SmallTileSumLimit + 64:
            return False
        if count_32k == self.num_free_32k:
            return True
        large_tiles_sum = original_board_sum - total_sum - ((self.num_free_32k + self.num_fixed_32k) << 15)
        # assert large_tiles_sum % 64 == 0
        tiles_combinations = self.tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6),
                                                               np.uint8(count_32k - self.num_free_32k)), None)
        if tiles_combinations is None:
            return False
        if tiles_combinations[-1] == self.target:
            return False
        if count_32k < 2:
            return True
        if tiles_combinations[0] == tiles_combinations[1]:
            if total_sum >= SmallTileSumLimit:
                return False
        return True

    def unmask_board(self, board: np.uint64, original_board_sum: np.uint32) -> NDArray[np.uint64]:
        """
        将tiles_combinations的排列填回masked的格子
        original_board_sum 指mask之前的所有数字和
        total_sum          指mask之后所有小数字和
        large_tiles_sum    指mask之前的所有非32k格子数字和
        """
        total_sum, count_32k, pos_32k = self.tile_sum_and_32k_count(board)
        if count_32k == 0 or count_32k == self.num_free_32k:
            return np.array([board,], dtype=np.uint64)
        large_tiles_sum = original_board_sum - total_sum - ((self.num_free_32k + self.num_fixed_32k) << 15)
        # assert large_tiles_sum % 64 == 0
        tiles_combinations = self.tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6),
                                                               np.uint8(count_32k - self.num_free_32k)), None)
        if tiles_combinations is None:
            return np.empty(0, dtype=np.uint64)
        if len(tiles_combinations) > 1 and tiles_combinations[0] == tiles_combinations[1]:
            permutations = self.permutation_dict2[(np.uint8(count_32k), np.uint8(count_32k - self.num_free_32k))]
        else:
            permutations = self.permutation_dict1[(np.uint8(count_32k), np.uint8(count_32k - self.num_free_32k))]
        unmask_boards = np.empty(len(permutations), dtype=np.uint64)
        for ind, permutation in enumerate(permutations):
            masked = board
            # pos_32k, permutation, tiles_combinations 长度都等于 count_32k - self.num_free_32k
            indices = pos_32k[permutation]
            clear_mask = ~(np.sum(np.uint64(0xf) << indices))
            set_mask = np.sum(tiles_combinations << indices)
            masked = (masked & clear_mask) | set_mask
            unmask_boards[ind] = masked
        return unmask_boards


def generate_permutations(m: int, n: int) -> Tuple[NDArray[NDArray[int]], NDArray[NDArray[int]]]:
    """
    A(m,n)排列
    """
    num_permutations = math.factorial(m) // math.factorial(m - n)
    result1 = np.empty((num_permutations, n), dtype=np.uint8)
    if n > 1:
        result2 = np.empty((num_permutations // 2, n), dtype=np.uint8)
    else:
        result2 = np.empty((0, n), dtype=np.uint8)
    permutations = itertools.permutations(range(m), n)
    count_r2 = 0
    for ind, perm in enumerate(permutations):
        result1[ind] = perm
        if n > 1 and perm[1] < perm[0]:
            result2[count_r2] = perm
            count_r2 += 1
    result1 = _resort_permutations(m,n,result1,True)
    if n > 1:
        result2 = _resort_permutations(m, n, result2, False)
    return result1, result2


@njit(nogil=True)
def _resort_permutations(m,n,permutation,type_flag):
    """
    保证unmask后的boards是有序的（但permutation不再是字典序）
    """
    index = np.empty(len(permutation), dtype=np.uint64)
    pos = (np.arange(m)[::-1] * 4).astype(np.uint8)
    if type_flag:
        tiles = np.arange(n,dtype=np.uint64)
    else:
        tiles = np.array([0]+[i for i in range(n-1)],dtype=np.uint64)
    for i in range(len(permutation)):
        p = permutation[i]
        indices = pos[p]
        clear_mask = ~(np.sum(np.uint64(0xf) << indices))
        set_mask = np.sum(tiles << indices)
        masked = clear_mask | set_mask
        index[i] = masked
    sorted_indices = np.argsort(index)
    return permutation[sorted_indices]


def init_masker(num_free_32k: int, target_num: int, pos_fixed_32k: NDArray[np.uint8]) -> BoardMasker:
    """
    :param num_free_32k: 可以移动的32k个数
    :param target_num: 目标数字对数值
    :param pos_fixed_32k: 不可移动的32k的位置
    :return: 初始化的LineMasker
    """
    permutation_dict1 = Dict.empty(KeyType, ValueType1)
    permutation_dict2 = Dict.empty(KeyType, ValueType1)
    for n in range(1, target_num - 4):
        m = n + num_free_32k
        d1, d2 = generate_permutations(m, n)
        permutation_dict1[(np.uint8(m), np.uint8(n))], permutation_dict2[(np.uint8(m), np.uint8(n))] = d1, d2
    return BoardMasker(target_num, permutation_dict1, permutation_dict2, num_free_32k, pos_fixed_32k)


@njit(nogil=True, parallel=True, boundscheck=True)
def gen_boards_ad(arr0: NDArray[np.uint64],
                  mbm: MaskedBoardMover,
                  pattern_check_func: PatternCheckFunc,
                  to_find_func: ToFindFunc,
                  seg: NDArray[float],
                  hashmap1: NDArray[np.uint64],
                  hashmap2: NDArray[np.uint64],
                  lm: BoardMasker,
                  original_board_sum: np.uint32,
                  n: int = 8,
                  length_factor: float = 8,
                  isfree: bool = False
                  ) -> \
        Tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[float], NDArray[float],
              NDArray[np.uint64], NDArray[np.uint64]]:
    """
    0.遍历生成新board

    1.mask new tiles?(待合并?拆分暴露待合并tiles:不变):不变

    另，每隔n步检查一次所有局面合法性(不在此函数中)
    """
    # 初始化两个arr，分别对应填充数字2和4后的棋盘状态
    min_length = 9999999 if isfree else 6999999
    length = max(min_length, int(len(arr0) * length_factor))
    arr1 = np.empty(length, dtype=np.uint64)
    arr2 = np.empty(length, dtype=np.uint64)
    starts = np.array([length // n * i for i in range(n)], dtype=np.uint64)
    c1, c2 = starts.copy(),  starts.copy()
    hashmap1_length = len(hashmap1) - 1  # 要减一，这个长度用于计算哈希的时候取模
    hashmap2_length = len(hashmap2) - 1

    for s in prange(n):
        start, end = int(seg[s] * len(arr0)), int(seg[s + 1] * len(arr0))
        c1t, c2t = length // n * s, length // n * s
        for b in range(start, end):
            t = arr0[b]

            for i in range(16):  # 遍历每个位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):
                    t1 = t | (np.uint64(1) << np.uint64(4 * i))  # 填充数字2
                    for newt, mnt in mbm.move_all_dir(t1):
                        newt = np.uint64(newt)
                        if newt == t1 or not pattern_check_func(newt):
                            continue
                        newt = to_find_func(newt)
                        hashed_newt = (hash_(np.uint64(newt + mnt))) & hashmap1_length
                        if hashmap1[hashed_newt] == newt:
                            continue
                        hashmap1[hashed_newt] = newt
                        if not mnt:
                            arr1[c1t] = newt
                            c1t += 1
                            continue
                        is_valid, is_derived, derived_boards = lm.derive(newt, np.uint32(original_board_sum + 2))
                        if not is_valid:
                            break
                        if not is_derived:
                            arr1[c1t] = newt
                            c1t += 1
                            continue
                        for derived_board in derived_boards:
                            arr1[c1t] = to_find_func(derived_board)
                            c1t += 1


                    t1 = t | (np.uint64(2) << np.uint64(4 * i))  # 填4
                    for newt, mnt in mbm.move_all_dir(t1):
                        newt = np.uint64(newt)
                        if newt == t1 or not pattern_check_func(newt):
                            continue
                        newt = to_find_func(newt)
                        hashed_newt = (hash_(np.uint64(newt + mnt))) & hashmap2_length
                        if hashmap2[hashed_newt] == newt:
                            continue
                        hashmap2[hashed_newt] = newt
                        if not mnt:
                            arr2[c2t] = newt
                            c2t += 1
                            continue
                        is_valid, is_derived, derived_boards = lm.derive(newt, np.uint32(original_board_sum + 4))
                        if not is_valid:
                            break
                        if not is_derived:
                            arr2[c2t] = newt
                            c2t += 1
                            continue
                        for derived_board in derived_boards:
                            arr2[c2t] = to_find_func(derived_board)
                            c2t += 1

        c1[s] = np.uint64(c1t)
        c2[s] = np.uint64(c2t)

    # 统计每个分段生成的新局面占比，用于平衡下一层的分组seg
    all_length = c2 - starts
    percents2 = all_length / all_length.sum() if all_length.sum() > 0 else np.array([1 / n for _ in range(n)])

    all_length = c1 - starts
    percents1 = all_length / all_length.sum() if all_length.sum() > 0 else np.array([1 / n for _ in range(n)])

    arr1 = merge_inplace(arr1, c1, starts.copy())
    arr2 = merge_inplace(arr2, c2, starts.copy())

    # 返回包含可能的新棋盘状态的两个array
    return arr1, arr2, percents2, percents1, hashmap1, hashmap2


@njit(nogil=True, parallel=True)
def validate_layer(arr: NDArray[np.uint64], lm: BoardMasker, original_board_sum: np.uint32):
    """每隔n步检查一次所有局面合法性"""
    valid = np.empty(len(arr), dtype=np.bool_)
    for i in prange(len(arr)):
        valid[i] = lm.validate(arr[i], original_board_sum)
    return arr[valid]


def handle_restart_ad(i, pathname, arr_init, started, d0, d1):
    """
    处理断点重连逻辑
    """
    if (os.path.exists(pathname + str(i + 1)) and os.path.exists(pathname + str(i))) \
            or (os.path.exists(pathname + str(i + 1) + 'b') and os.path.exists(pathname + str(i))) \
            or (os.path.exists(pathname + str(i + 1) + '.z') and os.path.exists(pathname + str(i))) \
            or os.path.exists(pathname + str(i) + 'b') \
            or os.path.exists(pathname + str(i) + '.z') \
            or os.path.exists(pathname + str(i) + 'b.7z') \
            or os.path.exists(pathname + str(i) + '.7z'):
        logger.debug(f"skipping step {i}")
        return False, None, None
    if i == 1:
        arr_init.tofile(pathname + str(i - 1))
        return True, arr_init, np.empty(0, dtype=np.uint64)
    elif not started:
        d0 = np.fromfile(pathname + str(i - 1), dtype=np.uint64)
        d1 = np.fromfile(pathname + str(i), dtype=np.uint64)
        return True, d0, d1

    return True, d0, d1


def generate_process_ad(
        arr_init: NDArray[np.uint64],
        pattern_check_func: PatternCheckFunc,
        to_find_func: ToFindFunc,
        steps: int,
        pathname: str,
        mbm: MaskedBoardMover,
        lm: BoardMasker,
        isfree: bool
) -> Tuple[bool, NDArray[np.uint64], NDArray[np.uint64]]:
    started = False  # 是否进行了计算，如果是则需要进行final_steps处理最后一批局面
    d0, d1 = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    pivots, pivots_list = None, None  # 用于快排的分割点
    hashmap1, hashmap2 = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    n = max(4, min(32, os.cpu_count()))  # 并行线程数
    seg, seg_path, length_factor, length_factors, length_factors_path, length_factor_multiplier = \
        initialize_parameters_ad(n, pathname)
    ini_board_sum = np.sum(mbm.decode_board(arr_init[0]))
    for b in range(len(arr_init)):
        arr_init[b] = lm.mask_board(arr_init[b])

    # 从前向后遍历，生成新的棋盘状态并保存到相应的array中
    for i in range(1, steps - 1):
        started, d0, d1 = handle_restart_ad(i, pathname, arr_init, started, d0, d1)
        if not started:
            continue

        if pivots is None:
            pivots = d0[[len(d0) // 8 * i for i in range(1, 8)]] if len(d0) > 0 else np.zeros(7, dtype='uint64')

        t0 = time.time()
        # 先预测预分配数组的长度乘数
        length_factor = predict_next_length_factor_quadratic(length_factors)
        length_factor *= 2.0 if isfree else 1.75
        length_factor *= length_factor_multiplier
        if len(hashmap1) == 0:
            hashmap1, hashmap2 = update_hashmap_length(hashmap1, d0), update_hashmap_length(hashmap2, d0)  # 初始化

        board_sum = 2 * i + ini_board_sum - 2
        d1t, d2, percents2, percents1, hashmap1, hashmap2 = \
            gen_boards_ad(d0, mbm, pattern_check_func, to_find_func, seg,
                          hashmap1, hashmap2, lm, board_sum, n, length_factor, isfree)

        validate_length_and_balance(d0, d2, d1t, seg, percents2, percents1, length_factor)

        t1 = time.time()
        # 排序
        sort_array(d1t, pivots)
        sort_array(d2, pivots)

        seg, length_factors, length_factor_multiplier, pivots \
            = update_parameters_ad(d0, d2, seg, percents2, percents1, length_factors, seg_path, length_factors_path)

        t2 = time.time()

        # 去重
        d1t, d2 = parallel_unique(d1t, n), parallel_unique(d2, n)
        dedup_pivots = d0[np.arange(1, n) * len(d0) // n].copy() if len(d0) > 0 else \
            (np.arange(1, n) * (1 << 50) // n).astype(np.uint64)
        d1 = merge_deduplicate_all([d1, d1t], dedup_pivots, n)
        del d1t
        d1 = concatenate(d1)
        # check_sorted(d1)

        t3 = time.time()
        log_performance(i, t0, t1, t2, t3, d1)

        d0, d1 = d1, d2
        del d2
        if len(hashmap1) > 0:
            hashmap1, hashmap2 = update_hashmap_length(hashmap1, d1), update_hashmap_length(hashmap2, d1)

        if (i + ini_board_sum % 64 // 2)%32 == (SmallTileSumLimit//2)%32+1:
            d0 = validate_layer(d0, lm, board_sum + 2)
            d1 = validate_layer(d1, lm, board_sum + 4)
            logger.debug(f'validate step {i}')
        hashmap1.fill(0)

        d0.tofile(pathname + str(i))
        if SingletonConfig().config['compress_temp_files'] and i > 5:
            compress_with_7z(pathname + str(i - 2))
        hashmap1, hashmap2 = hashmap2, hashmap1

    return started, d0, d1


def initialize_parameters_ad(n, pathname):
    """
    初始化分段间隔和长度乘数
    """
    seg_path = os.path.join(os.path.dirname(pathname), "seg.txt")
    length_factors_path = os.path.join(os.path.dirname(pathname), 'length_factors.txt')
    try:
        seg = np.loadtxt(seg_path, delimiter=',', dtype=np.float64)
    except FileNotFoundError:
        seg = np.array([round(1 / n * i, 4) for i in range(n + 1)], dtype=float)
    # 预分配的储存局面的数组长度乘数
    length_factor_multiplier = 2.5

    try:
        # noinspection PyUnresolvedReferences
        length_factors = np.loadtxt(length_factors_path, delimiter=',', dtype=np.float64).tolist()
        length_factor = predict_next_length_factor_quadratic(length_factors) * 1.2
    except FileNotFoundError:
        length_factor = 3.2
        length_factors = [length_factor, length_factor, length_factor]

    return seg, seg_path, length_factor, length_factors, length_factors_path, length_factor_multiplier


def update_parameters_ad(d0, d2, seg, percents, percents1, length_factors, seg_path, length_factors_path):
    # 根据上一层实际分段情况调整下一层分段间隔
    seg = update_seg(seg, percents, 0.4)
    np.savetxt(seg_path, seg, fmt='%.6f', delimiter=',')  # type: ignore

    # 更新数组长度乘数
    length_factors = length_factors[1:] + [max(len(d2) / (1 + len(d0)), 0.8)]
    length_factor_multiplier = max(max(percents) / np.mean(percents), max(percents1) / np.mean(percents1))
    np.savetxt(length_factors_path, length_factors, fmt='%.6f', delimiter=',')  # type: ignore
    # 选取更准确的分割点使下一次快排更有效
    pivots = d2[[len(d2) // 8 * i for i in range(1, 8)]].copy() if len(d2) > 0 else np.zeros(7, dtype='uint64')
    return seg, length_factors, length_factor_multiplier, pivots


if __name__ == '__main__':

    def d(encoded_board: np.uint64) -> NDArray:
        encoded_board = np.uint64(encoded_board)
        board = np.zeros((4, 4), dtype=np.int32)
        for i in range(3, -1, -1):
            for j in range(3, -1, -1):
                encoded_num = (encoded_board >> np.uint64(4 * ((3 - i) * 4 + (3 - j)))) & np.uint64(0xF)
                if encoded_num > 0:
                    board[i, j] = 2 ** encoded_num
                else:
                    board[i, j] = 0
        return board

    # lm_ = init_masker(0, 12, np.array([0, 4], dtype=np.uint8))  # 4442-4096
    # b_ = np.uint64(0x10ff31203fff2fff)
    # r = lm_.unmask_board(b_, 69660-64)

    # lm_ = init_masker(4, 12, np.array([], dtype=np.uint8))  # free12w
    # b_ = np.uint64(0x1fff2fff002432ff)
    # r = lm_.unmask_board(b_, 132838+64)
    # p = lm_.permutation_dict2[(np.uint8(8), np.uint8(4))]
    # r = r[(p[:,0]==3)&(p[:,1]==1)]

    # lm_ = init_masker(0, 10, np.array([0, 4, 16, 20], dtype=np.uint8))  # LL-1024
    # _1,_2,r = lm_.derive(np.uint64(0x3ffffff), 200+131072)
    # print(_1,_2)

    # print(len(r))
    # print(np.all(r[1:]>r[:-1]))
    # # r.sort()
    # for _,ii in enumerate(r):
    #     print(d(ii))
    #     print()
