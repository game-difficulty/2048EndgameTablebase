import os
import time
from typing import Callable, Tuple, List
import gc

import psutil
import numpy as np
from numba import njit, prange
from numba import types
from numpy.typing import NDArray

import Config
from BoardMaskerAD import BoardMasker
from BoardMoverAD import MaskedBoardMover
from BookGenerator import predict_next_length_factor_quadratic, largest_power_of_2, extract_uniform_elements, \
    update_hashmap_length, validate_length_and_balance, log_performance, initialize_parameters, update_parameters, \
    harmonic_mean_by_column, update_parameters_big
from BookGeneratorUtils import merge_inplace, hash_, parallel_unique, sort_array, concatenate, merge_deduplicate_all
from Config import SingletonConfig, clock
from LzmaCompressor import compress_with_7z

logger = Config.logger

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]
SymFindFunc = Callable[[np.uint64], Tuple[np.uint64, int]]

KeyType = types.UniTuple(types.uint8, 2)
ValueType1 = types.uint8[:, :]
ValueType2 = types.uint8[:]


@njit(nogil=True, parallel=True)
def gen_boards_ad(arr0: NDArray[np.uint64],
                  mbm: MaskedBoardMover,
                  pattern_check_func: PatternCheckFunc,
                  to_find_func: ToFindFunc,
                  sym_func: SymFindFunc,
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
              NDArray[np.uint64], NDArray[np.uint64], NDArray[np.float64]]:
    """
    0.遍历生成新board

    1.mask new tiles?(待合并?拆分暴露待合并tiles:不变):不变

    另，每隔n步检查一次所有局面合法性(不在此函数中)
    """
    # 初始化两个arr，分别对应填充数字2和4后的棋盘状态
    min_length = 99999999 if isfree else 69999999
    length = max(min_length, int(len(arr0) * length_factor))
    arr1 = np.empty(length, dtype=np.uint64)
    arr2 = np.empty(length, dtype=np.uint64)
    starts = np.array([length // n * i for i in range(n)], dtype=np.uint64)
    c1, c2 = starts.copy(), starts.copy()
    hashmap1_length = len(hashmap1) - 1  # 要减一，这个长度用于计算哈希的时候取模
    hashmap2_length = len(hashmap2) - 1
    times = np.empty(n, dtype=np.float64)
    t0 = clock()

    for s in prange(n):
        start, end = int(seg[s] * len(arr0)), int(seg[s + 1] * len(arr0))
        c1t, c2t = length // n * s, length // n * s
        for b in range(start, end):
            t = arr0[b]

            for i in range(16):  # 遍历每个位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):
                    t1 = t | (np.uint64(1) << np.uint64(4 * i))  # 填充数字2
                    for newt, mnt in mbm.move_all_dir(t1):
                        if newt == t1 or not pattern_check_func(newt):
                            continue
                        newt, symm_index = sym_func(newt)
                        hashed_newt = (hash_(newt) & hashmap1_length)
                        if hashmap1[hashed_newt] == newt:
                            continue
                        hashmap1[hashed_newt] = newt

                        if not mnt:
                            arr1[c1t] = newt
                            c1t += 1
                            continue
                        is_valid, is_derived, derived_boards = derive(lm, newt, np.uint32(original_board_sum + 2))
                        if not is_valid:
                            continue
                        if not is_derived:
                            arr1[c1t] = newt
                            c1t += 1
                            continue
                        for derived_board in derived_boards:
                            if pattern_check_func(derived_board):
                                arr1[c1t] = to_find_func(derived_board)
                                c1t += 1

                    t1 = t | (np.uint64(2) << np.uint64(4 * i))  # 填4
                    for newt, mnt in mbm.move_all_dir(t1):
                        if newt == t1 or not pattern_check_func(newt):
                            continue
                        newt, symm_index = sym_func(newt)
                        hashed_newt = (hash_(newt) & hashmap2_length)
                        if hashmap2[hashed_newt] == newt:
                            continue
                        hashmap2[hashed_newt] = newt

                        if not mnt:
                            arr2[c2t] = newt
                            c2t += 1
                            continue
                        is_valid, is_derived, derived_boards = derive(lm, newt, np.uint32(original_board_sum + 4))
                        if not is_valid:
                            continue
                        if not is_derived:
                            arr2[c2t] = newt
                            c2t += 1
                            continue
                        for derived_board in derived_boards:
                            if pattern_check_func(derived_board):
                                arr2[c2t] = to_find_func(derived_board)
                                c2t += 1

        c1[s] = np.uint64(c1t)
        c2[s] = np.uint64(c2t)
        times[s] = clock() - t0

    # 统计每个分段生成的新局面占比，用于平衡下一层的分组seg
    all_length = c2 - starts
    percents2 = all_length / all_length.sum() if all_length.sum() > 0 else np.array([1 / n for _ in range(n)])

    all_length = c1 - starts
    percents1 = all_length / all_length.sum() if all_length.sum() > 0 else np.array([1 / n for _ in range(n)])

    arr1 = merge_inplace(arr1, c1, starts.copy())
    arr2 = merge_inplace(arr2, c2, starts.copy())

    # 返回包含可能的新棋盘状态的两个array
    return arr1, arr2, percents2, percents1, hashmap1, hashmap2, times


@njit(nogil=True, parallel=True)
def validate_layer(arr: NDArray[np.uint64], lm: BoardMasker, original_board_sum: np.uint32):
    """每隔n步检查一次所有局面合法性"""
    valid = np.empty(len(arr), dtype=np.bool_)
    for i in prange(len(arr)):
        valid[i] = lm.validate(arr[i], original_board_sum)
    return arr[valid]


@njit(nogil=True)
def derive(lm: BoardMasker, board: np.uint64, original_board_sum: np.uint32) -> Tuple[bool, bool, NDArray[np.uint64]]:
    """
    在tiles_combinations最小两个数字相等的前提下，暴露这两个tiles，返回全部可能组合
    第一个返回值是board是否合法，第二个返回值是是否derive，第三个返回值是全部可能组合（如有，否则为空数组）

    0、需剪枝局面 1、存在两个相同大数 2、不存在两个相同大数 3、存在3个64 4、成功
    # 1.1 正常 1.2 一次合出两个64
    # 2.1 不由3个64合并而来 2.2 由3个64合并而来，需要把剩下的64 mask掉
    """
    total_sum, count_32k, pos_32k, board_, tile64_count = lm.tile_sum_and_32k_count3(board)

    if total_sum >= lm.small_tile_sum_limit + 64:
        # 0
        return False, False, np.empty(0, dtype=np.uint64)
    large_tiles_sum = original_board_sum - total_sum - ((lm.num_free_32k + lm.num_fixed_32k) << 15)

    # assert large_tiles_sum % 64 == 0
    tiles_combinations = lm.tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6),
                                                         np.uint8(count_32k - lm.num_free_32k)), None)
    if tiles_combinations is None:
        # 0
        return False, False, np.empty(0, dtype=np.uint64)
    if tiles_combinations[-1] == lm.target:
        # 4
        return False, False, np.empty(0, dtype=np.uint64)
    if count_32k - lm.num_free_32k < 2:
        # 2.1
        return True, False, np.empty(0, dtype=np.uint64)

    if tiles_combinations[0] == tiles_combinations[1]:
        if total_sum >= lm.small_tile_sum_limit:
            # 0
            return False, False, np.empty(0, dtype=np.uint64)
        if len(tiles_combinations) > 2 and tiles_combinations[0] == tiles_combinations[2]:
            # 3
            if total_sum >= lm.small_tile_sum_limit - 64:
                # 0
                return False, False, np.empty(0, dtype=np.uint64)
            if tile64_count == 0:
                # 1.2
                return True, True, np.empty(0, dtype=np.uint64)
            # 1.1
            result = _derive_3x64(board, pos_32k)
            return True, True, result
        else:
            # 1
            # 由于哈希表预去重，这里不能只暴露另一个最小大数的所有可能位置。
            result = np.empty(count_32k * (count_32k - 1) // 2, dtype=np.uint64)
            ind = 0
            for pos1 in range(count_32k - 1):
                for pos2 in range(pos1 + 1, count_32k):
                    derived = board & ~(np.uint64(0xf) << pos_32k[pos1])
                    derived |= tiles_combinations[0] << pos_32k[pos1]
                    derived &= ~(np.uint64(0xf) << pos_32k[pos2])
                    derived |= tiles_combinations[0] << pos_32k[pos2]
                    result[ind] = derived
                    ind += 1
            return True, True, result

    if board_ != board:
        # 2.2
        return True, True, np.array([board_], dtype=np.uint64)
    return True, False, np.empty(0, dtype=np.uint64)


@njit(nogil=True)
def _derive_3x64(masked: np.uint64, pos_32k: NDArray[np.uint64]) -> NDArray[np.uint64]:
    ind = 0
    result = np.empty(len(pos_32k) - 2, dtype=np.uint64)
    for i in pos_32k:
        tile_value = (masked >> i) & np.uint64(0xf)
        if tile_value == 6:
            continue
        derived = masked & ~(np.uint64(0xf) << i)
        derived |= np.uint64(6) << i
        result[ind] = derived
        ind += 1
    return result


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
        sym_func: SymFindFunc,
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
    seg, seg_list, seg_list_path, length_factor, length_factors, length_factors_list, length_factors_list_path, \
        percents2, length_factor_multiplier, segment_size = initialize_parameters(n, pathname, isfree)
    ini_board_sum = np.sum(mbm.decode_board(arr_init[0]))
    for b in range(len(arr_init)):
        arr_init[b] = lm.mask_board(arr_init[b])
    small_tile_sum_limit = SingletonConfig().config.get('SmallTileSumLimit', 56)

    # 从前向后遍历，生成新的棋盘状态并保存到相应的array中
    for i in range(1, steps - 1):
        started, d0, d1 = handle_restart_ad(i, pathname, arr_init, started, d0, d1)
        if not started:
            continue

        if pivots is None:
            pivots = d0[[len(d0) // 8 * i for i in range(1, 8)]] if len(d0) > 0 else np.zeros(7, dtype='uint64')
            pivots_list = [pivots] * int(len(seg_list) // n)

        board_sum = 2 * i + ini_board_sum - 2

        if len(d0) < segment_size:
            t0 = time.time()
            # 先预测预分配数组的长度乘数
            length_factor = predict_next_length_factor_quadratic(length_factors)
            length_factor *= 1.5 if isfree else 1.33
            length_factor *= length_factor_multiplier
            if len(hashmap1) == 0:
                hashmap1, hashmap2 = update_hashmap_length(hashmap1, d0), update_hashmap_length(hashmap2, d0)  # 初始化

            d1t, d2, percents2, percents1, hashmap1, hashmap2, times = \
                gen_boards_ad(d0, mbm, pattern_check_func, to_find_func, sym_func, seg,
                              hashmap1, hashmap2, lm, board_sum, n, length_factor, isfree)

            validate_length_and_balance(d0, d2, d1t, seg, percents2, percents1, length_factor)

            t1 = time.time()
            # 排序
            sort_array(d1t, pivots)
            sort_array(d2, pivots)

            seg, seg_list, length_factors, length_factors_list, length_factor_multiplier, pivots, pivots_list \
                = update_parameters(d0, d2, seg, percents2, percents1, length_factors, seg_list_path,
                                    length_factors_list_path)

            t2 = time.time()

            # 去重
            d1t = parallel_unique(d1t, n)
            d2 = parallel_unique(d2, n)
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

        else:
            if len(hashmap1) == 0:
                hashmap_max_length = 20971520 * (round(psutil.virtual_memory().total / (1024 ** 3), 0) * 0.75)
                hashmap1, hashmap2 = (np.empty(largest_power_of_2(hashmap_max_length), dtype=np.uint64),
                                      np.empty(largest_power_of_2(hashmap_max_length), dtype=np.uint64))  # 初始化
            (d1s, d2s, pivots_list, seg_list, length_factors_list, length_factor_multiplier, hashmap1, hashmap2,
             t0, gen_time, t2) = \
                gen_boards_big_ad(d0, mbm, lm, pattern_check_func, sym_func, to_find_func,
                               seg_list, pivots_list, hashmap1, hashmap2, board_sum, n, length_factors_list,
                               length_factor_multiplier, isfree)

            dedup_pivots = d0[np.arange(1, n) * len(d0) // n].copy() if len(d0) > 0 else \
                (np.arange(1, n) * (1 << 50) // n).astype(np.uint64)

            del d0
            d2 = merge_deduplicate_all(d2s, dedup_pivots, n)
            del d2s
            d2 = concatenate(d2)
            d1s.append(d1)
            d1 = merge_deduplicate_all(d1s, dedup_pivots, n)
            del d1s
            d1 = concatenate(d1)

            d0, d1 = d1, d2

            t3 = time.time()
            log_performance(i, t0, gen_time + t0, t2, t3, d0)

            np.savetxt(seg_list_path, seg_list, fmt='%.6f', delimiter=',')  # type: ignore
            np.savetxt(length_factors_list_path, length_factors_list, fmt='%.6f', delimiter=',')  # type: ignore
            seg = extract_uniform_elements(n, seg_list)
            length_factors = harmonic_mean_by_column(length_factors_list)

        if (i + ini_board_sum % 64 // 2) % 32 == (small_tile_sum_limit // 2) % 32 + 1:
            d0 = validate_layer(d0, lm, board_sum + 2)
            d1 = validate_layer(d1, lm, board_sum + 4)
            logger.debug(f'validate step {i}')
        hashmap1.fill(0)
        d0.tofile(pathname + str(i))
        if SingletonConfig().config['compress_temp_files'] and i > 5:
            compress_with_7z(pathname + str(i - 2))
        hashmap1, hashmap2 = hashmap2, hashmap1

    return started, d0, d1


def gen_boards_big_ad(arr0: NDArray[np.uint64],
                   mbm: MaskedBoardMover,
                   lm: BoardMasker,
                   pattern_check_func: PatternCheckFunc,
                   sym_func: SymFindFunc,
                   to_find_func: ToFindFunc,
                   seg_list: NDArray[float],
                   pivots_list: List[NDArray[np.uint64]],
                   hashmap1: NDArray[np.uint64],
                   hashmap2: NDArray[np.uint64],
                   board_sum: np.uint32,
                   n: int,
                   length_factors_list: List[List[float]],
                   length_factor_multiplier: float = 1.5,
                   isfree: bool = False,
                   ) -> Tuple[
    List[NDArray[np.uint64]], List[NDArray[np.uint64]], List[NDArray[np.uint64]], NDArray[float], List[
        List[float]], float, NDArray[np.uint64], NDArray[np.uint64], float, float, float]:
    """
    将arr0分段放入gen_boards生成排序去重后的局面，然后归并
    """
    arr1s: List[NDArray[np.uint64]] = []
    arr2s: List[NDArray[np.uint64]] = []
    actual_lengths2: NDArray[np.uint64] = np.empty(len(seg_list) - 1, dtype=np.uint64)
    actual_lengths1: NDArray[np.uint64] = np.empty(len(seg_list) - 1, dtype=np.uint64)
    t0 = time.time()

    gen_time = 0
    for seg_index in range(int(len(seg_list) // n)):
        t_ = time.time()

        start_index = int(seg_list[seg_index * n] * len(arr0))
        end_index = int(seg_list[(seg_index + 1) * n] * len(arr0))
        arr0t = arr0[start_index:end_index]

        seg = seg_list[(seg_index * n): ((seg_index + 1) * n + 1)]
        scaled_seg = (seg - seg[0]) / (seg[-1] - seg[0])

        length_factors = length_factors_list[seg_index]
        length_factor = predict_next_length_factor_quadratic(length_factors)
        length_factor *= 1.5 if isfree else 1.33
        length_factor *= length_factor_multiplier  # type: ignore

        arr1t, arr2t, percents2, percents1, hashmap1, hashmap2, times = \
            gen_boards_ad(arr0t, mbm, pattern_check_func, to_find_func, sym_func, scaled_seg,
                       hashmap1, hashmap2, lm, board_sum, n, length_factor, isfree)

        validate_length_and_balance(arr0t, arr2t, arr1t, seg, percents2, percents1, length_factor, True)

        actual_lengths2[(seg_index * n): ((seg_index + 1) * n)] = percents2 * len(arr2t)
        actual_lengths1[(seg_index * n): ((seg_index + 1) * n)] = percents1 * len(arr1t)
        length_factors_list[seg_index] = length_factors[1:] + [len(arr2t) / (1 + len(arr0t))]

        gen_time += time.time() - t_

        pivots = pivots_list[seg_index]
        sort_array(arr1t, pivots)
        sort_array(arr2t, pivots)
        pivots_list[seg_index] = arr2t[[len(arr2t) // 8 * i for i in range(1, 8)]] if len(arr2t) > 0 else \
            pivots_list[0].copy()

        arr1t = parallel_unique(arr1t, n)
        arr2t = parallel_unique(arr2t, n)
        arr1s.append(arr1t)
        arr2s.append(arr2t)

        del arr1t, arr2t, arr0t
        gc.collect()

    seg_list, length_factors_list, pivots_list, length_factor_multiplier \
        = update_parameters_big(actual_lengths2, actual_lengths1, seg_list, n, length_factors_list, pivots_list)

    t2 = time.time()
    gc.collect()

    return (arr1s, arr2s, pivots_list, seg_list, length_factors_list, length_factor_multiplier, hashmap1, hashmap2, t0,
            gen_time, t2)


if __name__ == '__main__':
    pass
    # from BoardMaskerAD import init_masker
    # lm_ = init_masker(0, 12, np.array([0, 4], dtype=np.uint8))  # 4442-4096
    # b_ = np.uint64(0x10ff31203fff2fff)
    # r = lm_.unmask_board(b_, 69660-64)
    #
    # lm_ = init_masker(4, 12, np.array([], dtype=np.uint8))  # free12w
    # b_ = np.uint64(0x1fff2fff002432ff)
    # r = lm_.unmask_board(b_, 132838+64)
    #
    # lm_ = init_masker(2, 10, np.array([0, 4, 16, 20], dtype=np.uint8))  # t
    # bm=BoardMover()
    # _1,_2,r = derive(lm_,np.uint64(0x201033ffffffffff), 918+131072,bm,4,np.uint64(0x231035fff5ffffff),0)
    # print(_1,_2)
    # for i in r:
    #     print(d(i))
    #
    # print(len(r))
    # print(np.all(r[1:]>r[:-1]))
    # # r.sort()
    # for _,ii in enumerate(r):
    #     print(bm.decode_board(ii))
    #     print()
