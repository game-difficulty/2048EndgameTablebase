import os
import time
from datetime import datetime
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from numba import types
from numba.typed import Dict

import Config
from BoardMover import move_board2
from BoardMoverAD import decode_board, m_move_all_dir2, reverse
from BoardMaskerAD import unmask_board, tile_sum_and_32k_count2, tile_sum_and_32k_count4, ParamType, extract_f_positions
from Calculator import ReverseLR, ReverseUD, ReverseUL, ReverseUR, Rotate180, RotateL, RotateR
from Config import SingletonConfig
from BookSolverADUtils import (dict_to_structured_array1, dict_to_structured_array2, dict_to_structured_array3,
                               get_array_view10, get_array_view11, get_array_view2, get_array_view3)
from LzmaCompressor import decompress_with_7z
from SignalHub import progress_signal

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]
SymFindFunc = Callable[[np.uint64], Tuple[np.uint64, int]]

KeyType1 = types.int8
ValueType1 = types.uint32[:, :]
BookDictType = Dict[KeyType1, ValueType1]
ValueType2 = types.uint64[:]
IndexDictType = Dict[KeyType1, ValueType2]
ValueType3 = types.uint32[:]
IndexIndexDictType = Dict[KeyType1, ValueType3]
ValueType4 = types.Tuple((types.uint64[:], types.uint32[:, :]))
KeyType2 = types.uint32
MatchDictType = Dict[KeyType2, ValueType4]

logger = Config.logger


def handle_solve_restart_ad(i, pathname, steps, b1, b2, i1, i2, started):
    """
    处理断点重连逻辑
    """
    if os.path.exists(pathname + str(i) + 'b'):
        logger.debug(f"skipping step {i}")
        return False, None, None, None, None
    elif not started:
        started = True
        if i != steps - 3 or b1 is None or b2 is None:
            b1, i1 = dict_fromfile(pathname, i + 1)
            b2, i2 = dict_fromfile(pathname, i + 2)
        return started, b1, b2, i1, i2

    return True, b1, b2, i1, i2


def recalculate_process_ad(
        arr_init: NDArray[np.uint64],
        book_dict1: BookDictType,
        book_dict2: BookDictType,
        ind_dict1: IndexDictType,
        ind_dict2: IndexDictType,
        tiles_combinations_dict,
        permutation_dict,
        param:ParamType, 
        pattern_check_func: PatternCheckFunc,
        sym_func: SymFindFunc,
        steps: int,
        pathname: str,
        spawn_rate4: float = 0.1
) -> None:
    started = False
    ini_board_sum = np.sum(decode_board(arr_init[0]))
    indind_dict1, indind_dict2 = None, None
    match_dict = None

    permutation_arr = dict_to_structured_array2(permutation_dict)
    tiles_combinations_arr = dict_to_structured_array3(tiles_combinations_dict)
    if not os.path.exists(pathname + 'stats.txt'):
        with open(pathname + 'stats.txt', 'a', encoding='utf-8') as file:
            file.write(','.join(['layer', 'length', 'max success rate', 'speed', 'deletion_threshold', 'time']) + '\n')

    # 从后向前更新ds中的array
    for i in range(steps - 3, -1, -1):
        started, book_dict1, book_dict2, ind_dict1, ind_dict2 \
            = handle_solve_restart_ad(i, pathname, steps, book_dict1, book_dict2, ind_dict1, ind_dict2, started)
        if not started:
            continue

        progress_signal.progress_updated.emit(steps * 2 - i - 2, steps * 2)

        if match_dict is None:
            match_dict = Dict.empty(KeyType2, ValueType4)

        original_board_sum = 2 * i + ini_board_sum
        deletion_threshold = np.uint32(SingletonConfig().config.get('deletion_threshold', 0.0) * 4e9)
        if SingletonConfig().config.get('compress_temp_files', False):
            decompress_with_7z(pathname + str(i) + '.7z')
        d0 = np.fromfile(pathname + str(i), dtype=np.uint64)

        t0 = time.time()

        book_dict0, ind_dict0 = expand_ad(d0, original_board_sum, tiles_combinations_arr, param)
        del d0

        # 创建、更新查找索引
        if indind_dict1 is not None:
            indind_dict2 = indind_dict1
        else:
            indind_dict2 = create_index_ad(ind_dict2)
        indind_dict1 = create_index_ad(ind_dict1)

        t1 = time.time()

        # 回算
        book_arr0 = dict_to_structured_array1(book_dict0)
        book_arr1 = dict_to_structured_array1(book_dict1)
        book_arr2 = dict_to_structured_array1(book_dict2)
        ind_arr0 = dict_to_structured_array1(ind_dict0)
        ind_arr1 = dict_to_structured_array1(ind_dict1)
        ind_arr2 = dict_to_structured_array1(ind_dict2)
        indind_arr1 = dict_to_structured_array1(indind_dict1)
        indind_arr2 = dict_to_structured_array1(indind_dict2)

        recalculate_ad(book_arr0, ind_arr0, book_arr1, ind_arr1, indind_arr1, book_arr2, ind_arr2, indind_arr2,
                       match_dict, tiles_combinations_arr, permutation_arr, param,
                       original_board_sum, pattern_check_func, sym_func, spawn_rate4)
        length, max_rate = length_count(book_dict0)
        t2 = time.time()

        if deletion_threshold > 0:
            remove_died_ad(book_dict2, ind_dict2, deletion_threshold)
        remove_died_ad(book_dict0, ind_dict0)
        t3 = time.time()
        if t3 > t0:
            logger.debug(f'step {i} recalculated: {round(length / (t3 - t0) / 1e6, 2)} mbps')
            logger.debug(f'index/solve/remove: {round((t1 - t0) / (t3 - t0), 2)}/'
                         f'{round((t2 - t1) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}')

        with open(pathname + 'stats.txt', 'a', encoding='utf-8') as file:
            file.write(','.join([str(i), str(length), str(max_rate),
                                 f'{round(length / (t3 - t0 + 0.0000001) / 1e6, 2)} mbps',
                                 str(deletion_threshold / 4e9), str(datetime.now())]) + '\n')
        if deletion_threshold > 0:
            dict_tofile(book_dict2, ind_dict2, pathname, i + 2)  # 再写一次，把成功率低于阈值的局面去掉

        del book_dict2, ind_dict2
        dict_tofile(book_dict0, ind_dict0, pathname, i)
        if os.path.exists(pathname + str(i)):
            os.remove(pathname + str(i))
        logger.debug(f'step {i} written\n')

        book_dict2, ind_dict2 = book_dict1, ind_dict1
        book_dict1, ind_dict1 = book_dict0, ind_dict0


@njit(nogil=True, parallel=True)
def expand_ad(arr: NDArray[np.uint64], original_board_sum: np.uint32, tiles_combinations_dict, param:ParamType, 
              expand_book_dict: bool = True) -> Tuple[BookDictType, IndexDictType]:
    """
    数据结构：
    ind_dict存储所有masked局面，每个键对应一组具有相同unmasked数量的masked局面：
    - 键：象征masked局面所代表的unmasked局面数量（由大数组合tiles_combinations决定）
    - 值：包含所有具有该unmasked数量的masked局面数组

    book_dict存储对应的成功率数据：
    - 键：与ind_dict共享键集合
    - 值：一个二维数组，包含每个masked组对应的成功率结果

    关于键：
    - 如果所有大数均不超过一个，键为count_32k
    - 如果最小大数有两个，键为-count_32k,
    - 如果有3个64，键为count_32k - 3 + 16
    - 如果局面不合法，暂时用-1表示
    """
    count32ks = np.empty(len(arr), dtype=np.int8)
    for i in prange(len(arr)):
        board = arr[i]
        total_sum, count_32k, pos_32k = tile_sum_and_32k_count2(board, param)
        if count_32k == param.num_free_32k:
            count32ks[i] = count_32k
            continue
        large_tiles_sum = original_board_sum - total_sum - ((param.num_free_32k + param.num_fixed_32k) << 15)
        tiles_combinations = get_array_view3(tiles_combinations_dict, np.uint8(large_tiles_sum >> 6), np.uint8(count_32k - param.num_free_32k))  # tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6), np.uint8(count_32k - param.num_free_32k)), None)
        if tiles_combinations is None:
            count32ks[i] = -1
            continue
        if count_32k - param.num_free_32k == 1:  # 也即 len(tiles_combinations) == 1
            count32ks[i] = count_32k
            continue
        if tiles_combinations[0] == tiles_combinations[1]:
            if count_32k - param.num_free_32k > 2 and tiles_combinations[0] == tiles_combinations[2]:
                # 3个64
                count32ks[i] = count_32k - 3 + 16
            else:
                count32ks[i] = -count_32k
            continue
        count32ks[i] = count_32k

    flags = np.zeros(48, dtype=np.bool_)
    for i in range(len(count32ks)):
        flags[count32ks[i] + 15] = True

    book_dict = Dict.empty(KeyType1, ValueType1)
    ind_dict = Dict.empty(KeyType1, ValueType2)

    factorials = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 1, 1, 1, 1, 1, 1, 1],
                          dtype=np.uint32)

    for flag in range(48):
        count_32k = np.int8(flag - 15)
        if count_32k == -1:
            continue
        if count_32k < 0:
            derive_size = np.uint32(factorials[-count_32k - 2] // factorials[param.num_free_32k])
        elif count_32k > 15:
            derive_size = np.uint32(factorials[count_32k - 16] // factorials[param.num_free_32k])
        else:
            derive_size = np.uint32(factorials[count_32k] // factorials[param.num_free_32k])

        if not flags[flag]:
            ind_dict[count_32k] = np.empty(0, dtype=np.uint64)
            if expand_book_dict:
                book_dict[count_32k] = np.empty((0, derive_size), dtype=np.uint32)
            continue

        positions = arr[count32ks == count_32k]

        ind_dict[count_32k] = positions
        if expand_book_dict:
            book_dict[count_32k] = np.empty((len(positions), derive_size), dtype=np.uint32)

    return book_dict, ind_dict


@njit(parallel=True, nogil=True)
def recalculate_ad(
        book_dict0: BookDictType, ind_dict0: IndexDictType,
        book_dict1: BookDictType, ind_dict1: IndexDictType, indind_dict1: IndexIndexDictType,
        book_dict2: BookDictType, ind_dict2: IndexDictType, indind_dict2: IndexIndexDictType,
        match_dict: MatchDictType,
        tiles_combinations_dict,
        permutation_dict,
        param:ParamType, 
        original_board_sum: np.uint32,
        pattern_check_func: PatternCheckFunc,
        sym_func: SymFindFunc,
        spawn_rate4: float = 0.1
):
    """
    回算方法的核心逻辑（批量回算优化）

    当masked局面移动后未生成新大数时：
    1. **unmasked基数不变**：
       - tiles_combinations的大数组合未变化 → 该masked局面代表的unmasked局面总数保持不变

    2. **顺序破坏问题**：
       - 先对masked局面进行变换后再unmask，与先unmask再对unmasked局面逐个变换的结果集合相同（变换是指出数、移动、对称系列操作）
       - 但后者的操作顺序会导致unmasked局面的自然升序排列被破坏（不可交换性）
       - 而批量查询得到的成功率数组对应的是自然升序排列下的unmasked局面的成功率
       → 批量查询时无法直接使用原有映射关系
       - 核心问题在于如何得到每种变换对unmasked升序排列的扰动模式

    3. **朴素解决方案**：
        - 每次实时计算: 对masked局面进行变换 → unmask → np.argsort重排
        - 缺点: unmask、np.argsort操作成本较高，时间复杂度是O(nlog(n))
        - 所有局面必须unmask才能回算，无法充分利用unmasked基数不变特性，无法体现批量回算的优势

    4. **缓存表解决方案**：
       - 设计一种对变换操作进行编码的方式，采用哈希表缓存
       - 结构示意: {unmasked基数: [变换编码: 索引映射数组 例如[2,0,3,1,...],...] ,...}  详见注意事项
       - 对masked局面进行变换的同时计算变换编码，复杂度O(1)，在批量查询后通过缓存表快速重建变换后的顺序关系
       - 平均时间复杂度是O(n)，且系数较小；unmasked基数越大计算速度提升越明显


    当masked局面移动后生成了新大数：
    1. **unmasked基数变化**：
       - 需重新计算新棋盘的大数组合（tiles_combinations）
       - 根据新局面大数组合确定批量查询策略

    2. **存在重复大数**：
       - 多次查询，结果合并；不使用缓存

    3. **无重复大数**：
       - 单次查询，从结果中筛选匹配对应的成功率；使用缓存

    4. **特殊场景（三个64）**：
       - 仍满足基数不变性，使用缓存


    *注意：
       - 先unmask再对unmasked局面逐个变换时，需要保证所有变换与masked局面完全相同。因此对称函数不能用ToFindFunc，需要用sym_like
       - 目前使用缓存表解决方案，哈希表不扩容、不处理碰撞，但是理论上仍然存在数据竞争导致结果错误的可能。实测结果无误
       - unmasked基数等于1时：
           1. 不涉及unmask操作和顺序破坏问题，无法批量回算
           2. 单独写计算逻辑，核心逻辑与老算法基本一致
           3. 注意移动后生成了新大数时，需要在查询后从结果中筛选匹配对应的成功率
       - 缓存表数据结构：
           1. 外层字典: 按unmasked基数分桶存储
              key: unmasked基数
              value: 数组模拟哈希表
           2. 哈希表数组:
              索引: 变换编码原始值数组
              值: 索引映射数组数组 (np.ndarray[np.ndarray[uint32]])
    """
    pos_rev = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.uint8)

    for i_c32k in range(48):
        count_32k = i_c32k - 16
        index = get_array_view10(ind_dict0, count_32k, np.uint64)
        if len(index) == 0:
            continue

        # print(count_32k)
        derive_size = book_dict0[i_c32k]['cols']
        map_length = 33331 if derive_size < 5000 else 11113

        match_index, match_mat = match_dict.setdefault(
            np.uint32(derive_size), (np.full(map_length, np.uint64(0xffffffffffffffff), dtype=np.uint64),
                                     np.zeros((map_length, derive_size), dtype=np.uint32)))

        book_mat0 = get_array_view11(book_dict0, count_32k, np.uint32)
        book_mat1 = get_array_view11(book_dict1, count_32k, np.uint32)  # book_dict1.setdefault(count_32k, np.empty((0, derive_size), dtype=np.uint32))
        ind_arr1 = get_array_view10(ind_dict1, count_32k, np.uint64)  # ind_dict1.setdefault(count_32k, np.empty(0, dtype=np.uint64))
        indind_arr1 = get_array_view10(indind_dict1, count_32k, np.uint32)  # indind_dict1.setdefault(count_32k, np.empty(0, dtype=np.uint32))
        book_mat2 = get_array_view11(book_dict2, count_32k, np.uint32) # book_dict2.setdefault(count_32k, np.empty((0, derive_size), dtype=np.uint32))
        ind_arr2 = get_array_view10(ind_dict2, count_32k, np.uint64)  # ind_dict2.setdefault(count_32k, np.empty(0, dtype=np.uint64))
        indind_arr2 = get_array_view10(indind_dict2, count_32k, np.uint32)  # indind_dict2.setdefault(count_32k, np.empty(0, dtype=np.uint32))

        chunk_count = max(min(1024, int(len(index) * round(np.log2(derive_size + 1)) // 1048576)), 1)
        chunk_size = len(index) // chunk_count
        for chunk in range(chunk_count):
            start, end = chunk_size * chunk, chunk_size * (chunk + 1)
            if chunk == chunk_count - 1:
                end = len(index)
            for k in prange(start, end):

                if derive_size == 1:
                    # derive_size == 1有些计算可以省略，因此单独写逻辑
                    t = index[k]
                    # 初始化概率和权重
                    success_probability = 0
                    empty_slots = 0

                    for i in range(16):  # 遍历所有位置
                        if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):  # 如果当前位置为空
                            empty_slots += 1

                            # 对于每个空位置，尝试填充2和4
                            new_value, probability = np.uint64(1), 1 - spawn_rate4
                            board_sum = original_board_sum + np.uint32(2)
                            optimal_success_rate = solve_optimal_success_rate(t, new_value, i, board_sum,
                                                                              pattern_check_func, sym_func,
                                                                              count_32k, book_dict1, ind_dict1,
                                                                              indind_dict1, book_mat1,
                                                                              ind_arr1, indind_arr1,
                                                                              tiles_combinations_dict, param)

                            # 对最佳移动下的成功概率加权平均
                            success_probability += optimal_success_rate * probability

                            # 填4
                            new_value, probability = np.uint64(2), spawn_rate4
                            board_sum = original_board_sum + np.uint32(4)
                            optimal_success_rate = solve_optimal_success_rate(t, new_value, i, board_sum,
                                                                              pattern_check_func, sym_func,
                                                                              count_32k, book_dict2, ind_dict2,
                                                                              indind_dict2, book_mat2,
                                                                              ind_arr2, indind_arr2,
                                                                              tiles_combinations_dict, param)

                            success_probability += optimal_success_rate * probability
                    book_mat0[k][0] = success_probability / empty_slots

                else:
                    t = index[k]

                    # rep_t, rep_v 用于计算变换编码，需要同步后续newt进行的所有变换，包括出数、移动、对称
                    rep_t, rep_v = replace_val(t)
                    rep_t_rev = reverse(rep_t)
                    # 初始化概率和权重
                    success_probability = np.zeros(derive_size, dtype=np.float64)
                    empty_slots = 0

                    for i in range(16):  # 遍历所有位置
                        if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):  # 如果当前位置为空
                            empty_slots += 1

                            # 对于每个空位置，尝试填充2和4
                            new_value, probability = 1, 1 - spawn_rate4
                            board_sum = original_board_sum + np.uint32(2)
                            optimal_success_rate = \
                                solve_optimal_success_rate_arr(t, new_value, i, rep_t, rep_t_rev, pos_rev, derive_size,
                                                               pattern_check_func,
                                                               sym_func, rep_v, count_32k, book_dict1, ind_dict1,
                                                               indind_dict1, board_sum,
                                                               match_index, match_mat, book_mat1, ind_arr1, indind_arr1,
                                                               tiles_combinations_dict, permutation_dict, param)

                            # 对最佳移动下的成功概率加权平均
                            success_probability += optimal_success_rate * probability

                            # 填4
                            new_value, probability = 2, spawn_rate4
                            board_sum = original_board_sum + np.uint32(4)
                            optimal_success_rate = \
                                solve_optimal_success_rate_arr(t, new_value, i, rep_t, rep_t_rev, pos_rev, derive_size,
                                                               pattern_check_func,
                                                               sym_func, rep_v, count_32k, book_dict2, ind_dict2,
                                                               indind_dict2, board_sum,
                                                               match_index, match_mat, book_mat2, ind_arr2, indind_arr2,
                                                               tiles_combinations_dict, permutation_dict, param)

                            success_probability += optimal_success_rate * probability
                    book_mat0[k] = (success_probability / empty_slots).astype(np.uint32)
            # 当大数数量（n）较多时，如果预先计算所有的变换编码（共n!种可能）会占用大量的空间（n!*n!*2 bytes）
            # 但结果表明，当大数数量（n）较多时，只有少数变换编码是可能用到的，因此采用缓存表
            # print(f'load {np.sum(match_index != np.uint64(0xffffffffffffffff))}')


@njit(nogil=True)
def solve_optimal_success_rate_arr(t, new_value, i, rep_t, rep_t_rev, pos_rev, derive_size, pattern_check_func,
                                   sym_func, rep_v, count_32k, book_dict, ind_dict, indind_dict, board_sum,
                                   match_index, match_mat, book_mat, ind_arr, indind_arr, tiles_combinations_dict,
                                   permutation_dict, param:ParamType):
    t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
    rep_t_gen = rep_t | (np.uint64(new_value) << np.uint64(4 * i))
    rep_t_gen_rev = rep_t_rev | (np.uint64(new_value) << np.uint64(4 * pos_rev[i]))
    board_rev = np.uint64(reverse(t_gen))
    optimal_success_rate = np.zeros(derive_size, dtype=np.uint64)

    #  mnt指移动后是否生成新大数
    for direc, (newt, mnt) in enumerate(m_move_all_dir2(t_gen, board_rev)):
        if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
            newt, symm_index = sym_func(newt)
            rep_t_gen_m = np.uint64(move_board2(rep_t_gen, rep_t_gen_rev, direc + 1))
            rep_t_gen_m = np.uint64(sym_like(rep_t_gen_m, symm_index))
            match_ind = ind_match(rep_t_gen_m, rep_v)
            hashed_match_ind = match_ind % np.uint64(len(match_index))

            if mnt:
                # 0、3x64 -> 64 128 1、合成后不存在两个相同大数 2、合成后存在两个相同大数 3、合出第三个64 4、成功
                unmasked_newt = np.uint64(move_board2(t_gen, board_rev, direc + 1))
                unmasked_newt = np.uint64(sym_like(unmasked_newt, symm_index))
                if count_32k > 15:
                    # 0
                    if match_index[hashed_match_ind] == match_ind:
                        # 0.1
                        ranked_array = match_mat[hashed_match_ind]
                        update_mnt_osr_364_arr_ad(book_dict, newt, unmasked_newt,
                                                  ind_dict, indind_dict,
                                                  optimal_success_rate, ranked_array, permutation_dict, param)
                    else:
                        # 0.2
                        new_derived = process_derived(t_gen, board_sum, direc, symm_index, tiles_combinations_dict,
                                                      permutation_dict, param)
                        ranked_array = match_arr(new_derived)

                        if match_index[hashed_match_ind] == np.uint64(0xffffffffffffffff):
                            match_index[hashed_match_ind] = match_ind
                            match_mat[hashed_match_ind] = ranked_array
                        update_mnt_osr_364_arr_ad(book_dict, newt, unmasked_newt,
                                                  ind_dict, indind_dict,
                                                  optimal_success_rate, ranked_array, permutation_dict, param)
                    continue
                need_process, tiles_combinations, pos_rank, pos_32k, tile_value, count32k = (
                    dispatch_mnt_osr_ad_arr(unmasked_newt, board_sum,
                                            optimal_success_rate, tiles_combinations_dict, param))

                if not need_process:
                    # 4
                    continue
                if len(tiles_combinations) > 2 and tiles_combinations[0] == tiles_combinations[1] \
                        and tiles_combinations[0] == tiles_combinations[2]:
                    # 3
                    if match_index[hashed_match_ind] == match_ind:
                        # 3.1
                        ranked_array = match_mat[hashed_match_ind]
                        update_mnt_osr_ad_arr3(book_dict, unmasked_newt, count32k, ind_dict,
                                               indind_dict, ranked_array, optimal_success_rate)
                    else:
                        # 3.2
                        new_derived = process_derived(t_gen, board_sum, direc, symm_index, tiles_combinations_dict,
                                                      permutation_dict, param)
                        ranked_array = match_arr(new_derived)

                        if match_index[hashed_match_ind] == np.uint64(0xffffffffffffffff):
                            match_index[hashed_match_ind] = match_ind
                            match_mat[hashed_match_ind] = ranked_array
                        update_mnt_osr_ad_arr3(book_dict, unmasked_newt, count32k, ind_dict,
                                               indind_dict, ranked_array, optimal_success_rate)
                    pass
                elif len(tiles_combinations) > 1 and tiles_combinations[0] == tiles_combinations[1]:
                    # 2
                    new_derived = process_derived(t_gen, board_sum, direc, symm_index, tiles_combinations_dict,
                                                  permutation_dict, param)
                    update_mnt_osr_ad_arr1(book_dict, unmasked_newt, ind_dict, indind_dict,
                                           new_derived, optimal_success_rate, sym_func,
                                           tiles_combinations, pos_rank, pos_32k, tile_value,
                                           count32k)
                else:
                    # 1
                    if match_index[hashed_match_ind] == match_ind:
                        # 1.1
                        ranked_array = match_mat[hashed_match_ind]
                        update_mnt_osr_ad_arr2(book_dict, newt, pos_rank, count32k, ind_dict,
                                               indind_dict, ranked_array, optimal_success_rate, permutation_dict, param)
                    else:
                        # 1.2
                        new_derived = process_derived(t_gen, board_sum, direc, symm_index, tiles_combinations_dict,
                                                      permutation_dict, param)
                        ranked_array = match_arr(new_derived)

                        if match_index[hashed_match_ind] == np.uint64(0xffffffffffffffff):
                            match_index[hashed_match_ind] = match_ind
                            match_mat[hashed_match_ind] = ranked_array
                        update_mnt_osr_ad_arr2(book_dict, newt, pos_rank, count32k, ind_dict,
                                               indind_dict, ranked_array, optimal_success_rate, permutation_dict, param)
            elif match_index[hashed_match_ind] == match_ind:
                ranked_array = match_mat[hashed_match_ind]
                update_osr_ad_arr(book_mat, newt, ind_arr, indind_arr,
                                  optimal_success_rate, ranked_array)
            else:
                new_derived = process_derived(t_gen, board_sum, direc, symm_index, tiles_combinations_dict,
                                              permutation_dict, param)
                ranked_array = match_arr(new_derived)
                if match_index[hashed_match_ind] == np.uint64(0xffffffffffffffff):
                    match_index[hashed_match_ind] = match_ind
                    match_mat[hashed_match_ind] = ranked_array
                update_osr_ad_arr(book_mat, newt, ind_arr, indind_arr,
                                  optimal_success_rate, ranked_array)
    return optimal_success_rate


@njit(nogil=True)
def solve_optimal_success_rate(t, new_value, i, board_sum, pattern_check_func, sym_func, count_32k,
                               book_dict, ind_dict, indind_dict, book_mat, ind_arr, indind_arr,
                               tiles_combinations_dict, param:ParamType):
    t_gen = t | (new_value << np.uint64(4 * i))
    board_rev = reverse(t_gen)
    optimal_success_rate = np.uint32(0)

    for direc, (newt, mnt) in enumerate(m_move_all_dir2(t_gen, board_rev)):
        if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
            newt, symm_index = sym_func(newt)
            if mnt:

                unmasked_newt = move_board2(t_gen, board_rev, direc + 1)
                unmasked_newt = np.uint64(sym_like(unmasked_newt, symm_index))

                if count_32k < 16:
                    optimal_success_rate = update_mnt_osr_ad(book_dict, newt, unmasked_newt,
                                                             ind_dict, indind_dict, optimal_success_rate, board_sum,
                                                             tiles_combinations_dict, param)
                else:
                    optimal_success_rate = update_mnt_osr_364_ad(book_dict, newt, unmasked_newt,
                                                                 ind_dict, indind_dict, optimal_success_rate, param)
            else:
                optimal_success_rate = update_osr_ad(book_mat, newt, ind_arr, indind_arr,
                                                     optimal_success_rate)
    return optimal_success_rate


@njit(nogil=True)
def update_mnt_osr_ad(book_dict: BookDictType, b: np.uint64, unmasked_b: np.uint64,
                      ind_dict: IndexDictType, indind_dict: IndexIndexDictType, osr: np.uint64, board_sum: np.uint32,
                      tiles_combinations_dict, param:ParamType) -> np.uint64:
    total_sum, count_32k, pos_32k, pos_rank, merged_tile_found, merged_tile, is_success = tile_sum_and_32k_count4(
        unmasked_b, param)

    if is_success:
        return np.uint64(4e9)
    large_tiles_sum = board_sum - total_sum - ((param.num_free_32k + param.num_fixed_32k) << 15)
    tiles_combinations = get_array_view3(tiles_combinations_dict, np.uint8(large_tiles_sum >> 6), np.uint8(count_32k - param.num_free_32k))  # tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6), np.uint8(count_32k - param.num_free_32k)), None)
    if tiles_combinations is None:
        return osr

    if merged_tile_found == 2:
        # 5 6
        return osr

    if len(tiles_combinations) > 1 and tiles_combinations[0] == tiles_combinations[1]:
        if len(tiles_combinations) > 2 and tiles_combinations[0] == tiles_combinations[2]:
            # 4
            count_32k = np.int8(count_32k + 16 - 3)  # 实际上就是16
            b = unmasked_b
        else:
            # 3
            count_32k = np.int8(-count_32k)  # 实际上是-2

            tile = np.uint64(
                tiles_combinations[0])  # = tiles_combinations[1] = log2(large_tiles_sum >> 1)
            tiles_all_positions = tile * np.uint64(0x1111111111111111)
            b = (b & (~pos_32k)) | (tiles_all_positions & pos_32k)

    ind_arr = get_array_view10(ind_dict, count_32k, np.uint64)  # ind_dict.get(np.int8(count_32k), np.empty(0, dtype=np.uint64))
    indind_arr = get_array_view10(indind_dict, count_32k, np.uint32)  # indind_dict.get(np.int8(count_32k), np.empty(0, dtype=np.uint32))

    mid = search_arr_ad(ind_arr, np.uint64(b), indind_arr)
    if mid != 0xffffffff:
        book_mat = get_array_view11(book_dict, count_32k, np.uint32)  # book_dict[count_32k]
        if count_32k > 15:
            # 4
            return max(osr, book_mat[mid][0])
        elif count_32k > 0:
            # 1 2
            return max(osr, book_mat[mid][pos_rank])
        else:
            # 3
            return max(osr, book_mat[mid][0])
    return osr


@njit(nogil=True)
def process_derived(t_gen: np.uint64, board_sum: np.uint32, direc: int,
                    symm_index: int, tiles_combinations_dict, permutation_dict, param:ParamType) -> NDArray[np.uint64]:
    derived = unmask_board(t_gen, board_sum, tiles_combinations_dict, permutation_dict, param)
    derived_rev = reverse_arr(derived)
    new_derived = move_arr(derived, direc + 1, derived_rev)
    sym_arr_like(new_derived, symm_index)
    return new_derived


@njit(nogil=True, inline='always')
def arr_max(arr1: NDArray, arr2: NDArray):
    # assert len(arr1) - len(arr2) == 0
    for i in range(len(arr1)):
        arr1[i] = max(arr1[i], arr2[i])


@njit(nogil=True)
def match_arr(a: NDArray) -> NDArray[np.uint32]:
    sorted_indices = np.argsort(a)  # 事实上在Numba包装的函数内部调用np.argsort比原版更慢
    ranked_array = np.empty(len(a), dtype=np.uint32)
    ranked_array[sorted_indices] = np.arange(len(a))
    return ranked_array


@njit(nogil=True)
def update_osr_ad_arr(book_mat: NDArray[NDArray[np.uint32]], b: np.uint64, ind_ar: NDArray[np.uint64],
                      indind_ar: NDArray[np.uint32], osr: NDArray[np.uint64], ranked_array: NDArray[np.uint32]):
    mid = search_arr_ad(ind_ar, b, indind_ar)
    if mid != 0xffffffff:
        arr_max(osr, book_mat[mid][ranked_array])


@njit(nogil=True)
def update_osr_ad(book_mat: NDArray[NDArray[np.uint32]], b: np.uint64, ind_arr: NDArray[np.uint64],
                  indind_arr: NDArray[np.uint32], osr: np.uint64) -> np.uint64:
    mid = search_arr_ad(ind_arr, b, indind_arr)
    if mid != 0xffffffff:
        return max(osr, book_mat[mid][0])
    else:
        return osr


@njit(nogil=True)
def dispatch_mnt_osr_ad_arr(unmasked_b: np.uint64, original_board_sum: np.uint32,
                            osr: NDArray[np.uint64], tiles_combinations_dict, param:ParamType, 
                            ) -> Tuple[bool, NDArray[np.uint8], int, NDArray[np.uint64], int, np.uint8]:
    # 1、合成后不存在两个相同大数 2、合成后存在两个相同大数 3、合出第三个64 4、成功 5、3x64 -> 64 128 (已在之前处理)
    total_sum, count_32k, pos_32k, pos_rank, merged_tile_found, tile_value, is_success = tile_sum_and_32k_count4(unmasked_b, param)

    if is_success:
        # 4
        arr_max(osr, np.full(len(osr), 4e9, dtype=np.uint64))
        return False, np.empty(0, dtype=np.uint8), pos_rank, pos_32k, tile_value, count_32k

    large_tiles_sum = original_board_sum - total_sum - ((param.num_free_32k + param.num_fixed_32k) << 15)
    tiles_combinations = get_array_view3(tiles_combinations_dict, np.uint8(large_tiles_sum >> 6), np.uint8(count_32k - param.num_free_32k))  # tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6), np.uint8(count_32k - param.num_free_32k)), None)
    if tiles_combinations is None:
        return False, np.empty(0, dtype=np.uint8), pos_rank, pos_32k, tile_value, count_32k
    return True, tiles_combinations, pos_rank, pos_32k, tile_value, count_32k


@njit(nogil=True)
def update_mnt_osr_ad_arr3(book_dict: BookDictType, unmasked_b: np.uint64,
                           count_32k: np.uint8, ind_dict: IndexDictType, indind_dict: IndexIndexDictType,
                           ranked_array: NDArray[np.uint32], osr: NDArray[np.uint64]):
    count_32k = np.int8(count_32k - 3 + 16)
    ind_arr = get_array_view10(ind_dict, count_32k, np.uint64)  # ind_dict.get(count_32k, np.empty(0, dtype=np.uint64))
    indind_arr = get_array_view10(indind_dict, count_32k, np.uint32)  # indind_dict.get(count_32k, np.empty(0, dtype=np.uint32))

    mid = search_arr_ad(ind_arr, unmasked_b, indind_arr)
    if mid != 0xffffffff:
        book_mat = get_array_view11(book_dict, count_32k, np.uint32)  # book_dict[count_32k]
        arr_max(osr, book_mat[mid][ranked_array])


@njit(nogil=True)
def update_mnt_osr_ad_arr2(book_dict: BookDictType, b: np.uint64,
                           pos_rank: int, count_32k: np.uint8, ind_dict: IndexDictType, indind_dict: IndexIndexDictType,
                           ranked_array: NDArray[np.uint32], osr: NDArray[np.uint64], permutation_dict, param:ParamType):
    ind_arr = get_array_view10(ind_dict, count_32k, np.uint64)  # ind_dict.get(count_32k, np.empty(0, dtype=np.uint64))
    indind_arr = get_array_view10(indind_dict, count_32k, np.uint32)  # indind_dict.get(count_32k, np.empty(0, dtype=np.uint32))

    mid = search_arr_ad(ind_arr, b, indind_arr)
    if mid != 0xffffffff:
        permutations = get_array_view2(permutation_dict, np.uint8(count_32k), np.uint8(count_32k - param.num_free_32k))  # permutation_dict[(np.uint8(count_32k), np.uint8(count_32k - param.num_free_32k))]
        book_mat = get_array_view11(book_dict, count_32k, np.uint32)  # book_dict[count_32k]
        permutations_match = (permutations[:, 0] == pos_rank)
        arr_max(osr, book_mat[mid][permutations_match][ranked_array])


@njit(nogil=True)
def update_mnt_osr_ad_arr1(book_dict: BookDictType, unmasked_b: np.uint64, ind_dict: IndexDictType,
                           indind_dict: IndexIndexDictType, board_derived: NDArray[np.uint64], osr: NDArray[np.uint64],
                           sym_func: SymFindFunc, tiles_combinations: NDArray[np.uint8], pos_rank: int,
                           pos_32k: NDArray[np.uint64], tile_value: int, count_32k: np.uint8):
    pos_32k = extract_f_positions(pos_32k)
    for j in range(count_32k):
        if j == pos_rank:
            continue

        pos = pos_32k[j]
        unmasked_b_j = unmasked_b & ~(np.uint64(0xf) << pos)
        unmasked_b_j = unmasked_b_j | (np.uint64(tiles_combinations[0]) << pos)
        unmasked_b_j, symm_index = sym_func(unmasked_b_j)

        ind_arr = get_array_view10(ind_dict, np.int8(-count_32k), np.uint64)  # ind_dict.get(np.int8(-count_32k), np.empty(0, dtype=np.uint64))
        indind_arr = get_array_view10(indind_dict, np.int8(-count_32k), np.uint32)  # indind_dict.get(np.int8(-count_32k), np.empty(0, dtype=np.uint32))

        mid = search_arr_ad(ind_arr, unmasked_b_j, indind_arr)
        if mid != 0xffffffff:
            positions_match = (((board_derived >> pos) & np.uint64(0xf)) == tile_value)
            osr_match = osr[positions_match]
            board_derived_match = board_derived[positions_match]
            sym_arr_like(board_derived_match, symm_index)
            book_mat = get_array_view11(book_dict, -count_32k, np.uint32)  # book_dict[-count_32k]
            ranked_array = match_arr(board_derived_match)
            arr_max(osr_match, (book_mat[mid])[ranked_array])
            osr[positions_match] = osr_match


@njit(nogil=True)
def update_mnt_osr_364_ad(book_dict: BookDictType, b: np.uint64, unmasked_b: np.uint64,
                          ind_dict: IndexDictType, indind_dict: IndexIndexDictType, osr: np.uint64, 
                          param:ParamType
                          ) -> np.uint64:
    # 3 64:0、128和64的pos_rank；1、要找的不是b，需要把64mask掉；2、book_mat[mid]中用permutation找到128,64对应位置
    pos_rank64, pos_rank128, pos_rank, pos_64 = find_3x64_pos(unmasked_b, param)
    b |= (np.uint64(0xf) << pos_64)
    count_32k = np.int8(pos_rank)
    ind_arr = get_array_view10(ind_dict, count_32k, np.uint64)  # ind_dict.get(count_32k, np.empty(0, dtype=np.uint64))
    indind_arr = get_array_view10(indind_dict, count_32k, np.uint32)  # indind_dict.get(count_32k, np.empty(0, dtype=np.uint32))

    mid = search_arr_ad(ind_arr, b, indind_arr)
    if mid != 0xffffffff:
        book_mat = get_array_view11(book_dict, count_32k, np.uint32)  # book_dict[count_32k]
        permutations_match_ind = _permutations_mapping_364(pos_rank64, pos_rank128, pos_rank)
        return max(osr, book_mat[mid][permutations_match_ind])
        # 等价于:
        # permutations = permutation_dict[(np.uint8(count_32k), np.uint8(count_32k - num_free_32k))]
        # permutations_match = (permutations[:, 0] == pos_rank64) & (permutations[:, 1] == pos_rank128)
        # # assert np.sum(permutations_match) == 1
        # return max(osr, book_mat[mid][permutations_match][0])
    return osr


@njit(nogil=True)
def _permutations_mapping_364(x, y, n):
    if x < y:
        return 2 * n * x - x * x - 2 * x + y - 1
    else:
        return 2 * n * y - y * y - 3 * y + n + x - 2


@njit(nogil=True)
def update_mnt_osr_364_arr_ad(book_dict: BookDictType, b: np.uint64, unmasked_b: np.uint64,
                              ind_dict: IndexDictType, indind_dict: IndexIndexDictType, osr: NDArray[np.uint64],
                              ranked_array: NDArray[np.uint32], permutation_dict, param:ParamType):
    # 3 64 -> 128 64:0、128和64的pos_rank；1、要找的不是b，需要把64mask掉；2、book_mat[mid]中用permutation找到128,64对应位置
    pos_rank64, pos_rank128, pos_rank, pos_64 = find_3x64_pos(unmasked_b, param)
    b |= (np.uint64(0xf) << pos_64)
    count_32k = np.int8(pos_rank)
    ind_arr = get_array_view10(ind_dict, count_32k, np.uint64)  # ind_dict.get(count_32k, np.empty(0, dtype=np.uint64))
    indind_arr = get_array_view10(indind_dict, count_32k, np.uint32)  # indind_dict.get(count_32k, np.empty(0, dtype=np.uint32))

    mid = search_arr_ad(ind_arr, b, indind_arr)
    if mid != 0xffffffff:
        book_mat = get_array_view11(book_dict, count_32k, np.uint32)  # book_dict[count_32k]
        permutations = get_array_view2(permutation_dict, np.uint8(count_32k), np.uint8(count_32k - param.num_free_32k))  # permutation_dict[(np.uint8(count_32k), np.uint8(count_32k - param.num_free_32k))]
        permutations_match = (permutations[:, 0] == pos_rank64) & (permutations[:, 1] == pos_rank128)
        arr_max(osr, book_mat[mid][permutations_match][ranked_array])


@njit(nogil=True)
def find_3x64_pos(unmasked_b: np.uint64, param:ParamType) -> Tuple[int, int, int, int]:
    pos_rank = 0
    for i in range(60, -4, -4):
        tile_value = (unmasked_b >> np.uint64(i)) & np.uint64(0xf)
        if tile_value == np.uint64(0xf):
            if (param.pos_fixed_32k >> i) & np.uint64(0xf) == 0:
                pos_rank += 1
        elif tile_value == 6:
            pos_rank64 = pos_rank
            pos_64 = i
            pos_rank += 1
        elif tile_value == 7:
            pos_rank128 = pos_rank
            pos_rank += 1
    # 如果在赋值前引用说明其他代码有Bug
    # noinspection PyUnboundLocalVariable
    return pos_rank64, pos_rank128, pos_rank, pos_64


@njit(nogil=True)
def reverse_arr(board: NDArray[np.uint64]) -> NDArray[np.uint64]:
    board = (board & np.uint64(0xFF00FF0000FF00FF)) | ((board & np.uint64(0x00FF00FF00000000)) >> np.uint64(24)) | (
            (board & np.uint64(0x00000000FF00FF00)) << np.uint64(24))
    board = (board & np.uint64(0xF0F00F0FF0F00F0F)) | ((board & np.uint64(0x0F0F00000F0F0000)) >> np.uint64(12)) | (
            (board & np.uint64(0x0000F0F00000F0F0)) << np.uint64(12))
    return board


@njit(nogil=True)
def move_arr(board: NDArray[np.uint64], direc: int, board_rev: NDArray[np.uint64]) -> NDArray[np.uint64]:
    moved = np.empty(len(board), dtype=np.uint64)
    for i in range(len(board)):
        moved[i] = move_board2(board[i], board_rev[i], direc)
    return moved


@njit(nogil=True)
def sym_like(bd2: np.uint64, symm_index: int) -> np.uint64:
    if symm_index == 0:
        bd2 = bd2
    elif symm_index == 1:
        bd2 = ReverseLR(bd2)
    elif symm_index == 2:
        bd2 = ReverseUD(bd2)
    elif symm_index == 3:
        bd2 = ReverseUL(bd2)
    elif symm_index == 4:
        bd2 = ReverseUR(bd2)
    elif symm_index == 5:
        bd2 = Rotate180(bd2)
    elif symm_index == 6:
        bd2 = RotateL(bd2)
    elif symm_index == 7:
        bd2 = RotateR(bd2)
    return bd2


@njit(nogil=True)
def sym_arr_like(bd_arr: NDArray[np.uint64], symm_index: int):
    if symm_index == 0:
        return
    elif symm_index == 1:
        for i in range(len(bd_arr)):
            bd_arr[i] = ReverseLR(bd_arr[i])
    elif symm_index == 2:
        for i in range(len(bd_arr)):
            bd_arr[i] = ReverseUD(bd_arr[i])
    elif symm_index == 3:
        for i in range(len(bd_arr)):
            bd_arr[i] = ReverseUL(bd_arr[i])
    elif symm_index == 4:
        for i in range(len(bd_arr)):
            bd_arr[i] = ReverseUR(bd_arr[i])
    elif symm_index == 5:
        for i in range(len(bd_arr)):
            bd_arr[i] = Rotate180(bd_arr[i])
    elif symm_index == 6:
        for i in range(len(bd_arr)):
            bd_arr[i] = RotateL(bd_arr[i])
    elif symm_index == 7:
        for i in range(len(bd_arr)):
            bd_arr[i] = RotateR(bd_arr[i])


@njit(nogil=True)
def find_mnt_pos(unmasked: np.uint64) -> int:
    pos_rank = 0
    for i in range(60, -4, -4):
        tile_value = (unmasked >> np.uint64(i)) & np.uint64(0xf)
        if tile_value == np.uint64(0xf):
            pos_rank += 1
        elif tile_value > 5:
            return pos_rank
    return pos_rank


@njit(nogil=True, parallel=True)
def remove_died_ad(book_dict: BookDictType, ind_dict: IndexDictType, deletion_threshold: np.uint32 = 0):
    for count32k in ind_dict.keys():
        ind = ind_dict[count32k]
        if len(ind) == 0:
            continue
        mat = book_dict[count32k]
        result = np.zeros(len(ind), dtype=np.bool_)
        for i in prange(len(ind)):
            arr = mat[i]
            for j in arr:
                if j > deletion_threshold:
                    result[i] = True
                    break
        book_dict[count32k] = filter_arr(mat, result)
        ind_dict[count32k] = ind[result]  # 不需要.copy()，直接就是副本


@njit(nogil=True, parallel=True)
def filter_arr(arr: NDArray[NDArray[np.uint32]], f: NDArray[np.bool_]):
    length = np.sum(f)
    lines = np.cumsum(f)
    result = np.empty((length, len(arr[0])), dtype=np.uint32)
    for i in prange(len(arr)):
        if f[i]:
            result[lines[i] - 1] = arr[i]
    return result


def create_index_ad(ind_dict: IndexDictType) -> IndexIndexDictType:
    indind_dict = Dict.empty(KeyType1, ValueType3)
    for k in ind_dict.keys():
        if len(ind_dict[k]) > 100000:
            indind_dict[k] = _create_index_ad(ind_dict[k])
        else:
            indind_dict[k] = np.empty(0, dtype=np.uint32)
    return indind_dict


@njit(parallel=True, nogil=True)
def _create_index_ad(arr: NDArray[np.uint64]) -> NDArray[np.uint32]:
    """
    根据uint64数据的前24位的分段位置创建一个索引，长度16777216+1
    """
    n = 16777217
    ind1: NDArray = np.empty(n, dtype='uint32')
    header = arr[0] >> np.uint32(40)
    ind1[:header + 1] = 0

    for i in prange(1, len(arr)):
        header = arr[i] >> np.uint32(40)
        header_pre = arr[i - 1] >> np.uint32(40)
        if header != header_pre:
            ind1[header_pre + 1: header + 1] = i
    header = arr[-1] >> np.uint32(40)
    ind1[header + 1:] = len(arr)
    return ind1


@njit(nogil=True)
def binary_search_arr_ad(arr: NDArray[np.uint64],
                         target: np.uint64, low: np.uint32, high: np.uint32) -> np.uint32:
    while low <= high:
        mid = np.uint32((low + high) // 2)
        mid_val = arr[mid]
        if mid_val < target:
            low = mid + 1
        elif mid_val > target:
            high = mid - 1
        else:
            return mid  # 找到匹配，返回索引

    return np.uint32(0xffffffff)  # 如果没有找到匹配项


@njit(nogil=True)
def search_arr_ad(arr: NDArray[np.uint64],
                  b: np.uint64, ind: NDArray[np.uint32]) -> np.uint64:
    if arr is None or len(arr) == 0:
        return np.uint32(0xffffffff)
    if ind is None or len(ind) == 0:
        return binary_search_arr_ad(arr, b, np.uint32(0), np.uint32(len(arr) - 1))
    header = b >> np.uint32(40)
    low, high = ind[header], ind[header + 1] - 1
    return binary_search_arr_ad(arr, b, low, high)


def length_count(book_dict: BookDictType) -> (int, float):
    length = 0
    max_rate = 0
    for mat in book_dict.values():
        s1, s2 = mat.shape
        length += s1 * s2
        if s1:
            max_rate = max(max_rate, np.max(mat))
    return length, max_rate / 4e9


def dict_tofile(book_dict: BookDictType, ind_dict: IndexDictType, path: str, step: int):
    # Create the target directory if it doesn't exist
    folder_path = path + str(step) + 'b'
    os.makedirs(folder_path, exist_ok=True)

    for k in ind_dict.keys():
        ind_filename = os.path.join(folder_path, f"{str(k)}.i")
        book_filename = os.path.join(folder_path, f"{str(k)}.b")
        if len(ind_dict[k]) == 0:
            if os.path.exists(ind_filename):
                os.remove(ind_filename)
            if os.path.exists(book_filename):
                os.remove(book_filename)
            continue
        # Write the ind_dict[k] data to a file named str(k).i
        ind_dict[k].tofile(ind_filename)
        # Write the book_dict[k] data to a file named str(k).b
        book_dict[k].tofile(book_filename)


def dict_fromfile(path: str, step: int) -> (BookDictType, IndexDictType):
    folder_path = path + str(step) + 'b'
    book_dict = Dict.empty(KeyType1, ValueType1)
    ind_dict = Dict.empty(KeyType1, ValueType2)

    # First, read all the .i files to populate ind_dict
    for filename in os.listdir(folder_path):
        if filename.endswith('.i'):
            key = np.int8(filename.split('.')[0])  # Extract key from filename (before .i)
            ind_data = np.fromfile(os.path.join(folder_path, filename), dtype=np.uint64)
            if len(ind_data) > 0:
                ind_dict[key] = ind_data

    # Now, read the .b files and use the corresponding ind_data to reshape book_data
    for filename in os.listdir(folder_path):
        if filename.endswith('.b'):
            key = np.int8(filename.split('.')[0])  # Extract key from filename (before .b)
            ind_length = len(
                ind_dict.get(key, np.empty(0, dtype=np.uint64)))  # Get the length of the corresponding ind_data
            if ind_length > 0:
                book_data = np.fromfile(os.path.join(folder_path, filename), dtype=np.uint32)
                book_data_reshaped = book_data.reshape((ind_length, -1))  # Reshape to (ind_length, columns)
                book_dict[key] = book_data_reshaped

    return book_dict, ind_dict


"""非查表实现"""


@njit(nogil=True)
def replace_val(encoded_board: np.uint64) -> Tuple[np.uint64, np.uint64]:
    replace_value = np.uint64(0xf)
    for k in range(0, 64, 4):
        encoded_num = (encoded_board >> np.uint64(k)) & np.uint64(0xf)
        if encoded_num == np.uint64(0xf):
            encoded_board &= ~(np.uint64(0xf) << np.uint64(k))  # 清除当前位置的值
            encoded_board |= (replace_value << np.uint64(k))  # 设置新的值
            replace_value -= np.uint64(1)
    return encoded_board, replace_value


@njit(nogil=True)
def ind_match(encoded_board: np.uint64, replacement_value: np.uint64) -> np.uint64:
    ind = 0
    x = np.uint64(0xf) - replacement_value
    for k in range(60, -4, -4):
        encoded_num = (encoded_board >> np.uint64(k)) & np.uint64(0xf)
        if encoded_num > replacement_value:
            ind *= x
            ind += (encoded_num - replacement_value)
    return ind


if __name__ == '__main__':
    pass
    # from BoardMaskerAD import init_masker
    # _d1,_d2,param = init_masker(2, 10, np.array([0, 4, 16, 20], dtype=np.uint8))  # t
    # derives = process_derived(np.uint64(0x1132656fffff6ff),131072+65536+256+64*3+32+16,0,0, _d2, _d1,param)
    # for i in derives:
    #     print(decode_board(i))
    #     print()
