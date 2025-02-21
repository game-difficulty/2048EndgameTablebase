import os
import time
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
import numba as nb
from numba import njit, prange
from numba import types
from numba.typed import Dict

import Config
from BoardMover import BoardMover
from BoardMoverAD import MaskedBoardMover
from BookGeneratorAD import BoardMasker
from Calculator import ReverseLR, ReverseUD, ReverseUL, ReverseUR, Rotate180, RotateL, RotateR
from Config import SingletonConfig
from LzmaCompressor import compress_with_7z, decompress_with_7z

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
ValueType4 = types.Tuple((types.uint64[:], types.uint16[:, :]))
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
        pattern_check_func: PatternCheckFunc,
        sym_func: SymFindFunc,
        steps: int,
        pathname: str,
        bm: BoardMover,
        mbm: MaskedBoardMover,
        lm: BoardMasker,
        spawn_rate4: float = 0.1
) -> None:
    started = False
    deletion_threshold = np.uint32(SingletonConfig().config.get('deletion_threshold', 0.0) * 4e9)
    ini_board_sum = np.sum(mbm.decode_board(arr_init[0]))
    indind_dict1, indind_dict2 = None, None
    match_dict = None

        # 从后向前更新ds中的array
    for i in range(steps - 3, -1, -1):
        started, book_dict1, book_dict2, ind_dict1, ind_dict2 \
            = handle_solve_restart_ad(i, pathname, steps, book_dict1, book_dict2, ind_dict1, ind_dict2, started)
        if not started:
            continue

        if match_dict is None:
            match_dict = Dict.empty(KeyType2, ValueType4)

        original_board_sum = 2 * i + ini_board_sum

        if SingletonConfig().config.get('compress_temp_files', False):
            decompress_with_7z(pathname + str(i) + '.7z')
        d0 = np.fromfile(pathname + str(i), dtype=np.uint64)

        t0 = time.time()

        book_dict0, ind_dict0 = expand_ad(d0, lm, original_board_sum)
        del d0

        # 创建、更新查找索引
        if indind_dict1 is not None:
            indind_dict2 = indind_dict1
        else:
            indind_dict2 = create_index_ad(ind_dict2)
        indind_dict1 = create_index_ad(ind_dict1)

        t1 = time.time()

        # 回算
        recalculate_ad(book_dict0, ind_dict0, book_dict1, ind_dict1, indind_dict1, book_dict2, ind_dict2, indind_dict2,
                       match_dict, bm, mbm, lm, original_board_sum, pattern_check_func, sym_func, spawn_rate4)
        length = length_count(book_dict0)
        t2 = time.time()

        if deletion_threshold > 0:
            remove_died_ad(book_dict2, ind_dict2, deletion_threshold)
        remove_died_ad(book_dict0, ind_dict0)
        t3 = time.time()
        if t3 > t0:
            logger.debug(f'step {i} recalculated: {round(length / (t3 - t0) / 1e6, 2)} mbps')
            logger.debug(f'index/solve/remove: {round((t1 - t0) / (t3 - t0), 2)}/'
                         f'{round((t2 - t1) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}')
        if deletion_threshold > 0:
            dict_tofile(book_dict2, ind_dict2, pathname, i + 2)  # 再写一次，把成功率低于阈值的局面去掉
        del book_dict2, ind_dict2
        dict_tofile(book_dict0, ind_dict0, pathname, i)
        # if os.path.exists(pathname + str(i)):
        #     os.remove(pathname + str(i))
        logger.debug(f'step {i} written\n')

        if SingletonConfig().config.get('compress_temp_files', False):
            compress_with_7z(pathname + str(i + 2) + 'b')
        # elif not SingletonConfig().config.get('optimal_branch_only', False):
        # do_compress(pathname + str(i + 2) + '.book')

        if i > 0:
            book_dict2, ind_dict2 = book_dict1, ind_dict1
            book_dict1, ind_dict1 = book_dict0, ind_dict0


@njit(nogil=True, parallel=True)
def expand_ad(arr: NDArray[np.uint64], lm: BoardMasker, original_board_sum: np.uint32
              ) -> Tuple[BookDictType, IndexDictType]:
    count32ks = np.empty(len(arr), dtype=np.int8)
    for i in prange(len(arr)):
        board = arr[i]
        total_sum, count_32k, pos_32k = lm.tile_sum_and_32k_count2(board)
        if count_32k == lm.num_free_32k:
            count32ks[i] = count_32k
            continue
        large_tiles_sum = original_board_sum - total_sum - ((lm.num_free_32k + lm.num_fixed_32k) << 15)
        tiles_combinations = lm.tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6),
                                                             np.uint8(count_32k - lm.num_free_32k)), None)
        if tiles_combinations is None:
            count32ks[i] = 16
            continue
        if count_32k - lm.num_free_32k == 1:  # len(tiles_combinations) == 1
            count32ks[i] = count_32k
            continue
        if tiles_combinations[0] == tiles_combinations[1]:
            count32ks[i] = -count_32k
            continue
        count32ks[i] = count_32k

    flags = np.zeros(32, dtype=np.bool_)
    for i in range(len(count32ks)):
        flags[count32ks[i] + 15] = True

    book_dict = Dict.empty(KeyType1, ValueType1)
    ind_dict = Dict.empty(KeyType1, ValueType2)

    factorials = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600],
                          dtype=np.uint32)

    for flag in range(31):
        if not flags[flag]:
            continue
        count_32k = np.int8(flag - 15)

        if count_32k < 0:
            derive_size = np.uint32(factorials[-count_32k - 2] // factorials[lm.num_free_32k])
        else:
            derive_size = np.uint32(factorials[count_32k] // factorials[lm.num_free_32k])

        positions = arr[count32ks == count_32k]

        ind_dict[count_32k] = positions
        book_dict[count_32k] = np.empty((len(positions), derive_size), dtype=np.uint32)

    return book_dict, ind_dict


@njit(parallel=True, nogil=True)
def recalculate_ad(
        book_dict0: BookDictType, ind_dict0: IndexDictType,
        book_dict1: BookDictType, ind_dict1: IndexDictType, indind_dict1: IndexIndexDictType,
        book_dict2: BookDictType, ind_dict2: IndexDictType, indind_dict2: IndexIndexDictType,
        match_dict: MatchDictType,
        bm: BoardMover,
        mbm: MaskedBoardMover,
        lm: BoardMasker,
        original_board_sum: np.uint32,
        pattern_check_func: PatternCheckFunc,
        sym_func: SymFindFunc,
        spawn_rate4: float = 0.1
):
    pos_rev = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.uint8)
    factorials = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800],
                          dtype=np.uint32)

    for count_32k, index in ind_dict0.items():
        if np.all(book_dict0[count_32k][: min(len(index), 10), 0] == 4e9):
            continue
        if count_32k < 0:
            derive_size = np.uint32(factorials[-count_32k - 2] // factorials[lm.num_free_32k])
        else:
            derive_size = np.uint32(factorials[count_32k] // factorials[lm.num_free_32k])

        match_index, match_mat = match_dict.setdefault(
            derive_size, (np.full(33331, np.uint64(0xffffffffffffffff), dtype=np.uint64),
                          np.zeros((33331, derive_size), dtype=np.uint16)))

        book_mat1 = book_dict1.setdefault(count_32k, np.empty((0, derive_size), dtype=np.uint32))
        ind_arr1 = ind_dict1.setdefault(count_32k, np.empty(0, dtype=np.uint64))
        indind_arr1 = indind_dict1.setdefault(count_32k, np.empty(0, dtype=np.uint32))
        book_mat2 = book_dict2.setdefault(count_32k, np.empty((0, derive_size), dtype=np.uint32))
        ind_arr2 = ind_dict2.setdefault(count_32k, np.empty(0, dtype=np.uint64))
        indind_arr2 = indind_dict2.setdefault(count_32k, np.empty(0, dtype=np.uint32))

        nb.set_parallel_chunksize(max(1024, len(index) // 1024))
        hit, miss = 0, 0

        for k in prange(len(index)):
            if derive_size == 1:
                t = index[k]
                # 初始化概率和权重
                success_probability = np.float64(0)
                empty_slots = 0

                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):  # 如果当前位置为空
                        empty_slots += 1

                        # 对于每个空位置，尝试填充2和4
                        new_value, probability = 1, 1 - spawn_rate4
                        t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                        board_rev = np.uint64(mbm.reverse(t_gen))
                        optimal_success_rate = np.uint32(0)

                        for direc, (newt, mnt) in enumerate(mbm.move_all_dir2(t_gen, board_rev)):
                            newt = np.uint64(newt)
                            if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
                                newt, symm_index = sym_func(newt)
                                newt = np.uint64(newt)
                                if mnt:
                                    unmasked_newt = np.uint64(bm.move_board2(t_gen, board_rev, direc + 1))
                                    unmasked_newt = np.uint64(sym_like(unmasked_newt, symm_index))
                                    optimal_success_rate = update_mnt_osr_ad(book_dict1, newt, unmasked_newt, lm,
                                                                             ind_dict1, indind_dict1,
                                                                             optimal_success_rate)
                                else:
                                    optimal_success_rate = update_osr_ad(book_mat1, newt, ind_arr1, indind_arr1,
                                                                         optimal_success_rate)

                        # 对最佳移动下的成功概率加权平均
                        success_probability += optimal_success_rate * probability

                        # 填4
                        new_value, probability = 2, spawn_rate4
                        t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                        board_rev = np.uint64(mbm.reverse(t_gen))
                        optimal_success_rate = np.uint32(0)

                        for direc, (newt, mnt) in enumerate(mbm.move_all_dir2(t_gen, board_rev)):
                            newt = np.uint64(newt)
                            if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
                                newt, symm_index = sym_func(newt)
                                newt = np.uint64(newt)
                                if mnt:
                                    unmasked_newt = np.uint64(bm.move_board2(t_gen, board_rev, direc + 1))
                                    unmasked_newt = np.uint64(sym_like(unmasked_newt, symm_index))
                                    optimal_success_rate = update_mnt_osr_ad(book_dict2, newt, unmasked_newt, lm,
                                                                             ind_dict2, indind_dict2,
                                                                             optimal_success_rate)
                                else:
                                    optimal_success_rate = update_osr_ad(book_mat2, newt, ind_arr2, indind_arr2,
                                                                         optimal_success_rate)

                        success_probability += optimal_success_rate * probability
                book_dict0[count_32k][k][0] = success_probability / empty_slots

            else:
                t = index[k]
                rep_t, rep_v = replace_val(t)
                rep_t_rev = np.uint64(bm.reverse(rep_t))
                # 初始化概率和权重
                success_probability = np.zeros(derive_size, dtype=np.float64)
                empty_slots = 0

                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):  # 如果当前位置为空
                        empty_slots += 1

                        # 对于每个空位置，尝试填充2和4
                        new_value, probability = 1, 1 - spawn_rate4
                        board_sum = original_board_sum + np.uint32(2)
                        t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                        rep_t_gen = rep_t | (np.uint64(new_value) << np.uint64(4 * i))
                        rep_t_gen_rev = rep_t_rev | (np.uint64(new_value) << np.uint64(4 * pos_rev[i]))
                        board_rev = np.uint64(mbm.reverse(t_gen))
                        optimal_success_rate = np.zeros(derive_size, dtype=np.uint64)

                        for direc, (newt, mnt) in enumerate(mbm.move_all_dir2(t_gen, board_rev)):
                            newt = np.uint64(newt)
                            if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
                                newt, symm_index = sym_func(newt)
                                newt = np.uint64(newt)
                                rep_t_gen_m = np.uint64(bm.move_board2(rep_t_gen, rep_t_gen_rev, direc + 1))
                                rep_t_gen_m = np.uint64(sym_like(rep_t_gen_m, symm_index))
                                match_ind = ind_match(rep_t_gen_m, rep_v)
                                hashed_match_ind = match_ind % np.uint64(33331)

                                if mnt:
                                    unmasked_newt = np.uint64(bm.move_board2(t_gen, board_rev, direc + 1))
                                    unmasked_newt = np.uint64(sym_like(unmasked_newt, symm_index))
                                    need_process, tiles_combinations, pos_rank, pos_32k, tile_value, count32k = (
                                        dispatch_mnt_osr_ad_arr(newt, unmasked_newt, lm, board_sum, optimal_success_rate))
                                    if not need_process:
                                        continue
                                    if len(tiles_combinations) > 1 and tiles_combinations[0] == tiles_combinations[1]:
                                        new_derived = process_derived(t_gen, board_sum, lm, direc, bm, symm_index)
                                        update_mnt_osr_ad_arr1(book_dict1, unmasked_newt, ind_dict1, indind_dict1,
                                                               new_derived, optimal_success_rate, sym_func,
                                                               tiles_combinations, pos_rank, pos_32k, tile_value,
                                                               count32k)
                                    elif match_index[hashed_match_ind] == match_ind:
                                        hit += 1
                                        ranked_array = match_mat[hashed_match_ind]
                                        update_mnt_osr_ad_arr2(book_dict1, newt, lm, pos_rank, count32k, ind_dict1,
                                                               indind_dict1, ranked_array, optimal_success_rate)
                                    else:
                                        miss += 1
                                        new_derived = process_derived(t_gen, board_sum, lm, direc, bm, symm_index)
                                        ranked_array = match_arr(new_derived)

                                        if match_index[hashed_match_ind] == np.uint64(0xffffffffffffffff):
                                            match_index[hashed_match_ind] = match_ind
                                            match_mat[hashed_match_ind] = ranked_array
                                        update_mnt_osr_ad_arr2(book_dict1, newt, lm, pos_rank, count32k, ind_dict1,
                                                               indind_dict1, ranked_array, optimal_success_rate)
                                elif match_index[hashed_match_ind] == match_ind:
                                    hit += 1
                                    ranked_array = match_mat[hashed_match_ind]
                                    update_osr_ad_arr(book_mat1, newt, ind_arr1, indind_arr1,
                                                       optimal_success_rate, ranked_array)
                                else:
                                    miss += 1
                                    new_derived = process_derived(t_gen, board_sum, lm, direc, bm, symm_index)
                                    ranked_array = match_arr(new_derived)
                                    if match_index[hashed_match_ind] == np.uint64(0xffffffffffffffff):
                                        match_index[hashed_match_ind] = match_ind
                                        match_mat[hashed_match_ind] = ranked_array
                                    update_osr_ad_arr(book_mat1, newt, ind_arr1, indind_arr1,
                                                      optimal_success_rate, ranked_array)

                        # 对最佳移动下的成功概率加权平均
                        success_probability += optimal_success_rate * probability

                        # 填4
                        new_value, probability = 2, spawn_rate4
                        board_sum = original_board_sum + np.uint32(4)
                        t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                        rep_t_gen = rep_t | (np.uint64(new_value) << np.uint64(4 * i))
                        rep_t_gen_rev = rep_t_rev | (np.uint64(new_value) << np.uint64(4 * pos_rev[i]))
                        board_rev = np.uint64(mbm.reverse(t_gen))
                        optimal_success_rate = np.zeros(derive_size, dtype=np.uint64)

                        for direc, (newt, mnt) in enumerate(mbm.move_all_dir2(t_gen, board_rev)):
                            newt = np.uint64(newt)
                            if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
                                newt, symm_index = sym_func(newt)
                                newt = np.uint64(newt)
                                rep_t_gen_m = np.uint64(bm.move_board2(rep_t_gen, rep_t_gen_rev, direc + 1))
                                rep_t_gen_m = np.uint64(sym_like(rep_t_gen_m, symm_index))
                                match_ind = ind_match(rep_t_gen_m, rep_v)
                                hashed_match_ind = match_ind % np.uint64(33331)

                                if mnt:
                                    unmasked_newt = np.uint64(bm.move_board2(t_gen, board_rev, direc + 1))
                                    unmasked_newt = np.uint64(sym_like(unmasked_newt, symm_index))
                                    need_process, tiles_combinations, pos_rank, pos_32k, tile_value, count32k = (
                                        dispatch_mnt_osr_ad_arr(newt, unmasked_newt, lm, board_sum, optimal_success_rate))
                                    if not need_process:
                                        continue
                                    if len(tiles_combinations) > 1 and tiles_combinations[0] == tiles_combinations[1]:
                                        new_derived = process_derived(t_gen, board_sum, lm, direc, bm, symm_index)
                                        update_mnt_osr_ad_arr1(book_dict2, unmasked_newt, ind_dict2, indind_dict2,
                                                               new_derived, optimal_success_rate, sym_func,
                                                               tiles_combinations, pos_rank, pos_32k, tile_value,
                                                               count32k)
                                    elif match_index[hashed_match_ind] == match_ind:
                                        hit += 1
                                        ranked_array = match_mat[hashed_match_ind]
                                        update_mnt_osr_ad_arr2(book_dict2, newt, lm, pos_rank, count32k, ind_dict2,
                                                               indind_dict2, ranked_array, optimal_success_rate)
                                    else:
                                        miss += 1
                                        new_derived = process_derived(t_gen, board_sum, lm, direc, bm, symm_index)
                                        ranked_array = match_arr(new_derived)

                                        if match_index[hashed_match_ind] == np.uint64(0xffffffffffffffff):
                                            match_index[hashed_match_ind] = match_ind
                                            match_mat[hashed_match_ind] = ranked_array
                                        update_mnt_osr_ad_arr2(book_dict2, newt, lm, pos_rank, count32k, ind_dict2,
                                                               indind_dict2, ranked_array, optimal_success_rate)
                                elif match_index[hashed_match_ind] == match_ind:
                                    hit += 1
                                    ranked_array = match_mat[hashed_match_ind]
                                    update_osr_ad_arr(book_mat2, newt, ind_arr2, indind_arr2,
                                                       optimal_success_rate, ranked_array)
                                else:
                                    miss += 1
                                    new_derived = process_derived(t_gen, board_sum, lm, direc, bm, symm_index)
                                    ranked_array = match_arr(new_derived)
                                    if match_index[hashed_match_ind] == np.uint64(0xffffffffffffffff):
                                        match_index[hashed_match_ind] = match_ind
                                        match_mat[hashed_match_ind] = ranked_array
                                    update_osr_ad_arr(book_mat2, newt, ind_arr2, indind_arr2,
                                                      optimal_success_rate, ranked_array)

                        success_probability += optimal_success_rate * probability
                book_dict0[count_32k][k] = (success_probability / empty_slots).astype(np.uint32)
        print(f'hit:{int(hit // 1e3)}k,miss:{int(miss // 1e3)}k, {count_32k}')
        print(f'load {np.sum(match_index != np.uint64(0xffffffffffffffff))}')


@njit(nogil=True)
def process_derived(t_gen: np.uint64, board_sum: np.uint32, lm: BoardMasker, direc: int, bm: BoardMover,
                    symm_index: int):
    derived = lm.unmask_board(t_gen, board_sum)
    derived_rev = reverse_arr(derived)
    new_derived = move_arr(derived, direc + 1, derived_rev, bm)
    sym_arr_like(new_derived, symm_index)
    return new_derived


@njit(nogil=True)  
def arr_max(arr1: NDArray, arr2: NDArray):
    # assert len(arr1)-len(arr2) == 0
    for i in range(len(arr1)):
        arr1[i] = max(arr1[i], arr2[i])


@njit(nogil=True)  
def match_arr(a: NDArray) -> NDArray:
    # assert len(a)-len(b) == 0
    sorted_indices = np.argsort(a)
    ranked_array = np.empty(len(a), dtype=np.uint32)
    ranked_array[sorted_indices] = np.arange(len(a))
    return ranked_array


@njit(nogil=True, boundscheck=True)
def update_osr_ad_arr(book_mat: NDArray[NDArray[np.uint32]], b: np.uint64, ind_ar: NDArray[np.uint64],
                       indind_ar: NDArray[np.uint32], osr: NDArray[np.uint64], ranked_array: NDArray[np.uint16]):
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


@njit(nogil=True, boundscheck=True)
def dispatch_mnt_osr_ad_arr(b: np.uint64, unmasked_b: np.uint64, lm: BoardMasker, original_board_sum: np.uint32,
                            osr: NDArray[np.uint64]
                            ) -> Tuple[bool, NDArray[np.uint8], int, NDArray[np.uint64], int, np.uint8]:
    total_sum, count_32k, pos_32k = lm.tile_sum_and_32k_count(b)

    pos_rank = 0
    tile_value = 0
    for i in range(60, -4, -4):
        tile_value = (unmasked_b >> i) & 0xf
        if tile_value == 0xf:
            if i not in lm.pos_fixed_32k:
                pos_rank += 1
        elif tile_value == lm.target:
            arr_max(osr, np.full(len(osr), 4e9, dtype=np.uint64))
            return False, np.empty(0, dtype=np.uint8), pos_rank, pos_32k, tile_value, count_32k
        elif tile_value > 5:
            break

    large_tiles_sum = original_board_sum - total_sum - ((lm.num_free_32k + lm.num_fixed_32k) << 15)
    tiles_combinations = lm.tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6),
                                                         np.uint8(count_32k - lm.num_free_32k)), None)
    if tiles_combinations is None:
        return False, np.empty(0, dtype=np.uint8), pos_rank, pos_32k, tile_value, count_32k
    return True, tiles_combinations, pos_rank, pos_32k, tile_value, count_32k


@njit(nogil=True, boundscheck=True)
def update_mnt_osr_ad_arr2(book_dict: BookDictType, b: np.uint64, lm: BoardMasker,
                           pos_rank: int, count_32k: np.uint8, ind_dict: IndexDictType, indind_dict: IndexIndexDictType,
                           ranked_array: NDArray[np.uint16], osr: NDArray[np.uint64]):
    ind_arr = ind_dict.setdefault(count_32k, np.empty(0, dtype=np.uint64))
    indind_arr = indind_dict.setdefault(count_32k, np.empty(0, dtype=np.uint32))

    mid = search_arr_ad(ind_arr, b, indind_arr)
    if mid != 0xffffffff:
        permutations = lm.permutation_dict1[(np.uint8(count_32k), np.uint8(count_32k - lm.num_free_32k))]
        book_mat = book_dict.setdefault(count_32k, np.empty((0, len(permutations)), dtype=np.uint32))
        permutations_match = (permutations[:, 0] == pos_rank)
        arr_max(osr, book_mat[mid][permutations_match][ranked_array])



@njit(nogil=True)  
def update_mnt_osr_ad_arr1(book_dict: BookDictType, unmasked_b: np.uint64, ind_dict: IndexDictType,
                           indind_dict: IndexIndexDictType, board_derived: NDArray[np.uint64], osr: NDArray[np.uint64],
                           sym_func: SymFindFunc, tiles_combinations: NDArray[np.uint8], pos_rank: int,
                           pos_32k: NDArray[np.uint64], tile_value: int, count_32k: np.uint8):
    for j in range(count_32k):
        if j == pos_rank:
            continue

        unmasked_b_j = unmasked_b & ~(np.uint64(0xf) << pos_32k[j])
        unmasked_b_j = unmasked_b_j | (np.uint64(tiles_combinations[0]) << pos_32k[j])
        unmasked_b_j, symm_index = sym_func(unmasked_b_j)

        ind_arr = ind_dict.setdefault(-count_32k, np.empty(0, dtype=np.uint64))
        indind_arr = indind_dict.setdefault(-count_32k, np.empty(0, dtype=np.uint32))

        mid = search_arr_ad(ind_arr, unmasked_b_j, indind_arr)
        if mid != 0xffffffff:
            positions_match = (((board_derived >> pos_32k[j]) & np.uint64(0xf)) == tile_value)
            osr_match = osr[positions_match]
            board_derived_match = board_derived[positions_match]
            sym_arr_like(board_derived_match, symm_index)
            book_mat = book_dict.setdefault(-count_32k, np.empty((0, len(board_derived_match)), dtype=np.uint32))
            ranked_array = match_arr(board_derived_match)
            arr_max(osr_match, (book_mat[mid])[ranked_array])
            osr[positions_match] = osr_match


@njit(nogil=True)  
def update_mnt_osr_ad(book_dict: BookDictType, b: np.uint64, unmasked_b: np.uint64, lm: BoardMasker,
                      ind_dict: IndexDictType, indind_dict: IndexIndexDictType, osr: np.uint64) -> np.uint64:
    total_sum, count_32k, pos_32k = lm.tile_sum_and_32k_count(b)

    pos_rank = 0
    for i in range(60, -4, -4):
        tile_value = (unmasked_b >> i) & 0xf
        if tile_value == 0xf:
            if i not in lm.pos_fixed_32k:
                pos_rank += 1
        elif tile_value == lm.target:
            return np.uint64(4e9)
        elif tile_value > 5:
            break

    ind_arr = ind_dict.setdefault(count_32k, np.empty(0, dtype=np.uint64))
    indind_arr = indind_dict.setdefault(count_32k, np.empty(0, dtype=np.uint32))

    mid = search_arr_ad(ind_arr, b, indind_arr)
    if mid != 0xffffffff:
        book_mat = book_dict.setdefault(count_32k, np.empty((0, 1), dtype=np.uint32))
        return max(osr, book_mat[mid][pos_rank])
    return osr


@njit(nogil=True)
def reverse_arr(board: NDArray[np.uint64]) -> NDArray[np.uint64]:
    board = (board & np.uint64(0xFF00FF0000FF00FF)) | ((board & np.uint64(0x00FF00FF00000000)) >> np.uint64(24)) | (
            (board & np.uint64(0x00000000FF00FF00)) << np.uint64(24))
    board = (board & np.uint64(0xF0F00F0FF0F00F0F)) | ((board & np.uint64(0x0F0F00000F0F0000)) >> np.uint64(12)) | (
            (board & np.uint64(0x0000F0F00000F0F0)) << np.uint64(12))
    return board


@njit(nogil=True)
def move_arr(board: NDArray[np.uint64], direc: int, board_rev: NDArray[np.uint64],
             bm: BoardMover) -> NDArray[np.uint64]:
    moved = np.empty(len(board), dtype=np.uint64)
    for i in range(len(board)):
        moved[i] = bm.move_board2(board[i], board_rev[i], direc)
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
        tile_value = (unmasked >> i) & 0xf
        if tile_value == 0xf:
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
        ind_dict[count32k] = ind[result]  #不需要.copy()，直接就是副本


@njit(nogil=True)
def filter_arr(arr: NDArray[NDArray[np.uint32]], f: NDArray[np.bool_]):
    length = np.sum(f)
    result = np.empty((length, len(arr[0])), dtype=np.uint32)
    line = 0
    for i in range(len(arr)):
        if f[i]:
            result[line] = arr[i]
            line += 1
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
    ind1 = np.full(n, 0xffffffff, dtype='uint32')
    header = arr[0] >> np.uint32(40)
    ind1[header] = 0

    for i in prange(1, len(arr)):
        header = arr[i] >> np.uint32(40)
        header_pre = arr[i - 1] >> np.uint32(40)
        if header != header_pre:
            ind1[header] = i
    # 向前填充
    num_threads = 8
    segment_size = (n + num_threads - 1) // num_threads

    # 记录每个段最后一个非零值
    last_values = np.empty(num_threads, dtype='uint32')

    # 每段并行处理
    for i in prange(num_threads):
        start = i * segment_size
        end = min(start + segment_size, n)

        # 初始化 last_value 为段末尾的第一个非零值
        last_value = len(arr)
        for j in range(end - 1, start - 1, -1):
            if ind1[j] != 0xffffffff:
                last_value = ind1[j]
            else:
                ind1[j] = last_value

        last_values[i] = last_value

    # 处理边界，确保每段的填充是正确的
    for i in prange(1, num_threads):
        if last_values[i] != len(arr):
            start = i * segment_size
            for j in range(start - 1, start - segment_size, -1):
                if ind1[j] == len(arr):
                    ind1[j] = last_values[i]
                else:
                    break

    ind1[0] = 0
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



@njit(nogil=True, boundscheck=True)
def replace_val(encoded_board: np.uint64) -> Tuple[np.uint64, np.uint64]:
    replace_value = 0xf
    for k in range(0, 64, 4):
        encoded_num = (encoded_board >> k) & 0xF
        if encoded_num == 0xF:
            encoded_board &= ~np.uint64(0xF << k)  # 清除当前位置的值
            encoded_board |= np.uint64(replace_value << k)  # 设置新的值
            replace_value -= 1
    return encoded_board, replace_value


@njit(nogil=True, boundscheck=True)
def ind_match(encoded_board: np.uint64, replacement_value: np.uint64) -> np.uint64:
    ind = 0
    x = 0xf - replacement_value
    for k in range(60, -4, -4):
        encoded_num = (np.uint64(encoded_board) >> k) & 0xF
        if encoded_num > replacement_value:
            ind *= x
            ind += (encoded_num - replacement_value)
    return ind


def length_count(book_dict: BookDictType) -> int:
    length = 0
    for mat in book_dict.values():
        s1, s2 = mat.shape
        length += s1 * s2
    return length


def dict_tofile(book_dict: BookDictType, ind_dict: IndexDictType, path: str, step: int):
    # Create the target directory if it doesn't exist
    folder_path = path + str(step) + 'b'
    os.makedirs(folder_path, exist_ok=True)

    for k in ind_dict.keys():
        if len(ind_dict[k]) == 0:
            continue
        # Write the ind_dict[k] data to a file named str(k).i
        ind_filename = os.path.join(folder_path, f"{str(k)}.i")
        ind_dict[k].tofile(ind_filename)

        # Write the book_dict[k] data to a file named str(k).b
        book_filename = os.path.join(folder_path, f"{str(k)}.b")
        book_dict[k].tofile(book_filename)


def dict_fromfile(path: str, step: int) -> (BookDictType, IndexDictType):
    folder_path = path + str(step) + 'b'
    book_dict = Dict.empty(KeyType1, ValueType1)
    ind_dict = Dict.empty(KeyType1, ValueType2)

    # First, read all the .i files to populate ind_dict
    for filename in os.listdir(folder_path):
        if filename.endswith('.i'):
            key = int(filename.split('.')[0])  # Extract key from filename (before .i)
            ind_data = np.fromfile(os.path.join(folder_path, filename), dtype=np.uint64)
            if len(ind_data) > 0:
                ind_dict[key] = ind_data

    # Now, read the .b files and use the corresponding ind_data to reshape book_data
    for filename in os.listdir(folder_path):
        if filename.endswith('.b'):
            key = int(filename.split('.')[0])  # Extract key from filename (before .b)
            ind_length = len(
                ind_dict.get(key, np.empty(0, dtype=np.uint64)))  # Get the length of the corresponding ind_data
            if ind_length > 0:
                book_data = np.fromfile(os.path.join(folder_path, filename), dtype=np.uint32)
                book_data_reshaped = book_data.reshape((ind_length, -1))  # Reshape to (ind_length, columns)
                book_dict[key] = book_data_reshaped

    return book_dict, ind_dict




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


if __name__ == '__main__':
    pass
    # from BookGeneratorAD import init_masker
    #
    # lm_ = init_masker(1, 12, np.array([0, 4, 20], dtype=np.uint8))  # 4432f_2048
    # arrrr = np.fromfile(r"D:\2048calculates\table\testtable\4432f_2048_526", dtype='uint64')
    # obs = 526 * 2 + 131072 + 20
    # b_, i_ = expand_ad(arrrr, lm_, obs)
    # pass
