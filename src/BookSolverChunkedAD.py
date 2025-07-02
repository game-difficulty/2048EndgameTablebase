import os
import time
import shutil
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from numba import types
from numba.typed import Dict
import psutil

import Config
from BoardMover import BoardMover
from BoardMoverAD import MaskedBoardMover
from BoardMaskerAD import BoardMasker
from Config import SingletonConfig
from LzmaCompressor import compress_with_7z, decompress_with_7z
from BookSolverAD import remove_died_ad, create_index_ad, \
    solve_optimal_success_rate, solve_optimal_success_rate_arr, replace_val, dict_fromfile, expand_ad, dict_tofile

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

# 用于统计回算速度
_timer, _counter = 0, 0


def handle_solve_restart_ad_c(i, pathname, steps, b2, i2, started):
    if os.path.exists(pathname + str(i) + 'b'):
        logger.debug(f"skipping step {i}")
        return False, None, None
    elif not started:
        started = True
        if i != steps - 3 or b2 is None:
            b2, i2 = dict_fromfile(pathname, i + 2)
            remove_died_ad(b2, i2)
        return started, b2, i2
    return True, b2, i2


def recalculate_process_ad_c(
        arr_init: NDArray[np.uint64],
        _0: BookDictType,
        book_dict2: BookDictType,
        _1: IndexDictType,
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
    global _timer, _counter

    # 从后向前更新ds中的array
    for i in range(steps - 3, -1, -1):
        started, book_dict2, ind_dict2 \
            = handle_solve_restart_ad_c(i, pathname, steps, book_dict2, ind_dict2, started)
        if not started:
            continue

        if match_dict is None:
            match_dict = Dict.empty(KeyType2, ValueType4)

        original_board_sum = 2 * i + ini_board_sum

        if SingletonConfig().config.get('compress_temp_files', False):
            decompress_with_7z(pathname + str(i) + '.7z')
        d0 = np.fromfile(pathname + str(i), dtype=np.uint64)

        book_dict0, ind_dict0 = expand_ad(d0, lm, original_board_sum, False)
        ind_dict0_keys = write_ind(ind_dict0, pathname, i)
        del d0, book_dict0, ind_dict0

        # 创建、更新查找索引
        if indind_dict1 is not None:
            indind_dict2 = indind_dict1
        else:
            indind_dict2 = create_index_ad(ind_dict2)

        # 出4
        iter_ind_dict4(ind_dict0_keys, lm, book_dict2, ind_dict2, indind_dict2, match_dict, bm, mbm, original_board_sum,
                       pattern_check_func, sym_func, spawn_rate4, pathname, i)
        if deletion_threshold > 0:
            remove_died_ad(book_dict2, ind_dict2, deletion_threshold)
        dict_tofile(book_dict2, ind_dict2, pathname, i + 2)
        del book_dict2, ind_dict2, indind_dict2

        # 出2
        book_dict1, ind_dict1 = dict_fromfile(pathname, i + 1)
        remove_died_ad(book_dict1, ind_dict1)
        indind_dict1 = create_index_ad(ind_dict1)
        iter_ind_dict2(pathname, i, ind_dict0_keys, book_dict1, ind_dict1, indind_dict1, match_dict, bm, mbm, lm,
                       original_board_sum, pattern_check_func, sym_func, spawn_rate4)

        os.rename(pathname + str(i) + 'bt', pathname + str(i) + 'b')

        # if os.path.exists(pathname + str(i)):
        #     os.remove(pathname + str(i))
        logger.debug(f'step {i} done, solving avg {round(_counter / max(_timer, 0.0001) / 2e6, 2)} mbps\n')  # todo

        if SingletonConfig().config.get('compress_temp_files', False):
            compress_with_7z(pathname + str(i + 2) + 'b')

        book_dict2, ind_dict2 = book_dict1, ind_dict1
        del book_dict1, ind_dict1
        _timer, _counter = 0.0, 0


def iter_ind_dict4(ind_dict0_keys: list, lm: BoardMasker, book_dict2, ind_dict2, indind_dict2,
                   match_dict, bm, mbm, original_board_sum, pattern_check_func, sym_func, spawn_rate4,
                   path, step):
    """
    将输入字典拆分计算出4成功率
    chunk_length: 每个切片的最大长度
    """
    folder = path + str(step) + 'bt'
    factorials = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 1, 1, 1, 1, 1, 1, 1],
                          dtype=np.uint32)
    for count_32k in ind_dict0_keys:
        positions_array = np.fromfile(os.path.join(folder, f"{count_32k}.i"), dtype='uint64')
        if count_32k < 0:
            derive_size = int(factorials[-count_32k - 2] // factorials[lm.num_free_32k])
        elif count_32k > 15:
            derive_size = int(factorials[count_32k - 16] // factorials[lm.num_free_32k])
        else:
            derive_size = int(factorials[count_32k] // factorials[lm.num_free_32k])

        # 剩余可用物理内存（单位：字节）
        mem = psutil.virtual_memory()
        free_mem = mem.available
        chunk_size = max(2 ** 28, int(free_mem * 0.9 / 4))
        chunk_length = np.uint64(chunk_size) // derive_size
        global _timer, _counter

        for start_idx in range(0, len(positions_array), chunk_length):
            t0 = time.time()
            end_idx = min(start_idx + chunk_length, len(positions_array))
            positions_chunk = positions_array[start_idx: end_idx]
            # 这里需要赋0，后面计算结果直接相加
            book_chunk = np.zeros((len(positions_chunk), derive_size), dtype=np.uint32)

            recalculate_ad_c(count_32k, positions_chunk, book_chunk, book_dict2, ind_dict2, indind_dict2,
                         match_dict, bm, mbm, lm, original_board_sum, pattern_check_func, sym_func, False,
                         spawn_rate4)
            t1 = time.time()
            if t1 > t0:
                logger.debug(f'step {step}, gen4, count_32k {count_32k}, chunk {start_idx // chunk_length}, '
                             f'done {round(len(positions_chunk) * derive_size / (t1 - t0) / 1e6, 2)} mbps')
            _timer += t1 - t0
            _counter += len(positions_chunk) * derive_size
            # 将book_chunk写入文件末尾
            file_path = os.path.join(folder, f"{count_32k}.b")
            with open(file_path, 'ab') as f:  # 'ab'模式表示二进制追加
                # 这里不去除0成功率局面，因为出2的情况还没考虑
                book_chunk.tofile(f)
            del book_chunk, positions_chunk


def iter_ind_dict2(path: str, step: int, ind_dict0_keys: list, book_dict1, ind_dict1, indind_dict1,
                   match_dict, bm, mbm, lm, original_board_sum, pattern_check_func, sym_func,
                   spawn_rate4):
    """
    以迭代器方式分块读取成功率文件用于分块回算
    """
    folder = path + str(step) + 'bt'

    for count_32k in ind_dict0_keys:
        positions_array = np.fromfile(os.path.join(folder, f"{count_32k}.i"), dtype='uint64')
        book_path = os.path.join(folder, f"{count_32k}.b")
        file_size = os.path.getsize(book_path)
        total_elements = file_size // 4
        derive_size = total_elements // len(positions_array)
        # 剩余可用物理内存（单位：字节）
        mem = psutil.virtual_memory()
        free_mem = mem.available
        chunk_size = max(2 ** 28, int(free_mem * 0.9 / 4))
        chunk_length = chunk_size // derive_size
        global _timer, _counter
        # 分块读取逻辑
        with open(book_path, 'r+b') as f:
            for start_idx in range(0, len(positions_array), chunk_length):
                row_end = min(start_idx + chunk_length, len(positions_array))
                positions_chunk = positions_array[start_idx:row_end]

                # 定位并读取数据块
                f.seek(start_idx * derive_size * 4)
                book_chunk = np.fromfile(
                    f,
                    dtype=np.uint32,
                    count=len(positions_chunk) * derive_size,
                ).reshape(len(positions_chunk), derive_size)

                t0 = time.time()
                # 提取对应索引块

                recalculate_ad_c(count_32k, positions_chunk, book_chunk, book_dict1, ind_dict1, indind_dict1,
                             match_dict, bm, mbm, lm, original_board_sum, pattern_check_func, sym_func, True,
                             spawn_rate4)
                t1 = time.time()
                if t1 > t0:
                    logger.debug(f'step {step}, gen2, count_32k {count_32k}, chunk {start_idx // chunk_length}, '
                                 f'done {round(len(positions_chunk) * derive_size / (t1 - t0) / 1e6, 2)} mbps')
                _timer += t1 - t0
                _counter += len(positions_chunk) * derive_size
                # 将计算结果写回原位置
                # 这里不去除0成功率局面，保证写回大小相同
                f.seek(start_idx * derive_size * 4)
                book_chunk.tofile(f)
                f.flush()
                del book_chunk, positions_chunk


def write_ind(ind_dict: IndexDictType, path: str, step: int) -> list:
    ind_dict0_keys = []
    folder = path + str(step) + 'bt'
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    for k,v in ind_dict.items():
        if len(v) > 0:
            v.tofile(os.path.join(folder, f"{k}.i"))
            ind_dict0_keys.append(k)
    return ind_dict0_keys


@njit(parallel=True, nogil=True)
def recalculate_ad_c(
        count_32k: np.uint8,
        positions_chunk: ValueType2, book_chunk: ValueType1,
        book_dict: BookDictType, ind_dict: IndexDictType, indind_dict: IndexIndexDictType,
        match_dict: MatchDictType,
        bm: BoardMover,
        mbm: MaskedBoardMover,
        lm: BoardMasker,
        original_board_sum: np.uint32,
        pattern_check_func: PatternCheckFunc,
        sym_func: SymFindFunc,
        is_gen2_step: bool,
        spawn_rate4: float = 0.1
):
    pos_rev = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.uint8)
    derive_size = len(book_chunk[0])

    match_index, match_mat = match_dict.setdefault(
        np.uint32(derive_size), (np.full(33331, np.uint64(0xffffffffffffffff), dtype=np.uint64),
                      np.zeros((33331, derive_size), dtype=np.uint16)))

    book_mat = book_dict.setdefault(count_32k, np.empty((0, derive_size), dtype=np.uint32))
    ind_arr = ind_dict.setdefault(count_32k, np.empty(0, dtype=np.uint64))
    indind_arr = indind_dict.setdefault(count_32k, np.empty(0, dtype=np.uint32))
    if is_gen2_step:
        new_value, probability = np.uint64(1), 1 - spawn_rate4
        board_sum = original_board_sum + np.uint32(2)
    else:
        new_value, probability = np.uint64(2), spawn_rate4
        board_sum = original_board_sum + np.uint32(4)

    chunk_count = max(min(1024, int(len(positions_chunk) * round(np.log2(derive_size + 1)) // 1048576)), 1)
    chunk_size = len(positions_chunk) // chunk_count
    for chunk in range(chunk_count):
        start, end = chunk_size * chunk, chunk_size * (chunk + 1)
        if chunk == chunk_count - 1:
            end = len(positions_chunk)
        for k in prange(start, end):

            if derive_size == 1:
                t = positions_chunk[k]
                success_probability = 0
                empty_slots = 0

                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):  # 如果当前位置为空
                        empty_slots += 1
                        optimal_success_rate = solve_optimal_success_rate(t, new_value, mbm, i, board_sum,
                                                                          pattern_check_func, sym_func, bm,
                                                                          count_32k, lm, book_dict, ind_dict,
                                                                          indind_dict, book_mat, ind_arr, indind_arr)

                        success_probability += optimal_success_rate
                if is_gen2_step:
                    book_chunk[k][0] = book_chunk[k][0] * spawn_rate4 + success_probability * probability / empty_slots
                else:
                    book_chunk[k][0] = success_probability / empty_slots

            else:
                t = positions_chunk[k]
                rep_t, rep_v = replace_val(t)
                rep_t_rev = bm.reverse(rep_t)
                success_probability = np.zeros(derive_size, dtype=np.float64)
                empty_slots = 0

                for i in range(16):  # 遍历所有位置
                    if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):  # 如果当前位置为空
                        empty_slots += 1
                        optimal_success_rate = \
                            solve_optimal_success_rate_arr(t, new_value, i, rep_t, rep_t_rev, pos_rev, mbm, derive_size,
                                                           pattern_check_func, sym_func, bm, rep_v, count_32k, book_dict,
                                                           lm, ind_dict, indind_dict, board_sum,
                                                           match_index, match_mat, book_mat, ind_arr, indind_arr)

                        success_probability += optimal_success_rate
                if is_gen2_step:
                    book_chunk[k] = book_chunk[k] * spawn_rate4 + success_probability * probability / empty_slots
                else:
                    book_chunk[k] = success_probability / empty_slots
