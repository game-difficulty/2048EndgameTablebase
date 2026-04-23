import ctypes
import os
import time
from ctypes import c_int, c_size_t, c_uint32, c_uint64, c_void_p
from datetime import datetime
from typing import Callable, Tuple, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

import Config
from Config import SingletonConfig
from egtb_core.BookGeneratorUtils import native_dll, native_has_avx512
from egtb_core.TrieCompressor import trie_compress_progress
from egtb_core.LzmaCompressor import compress_with_7z, decompress_with_7z
from SignalHub import progress_signal

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int], bool]
CanonicalFunc = Callable[[np.uint64], np.uint64]

ValidDType: TypeAlias = Union[
    type[np.uint32], type[np.uint64], type[np.float32], type[np.float64]
]
SuccessRateDType: TypeAlias = Union[np.uint32, np.uint64, np.float32, np.float64]

move_all_dir: Callable[[np.uint64], tuple[np.uint64, np.uint64, np.uint64, np.uint64]]

logger = Config.logger

SOLVER_VALUE_KIND_U32 = 0
SOLVER_VALUE_KIND_U64 = 1
SOLVER_VALUE_KIND_F32 = 2
SOLVER_VALUE_KIND_F64 = 3

_SOLVER_RECORD_DTYPE_MAP = {
    np.dtype(np.uint32): np.dtype([("f0", np.uint64), ("f1", np.uint32)]),
    np.dtype(np.uint64): np.dtype([("f0", np.uint64), ("f1", np.uint64)]),
    np.dtype(np.float32): np.dtype([("f0", np.uint64), ("f1", np.float32)]),
    np.dtype(np.float64): np.dtype([("f0", np.uint64), ("f1", np.float64)]),
}
_SOLVER_VALUE_KIND_MAP = {
    np.dtype(np.uint32): SOLVER_VALUE_KIND_U32,
    np.dtype(np.uint64): SOLVER_VALUE_KIND_U64,
    np.dtype(np.float32): SOLVER_VALUE_KIND_F32,
    np.dtype(np.float64): SOLVER_VALUE_KIND_F64,
}

_solver_native_ready = False
if native_dll is not None:
    try:
        native_dll.booksolver_expand_u64_records.argtypes = [
            c_void_p,
            c_size_t,
            c_void_p,
            c_size_t,
        ]
        native_dll.booksolver_expand_u64_records.restype = c_size_t
        native_dll.booksolver_filter_records.argtypes = [
            c_void_p,
            c_size_t,
            c_size_t,
            c_int,
            c_uint64,
            c_void_p,
        ]
        native_dll.booksolver_filter_records.restype = c_size_t
        native_dll.booksolver_filter_records_inplace.argtypes = [
            c_void_p,
            c_size_t,
            c_size_t,
            c_int,
            c_uint64,
        ]
        native_dll.booksolver_filter_records_inplace.restype = c_size_t
        native_dll.booksolver_create_index.argtypes = [
            c_void_p,
            c_size_t,
            c_size_t,
            ctypes.POINTER(c_uint32),
        ]
        native_dll.booksolver_create_index.restype = c_int
        _solver_native_ready = True
    except AttributeError:
        _solver_native_ready = False


def solver_native_available() -> bool:
    return _solver_native_ready


def solver_native_has_avx512() -> bool:
    return _solver_native_ready and native_has_avx512()


def _as_void_p(array) -> c_void_p:
    return c_void_p(int(array.ctypes.data))


def solver_record_dtype(value_dtype) -> np.dtype:
    return _SOLVER_RECORD_DTYPE_MAP[np.dtype(value_dtype)]


def read_solver_book(path: str, value_dtype) -> NDArray:
    return np.fromfile(path, dtype=solver_record_dtype(value_dtype))


def write_solver_book(path: str, data: NDArray, value_dtype) -> None:
    target_dtype = solver_record_dtype(value_dtype)
    if data.dtype == target_dtype:
        data.tofile(path)
        return

    converted = np.empty(data.shape, dtype=target_dtype)
    if len(data):
        converted["f0"] = data["f0"]
        converted["f1"] = data["f1"]
    converted.tofile(path)


def _solver_native_record_info(arr) -> tuple[np.dtype, int] | None:
    if (
        not _solver_native_ready
        or arr.ndim != 1
        or not arr.flags.c_contiguous
        or arr.dtype.itemsize not in (12, 16)
    ):
        return None

    arr_dtype = arr.dtype
    if arr_dtype.names is None or "f1" not in arr_dtype.fields or "f0" not in arr_dtype.fields:
        return None

    field_dtype, field_offset = arr_dtype.fields["f1"][:2]
    key_dtype, key_offset = arr_dtype.fields["f0"][:2]
    if key_dtype != np.dtype(np.uint64) or key_offset != 0 or field_offset != 8:
        return None

    success_dtype = np.dtype(field_dtype)
    kind = _SOLVER_VALUE_KIND_MAP.get(success_dtype)
    if kind is None:
        return None

    return success_dtype, kind


def _solver_threshold_bits(value, dtype: np.dtype) -> int:
    if dtype == np.dtype(np.uint32):
        return int(np.uint32(value))
    if dtype == np.dtype(np.uint64):
        return int(np.uint64(value))
    if dtype == np.dtype(np.float32):
        return int(np.array([value], dtype=np.float32).view(np.uint32)[0])
    if dtype == np.dtype(np.float64):
        return int(np.array([value], dtype=np.float64).view(np.uint64)[0])
    raise TypeError(f"unsupported success-rate dtype: {dtype}")


def handle_restart(i, pathname, steps, d1, d2, started, dtype):
    """
    处理断点重连逻辑
    """
    if os.path.exists(pathname + str(i) + ".book"):
        logger.debug(f"skipping step {i}")
        if not SingletonConfig().config.get("optimal_branch_only", False):
            do_compress(pathname + str(i + 2) + ".book", dtype)
        return False, None, None
    elif os.path.exists(pathname + str(i) + ".z") or os.path.exists(
        pathname + str(i) + ".book.7z"
    ):
        logger.debug(f"skipping step {i}")
        return False, None, None
    elif not started:
        started = True
        if i != steps - 3 or d1 is None or d2 is None:
            d1 = read_solver_book(pathname + str(i + 1) + ".book", dtype)
            d2 = read_solver_book(pathname + str(i + 2) + ".book", dtype)
        return started, d1, d2

    return True, d1, d2


def recalculate_process(
    d1: NDArray[[np.uint64, SuccessRateDType]],
    d2: NDArray[[np.uint64, SuccessRateDType]],
    pattern_check_func: PatternCheckFunc,
    success_check_func: SuccessCheckFunc,
    canonical_func: CanonicalFunc,
    target: int,
    steps: int,
    pathname: str,
    docheck_step: int,
    spawn_rate4: float = 0.1,
    is_variant: bool = False,
) -> None:
    success_rate_dtype = SingletonConfig().config.get("success_rate_dtype", "uint32")
    _, current_dtype, max_scale, zero_val = Config.DTYPE_CONFIG[
        success_rate_dtype
    ]
    global move_all_dir
    if is_variant:
        from egtb_core.VBoardMover import move_all_dir
    else:
        from egtb_core.BoardMover import move_all_dir

    started = False
    # 回算搜索时可能需要使用索引进行加速
    ind1: NDArray[np.uint32] | None = None

    if not os.path.exists(pathname + "stats.txt"):
        with open(pathname + "stats.txt", "a", encoding="utf-8") as file:
            file.write(
                ",".join(
                    [
                        "layer",
                        "length",
                        "max success rate",
                        "speed",
                        "deletion_threshold",
                        "time",
                    ]
                )
                + "\n"
            )

    # 从后向前更新ds中的array
    for i in range(steps - 3, -1, -1):
        started, d1, d2 = handle_restart(
            i, pathname, steps, d1, d2, started, current_dtype
        )
        if not started:
            continue

        progress_signal.progress_updated.emit(steps * 2 - i - 2, steps * 2)

        deletion_threshold_n = SingletonConfig().config.get("deletion_threshold", 0.0)
        deletion_threshold = current_dtype(
            deletion_threshold_n * (max_scale - zero_val) + zero_val
        )
        if SingletonConfig().config.get("compress_temp_files", False):
            decompress_with_7z(pathname + str(i) + ".7z")
        d0 = np.fromfile(pathname + str(i), dtype=np.uint64)

        t0 = time.perf_counter()

        arr0 = np.empty(len(d0), dtype=solver_record_dtype(current_dtype))
        expanded_arr0 = expand(d0, arr0)
        del d0

        # 创建、更新查找索引
        if ind1 is not None:
            ind2 = ind1
        elif len(d2) < 100000:
            ind2 = None
        else:
            ind2 = create_index(d2)
        if len(d1) < 100000:
            ind1 = None
        else:
            ind1 = create_index(d1)
        t1 = time.perf_counter()

        # 回算
        d0 = recalculate(
            expanded_arr0,
            d1,
            d2,
            target,
            pattern_check_func,
            success_check_func,
            canonical_func,
            ind1,
            ind2,
            max_scale,
            zero_val,
            i > docheck_step,
            spawn_rate4,
        )
        length = len(d0)
        t2 = time.perf_counter()
        d0 = remove_died(d0, zero_val).copy()  # 去除0成功率的局面

        t3 = time.perf_counter()
        if t3 > t0:
            logger.debug(
                f"step {i} recalculated: {round(length / (t3 - t0) / 1e6, 2)} mbps"
            )
            logger.debug(
                f"index/solve/remove: {round((t1 - t0) / (t3 - t0), 2)}/"
                f"{round((t2 - t1) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}"
            )

        if len(d0):
            with open(pathname + "stats.txt", "a", encoding="utf-8") as file:
                file.write(
                    ",".join(
                        [
                            str(i),
                            str(length),
                            str((np.max(d0["f1"]) - zero_val) / (max_scale - zero_val)),
                            f"{round(length / max((t3 - t0), 0.01) / 1e6, 2)} mbps",
                            str(deletion_threshold_n),
                            str(datetime.now()),
                        ]
                    )
                    + "\n"
                )

        write_solver_book(pathname + str(i) + ".book", d0, current_dtype)
        if os.path.exists(pathname + str(i)):
            os.remove(pathname + str(i))
        logger.debug(f"step {i} written\n")

        if deletion_threshold_n > 0:
            write_solver_book(
                pathname + str(i + 2) + ".book",
                remove_died(d2, deletion_threshold),
                current_dtype,
            )  # 再写一次，把成功率低于阈值的局面去掉

        if SingletonConfig().config.get(
            "compress_temp_files", False
        ) and SingletonConfig().config.get("optimal_branch_only", False):
            compress_with_7z(pathname + str(i + 2) + ".book")
        elif SingletonConfig().config.get(
            "compress", False
        ) and not SingletonConfig().config.get("optimal_branch_only", False):
            do_compress(
                pathname + str(i + 2) + ".book", current_dtype
            )  # 如果设置了压缩，则压缩i+2的book，其已经不需要再频繁查找

        if i > 0:
            d1, d2 = d0, d1


@njit(nogil=True, parallel=True, cache=True)
def _expand_numba(
    arr: NDArray[np.uint64], arr0
) -> NDArray[[np.uint64, SuccessRateDType]]:
    for i in prange(len(arr)):
        arr0[i]["f0"] = arr[i]
    return arr0


@njit(nogil=True, cache=True)
def _remove_died_numba(
    arr: NDArray[[np.uint64, SuccessRateDType]],
    deletion_threshold: SuccessRateDType = 0,
) -> NDArray[[np.uint64, SuccessRateDType]]:
    count = 0
    # 原地移动成功率高于阈值的元素
    for i in range(len(arr)):
        if arr[i]["f1"] > deletion_threshold:
            arr[count] = arr[i]
            count += 1
    return arr[:count]


@njit(parallel=True, nogil=True, cache=True)
def _create_index_numba(
    arr: NDArray[[np.uint64, SuccessRateDType]],
) -> NDArray[np.uint32] | None:
    """
    根据uint64数据的前24位的分段位置创建一个索引，长度16777216+1
    """
    n = 16777217
    ind1: NDArray = np.empty(n, dtype="uint32")
    header = arr[0][0] >> np.uint32(40)
    ind1[: header + 1] = 0

    for i in prange(1, len(arr)):
        header = arr[i][0] >> np.uint32(40)
        header_pre = arr[i - 1][0] >> np.uint32(40)
        if header != header_pre:
            ind1[header_pre + 1 : header + 1] = i

    header = arr[-1][0] >> np.uint32(40)
    ind1[header + 1 :] = len(arr)
    return ind1


def _expand_native(
    arr: NDArray[np.uint64], arr0
) -> NDArray[[np.uint64, SuccessRateDType]] | None:
    record_info = _solver_native_record_info(arr0)
    if (
        arr.dtype != np.uint64
        or arr.ndim != 1
        or not arr.flags.c_contiguous
        or record_info is None
    ):
        return None

    native_dll.booksolver_expand_u64_records(
        _as_void_p(arr),
        arr.size,
        _as_void_p(arr0),
        arr0.dtype.itemsize,
    )
    return arr0


def expand(arr: NDArray[np.uint64], arr0) -> NDArray[[np.uint64, SuccessRateDType]]:
    native_result = _expand_native(arr, arr0)
    if native_result is not None:
        return native_result
    return _expand_numba(arr, arr0)


def _remove_died_native(
    arr: NDArray[[np.uint64, SuccessRateDType]],
    deletion_threshold: SuccessRateDType = 0,
) -> NDArray[[np.uint64, SuccessRateDType]] | None:
    record_info = _solver_native_record_info(arr)
    if record_info is None:
        return None

    success_dtype, kind = record_info
    threshold_bits = _solver_threshold_bits(deletion_threshold, success_dtype)
    filtered_len = native_dll.booksolver_filter_records_inplace(
        _as_void_p(arr),
        arr.size,
        arr.dtype.itemsize,
        kind,
        threshold_bits,
    )
    return arr[:filtered_len]


def remove_died(
    arr: NDArray[[np.uint64, SuccessRateDType]],
    deletion_threshold: SuccessRateDType = 0,
) -> NDArray[[np.uint64, SuccessRateDType]]:
    if len(arr) < 65536:
        return _remove_died_numba(arr, deletion_threshold)
    native_result = _remove_died_native(arr, deletion_threshold)
    if native_result is not None:
        return native_result
    return _remove_died_numba(arr, deletion_threshold)


def _create_index_native(
    arr: NDArray[[np.uint64, SuccessRateDType]],
) -> NDArray[np.uint32] | None:
    if len(arr) == 0:
        return None
    if _solver_native_record_info(arr) is None:
        return None

    ind1 = np.empty(16777217, dtype=np.uint32)
    status = native_dll.booksolver_create_index(
        _as_void_p(arr),
        arr.size,
        arr.dtype.itemsize,
        ind1.ctypes.data_as(ctypes.POINTER(c_uint32)),
    )
    return ind1 if status != 0 else None


def create_index(
    arr: NDArray[[np.uint64, SuccessRateDType]],
) -> NDArray[np.uint32] | None:
    if len(arr) == 0:
        return None

    record_info = _solver_native_record_info(arr)
    if (
        record_info is not None
        and len(arr) >= 2_000_000
    ):
        first_header = int(arr[0]["f0"] >> np.uint64(40))
        last_header = int(arr[-1]["f0"] >> np.uint64(40))
        if first_header == last_header:
            native_result = _create_index_native(arr)
            if native_result is not None:
                return native_result

    return _create_index_numba(arr)


@njit(nogil=True, cache=True)
def binary_search_arr(
    arr: NDArray[[np.uint64, SuccessRateDType]],
    zero_val: SuccessRateDType,
    target: np.uint64,
    low: np.uint32 | None = None,
    high: np.uint32 | None = None,
) -> SuccessRateDType:
    if low is None:
        low = 0
        high = len(arr) - 1

    while low <= high:
        mid = low + np.uint32((high - low) // 2)
        mid_val = arr[mid][0]
        if mid_val < target:
            low = mid + 1
        elif mid_val > target:
            high = mid - 1
        else:
            return arr[mid][1]  # 找到匹配，返回对应的uint32值

    return zero_val  # 如果没有找到匹配项


@njit(nogil=True, cache=True)
def search_arr(
    arr: NDArray[[np.uint64, SuccessRateDType]],
    b: np.uint64,
    ind: NDArray[np.uint32] | None,
    zero_val: SuccessRateDType,
) -> SuccessRateDType:
    """
    没有索引就直接二分查找，否则先从索引中确定一个更窄的范围再查找
    """
    if ind is None:
        return binary_search_arr(arr, zero_val, b)
    header = b >> np.uint32(40)
    low, high = ind[header], ind[header + 1] - 1
    return binary_search_arr(arr, zero_val, b, low, high)


@njit(parallel=True, nogil=True, cache=True)
def recalculate(
    arr0: NDArray[[np.uint64, SuccessRateDType]],
    arr1: NDArray[[np.uint64, SuccessRateDType]],
    arr2: NDArray[[np.uint64, SuccessRateDType]],
    target: int,
    pattern_check_func: PatternCheckFunc,
    success_check_func: SuccessCheckFunc,
    canonical_func: CanonicalFunc,
    ind1: NDArray[np.uint32] | None,
    ind2: NDArray[np.uint32] | None,
    max_scale: SuccessRateDType,
    zero_val: SuccessRateDType,
    do_check: bool = False,
    spawn_rate4: float = 0.1,
) -> NDArray[[np.uint64, SuccessRateDType]]:
    """
    根据已经填充好成功概率的array回算前一批面板的成功概率。
    对于arr0中的每个面板，考虑在每个空位填充数字2或4（90%概率为2，10%概率为4），
    然后对于每个可能的填充，执行所有有效的移动操作，并基于移动结果的成功概率来更新当前面板的成功概率。
    ind1, ind2是预先计算的索引，用于加速二分查找过程
    """
    chunk_count = max(min(1024, len(arr0) // 1048576), 1)
    chunk_size = len(arr0) // chunk_count
    for chunk in range(chunk_count):
        start, end = chunk_size * chunk, chunk_size * (chunk + 1)
        if chunk == chunk_count - 1:
            end = len(arr0)
        for k in prange(start, end):
            t: np.uint64 = arr0[k][0]
            if do_check and success_check_func(t, target):
                arr0[k][1] = max_scale
                continue
            # 初始化概率和权重
            success_probability = 0.0
            empty_slots = 0
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(
                    0
                ):  # 如果当前位置为空
                    empty_slots += 1

                    # 对于每个空位置，尝试填充2和4
                    new_value, probability = 1, 1 - spawn_rate4
                    t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                    optimal_success_rate = (
                        zero_val  # 记录有效移动后的面板成功概率中的最大值
                    )
                    for newt in move_all_dir(t_gen):
                        if newt != t_gen and pattern_check_func(
                            newt
                        ):  # 只考虑有效的移动
                            # 获取移动后的面板成功概率
                            optimal_success_rate = max(
                                optimal_success_rate,
                                search_arr(arr1, canonical_func(newt), ind1, zero_val),
                            )
                    # 对最佳移动下的成功概率加权平均
                    success_probability += optimal_success_rate * probability

                    # 填4
                    new_value, probability = 2, spawn_rate4
                    t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                    optimal_success_rate = (
                        zero_val  # 记录有效移动后的面板成功概率中的最大值
                    )
                    for newt in move_all_dir(t_gen):
                        if newt != t_gen and pattern_check_func(
                            newt
                        ):  # 只考虑有效的移动
                            # 获取移动后的面板成功概率
                            optimal_success_rate = max(
                                optimal_success_rate,
                                search_arr(arr2, canonical_func(newt), ind2, zero_val),
                            )
                    # 对最佳移动下的成功概率加权平均
                    success_probability += optimal_success_rate * probability

            # t是进行一次有效移动后尚未生成新数字时的面板，因此不可能没有空位置
            # numba会自动转换类型
            arr0[k][1] = success_probability / empty_slots

    return arr0


def do_compress(bookpath: str, dtype: ValidDType) -> None:
    if bookpath[-4:] != "book":
        return
    if bookpath[-7:] in ["_0.book", "_1.book", "_2.book"]:
        return
    if SingletonConfig().config.get("compress", False) and os.path.exists(bookpath):
        if os.path.getsize(bookpath) > 2097152:
            trie_compress_progress(*os.path.split(bookpath), dtype)
            if os.path.exists(bookpath):
                os.remove(bookpath)


@njit(parallel=True, nogil=True, cache=True)
def find_optimal_branches(
    arr0: NDArray[[np.uint64, SuccessRateDType]],
    arr1: NDArray[[np.uint64, SuccessRateDType]],
    result: NDArray[bool],
    pattern_check_func: PatternCheckFunc,
    canonical_func: CanonicalFunc,
    ind1: NDArray[np.uint32] | None,
    new_value: int,
    zero_val: SuccessRateDType,
) -> NDArray[bool]:
    for start, end in (
        (0, len(arr0) // 10),
        (len(arr0) // 10, len(arr0) // 3),
        (len(arr0) // 3, len(arr0)),
    ):  # 缓解负载不均衡问题
        for k in prange(start, end):
            t = arr0[k][0]
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(
                    0
                ):  # 如果当前位置为空
                    t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                    optimal_success_rate = (
                        zero_val  # 记录有效移动后的面板成功概率中的最大值
                    )
                    optimal_pos_index = np.uint64(0)
                    for newt in move_all_dir(t_gen):
                        if newt != t_gen and pattern_check_func(
                            newt
                        ):  # 只考虑有效的移动
                            # 获取移动后的面板成功概率
                            success_rate, pos_index = search_arr2(
                                arr1, canonical_func(newt), ind1, zero_val
                            )
                            if success_rate > optimal_success_rate:
                                optimal_success_rate = success_rate
                                optimal_pos_index = np.uint64(pos_index)
                    result[optimal_pos_index] = True
    return result


@njit(nogil=True, cache=True)
def binary_search_arr2(
    arr: NDArray[[np.uint64, SuccessRateDType]],
    zero_val: SuccessRateDType,
    target: np.uint64,
    low: np.uint32 | None = None,
    high: np.uint32 | None = None,
) -> Tuple[SuccessRateDType, np.uint64]:
    """相比binary_search_arr，还会返回索引"""
    if low is None:
        low = 0
        high = len(arr) - 1

    while low <= high:
        mid = low + np.uint32((high - low) // 2)
        mid_val = arr[mid][0]
        if mid_val < target:
            low = mid + 1
        elif mid_val > target:
            high = mid - 1
        else:
            return arr[mid][1], mid  # 找到匹配，返回对应的值和索引位置

    return zero_val, np.uint64(0)  # 如果没有找到匹配项


@njit(nogil=True, cache=True)
def search_arr2(
    arr: NDArray[[np.uint64, SuccessRateDType]],
    b: np.uint64,
    ind: NDArray[np.uint32] | None,
    zero_val: SuccessRateDType,
) -> Tuple[SuccessRateDType, np.uint64]:
    """
    相比search_arr，还会返回索引
    """
    if ind is None:
        return binary_search_arr2(arr, zero_val, b)
    header = b >> np.uint32(40)
    low, high = ind[header], ind[header + 1] - 1
    return binary_search_arr2(arr, zero_val, b, low, high)


def keep_only_optimal_branches(
    pattern_check_func: PatternCheckFunc,
    canonical_func: CanonicalFunc,
    steps: int,
    pathname: str,
):
    success_rate_dtype = SingletonConfig().config.get("success_rate_dtype", "uint32")
    _, current_dtype, max_scale, zero_val = Config.DTYPE_CONFIG[
        success_rate_dtype
    ]
    d0, d1 = None, None
    started = False
    for i in range(0, steps):
        started, d0, d1 = handle_restart_opt_only(
            i, started, d0, d1, pathname, current_dtype
        )
        if SingletonConfig().config.get("compress_temp_files", False):
            decompress_with_7z(pathname + str(i) + ".book.7z")
        if i > 20 and started:
            d2 = read_solver_book(pathname + str(i) + ".book", current_dtype)
            ind = create_index(d2)
            is_in_optimal_branches_mask = np.zeros(len(d2), dtype="bool")
            is_in_optimal_branches_mask = find_optimal_branches(
                d0,
                d2,
                is_in_optimal_branches_mask,
                pattern_check_func,
                canonical_func,
                ind,
                2,
                zero_val,
            )
            is_in_optimal_branches_mask = find_optimal_branches(
                d1,
                d2,
                is_in_optimal_branches_mask,
                pattern_check_func,
                canonical_func,
                ind,
                1,
                zero_val,
            )
            d2 = d2[is_in_optimal_branches_mask].copy()
            write_solver_book(pathname + str(i) + ".book", d2, current_dtype)
            d0, d1 = d1, d2
            del is_in_optimal_branches_mask, d2
            logger.debug(f"step {i} retains only the optimal branch\n")
        do_compress(pathname + str(i - 2) + ".book", current_dtype)

    do_compress(pathname + str(steps - 2) + ".book", current_dtype)
    do_compress(pathname + str(steps - 1) + ".book", current_dtype)
    if os.path.exists(pathname + "optlayer"):
        os.remove(pathname + "optlayer")


def handle_restart_opt_only(i, started, d0, d1, pathname, dtype):
    if started and d0 is not None:
        with open(pathname + "optlayer", "w") as f:
            f.write(str(i - 1))
        return True, d0, d1
    try:
        with open(pathname + "optlayer", "r") as f:
            current_layer = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        current_layer = i - 1
    if current_layer >= i:
        return False, None, None
    else:
        if (i > 20) & (d0 is None):
            try:
                d0 = read_solver_book(pathname + str(i - 2) + ".book", dtype)
                d1 = read_solver_book(pathname + str(i - 1) + ".book", dtype)
                return True, d0, d1
            except FileNotFoundError:
                pass
        return False, d0, d1


# if __name__ == "__main__":
#     from Calculator import is_free_pattern, canonical_full
#     from BoardMover import BoardMover
#
#     keep_only_optimal_branches(is_free_pattern, canonical_full, 536,
#                                r"D:\2048calculates\table\free10w_1024 - 副本\free10w_1024_")
