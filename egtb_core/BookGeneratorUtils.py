import ctypes
import os
from ctypes import POINTER, c_size_t, c_uint64, c_void_p
from typing import Callable, List
import traceback
import sys

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

import Config
from Config import SingletonConfig


PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int], bool]
CanonicalFunc = Callable[[NDArray[np.uint64]], None]
CanonicalFunc1 = Callable[[np.uint64], np.uint64]

logger = Config.logger


def _resolve_native_library() -> str:
    project_root = os.path.dirname(os.path.dirname(__file__))
    ext = "dll" if sys.platform == "win32" else "so"
    candidates = (
        os.path.join(project_root, "ai_and_sort", f"bookgen_native.{ext}"),
        os.path.join("_internal", f"bookgen_native.{ext}"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"bookgen_native.{ext} not found.")


def initialize_sorting_library():
    _dll = None
    try:
        dll_path = _resolve_native_library()

        # 加载DLL
        _dll = ctypes.CDLL(dll_path)

        # 声明sort_uint64函数原型
        _dll.sort_uint64.argtypes = [POINTER(c_uint64), ctypes.c_size_t, ctypes.c_bool]
        _dll.sort_uint64.restype = None  # 无返回值

        # 测试排序操作
        _arr = np.random.randint(0, 1 << 64, 10000, dtype=np.uint64)
        _arr_ptr = _arr.ctypes.data_as(POINTER(c_uint64))
        descending = False

        _dll.sort_uint64(_arr_ptr, len(_arr), descending)

        # 验证排序结果是否正确
        if not np.all(_arr[:-1] <= _arr[1:]):
            raise ValueError("DLL sorting failed, results are not sorted correctly")

    except Exception as e:
        logger.info(f"avx support: {SingletonConfig.check_cpuinfo()}")
        logger.warning(
            f"Failed to load DLL or test sorting failed: {e}. Falling back to numpy sort."
        )
        logger.warning(traceback.format_exc())  # 输出完整的异常堆栈

    return _dll


# 在模块加载时执行初始化函数
def initialize_unique_library():
    _dll = None
    try:
        dll_path = _resolve_native_library()
        _dll = ctypes.CDLL(dll_path)

        _dll.unique_sorted_u64_inplace.argtypes = [POINTER(c_uint64), ctypes.c_size_t]
        _dll.unique_sorted_u64_inplace.restype = ctypes.c_size_t
        _dll.unique_sorted_u64_has_avx512.argtypes = []
        _dll.unique_sorted_u64_has_avx512.restype = ctypes.c_int

        _arr = np.array([1, 1, 2, 2, 5, 8, 8], dtype=np.uint64)
        _arr_ptr = _arr.ctypes.data_as(POINTER(c_uint64))
        _length = _dll.unique_sorted_u64_inplace(_arr_ptr, len(_arr))
        if _length != 4 or not np.array_equal(
            _arr[:_length], np.array([1, 2, 5, 8], dtype=np.uint64)
        ):
            raise ValueError(
                "DLL unique failed, results are not deduplicated correctly"
            )

    except Exception as e:
        logger.info(f"avx support: {SingletonConfig.check_cpuinfo()}")
        logger.warning(
            f"Failed to load unique DLL or test unique failed: {e}. Falling back to numba unique."
        )
        logger.warning(traceback.format_exc())

    return _dll


def initialize_merge_tree_library():
    _dll = None
    try:
        dll_path = _resolve_native_library()
        _dll = ctypes.CDLL(dll_path)

        uint64_ptr = POINTER(c_uint64)
        _dll.merge_tree_u64_dedup.argtypes = [
            POINTER(uint64_ptr),
            POINTER(c_size_t),
            c_size_t,
            uint64_ptr,
            uint64_ptr,
        ]
        _dll.merge_tree_u64_dedup.restype = c_size_t
        _dll.merge_tree_partitioned_u64_dedup.argtypes = [
            POINTER(uint64_ptr),
            POINTER(c_size_t),
            c_size_t,
            uint64_ptr,
            c_size_t,
            uint64_ptr,
            POINTER(c_size_t),
            POINTER(c_size_t),
            uint64_ptr,
        ]
        _dll.merge_tree_partitioned_u64_dedup.restype = c_size_t
        _dll.merge_two_u64_dedup.argtypes = [
            c_void_p,
            c_size_t,
            c_void_p,
            c_size_t,
            c_void_p,
        ]
        _dll.merge_two_u64_dedup.restype = c_size_t
        _dll.merge_two_u64_partitioned_dedup.argtypes = [
            c_void_p,
            c_size_t,
            c_void_p,
            c_size_t,
            c_void_p,
            c_size_t,
            c_void_p,
            c_void_p,
            c_void_p,
        ]
        _dll.merge_two_u64_partitioned_dedup.restype = c_size_t
        _dll.merge_tree_u64_dedup_has_avx512.argtypes = []
        _dll.merge_tree_u64_dedup_has_avx512.restype = ctypes.c_int

        _arr1 = np.array([1, 2, 2, 7], dtype=np.uint64)
        _arr2 = np.array([1, 3, 3, 7], dtype=np.uint64)
        _result = np.empty(len(_arr1) + len(_arr2), dtype=np.uint64)
        _scratch = np.empty(len(_arr1) + len(_arr2), dtype=np.uint64)
        _ptrs = (uint64_ptr * 2)(
            _arr1.ctypes.data_as(uint64_ptr),
            _arr2.ctypes.data_as(uint64_ptr),
        )
        _lengths = (c_size_t * 2)(len(_arr1), len(_arr2))
        _merged = _dll.merge_tree_u64_dedup(
            _ptrs,
            _lengths,
            2,
            _result.ctypes.data_as(uint64_ptr),
            _scratch.ctypes.data_as(uint64_ptr),
        )
        if _merged != 4 or not np.array_equal(
            _result[:_merged], np.array([1, 2, 3, 7], dtype=np.uint64)
        ):
            raise ValueError("DLL merge tree failed, results are not merged correctly")

        _partitioned = np.empty(len(_arr1) + len(_arr2), dtype=np.uint64)
        _offsets = np.empty(3, dtype=np.uintp)
        _sizes = np.empty(2, dtype=np.uintp)
        _pivots = np.array([3], dtype=np.uint64)
        _dll.merge_two_u64_partitioned_dedup(
            _arr1.ctypes.data,
            len(_arr1),
            _arr2.ctypes.data,
            len(_arr2),
            _pivots.ctypes.data,
            2,
            _partitioned.ctypes.data,
            _offsets.ctypes.data,
            _sizes.ctypes.data,
        )
        if (
            list(_sizes.tolist()) != [2, 2]
            or not np.array_equal(
                _partitioned[_offsets[0] : _offsets[0] + _sizes[0]],
                np.array([1, 2], dtype=np.uint64),
            )
            or not np.array_equal(
                _partitioned[_offsets[1] : _offsets[1] + _sizes[1]],
                np.array([3, 7], dtype=np.uint64),
            )
        ):
            raise ValueError(
                "DLL partitioned merge tree failed, results are not partitioned correctly"
            )

    except Exception as e:
        logger.info(f"avx support: {SingletonConfig.check_cpuinfo()}")
        logger.warning(
            f"Failed to load merge tree DLL or test merge failed: {e}. Falling back to numba merge."
        )
        logger.warning(traceback.format_exc())

    return _dll

def initialize_native_library():
    _dll = None
    try:
        dll_path = _resolve_native_library()
        _dll = ctypes.CDLL(dll_path)

        uint64_ptr = POINTER(c_uint64)
        _dll.sort_uint64.argtypes = [uint64_ptr, c_size_t, ctypes.c_bool]
        _dll.sort_uint64.restype = None
        _dll.unique_sorted_u64_inplace.argtypes = [uint64_ptr, c_size_t]
        _dll.unique_sorted_u64_inplace.restype = c_size_t
        _dll.merge_tree_u64_dedup.argtypes = [
            POINTER(uint64_ptr),
            POINTER(c_size_t),
            c_size_t,
            uint64_ptr,
            uint64_ptr,
        ]
        _dll.merge_tree_u64_dedup.restype = c_size_t
        _dll.merge_tree_partitioned_u64_dedup.argtypes = [
            POINTER(uint64_ptr),
            POINTER(c_size_t),
            c_size_t,
            uint64_ptr,
            c_size_t,
            uint64_ptr,
            POINTER(c_size_t),
            POINTER(c_size_t),
            uint64_ptr,
        ]
        _dll.merge_tree_partitioned_u64_dedup.restype = c_size_t
        _dll.merge_two_u64_dedup.argtypes = [
            c_void_p,
            c_size_t,
            c_void_p,
            c_size_t,
            c_void_p,
        ]
        _dll.merge_two_u64_dedup.restype = c_size_t
        _dll.merge_two_u64_partitioned_dedup.argtypes = [
            c_void_p,
            c_size_t,
            c_void_p,
            c_size_t,
            c_void_p,
            c_size_t,
            c_void_p,
            c_void_p,
            c_void_p,
        ]
        _dll.merge_two_u64_partitioned_dedup.restype = c_size_t
        _dll.bookgen_native_has_avx512.argtypes = []
        _dll.bookgen_native_has_avx512.restype = ctypes.c_int

        sort_probe = np.array([5, 2, 9, 1, 1, 3], dtype=np.uint64)
        _dll.sort_uint64(sort_probe.ctypes.data_as(uint64_ptr), sort_probe.size, False)
        if not np.all(sort_probe[:-1] <= sort_probe[1:]):
            raise ValueError("Native sorting failed, results are not sorted correctly")

        unique_probe = np.array([1, 1, 2, 2, 5, 8, 8], dtype=np.uint64)
        unique_length = _dll.unique_sorted_u64_inplace(
            unique_probe.ctypes.data_as(uint64_ptr), unique_probe.size
        )
        if unique_length != 4 or not np.array_equal(
            unique_probe[:unique_length], np.array([1, 2, 5, 8], dtype=np.uint64)
        ):
            raise ValueError(
                "Native unique failed, results are not deduplicated correctly"
            )

        left = np.array([1, 2, 2, 7], dtype=np.uint64)
        right = np.array([1, 3, 3, 7], dtype=np.uint64)
        merged = np.empty(left.size + right.size, dtype=np.uint64)
        scratch = np.empty(left.size + right.size, dtype=np.uint64)
        ptrs = (uint64_ptr * 2)(
            left.ctypes.data_as(uint64_ptr),
            right.ctypes.data_as(uint64_ptr),
        )
        lengths = (c_size_t * 2)(left.size, right.size)
        merged_length = _dll.merge_tree_u64_dedup(
            ptrs,
            lengths,
            2,
            merged.ctypes.data_as(uint64_ptr),
            scratch.ctypes.data_as(uint64_ptr),
        )
        if merged_length != 4 or not np.array_equal(
            merged[:merged_length], np.array([1, 2, 3, 7], dtype=np.uint64)
        ):
            raise ValueError("Native merge failed, results are not merged correctly")

        partitioned = np.empty(left.size + right.size, dtype=np.uint64)
        offsets = np.empty(3, dtype=np.uintp)
        sizes = np.empty(2, dtype=np.uintp)
        pivots = np.array([3], dtype=np.uint64)
        _dll.merge_two_u64_partitioned_dedup(
            c_void_p(int(left.ctypes.data)),
            left.size,
            c_void_p(int(right.ctypes.data)),
            right.size,
            c_void_p(int(pivots.ctypes.data)),
            2,
            c_void_p(int(partitioned.ctypes.data)),
            c_void_p(int(offsets.ctypes.data)),
            c_void_p(int(sizes.ctypes.data)),
        )
        if (
            list(sizes.tolist()) != [2, 2]
            or not np.array_equal(
                partitioned[offsets[0] : offsets[0] + sizes[0]],
                np.array([1, 2], dtype=np.uint64),
            )
            or not np.array_equal(
                partitioned[offsets[1] : offsets[1] + sizes[1]],
                np.array([3, 7], dtype=np.uint64),
            )
        ):
            raise ValueError(
                "Native partitioned merge failed, results are not partitioned correctly"
            )

    except Exception as e:
        logger.info(f"avx support: {SingletonConfig.check_cpuinfo()}")
        logger.warning(
            f"Failed to load native algorithms DLL or self-test failed: {e}. "
            "Falling back to numpy/numba implementations."
        )
        logger.warning(traceback.format_exc())
    return _dll


native_dll = initialize_native_library()


def _as_void_p(array: NDArray[np.uint64] | NDArray[np.uintp] | None) -> c_void_p:
    if array is None:
        return c_void_p()
    return c_void_p(int(array.ctypes.data))


@njit(nogil=True, cache=True)
def single_thread_unique(aux: NDArray[np.uint64]) -> NDArray[np.uint64]:
    if len(aux) == 0:
        return np.empty(0, dtype=np.uint64)

    c = 1

    for j in range(1, len(aux)):
        if aux[j] != aux[j - 1]:
            aux[c] = aux[j]
            c += 1

    return aux[:c]


def parallel_unique(aux: NDArray[np.uint64], n: int) -> NDArray[np.uint64]:
    _ = n
    return unique_native_inplace(aux)


def native_available() -> bool:
    return native_dll is not None


def native_has_avx512() -> bool:
    return bool(native_dll is not None and native_dll.bookgen_native_has_avx512())


def unique_native_inplace(aux: NDArray[np.uint64]) -> NDArray[np.uint64]:
    if (
        native_dll is None
        or aux.dtype != np.uint64
        or aux.ndim != 1
        or not aux.flags.c_contiguous
    ):
        return single_thread_unique(aux)

    arr_ptr = aux.ctypes.data_as(POINTER(c_uint64))
    new_length = native_dll.unique_sorted_u64_inplace(arr_ptr, aux.size)
    return aux[:new_length]


@njit(nogil=True, cache=True)
def largest_power_of_2(n):
    n = np.uint64(n)
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


@njit(nogil=True, cache=True)
def _merge_deduplicate_all_numba(arrays: List[NDArray], length: int = 0) -> NDArray:
    if len(arrays) == 1:
        return arrays[0]
    if length == 0:
        for arr in arrays:
            length += len(arr)
    num_arrays = len(arrays)
    indices = np.zeros(num_arrays, dtype="uint32")  # 每个数组的当前索引
    merged_array = np.empty(length, dtype="uint64")  # 合并后且去重的数组
    last_added = np.uint64(0xFFFFFFFFFFFFFFFF)  # 上一个添加到 merged_array 的元素
    c = 0  # 已添加的元素数量
    # 继续循环，直到所有数组都被完全处理
    while True:
        current_min = np.uint64(0xFFFFFFFFFFFFFFFF)
        min_index = -1
        # 寻找当前可用元素中的最小值
        for i in range(num_arrays):
            if indices[i] < len(arrays[i]):  # 确保索引不超出数组长度
                if arrays[i][indices[i]] < current_min:
                    current_min = arrays[i][indices[i]]
                    min_index = i
        # 如果找不到最小值，说明所有数组都已处理完成
        if current_min == np.uint64(0xFFFFFFFFFFFFFFFF):
            break
        # 检查是否需要将当前最小值添加到结果数组中
        if current_min != last_added:
            merged_array[c] = current_min
            last_added = current_min
            c += 1
        # 移动选中数组的索引
        indices[min_index] += 1
    return merged_array[:c]


def _prepare_merge_tree_inputs(
    arrays: List[NDArray[np.uint64]],
) -> list[NDArray[np.uint64]]:
    prepared: list[NDArray[np.uint64]] = []
    for arr in arrays:
        candidate = arr
        if candidate.dtype != np.uint64 or candidate.ndim != 1:
            candidate = np.asarray(candidate, dtype=np.uint64)
        if not candidate.flags.c_contiguous:
            candidate = np.ascontiguousarray(candidate)
        prepared.append(candidate)
    return prepared


def _build_native_merge_tree_inputs(
    arrays: List[NDArray[np.uint64]],
) -> tuple[list[NDArray[np.uint64]], object, object, int]:
    prepared = _prepare_merge_tree_inputs(arrays)
    uint64_ptr = POINTER(c_uint64)
    ptrs = (uint64_ptr * len(prepared))()
    lengths = (c_size_t * len(prepared))()
    total_length = 0
    for idx, arr in enumerate(prepared):
        ptrs[idx] = arr.ctypes.data_as(uint64_ptr)
        lengths[idx] = arr.size
        total_length += arr.size
    return prepared, ptrs, lengths, total_length


def _merge_deduplicate_all_native(
    arrays: List[NDArray[np.uint64]], length: int = 0
) -> NDArray[np.uint64]:
    if len(arrays) == 0:
        return np.empty(0, dtype=np.uint64)
    if len(arrays) == 1:
        return arrays[0]
    if native_dll is None:
        return _merge_deduplicate_all_numba(arrays, length)

    prepared = _prepare_merge_tree_inputs(arrays)
    total_length = sum(arr.size for arr in prepared)
    if length == 0:
        length = total_length

    result = np.empty(length, dtype=np.uint64)
    uint64_ptr = POINTER(c_uint64)
    if len(prepared) == 2:
        merged = native_dll.merge_two_u64_dedup(
            _as_void_p(prepared[0]),
            prepared[0].size,
            _as_void_p(prepared[1]),
            prepared[1].size,
            _as_void_p(result),
        )
    else:
        _, ptrs, lengths, _ = _build_native_merge_tree_inputs(prepared)
        scratch = np.empty(length, dtype=np.uint64)
        merged = native_dll.merge_tree_u64_dedup(
            ptrs,
            lengths,
            len(prepared),
            result.ctypes.data_as(uint64_ptr),
            scratch.ctypes.data_as(uint64_ptr),
        )
    return result[:merged]


def merge_tree_deduplicate_all_native(
    arrays: List[NDArray[np.uint64]], length: int = 0
) -> NDArray[np.uint64]:
    return _merge_deduplicate_all_native(arrays, length)


def _compute_merge_split_positions(
    arrays: List[NDArray], pivots, n_threads: int
) -> NDArray[np.int64]:
    num_arrays = len(arrays)
    split_positions = np.zeros((num_arrays, n_threads + 1), dtype=np.int64)
    for a in range(num_arrays):
        for t in range(n_threads - 1):
            split_positions[a][t + 1] = binary_search(arrays[a], pivots[t])
        split_positions[a][n_threads] = len(arrays[a])
    return split_positions


def _merge_deduplicate_all_partitions_native(
    arrays: List[NDArray[np.uint64]], pivots, n_threads: int
) -> list[NDArray[np.uint64]]:
    if native_dll is None:
        return _merge_deduplicate_all_partitions_numba(arrays, pivots, n_threads)
    if n_threads <= 0:
        return []
    if len(arrays) == 0:
        return [np.empty(0, dtype=np.uint64) for _ in range(n_threads)]

    prepared = _prepare_merge_tree_inputs(arrays)
    total_length = sum(arr.size for arr in prepared)
    result = np.empty(total_length, dtype=np.uint64)
    offsets = np.empty(n_threads + 1, dtype=np.uintp)
    sizes = np.empty(n_threads, dtype=np.uintp)

    pivots_array = np.ascontiguousarray(
        np.asarray(pivots[: max(0, n_threads - 1)], dtype=np.uint64)
    )
    uint64_ptr = POINTER(c_uint64)
    pivot_ptr = (
        pivots_array.ctypes.data_as(uint64_ptr)
        if pivots_array.size > 0
        else uint64_ptr()
    )

    if len(prepared) == 2:
        native_dll.merge_two_u64_partitioned_dedup(
            _as_void_p(prepared[0]),
            prepared[0].size,
            _as_void_p(prepared[1]),
            prepared[1].size,
            _as_void_p(pivots_array if pivots_array.size > 0 else None),
            n_threads,
            _as_void_p(result),
            _as_void_p(offsets),
            _as_void_p(sizes),
        )
    else:
        _, ptrs, lengths, _ = _build_native_merge_tree_inputs(prepared)
        scratch = np.empty(total_length, dtype=np.uint64)
        native_dll.merge_tree_partitioned_u64_dedup(
            ptrs,
            lengths,
            len(prepared),
            pivot_ptr,
            n_threads,
            result.ctypes.data_as(uint64_ptr),
            offsets.ctypes.data_as(POINTER(c_size_t)),
            sizes.ctypes.data_as(POINTER(c_size_t)),
            scratch.ctypes.data_as(uint64_ptr),
        )

    return [result[int(offsets[t]) : int(offsets[t] + sizes[t])] for t in range(n_threads)]


def _merge_deduplicate_all_native_partitioned_concat(
    arrays: List[NDArray[np.uint64]], pivots, n_threads: int
) -> NDArray[np.uint64]:
    if native_dll is None:
        return concatenate(_merge_deduplicate_all_partitions_numba(arrays, pivots, n_threads))
    if n_threads <= 0:
        return np.empty(0, dtype=np.uint64)
    if len(arrays) == 0:
        return np.empty(0, dtype=np.uint64)

    prepared = _prepare_merge_tree_inputs(arrays)
    total_length = sum(arr.size for arr in prepared)
    result = np.empty(total_length, dtype=np.uint64)
    pivots_array = np.ascontiguousarray(
        np.asarray(pivots[: max(0, n_threads - 1)], dtype=np.uint64)
    )

    offsets = np.empty(n_threads + 1, dtype=np.uintp)
    sizes = np.empty(n_threads, dtype=np.uintp)
    uint64_ptr = POINTER(c_uint64)

    if len(prepared) == 2:
        merged = native_dll.merge_two_u64_partitioned_dedup(
            _as_void_p(prepared[0]),
            prepared[0].size,
            _as_void_p(prepared[1]),
            prepared[1].size,
            _as_void_p(pivots_array if pivots_array.size > 0 else None),
            n_threads,
            _as_void_p(result),
            _as_void_p(offsets),
            _as_void_p(sizes),
        )
    else:
        _, ptrs, lengths, _ = _build_native_merge_tree_inputs(prepared)
        scratch = np.empty(total_length, dtype=np.uint64)
        pivot_ptr = (
            pivots_array.ctypes.data_as(uint64_ptr)
            if pivots_array.size > 0
            else uint64_ptr()
        )
        merged = native_dll.merge_tree_partitioned_u64_dedup(
            ptrs,
            lengths,
            len(prepared),
            pivot_ptr,
            n_threads,
            result.ctypes.data_as(uint64_ptr),
            offsets.ctypes.data_as(POINTER(c_size_t)),
            sizes.ctypes.data_as(POINTER(c_size_t)),
            scratch.ctypes.data_as(uint64_ptr),
        )

    return result[:merged]


@njit(nogil=True, cache=True)
def binary_search(arr, x):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left


def _merge_deduplicate_all_partitions_numba(
    arrays: List[NDArray], pivots, n_threads: int
) -> list[NDArray]:
    split_positions = _compute_merge_split_positions(arrays, pivots, n_threads)
    return _merge_deduplicate_all_partitions_numba_impl(
        arrays, split_positions, n_threads
    )


@njit(parallel=True, nogil=True)
def _merge_deduplicate_all_partitions_numba_impl(arrays, split_positions, n_threads):
    num_arrays = len(arrays)
    res = [np.empty(0, dtype="uint64") for _ in range(n_threads)]
    for t in prange(n_threads):
        temp_arrays = [np.empty(0, dtype="uint64") for _ in range(num_arrays)]
        for a in range(num_arrays):
            s = split_positions[a][t]
            e = split_positions[a][t + 1]
            temp_arrays[a] = arrays[a][s:e]
        res[t] = _merge_deduplicate_all_numba(temp_arrays)
    return res


def merge_deduplicate_partitions(
    arrays: List[NDArray], pivots, n_threads: int | None = None
) -> list[NDArray]:
    n_threads = (os.cpu_count() or 2) if n_threads is None else max(1, int(n_threads))
    if len(pivots) < max(0, n_threads - 1):
        raise ValueError("pivots length must be at least n_threads - 1")
    if native_dll is not None:
        return _merge_deduplicate_all_partitions_native(arrays, pivots, n_threads)
    return _merge_deduplicate_all_partitions_numba(arrays, pivots, n_threads)


def merge_deduplicate_all(
    arrays: List[NDArray], pivots, n_threads: int | None = None
) -> NDArray[np.uint64]:
    n_threads = (os.cpu_count() or 2) if n_threads is None else max(1, int(n_threads))
    if len(pivots) < max(0, n_threads - 1):
        raise ValueError("pivots length must be at least n_threads - 1")
    if len(arrays) == 2 and native_dll is not None:
        return _merge_deduplicate_all_native_partitioned_concat(arrays, pivots, n_threads)
    return concatenate(_merge_deduplicate_all_partitions_numba(arrays, pivots, n_threads))


@njit(parallel=True, nogil=True)
def concatenate(arrays):
    length = 0
    for array in arrays:
        length += len(array)
    res = np.empty(length, dtype="uint64")
    offset = 0
    for array in arrays:
        res[offset : len(array) + offset] = array
        offset += len(array)
    return res


def sort_array(arr: NDArray[np.uint64]) -> None:
    # 小数组直接调用numpy sort
    if len(arr) < 1e4 or native_dll is None:
        arr.sort()
    else:
        arr_ptr = arr.ctypes.data_as(POINTER(c_uint64))
        native_dll.sort_uint64(arr_ptr, arr.size, False)


@njit(nogil=True, cache=True)
def hash_(board: np.uint64) -> np.uint64:
    board = np.uint64((board ^ (board >> 27)) * 0x1A85EC53 + (board >> 23) + board)
    return np.uint64((board ^ (board >> 27)) * 0x1A85EC53 + (board >> 23) + board)


@njit(nogil=True, parallel=True, cache=True)
def is_sorted(arr: NDArray) -> np.bool:
    return np.all(arr[:-1] < arr[1:])


@njit(nogil=True, parallel=True, cache=True)
def is_sorted2(arr: NDArray) -> np.bool:
    return np.all(arr[:-1] <= arr[1:])


def check_sorted(arr):
    try:
        if is_sorted(arr):
            return arr
        elif is_sorted2(arr):
            logger.warning("sorted but not deduplicated")
            return parallel_unique(arr, os.cpu_count() or 2)
        raise ValueError("not sorted")
    except ValueError:
        arr = np.unique(arr)
        return arr


@njit(nogil=True, cache=True)
def merge_and_deduplicate(sorted_arr1: NDArray, sorted_arr2: NDArray) -> NDArray:
    # 结果数组的长度最多与两数组之和一样长
    unique_array = np.empty(len(sorted_arr1) + len(sorted_arr2), dtype=np.uint64)

    i, j, k = 0, 0, 0  # i, j 分别是两数组的索引，k 是结果数组的索引

    while i < len(sorted_arr1) and j < len(sorted_arr2):
        if sorted_arr1[i] < sorted_arr2[j]:
            if (
                k == 0 or unique_array[k - 1] != sorted_arr1[i]
            ):  # 添加新元素，并确保不重复
                unique_array[k] = sorted_arr1[i]
                k += 1
            i += 1
        elif sorted_arr1[i] > sorted_arr2[j]:
            if (
                k == 0 or unique_array[k - 1] != sorted_arr2[j]
            ):  # 添加新元素，并确保不重复
                unique_array[k] = sorted_arr2[j]
                k += 1
            j += 1
        else:  # sorted_arr1[i] == sorted_arr2[j]
            if (
                k == 0 or unique_array[k - 1] != sorted_arr1[i]
            ):  # 添加新元素，并确保不重复
                unique_array[k] = sorted_arr1[i]
                k += 1
            i += 1
            j += 1

    # 处理剩余的元素
    if i < len(sorted_arr1):
        unique_array[k : k + len(sorted_arr1) - i] = sorted_arr1[i:]
        k += len(sorted_arr1) - i
    if j < len(sorted_arr2):
        unique_array[k : k + len(sorted_arr2) - j] = sorted_arr2[j:]
        k += len(sorted_arr2) - j

    return unique_array[:k]  # 调整数组大小以匹配实际元素数


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def merge_inplace(
    arr: NDArray, segment_ends: NDArray, segment_starts: NDArray
) -> NDArray:
    """
    将每一段有效数据依次合并到前一段末尾。
    如果读写区间产生重叠，则仅移动非重叠部分。

    参数:
    - arr: 需要合并的原始数组。
    - segment_ends: 每个数据段的结束索引。
    - n: 数组的总长度（可选，根据需要使用）。
    - segment_starts: 每个数据段的起始索引。

    返回:
    - 合并后的数组切片。
    """
    counts = np.uint64(segment_ends[0])
    num_segments = len(segment_starts)

    for i in range(1, num_segments):
        start = segment_starts[i]
        end = segment_ends[i]
        size = np.uint64(end - start)
        dest_start = counts
        dest_end = np.uint64(counts + size)

        # 检查读写区间是否重叠
        if start < dest_end:
            arr[dest_start:start] = arr[dest_end:end]
        else:
            arr[dest_start:dest_end] = arr[start:end]
        counts += size

    return arr[:counts]
