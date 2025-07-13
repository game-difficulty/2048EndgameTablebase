import numba as nb
import numpy as np
from numba import njit
from numba.core import types, cgutils


@nb.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """ returns a void pointer from a given memory address """
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen


@njit(nogil=True)
def cast_arr(addr: int, shape: int | tuple, dtype):
    arr = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)
    return arr


def dict_to_structured_array1(data_dict):
    """
    将字典转换为结构化数组，键为[-16, 31]之间的int8整数，适用于book_dict ind_dict indind_dict
        包含以下字段:
        - addr: 数组内存地址 (uint64)
        - total_len: 数组总元素数 (uint64)
        - rows: 行数 (uint64)
        - cols: 列数 (uint64)
    """
    dtype = np.dtype([
        ('addr', 'u8'),  # uint64
        ('total_len', 'u8'),  # uint64
        ('rows', 'u8'),  # uint64
        ('cols', 'u8')  # uint64
    ])

    structured_arr = np.zeros(48, dtype=dtype)
    for i in range(-16, 32):
        idx = np.uint64(i + 16)
        arr = data_dict.get(i, None)

        if arr is None:
            continue

        addr = arr.ctypes.data
        total_len = arr.size
        shape = arr.shape

        if arr.ndim == 1:
            rows = 1
            cols = shape[0]
        else:
            rows = shape[0]
            cols = shape[1]

        structured_arr[idx] = (
            np.uint64(addr),
            np.uint64(total_len),
            np.uint64(rows),
            np.uint64(cols)
        )

    return structured_arr


def dict_to_structured_array2(data_dict):
    """
    将字典转换为结构化数组，键为两个uint8组成的元组（值均≤15），适用于permutation_dict
    输出结构化数组固定长度为256，包含以下字段:
        - addr: 数组内存地址 (uint64)
        - rows: 行数 (uint64)
        - cols: 列数 (uint64)
    """
    # 定义结构化数组的dtype
    dtype = np.dtype([
        ('addr', 'u8'),  # uint64
        ('rows', 'u8'),  # uint64
        ('cols', 'u8')  # uint64
    ])

    structured_arr = np.zeros(256, dtype=dtype)

    for a in range(16):
        for b in range(16):
            idx = np.uint64((a << 4) + b)
            arr = data_dict.get((a, b), None)

            if arr is None:
                continue

            addr = arr.ctypes.data
            shape = arr.shape

            if arr.ndim == 1:
                rows = 1
                cols = shape[0]
            else:
                rows = shape[0]
                cols = shape[1]

            structured_arr[idx] = (
                np.uint64(addr),
                np.uint64(rows),
                np.uint64(cols)
            )

    return structured_arr


def dict_to_structured_array3(data_dict):
    """
    将字典转换为结构化数组，键为两个uint8组成的元组（<256 & <10），适用于tiles_combinations_dict
    输出结构化数组固定长度为256，包含以下字段:
        - addr: 数组内存地址 (uint64)
        - total_len: 数组总元素数 (uint64)
    """
    # 定义结构化数组的dtype
    dtype = np.dtype([
        ('addr', 'u8'),  # uint64
        ('total_len', 'u8'),  # uint64
    ])

    structured_arr = np.zeros(2560, dtype=dtype)

    for a in range(256):
        for b in range(10):
            idx = np.uint64(a + (b << 8))
            arr = data_dict.get((a, b), None)

            if arr is None:
                continue

            addr = arr.ctypes.data
            total_len = arr.size

            structured_arr[idx] = (
                np.uint64(addr),
                np.uint64(total_len),
            )

    return structured_arr


@njit(nogil=True)
def get_array_view10(structured_arr, key, dtype=np.uint64):
    """
    从结构化数组中重建原始数组视图
    """
    idx = (key + 16)
    rec = structured_arr[idx]
    addr, total_len, rows, cols = rec['addr'], rec['total_len'], rec['rows'], rec['cols']

    if addr == 0:
        return np.empty(0, dtype=dtype)

    arr_view = cast_arr(addr, total_len, dtype)

    return arr_view


@njit(nogil=True)
def get_array_view11(structured_arr, key, dtype=np.uint64):
    """
    从结构化数组中重建原始数组视图
    """
    idx = (key + 16)
    rec = structured_arr[idx]
    addr, total_len, rows, cols = rec['addr'], rec['total_len'], rec['rows'], rec['cols']

    if addr == 0:
        return np.empty((0, 0), dtype=dtype)

    arr_view = cast_arr(addr, (rows, cols), dtype)

    return arr_view


@njit(nogil=True)
def get_array_view2(structured_arr, key_a, key_b):
    """
    从结构化数组中重建原始数组视图
    """
    idx = ((key_a << 4) + key_b)
    rec = structured_arr[idx]
    addr, rows, cols = rec['addr'], rec['rows'], rec['cols']

    shape = (rows, cols)
    arr_view = cast_arr(addr, shape, np.uint8)
    return arr_view


@njit(nogil=True)
def get_array_view3(structured_arr, key_a, key_b):
    """
    从结构化数组中重建原始数组视图
    """
    idx = (key_a + (key_b << 8))
    rec = structured_arr[idx]
    addr, length = rec['addr'], rec['total_len']

    if addr == 0:
        return None

    arr_view = cast_arr(addr, length, np.uint8)
    return arr_view



