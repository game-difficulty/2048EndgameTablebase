import itertools
from typing import Callable, Tuple

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from numpy.typing import NDArray

import Config
from BookSolverADUtils import get_array_view3, get_array_view2
from Config import SingletonConfig

logger = Config.logger

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]
SymFindFunc = Callable[[np.uint64], Tuple[np.uint64, int]]

KeyType = types.UniTuple(types.uint8, 2)
ValueType1 = types.uint8[:, :]
ValueType2 = types.uint8[:]

"""
    "permutation_dict": types.DictType(KeyType, ValueType1),  # A(m,n)排列字典
    "tiles_combinations_dict": types.DictType(KeyType, ValueType2),  # 将数字和拆分为数字组成的字典
"""


from collections import namedtuple

Param = namedtuple('Param', 
                   ['small_tile_sum_limit', 'target', 'pos_fixed_32k', 'num_free_32k', 'num_fixed_32k'])

ParamType = types.NamedTuple([
    ('small_tile_sum_limit', np.uint32),
    ('target', np.uint32),
    ('pos_fixed_32k', np.uint64),
    ('num_free_32k',np.uint8),
    ('num_fixed_32k',np.uint8),
],Param)


@njit(nogil=True)
def generate_board_info_table1():
    # 初始化查找表（65536个条目）
    lookup_table = np.zeros(65536, dtype='uint16,uint8,uint16,uint8,uint8,uint8')

    # 填充查找表
    for block in range(65536):
        total_sum = 0
        count_32k = 0
        pos_bitmap = 0
        merged_count = 0
        merged_xor = 0
        pos_rank = 0

        # 处理每个4位组（每个tile）
        for i in range(4, -1, -1):
            shift = i * 4
            tile_value = (block >> shift) & 0xF

            if tile_value > 5:
                pos_bitmap |= 0xF << shift
                count_32k += 1
                if tile_value == np.uint64(0xf):
                    if merged_count == 0:
                        pos_rank += 1

                else:
                    merged_count += 1
                    merged_xor ^= tile_value
            elif tile_value > 0:
                total_sum += 1 << tile_value

        # 存储结果
        lookup_table[block][0] = total_sum
        lookup_table[block][1] = count_32k
        lookup_table[block][2] = pos_bitmap
        lookup_table[block][3] = merged_count
        lookup_table[block][4] = merged_xor
        lookup_table[block][5] = pos_rank

    return lookup_table


# 生成并缓存查找表（全局常量）
INFO_TABLE1 = generate_board_info_table1()


@njit(nogil=True)
def generate_board_info_table2():
    # 初始化查找表（65536个条目）
    lookup_table = np.zeros(65536, dtype='uint16,uint8,uint16,uint8,uint8,uint8')

    # 填充查找表
    for block in range(65536):
        total_sum = 0
        count_32k = 0
        pos_bitmap = 0
        merged_count = 0
        merged_xor = 0
        pos_rank = 0

        # 处理每个4位组（每个tile）
        for i in range(4, -1, -1):
            shift = i * 4
            tile_value = (block >> shift) & 0xF

            if tile_value == 0xf:
                pos_bitmap |= 0xF << shift
                count_32k += 1
                if tile_value == np.uint64(0xf):
                    if merged_count == 0:
                        pos_rank += 1

                else:
                    merged_count += 1
                    merged_xor ^= tile_value
            elif tile_value > 0:
                total_sum += 1 << tile_value

        # 存储结果
        lookup_table[block][0] = total_sum
        lookup_table[block][1] = count_32k
        lookup_table[block][2] = pos_bitmap
        lookup_table[block][3] = merged_count
        lookup_table[block][4] = merged_xor
        lookup_table[block][5] = pos_rank

    return lookup_table


# 生成并缓存查找表（全局常量）
INFO_TABLE2 = generate_board_info_table2()


def _fixed_32k_mask(pos_fixed_32k):
    mask = np.uint64(0)
    for i in pos_fixed_32k:
        mask += np.uint64(0xf) << i
    return mask


@njit(nogil=True)
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



@njit(nogil=True)
def tile_sum_and_32k_count(masked_board: np.uint64, param:ParamType) -> Tuple[np.uint32, np.int8, np.uint64]:
    """
    统计一个masked board中非32k的数字和、可移动32k数量和位置
    """
    masked_board &= (~param.pos_fixed_32k)
    # 拆分64位棋盘为4个16位块
    block3 = (masked_board >> np.uint64(48)) & np.uint64(0xFFFF)
    block2 = (masked_board >> np.uint64(32)) & np.uint64(0xFFFF)
    block1 = (masked_board >> np.uint64(16)) & np.uint64(0xFFFF)
    block0 = masked_board & np.uint64(0xFFFF)

    r0 = INFO_TABLE2[block0]
    r1 = INFO_TABLE2[block1]
    r2 = INFO_TABLE2[block2]
    r3 = INFO_TABLE2[block3]

    total_sum = r0[0] + r1[0] + r2[0] + r3[0]
    count_32k = r0[1] + r1[1] + r2[1] + r3[1]
    pos_bitmap = (np.uint64(r0[2]) |
                  (np.uint64(r1[2]) << 16) |
                  (np.uint64(r2[2]) << 32) |
                  (np.uint64(r3[2]) << 48))

    return np.uint32(total_sum), np.int8(count_32k), np.uint64(pos_bitmap)


@njit(nogil=True)
def tile_sum_and_32k_count2(masked_board: np.uint64, param:ParamType) -> Tuple[np.uint32, np.int8, np.uint64]:
    """
    统计一个masked board中小数数字和、可移动大数数量和位置
    """
    masked_board &= (~param.pos_fixed_32k)
    # 拆分64位棋盘为4个16位块
    block3 = (masked_board >> 48) & 0xFFFF
    block2 = (masked_board >> 32) & 0xFFFF
    block1 = (masked_board >> 16) & 0xFFFF
    block0 = masked_board & 0xFFFF

    r0 = INFO_TABLE1[block0]
    r1 = INFO_TABLE1[block1]
    r2 = INFO_TABLE1[block2]
    r3 = INFO_TABLE1[block3]

    total_sum = r0[0] + r1[0] + r2[0] + r3[0]
    count_32k = r0[1] + r1[1] + r2[1] + r3[1]
    pos_bitmap = (np.uint64(r0[2]) |
                  (np.uint64(r1[2]) << 16) |
                  (np.uint64(r2[2]) << 32) |
                  (np.uint64(r3[2]) << 48))

    return np.uint32(total_sum), np.int8(count_32k), np.uint64(pos_bitmap)


@njit(nogil=True)
def tile_sum_and_32k_count3(masked_board: np.uint64, param:ParamType
                            ) -> Tuple[np.uint32, np.int8, NDArray, np.uint64, np.uint64]:
    """
    仅用于生成阶段derive，需要判断不存在两个相同大数情况是否是3个64合并为128+64
    统计一个masked board中小数数字和、可移动大数数量和位置，同时如上述问题为是，则将64 mask掉，返回masked_board
    """
    total_sum = np.uint32(0)
    count_32k = np.int8(0)
    pos_32k = np.empty(16, dtype=np.uint64)
    tile64_count = 0
    tile64_pos = 0
    for i in range(60, -4, -4):
        tile_value = (masked_board >> np.uint64(i)) & np.uint64(0xf)
        if tile_value > 5:
            if (param.pos_fixed_32k >> i) & np.uint64(0xf) == 0:
                pos_32k[count_32k] = i
                count_32k += 1
                if tile_value == 6:
                    tile64_count += 1
                    tile64_pos = i
        elif tile_value > 0:
            total_sum += 1 << tile_value
    if tile64_count == 1:
        # 证明由3个64合并而来
        masked_board |= (np.uint64(0xF) << tile64_pos)
    return total_sum, count_32k, pos_32k[:count_32k], masked_board, tile64_count


@njit(nogil=True)
def tile_sum_and_32k_count4(board: np.uint64, param:ParamType
                            ) -> Tuple[np.uint32, np.int8, np.uint64, np.uint8, np.uint8, np.uint8, bool]:
    """
    统计一个board中小数数字和、可移动大数数量和位置，同时专门统计合成数字位置和个数
    """
    board &= (~param.pos_fixed_32k)
    # 拆分64位棋盘为4个16位块
    block3 = (board >> 48) & 0xFFFF
    block2 = (board >> 32) & 0xFFFF
    block1 = (board >> 16) & 0xFFFF
    block0 = board & 0xFFFF

    r0 = INFO_TABLE1[block0]
    r1 = INFO_TABLE1[block1]
    r2 = INFO_TABLE1[block2]
    r3 = INFO_TABLE1[block3]

    total_sum = r0[0] + r1[0] + r2[0] + r3[0]
    count_32k = r0[1] + r1[1] + r2[1] + r3[1]
    pos_bitmap = (np.uint64(r0[2]) |
                  (np.uint64(r1[2]) << 16) |
                  (np.uint64(r2[2]) << 32) |
                  (np.uint64(r3[2]) << 48))
    merged_tile_found = r0[3] + r1[3] + r2[3] + r3[3]
    merged_tile = r0[4] ^ r1[4] ^ r2[4] ^ r3[4]


    s3 = r3[5]
    s2 = s3 + r2[5]
    s1 = s2 + r1[5]
    s0 = s1 + r0[5]

    f3 = (r3[3] != 0)
    f2 = (r3[3] == 0) & (r2[3] != 0)
    f1 = (r3[3] == 0) & (r2[3] == 0) & (r1[3] != 0)
    f0 = (r3[3] == 0) & (r2[3] == 0) & (r1[3] == 0)

    pos_rank = s0 * f0 + s1 * f1 + s2 * f2 + s3 * f3

    is_success = (merged_tile == param.target) & (merged_tile_found == 1)

    return np.uint32(total_sum), np.int8(count_32k), np.uint64(pos_bitmap), np.uint8(pos_rank), np.uint8(merged_tile_found), np.uint8(merged_tile), is_success


@njit(nogil=True)
def _masked_tiles_combinations(remaining_sum: np.uint64, remaining_count: int
                               ) -> NDArray[np.uint8] | None:
    """
    输入masked board中非32k的数字和和32k数量，输出被masked的格子的数字组成
    """
    tiles = np.array([8192, 4096, 2048, 1024, 512, 256, 128, 64])
    result = np.empty(10, dtype=np.uint8)
    index = 0
    for tile in tiles:
        if tile > remaining_sum:
            continue
        if tile == remaining_sum:
            if remaining_count == 1:
                result[index] = np.log2(tile)
                return result[:index + 1].copy()
        else:  # tile < remaining_sum:
            if (tile << 1) == remaining_sum and remaining_count == 2:
                result[index] = np.log2(tile)
                result[index + 1] = np.log2(tile)
                return result[:index + 2].copy()
            # 支持3个64
            elif tile != 64 and remaining_sum == 192 and remaining_count == 3:
                # tile != 64 确保当前result中没有128（不支持有128的情况下还有三个64）
                result[index] = 6
                result[index + 1] = 6
                result[index + 2] = 6
                return result[:index + 3].copy()
            else:  # 分配一个格子
                remaining_sum -= tile
                remaining_count -= 1
                result[index] = np.log2(tile)
                index += 1
    return None


def generate_tiles_combinations_dict() -> Dict[KeyType, ValueType2]:
    """
    combinations_dict: {(数字和//64, 数字个数): 数字组成array}
    确定数字和和个数的情况下，组成唯一
    """
    tiles_combinations_dict = Dict.empty(KeyType, ValueType2)
    for i in range(1, 256):
        for j in range(1, 10):
            tile = _masked_tiles_combinations(np.uint64(i << 6), j)
            if tile is not None:
                # 从小到大
                tiles_combinations_dict[(np.uint8(i), np.uint8(j))] = tile[::-1].astype(np.uint8)
    return tiles_combinations_dict


@njit(nogil=True)
def validate(board: np.uint64, original_board_sum: np.uint32, tiles_combinations_dict, param:ParamType) -> bool:
    """
    检验一个masked_board是否符合条件：
    1 小数字和不大于x
    2 大数组成满足--最小大数不超过2个（例外：允许3个64），其余大数不超过1个
    3 如最小大数为2个，小数字和不大于x-64
    4 如3个64，小数字和不大于x-128
    """
    total_sum, count_32k, pos_32k = tile_sum_and_32k_count2(board, param)
    if total_sum >= param.small_tile_sum_limit + 64:
        return False
    if count_32k == param.num_free_32k:
        return True
    large_tiles_sum = original_board_sum - total_sum - ((param.num_free_32k + param.num_fixed_32k) << 15)
    # assert large_tiles_sum % 64 == 0
    tiles_combinations = get_array_view3(tiles_combinations_dict, np.uint8(large_tiles_sum >> 6), np.uint8(count_32k - param.num_free_32k))  # tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6), np.uint8(count_32k - param.num_free_32k)), None)
    if tiles_combinations is None:
        return False
    if tiles_combinations[-1] == param.target:
        return False
    if count_32k < 2:
        return True
    if len(tiles_combinations) > 2 and tiles_combinations[0] == tiles_combinations[2]:
        if total_sum >= param.small_tile_sum_limit - 64:
            return False
    elif len(tiles_combinations) > 1 and tiles_combinations[0] == tiles_combinations[1]:
        if total_sum >= param.small_tile_sum_limit:
            return False
    return True


@njit(nogil=True, boundscheck=True )
def unmask_board(board: np.int64, original_board_sum: np.uint32, tiles_combinations_dict, permutation_dict, param:ParamType
                 ) -> NDArray[np.uint64]:
    """
    将tiles_combinations的排列填回masked的格子
    original_board_sum 指mask之前的所有数字和
    total_sum          指mask之后所有小数字和
    large_tiles_sum    指mask之前的所有非32k格子数字和
    """
    total_sum, count_32k, pos_32k = tile_sum_and_32k_count(board, param)
    if count_32k == 0 or count_32k == param.num_free_32k:
        return np.array([board, ], dtype=np.uint64)
    large_tiles_sum = original_board_sum - total_sum - ((param.num_free_32k + param.num_fixed_32k) << 15)
    # assert large_tiles_sum % 64 == 0

    tiles_combinations = get_array_view3(tiles_combinations_dict, np.uint8(large_tiles_sum >> 6), np.uint8(count_32k - param.num_free_32k))  # tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6),np.uint8(count_32k - param.num_free_32k)), None)

    if tiles_combinations is None:
        return np.empty(0, dtype=np.uint64)

    permutations = get_array_view2(permutation_dict, np.uint8(count_32k), np.uint8(count_32k - param.num_free_32k))  # permutation_dict[(np.uint8(count_32k), np.uint8(count_32k - param.num_free_32k))]
    if len(tiles_combinations) > 1 and tiles_combinations[0] == tiles_combinations[1]:
        permutations = permutations[permutations[:, 1] < permutations[:, 0]]

    unmask_boards = np.empty(len(permutations), dtype=np.uint64)
    pos_32k = extract_f_positions(pos_32k)
    for ind, permutation in enumerate(permutations):
        masked = board
        # pos_32k, permutation, tiles_combinations 长度都等于 count_32k - num_free_32k
        indices = pos_32k[permutation]
        clear_mask = ~(np.sum(np.uint64(0xf) << indices))
        set_mask = np.sum(tiles_combinations << indices)
        masked = (masked & clear_mask) | set_mask
        unmask_boards[ind] = masked
    return unmask_boards


@njit(nogil=True)
def extract_f_positions(pos_bitmap: np.uint64) -> NDArray:
    result = np.empty(16, dtype=np.uint64)
    count = 0
    for i in range(60, -4, -4):
        nibble = (pos_bitmap >> np.uint64(i)) & np.uint64(0xF)
        if nibble == 0xF:
            result[count] = i
            count += 1
    return result[:count]


def generate_permutations(m: int, n: int) -> NDArray[NDArray[np.uint8]]:
    """
    A(m,n)排列
    """
    result1 = np.array(list(itertools.permutations(range(m), n)), dtype=np.uint8)
    result1 = _resort_permutations(m, n, result1, True)
    return result1


@njit(nogil=True)
def _resort_permutations(m, n, permutation, type_flag):
    """
    保证unmask后的boards是有序的（但permutation不再是字典序）
    """
    index = np.empty(len(permutation), dtype=np.uint64)
    pos = (np.arange(m)[::-1] * 4).astype(np.uint8)
    if type_flag:
        tiles = np.arange(n, dtype=np.uint64)
    else:
        tiles = np.array([0] + [i for i in range(n - 1)], dtype=np.uint64)
    for i in range(len(permutation)):
        p = permutation[i]
        indices = pos[p]
        clear_mask = ~(np.sum(np.uint64(0xf) << indices))
        set_mask = np.sum(tiles << indices)
        masked = clear_mask | set_mask
        index[i] = masked
    sorted_indices = np.argsort(index)
    return permutation[sorted_indices]


def init_masker(num_free_32k: int, target_num: int, pos_fixed_32k: NDArray[np.uint8]):
    """
    修改全局变量，计算permutation_dict, tiles_combinations_dict
    :param num_free_32k: 可以移动的32k个数
    :param target_num: 目标数字对数值
    :param pos_fixed_32k: 不可移动的32k的位置
    :return: 初始化的字典
    """
    param = Param(
        small_tile_sum_limit=SingletonConfig().config.get('SmallTileSumLimit', 56),
        target = np.uint8(target_num),
        pos_fixed_32k=_fixed_32k_mask(pos_fixed_32k),
        num_free_32k = np.uint8(num_free_32k),
        num_fixed_32k = np.uint8(len(pos_fixed_32k)),
        )

    permutation_dict = Dict.empty(KeyType, ValueType1)
    for n in range(1, target_num - 4):
        m = n + num_free_32k
        permutation_dict[(np.uint8(m), np.uint8(n))] = generate_permutations(m, n)
    tiles_combinations_dict = generate_tiles_combinations_dict()

    return permutation_dict, tiles_combinations_dict, param
