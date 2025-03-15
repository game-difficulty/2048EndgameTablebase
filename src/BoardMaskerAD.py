import itertools
from typing import Callable, Tuple

import numpy as np
from numba import njit
from numba import types
from numba import uint8, uint32
from numba.experimental import jitclass
from numba.typed import Dict
from numpy.typing import NDArray

import Config
from Config import SingletonConfig

logger = Config.logger

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]
SymFindFunc = Callable[[np.uint64], Tuple[np.uint64, int]]

KeyType = types.UniTuple(types.uint8, 2)
ValueType1 = types.uint8[:, :]
ValueType2 = types.uint8[:]


spec = {
    "small_tile_sum_limit": uint32,
    "target": uint8,
    "num_free_32k": uint8,  # 可移动32k格子的数量
    "pos_fixed_32k": uint8[:],  # 不可移动32k格子的位置,右下角是0
    "num_fixed_32k": uint8,  # 不可移动32k格子的数量
    "permutation_dict1": types.DictType(KeyType, ValueType1),  # A(m,n)排列字典
    "tiles_combinations_dict": types.DictType(KeyType, ValueType2),  # 将数字和拆分为数字组成的字典
}


@jitclass(spec)
class BoardMasker:
    def __init__(self, small_tile_sum_limit, target, permutation_dict1, num_free_32k, pos_fixed_32k):
        self.small_tile_sum_limit = small_tile_sum_limit
        self.target = target
        self.num_free_32k = num_free_32k
        self.pos_fixed_32k = pos_fixed_32k
        self.num_fixed_32k = len(pos_fixed_32k)
        self.permutation_dict1 = permutation_dict1
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
            tile_value = (masked_board >> np.uint64(i)) & np.uint64(0xf)
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
            tile_value = (masked_board >> np.uint64(i)) & np.uint64(0xf)
            if tile_value > 5:
                if i not in self.pos_fixed_32k:
                    count_32k += 1
                    pos_32k[pos_ind] = i
                    pos_ind += 1
            elif tile_value > 0:
                total_sum += 2 ** tile_value
        return total_sum, count_32k, pos_32k[:pos_ind]

    def tile_sum_and_32k_count3(self, masked_board: np.uint64
                                ) -> Tuple[np.uint32, np.uint8, NDArray[np.uint64], np.uint64, np.uint64]:
        """
        仅用于生成阶段derive，需要判断不存在两个相同大数情况是否是3个64合并为128+64
        统计一个masked board中小数数字和、可移动大数数量和位置，同时如上述问题为是，则将64 mask掉，返回masked_board
        """
        total_sum = 0
        count_32k = 0
        pos_32k = np.empty(16, dtype=np.uint64)
        pos_ind = 0
        tile64_count = 0
        tile64_pos = 0
        for i in range(60, -4, -4):
            tile_value = (masked_board >> np.uint64(i)) & np.uint64(0xf)
            if tile_value > 5:
                if i not in self.pos_fixed_32k:
                    count_32k += 1
                    pos_32k[pos_ind] = i
                    pos_ind += 1
                    if tile_value == 6:
                        tile64_count += 1
                        tile64_pos = i
            elif tile_value > 0:
                total_sum += 2 ** tile_value
        if tile64_count == 1:
            # 证明由3个64合并而来
            masked_board |= (np.uint64(0xF) << tile64_pos)
        return total_sum, count_32k, pos_32k[:pos_ind], masked_board, tile64_count

    def tile_sum_and_32k_count4(self, board: np.uint64
                                ) -> Tuple[np.uint32, np.uint8, NDArray[np.uint64], int, int, bool]:
        """
        统计一个board中小数数字和、可移动大数数量和位置，同时专门统计合成数字位置和个数
        """
        total_sum = 0
        count_32k = 0
        pos_32k = np.empty(16, dtype=np.uint64)
        pos_ind = 0
        pos_rank = 0
        merged_tile_found = 0
        is_success = False
        for i in range(60, -4, -4):
            tile_value = (board >> np.uint64(i)) & np.uint64(0xf)
            if tile_value > 5:
                if tile_value == np.uint64(0xf):
                    if i not in self.pos_fixed_32k and merged_tile_found == 0:
                        pos_rank += 1
                elif tile_value == self.target:
                    is_success = True
                else:
                    merged_tile_found += 1

                if i not in self.pos_fixed_32k:
                    count_32k += 1
                    pos_32k[pos_ind] = i
                    pos_ind += 1
            elif tile_value > 0:
                total_sum += 2 ** tile_value
        return total_sum, count_32k, pos_32k[:pos_ind], pos_rank, merged_tile_found, is_success

    def tile_sum_and_32k_count5(self, board: np.uint64
                                ) -> Tuple[np.uint32, np.uint8, NDArray[np.uint64], int, int, bool]:
        """
        统计一个board中小数数字和、可移动大数数量和位置，同时专门找到合数位置和值
        """
        total_sum = 0
        count_32k = 0
        pos_32k = np.empty(16, dtype=np.uint64)
        pos_ind = 0
        pos_rank = 0
        tile_merged = 0
        is_success = False
        for i in range(60, -4, -4):
            tile_value = (board >> np.uint64(i)) & np.uint64(0xf)
            if tile_value > 5:
                if tile_value == np.uint64(0xf):
                    if i not in self.pos_fixed_32k and tile_merged == 0:
                        pos_rank += 1
                elif tile_value == self.target:
                    is_success = True
                else:
                    tile_merged = tile_value
                if i not in self.pos_fixed_32k:
                    count_32k += 1
                    pos_32k[pos_ind] = i
                    pos_ind += 1
            elif tile_value > 0:
                total_sum += 2 ** tile_value
        return total_sum, count_32k, pos_32k[:pos_ind], pos_rank, tile_merged, is_success

    @staticmethod
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

    def validate(self, board: np.uint64, original_board_sum: np.uint32) -> bool:
        """
        检验一个masked_board是否符合条件：
        1 小数字和不大于x
        2 大数组成满足--最小大数不超过2个（例外：允许3个64），其余大数不超过1个
        3 如最小大数为2个，小数字和不大于x-64
        4 如3个64，小数字和不大于x-128
        """
        total_sum, count_32k, pos_32k = self.tile_sum_and_32k_count2(board)
        if total_sum >= self.small_tile_sum_limit + 64:
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
        if len(tiles_combinations) > 2 and tiles_combinations[0] == tiles_combinations[2]:
            if total_sum >= self.small_tile_sum_limit - 64:
                return False
        elif len(tiles_combinations) > 1 and tiles_combinations[0] == tiles_combinations[1]:
            if total_sum >= self.small_tile_sum_limit:
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
        permutations = self.permutation_dict1[(np.uint8(count_32k), np.uint8(count_32k - self.num_free_32k))]
        if len(tiles_combinations) > 1 and tiles_combinations[0] == tiles_combinations[1]:
            permutations = permutations[permutations[:, 1] < permutations[:, 0]]
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


def init_masker(num_free_32k: int, target_num: int, pos_fixed_32k: NDArray[np.uint8]) -> BoardMasker:
    """
    :param num_free_32k: 可以移动的32k个数
    :param target_num: 目标数字对数值
    :param pos_fixed_32k: 不可移动的32k的位置
    :return: 初始化的LineMasker
    """
    small_tile_sum_limit = SingletonConfig().config.get('SmallTileSumLimit', 56)
    permutation_dict1 = Dict.empty(KeyType, ValueType1)
    for n in range(1, target_num - 4):
        m = n + num_free_32k
        permutation_dict1[(np.uint8(m), np.uint8(n))] = generate_permutations(m, n)
    return BoardMasker(small_tile_sum_limit, target_num, permutation_dict1, num_free_32k, pos_fixed_32k)
