import os
import struct
from typing import Callable, Dict, Tuple, Union, List, Optional

import numpy as np
from numpy.typing import NDArray

import Config
import Variants.vBoardMover as vbm
import BoardMover as bm
from BookSolverADUtils import dict_to_structured_array2, dict_to_structured_array3
from Config import SingletonConfig, formation_info, pattern_32k_tiles_map
import Calculator
from Calculator import re_self
from BoardMaskerAD import init_masker, mask_board, tile_sum_and_32k_count2, unmask_board
from BookSolverAD import sym_like

PatternCheckFunc = Callable[[np.uint64], bool]
SymFindFunc = Callable[[np.uint64], Tuple[np.uint64, int]]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]

logger = Config.logger


class BookReaderAD:
    last_operation = ('none', 'none', lambda x: x)

    def __new__(cls, pattern: str, target: int):
        # 检查 pattern 是否在 pattern_32k_tiles_map 中，是则创建新实例
        if pattern in pattern_32k_tiles_map.keys():
            return super(BookReaderAD, cls).__new__(cls)
        else:
            return None

    def __init__(self, pattern: str, target: int):
        self.pattern = pattern
        self.target = target
        _, num_free_32k, pos_fixed_32k = pattern_32k_tiles_map[pattern]
        self._permutation_dict, self._tiles_combinations_dict, self.param = init_masker(num_free_32k, target, pos_fixed_32k)

        self.permutation_dict = dict_to_structured_array2(self._permutation_dict)
        self.tiles_combinations_dict = dict_to_structured_array3(self._tiles_combinations_dict)

    def move_on_dic(self, board: np.typing.NDArray, pattern_full: str, pos: str = '0'
                    ) -> Dict[str, Union[str, float, int, None]]:
        bm_ = bm if self.pattern not in ('2x4', '3x3', '3x4') else vbm
        nums_adjust, pattern_check_func, to_find_func, success_check_func, _ = \
            formation_info.get(self.pattern, [0, None, re_self, None, None])
        path_list = SingletonConfig().config['filepath_map'].get(pattern_full, [])
        nums = (board.sum() + nums_adjust) // 2
        if self.pattern[:4] == 'free' and self.pattern[-1] != 'w':
            nums -= int(self.target) / 2
        if self.pattern == 'LL' and int(pos) == 1:
            to_find_func = re_self
        if nums < 0:
            return {'down': '', 'right': '', 'left': '', 'up': ''}
        if not pattern_check_func or not path_list:
            return {'?': '?'}
        sorted_results = {'down': '', 'right': '', 'left': '', 'up': ''}
        for path in path_list:
            if not os.path.exists(path):
                continue

            sym_func = {Calculator.re_self: Calculator.re_self_pair,
                        Calculator.minUL: Calculator.minUL_pair,
                        Calculator.min_all_symm: Calculator.min_all_symm_pair}[to_find_func]

            for operation in [self.last_operation] + self.gen_all_mirror(self.pattern):
                rotation, flip, operation_func = operation
                t_board = operation_func(board)
                encoded = np.uint64(bm_.encode_board(t_board))
                if pattern_check_func(encoded):
                    self.last_operation = operation
                    results = self.get_best_move(path, f'{pattern_full}_{int(nums)}b', encoded,
                                                 pattern_check_func, bm_, sym_func)
                    adjusted = {self.adjust_direction(flip, rotation, direction): success_rate
                                for direction, success_rate in results.items()}
                    float_items = {k: round(v, 10) for k, v in adjusted.items() if isinstance(v, (int, float))}
                    non_float_items = {k: v for k, v in adjusted.items() if not isinstance(v, (int, float))}
                    sorted_float_items = dict(sorted(float_items.items(), key=lambda item: item[1], reverse=True))
                    sorted_results = {**sorted_float_items, **non_float_items}
                    if float_items:
                        return sorted_results

        return sorted_results

    @staticmethod
    def gen_all_mirror(pattern: str) -> List[Tuple[str, str, Callable[[np.ndarray], np.ndarray]]]:
        if pattern in ('2x4', '3x3', '3x4'):
            return [('none', 'none', lambda x: x)]
        operations = [
            ('none', 'none', lambda x: x),
            ('rotate_90', 'none', lambda x: np.rot90(x)),
            ('rotate_180', 'none', lambda x: np.rot90(x, k=2)),
            ('rotate_270', 'none', lambda x: np.rot90(x, k=3)),
            ('none', 'horizontal', lambda x: np.flip(x, axis=1)),
            ('rotate_90', 'horizontal', lambda x: np.flip(np.rot90(x), axis=1)),
            ('rotate_180', 'horizontal', lambda x: np.flip(np.rot90(x, k=2), axis=1)),
            ('rotate_270', 'horizontal', lambda x: np.flip(np.rot90(x, k=3), axis=1)),
        ]
        return operations if pattern != 'LL' else operations[:4]

    def get_best_move(self, pathname: str, filename: str, board: np.uint64, pattern_check_func: PatternCheckFunc,
                      bm_, sym_func: SymFindFunc) -> Dict[str, Optional[float]]:
        result = {'down': None, 'right': None, 'left': None, 'up': None}
        for newt, d in zip(bm_.move_all_dir(board), ('left', 'right', 'up', 'down')):
            newt = np.uint64(newt)
            if newt != board and pattern_check_func(newt):
                result[d] = self.find_value(pathname, filename, newt, sym_func)
        return result

    @staticmethod
    def adjust_direction(flip: str, rotation: str, direction: str) -> str:
        if flip == 'horizontal':
            if direction == 'left':
                direction = 'right'
            elif direction == 'right':
                direction = 'left'

        direction_map = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
        inverse_map = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}

        # 处理旋转的逆过程
        rotation_steps = {
            'none': 0,
            'rotate_90': 1,  # 顺时针旋转90度的逆操作是逆时针旋转90度
            'rotate_180': 2,  # 旋转180度的逆操作还是旋转180度
            'rotate_270': 3,  # 顺时针旋转270度的逆操作是逆时针旋转90度，或顺时针旋转90度
        }
        direction_index = direction_map[direction]
        direction_index = (direction_index + rotation_steps[rotation]) % 4
        direction = inverse_map[direction_index]

        return direction

    @staticmethod
    def sum_board(board: np.uint64) -> int:
        total_sum = 0
        for k in range(16):
            encoded_num = (board >> (4 * k)) & np.uint64(0xF)
            total_sum += 2 ** encoded_num if encoded_num > 0 else 0
        return total_sum

    def find_value(self, pathname: str, filename: str, board: np.uint64, sym_func: SymFindFunc
                   ) -> Union[int, float, str, None]:
        total_sum, count_32k, pos_32k = tile_sum_and_32k_count2(board, self.param)
        original_board_sum = self.sum_board(board)
        large_tiles_sum = original_board_sum - total_sum - ((self.param.num_free_32k + self.param.num_fixed_32k) << 15)
        tiles_combinations = self._tiles_combinations_dict.get((np.uint8(large_tiles_sum >> 6),
                                                                  np.uint8(count_32k - self.param.num_free_32k)), None)

        if tiles_combinations is not None:
            if len(tiles_combinations) > 2 and tiles_combinations[0] == tiles_combinations[2]:
                count_32k = count_32k - 3 + 16
                masked_board = np.uint64(mask_board(board, 7))
            elif len(tiles_combinations) > 1 and tiles_combinations[0] == tiles_combinations[1]:
                count_32k = -count_32k
                masked_board = np.uint64(mask_board(board, tiles_combinations[0] + 1))
            else:
                masked_board = np.uint64(mask_board(board, 6))
            search_key, symm_index = sym_func(masked_board)
            search_key = np.uint64(search_key)
        else:
            search_key, symm_index = sym_func(board)
            search_key = np.uint64(search_key)

        filepath = os.path.join(pathname, filename)
        index_filepath = os.path.join(filepath, str(count_32k) + '.i')
        if not os.path.exists(index_filepath):
            return '?'
        ind = self.find_value_in_binary(index_filepath, search_key)
        if ind is None:
            return 0

        book_filepath = os.path.join(filepath, str(count_32k) + '.b')
        if not os.path.exists(book_filepath):
            logger.warning(f'Files in {filepath} may be corrupted.')
            return '?'

        if tiles_combinations is not None:
            board_derived = unmask_board(search_key, np.uint32(original_board_sum), self.tiles_combinations_dict,
                                         self.permutation_dict, self.param)
            symm_board = np.uint64(sym_like(board, symm_index))

            ind2 = np.searchsorted(board_derived, symm_board)

            with open(book_filepath, 'rb') as f:
                record_size = struct.calcsize('I')
                f.seek((ind2 + ind * len(board_derived)) * record_size)
                try:
                    data = f.read(record_size)
                    return struct.unpack('I', data)[0] / 4e9
                except Exception as e:
                    logger.critical(f"Full exception: {e}")
                    logger.critical(f"Current file position: {f.tell()}")
                    logger.critical(bm.decode_board(board))
                    logger.critical(f'book filepath: {book_filepath}')
                    logger.critical('This file may be corrupted.')
                    return '?'

        else:
            with open(book_filepath, 'rb') as f:
                record_size = struct.calcsize('I')
                f.seek(ind * record_size)
                return struct.unpack('I', f.read(record_size))[0] / 4e9

    @staticmethod
    def find_value_in_binary(index_filepath: str, search_key: np.uint64) -> Union[int, None]:
        """
        从二进制文件中读取数据，并根据给定的键查找对应的值。
        """
        with open(index_filepath, 'rb') as f:
            record_size = struct.calcsize('Q')
            f.seek(0, 2)
            file_size = f.tell()
            num_records = file_size // record_size
            # 二分查找
            left, right = 0, num_records - 1
            while left <= right:
                mid = (left + right) // 2
                f.seek(mid * record_size)
                key = struct.unpack('Q', f.read(record_size))[0]
                if np.uint64(key) == search_key:
                    return mid
                elif np.uint64(key) < search_key:
                    left = mid + 1
                else:
                    right = mid - 1
        # 没有找到局面
        return None

    @staticmethod
    def get_random_state(path_list: list, pattern_full: str) -> np.uint64:
        for path in path_list:
            book_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            while len(book_index) > 0:
                book_id = np.random.choice(book_index)
                book_index.remove(book_id)
                filepath = os.path.join(path, f'{pattern_full}_{int(book_id)}b')
                if not os.path.exists(filepath):
                    continue
                files = os.listdir(filepath)
                i_files = [file for file in files if file.endswith('.i')]

                if not i_files:
                    continue
                index_filepath = np.random.choice(i_files)
                with open(os.path.join(filepath, index_filepath), 'rb') as file:
                    record_size = struct.calcsize('Q')
                    file.seek(0, 2)
                    file_size = file.tell()
                    num_records = file_size // record_size
                    random_record_index = np.random.randint(0, num_records)
                    offset = random_record_index * record_size
                    file.seek(offset)
                    state = struct.unpack('Q', file.read(record_size))[0]
                    return np.uint64(bm.gen_new_num(np.uint64(state),
                                                                 SingletonConfig().config['4_spawn_rate'])[0])
        return np.uint64(0)


if __name__ == "__main__":
    BRAD = BookReaderAD('free8w', 7, )
    _result = BRAD.move_on_dic(np.array([[0, 0, 2, 0],
                                         [64, 16, 2, 2],
                                         [32768, 32768, 32768, 32768],
                                         [32768, 32768, 32768, 32768]]),
                               'free8w_128')
    print(_result)
    print(BRAD.get_random_state([r"Q:\tables\adtest"], 'free8w_128'))
