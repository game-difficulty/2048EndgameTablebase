import os
import struct
from typing import Callable, Dict, Tuple, Union, List, Optional

import numpy as np
from numpy.typing import NDArray

from BoardMover import SingletonBoardMover, BoardMoverWithScore
from Config import SingletonConfig, formation_info
from Calculator import re_self

PatternCheckFunc = Callable[[np.uint64], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]


class BookReader:
    bm: BoardMoverWithScore = SingletonBoardMover(2)
    last_operation = ('none', 'none', lambda x: x)

    @staticmethod
    def move_on_dic(board: np.typing.NDArray, pattern: str, target: str, pattern_full: str
                    ) -> Dict[str, Union[str, float, int]]:
        bm = BookReader.bm
        nums_adjust, pattern_check_func, to_find_func, base_pattern, _ = \
            formation_info.get(pattern, [0, None, re_self, None, None])
        path_list = SingletonConfig().config['filepath_map'].get(pattern_full, [])
        if not path_list or not pattern_check_func:
            return {'?': '?'}

        pattern_encoded = get_pattern_encoded(bm, target, base_pattern)
        pattern_sum = np.sum(bm.decode_board(base_pattern) * int(target)) // 8
        nums = (board.sum() + nums_adjust - pattern_sum) // 2

        if nums < 0:
            return {'down': '', 'right': '', 'left': '', 'up': ''}
        sorted_results = {'down': '', 'right': '', 'left': '', 'up': ''}

        for path in path_list:
            if not os.path.exists(path):
                continue

            for operation in [BookReader.last_operation] + BookReader.gen_all_mirror(pattern):
                rotation, flip, operation_func = operation
                t_board = operation_func(board)
                encoded = np.uint64(bm.encode_board(t_board))
                if pattern_check_func(encoded, pattern_encoded):
                    BookReader.last_operation = operation
                    results = BookReader.get_best_move(path, f'{pattern_full}_{int(nums)}.book', encoded,
                                                       pattern_check_func, bm, to_find_func, pattern_encoded, target)
                    adjusted = {BookReader.adjust_direction(flip, rotation, direction): success_rate
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
        return operations

    @staticmethod
    def get_best_move(pathname: str, filename: str, board: np.uint64, pattern_check_func: PatternCheckFunc,
                      bm: BoardMoverWithScore, to_find_func: ToFindFunc, pattern_encoded: np.uint64, target: str
                      ) -> Dict[str, Optional[float]]:
        result = {'down': None, 'right': None, 'left': None, 'up': None}

        for (newt, new_score), d in zip(bm.move_all_dir(board), ('left', 'right', 'up', 'down')):
            newt = np.uint64(newt)
            if newt != board and pattern_check_func(newt, pattern_encoded) and new_score < int(target):
                result[d] = BookReader.find_value(pathname, filename, to_find_func(newt))
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
    def find_value(pathname: str, filename: str, search_key: np.uint64) -> Union[int, float, str, None]:
        search_key = np.uint64(search_key)
        fullpath = os.path.join(pathname, filename)
        if os.path.exists(fullpath):
            return BookReader.find_value_in_binary(pathname, filename, search_key)

    @staticmethod
    def find_value_in_binary(pathname: str, filename: str, search_key: np.uint64) -> Union[int, float, str]:
        """
        从二进制文件中读取数据，并根据给定的键查找对应的值。
        """
        if not os.path.exists(os.path.join(pathname, filename)):
            return '?'
        with open(os.path.join(pathname, filename), 'rb') as f:
            record_size = struct.calcsize('QI')
            f.seek(0, 2)
            file_size = f.tell()
            num_records = file_size // record_size
            # 二分查找
            left, right = 0, num_records - 1
            while left <= right:
                mid = (left + right) // 2
                f.seek(mid * record_size)
                key, value = struct.unpack('QI', f.read(record_size))
                if np.uint64(key) == search_key:
                    return value / 4e9
                elif np.uint64(key) < search_key:
                    left = mid + 1
                else:
                    right = mid - 1
        # 没有找到局面
        return 0

    @staticmethod
    def get_random_state(path_list: list, pattern_full: str) -> np.uint64:
        for path in path_list:
            book_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            while len(book_index) > 0:
                book_id = np.random.choice(book_index)
                book_index.remove(book_id)
                filepath = os.path.join(path, pattern_full + f'_{book_id}.book')
                if not os.path.exists(filepath):
                    continue
                with open(filepath, 'rb') as file:
                    record_size = struct.calcsize('QI')
                    file.seek(0, 2)
                    file_size = file.tell()
                    num_records = file_size // record_size
                    random_record_index = np.random.randint(0, num_records)
                    offset = random_record_index * record_size
                    file.seek(offset)
                    state = struct.unpack('QI', file.read(record_size))[0]
                    return np.uint64(BookReader.bm.gen_new_num(np.uint64(state),
                                                               SingletonConfig().config['4_spawn_rate'])[0])
        return np.uint64(0)


def get_pattern_encoded(bm, target: str, base_pattern: np.uint64):
    base_pattern = bm.decode_board(base_pattern)
    pattern = base_pattern * np.uint32(int(target) / 8)
    return np.uint64(bm.encode_board(pattern))


if __name__ == "__main__":
    _result = BookReader.move_on_dic(np.array([[0, 0, 0, 8],
                                               [0, 0, 4, 4],
                                               [4, 128, 256, 128],
                                               [8, 256, 128, 256]]),
                                     '4411', '512', '4411_512')
    print(_result)
