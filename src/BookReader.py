import os
import struct
from typing import Callable, Dict, Tuple, Union, List, Optional

import numpy as np
from numpy.typing import NDArray

from BookReaderAD import BookReaderAD
from Calculator import canonical_identity
from Config import SingletonConfig, formation_info, DTYPE_CONFIG, category_info
from TrieCompressor import trie_decompress_search
import VBoardMover as vbm
import BoardMover as bm

PatternCheckFunc = Callable[[np.uint64], bool]
CanonicalFunc = Callable[[np.uint64], np.uint64]
SuccessCheckFunc = Callable[[np.uint64, int], bool]
_TYPE_MAP = {np.uint32: 'I', np.uint64: 'Q', np.float32: 'f', np.float64: 'd'}


class BookReader:
    last_operation = ('none', 'none', lambda x: x)

    @staticmethod
    def move_on_dic(board: NDArray, pattern: str, target: str, pattern_full: str
                    ) -> Tuple[Dict[str, Union[str, float, int]], str]:
        bm_ = bm if pattern not in category_info.get('variant', []) else vbm
        nums_adjust, pattern_check_func, canonical_func, success_check_func, _, _ = \
            formation_info.get(pattern, [0, None, canonical_identity, None, None, None])
        spawn_rate4 = SingletonConfig().config['4_spawn_rate']
        path_list = SingletonConfig().config['filepath_map'].get((pattern_full, spawn_rate4), [])
        nums = (board.sum() + nums_adjust) // 2

        if not path_list or not pattern_check_func:
            return {'?': '?'}, ''
        if nums < 0:
            return {'down': '', 'right': '', 'left': '', 'up': ''}, ''
        final_results = {'down': '', 'right': '', 'left': '', 'up': ''}
        max_success_rate = 0
        _success_rate_dtype = ''
        for path, success_rate_dtype in path_list:
            if not os.path.exists(path) or max_success_rate:
                continue
            precision_digits = 9 if '32' in success_rate_dtype else None

            for operation in [BookReader.last_operation] + BookReader.gen_all_mirror(pattern):
                rotation, flip, operation_func = operation
                t_board = operation_func(board)
                encoded = np.uint64(bm_.encode_board(t_board))
                if pattern_check_func(encoded):
                    results = BookReader.get_best_move(path, f'{pattern_full}_{int(nums)}.book', encoded,
                                                       pattern_check_func, bm_, canonical_func, success_rate_dtype)
                    adjusted = {BookReader.adjust_direction(flip, rotation, direction): success_rate
                                for direction, success_rate in results.items()}
                    float_items = {k: round(v, precision_digits) if (abs(v) > 1e-7 and precision_digits)
                                    else v for k, v in adjusted.items()
                                   if isinstance(v, (int, float, np.integer, np.floating))}
                    non_float_items = {k: v for k, v in adjusted.items()
                                       if not isinstance(v, (int, float, np.integer, np.floating))}
                    sorted_float_items = dict(sorted(float_items.items(), key=lambda item: item[1], reverse=True))
                    sorted_results = {**sorted_float_items, **non_float_items}
                    if pattern in ('4442ff', '4442f', '4tiler') and sorted_float_items:
                        first_value = sorted_float_items[next(iter(sorted_float_items))]
                        if first_value > max_success_rate:
                            BookReader.last_operation = operation
                            max_success_rate = first_value
                            final_results = sorted_results
                            _success_rate_dtype = success_rate_dtype
                    elif float_items:
                        BookReader.last_operation = operation
                        final_results = sorted_results
                        return final_results, success_rate_dtype

        return final_results, _success_rate_dtype

    @staticmethod
    def gen_all_mirror(pattern: str) -> List[Tuple[str, str, Callable[[np.ndarray], np.ndarray]]]:
        if pattern in category_info.get('variant', []):
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

    @staticmethod
    def get_best_move(pathname: str, filename: str, board: np.uint64, pattern_check_func: PatternCheckFunc,
                      bm_, canonical_func: CanonicalFunc, success_rate_dtype: str) -> Dict[str, Optional[float]]:
        result = {'down': None, 'right': None, 'left': None, 'up': None}
        fullpath = os.path.join(pathname, filename.replace('.book', '.z'))
        if os.path.exists(fullpath):
            path = os.path.join(fullpath, filename.replace('.book', '.'))
            ind = np.fromfile(path + 'i', dtype='uint8,uint32')
            segments = np.fromfile(path + 's', dtype='uint32,uint64')
        else:
            ind, segments = None, None

        for newt, d in zip(bm_.move_all_dir(board), ('left', 'right', 'up', 'down')):
            newt = np.uint64(newt)
            if newt != board and pattern_check_func(newt):
                result[d] = BookReader.find_value(pathname, filename, canonical_func(newt), ind, segments,
                                                  success_rate_dtype)
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
    def find_value(pathname: str, filename: str, search_key: np.uint64, ind: NDArray = None,
                   segments: NDArray = None, success_rate_dtype: str = 'uint32') -> Union[int, float, str, None]:
        search_key = np.uint64(search_key)
        fullpath = os.path.join(pathname, filename)
        if os.path.exists(fullpath):
            return BookReader.find_value_in_binary(pathname, filename, search_key, success_rate_dtype)
        elif ind is not None and segments is not None:
            path = os.path.join(fullpath.replace('.book', '.z'), filename.replace('.book', '.'))
            return trie_decompress_search(path, search_key, ind, segments, success_rate_dtype)
        elif os.path.exists(fullpath.replace('.book', '.z')):
            path = os.path.join(fullpath.replace('.book', '.z'), filename.replace('.book', '.'))
            ind = np.fromfile(path + 'i', dtype='uint8,uint32')
            segments = np.fromfile(path + 's', dtype='uint32,uint64')
            return trie_decompress_search(path, search_key, ind, segments, success_rate_dtype)
        return None

    @staticmethod
    def find_value_in_binary(pathname: str, filename: str, search_key: np.uint64, success_rate_dtype: str
                             ) -> Union[int, float, str]:
        """
        从二进制文件中读取数据，并根据给定的键查找对应的值。
        适配 uint32, uint64, float32, float64 等多种格式。
        """
        _, val_type, max_scale, zero_val = DTYPE_CONFIG.get(success_rate_dtype, DTYPE_CONFIG['uint32'])

        file_path = os.path.join(pathname, filename)
        if not os.path.exists(file_path):
            return '?'

        fmt_key = 'Q'
        fmt_val = _TYPE_MAP.get(val_type, 'I')
        fmt = fmt_key + fmt_val
        record_size = struct.calcsize(fmt)

        with open(file_path, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            num_records = file_size // record_size

            # 二分查找
            left, right = 0, num_records - 1
            while left <= right:
                mid = (left + right) // 2
                f.seek(mid * record_size)
                chunk = f.read(record_size)
                key_int, val_raw = struct.unpack(fmt, chunk)

                if np.uint64(key_int) == search_key:
                    return val_raw / max_scale if max_scale > 1.0 else val_raw

                elif key_int < search_key:
                    left = mid + 1
                else:
                    right = mid - 1

        # 没有找到局面，返回 zero_val
        return zero_val

    @staticmethod
    def get_random_state(path_list: list, pattern_full: str) -> np.uint64:
        for path, success_rate_dtype in path_list:
            book_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            _, val_type, max_scale, zero_val = DTYPE_CONFIG.get(success_rate_dtype, DTYPE_CONFIG['uint32'])
            fmt_val = _TYPE_MAP.get(val_type, 'I')
            fmt = 'Q' + fmt_val
            record_size = struct.calcsize(fmt)

            while len(book_index) > 0:
                book_id = np.random.choice(book_index)
                book_index.remove(book_id)
                filepath = os.path.join(path, pattern_full + f'_{book_id}.book')
                if not os.path.exists(filepath):
                    continue

                with open(filepath, 'rb') as file:
                    file.seek(0, 2)
                    file_size = file.tell()
                    num_records = file_size // record_size
                    if num_records == 0:
                        continue
                    random_record_index = np.random.randint(0, num_records)
                    offset = random_record_index * record_size
                    file.seek(offset)
                    state = struct.unpack(fmt, file.read(record_size))[0]
                    if state:
                        return np.uint64(bm.gen_new_num(np.uint64(state),
                                                               SingletonConfig().config['4_spawn_rate'])[0])
        return np.uint64(0)


class BookReaderDispatcher:
    _book_reader: BookReader = BookReader

    def __init__(self):
        self.book_reader_ad: BookReaderAD | None = None
        self.use_ad = False

    def set_book_reader_ad(self, pattern: str, target: int):
        if self.book_reader_ad is not None:
            if pattern == self.book_reader_ad.pattern and target == self.book_reader_ad.target:
                return
        self.book_reader_ad = BookReaderAD(pattern, target)

    def move_on_dic(self, board: NDArray, pattern: str, target: str, pattern_full: str
                    ) -> Tuple[Dict[str, Union[str, float, int]], str]:
        if self.use_ad and self.book_reader_ad is not None:
            return self.book_reader_ad.move_on_dic(board, pattern_full)
        else:
            return self._book_reader.move_on_dic(board, pattern, target, pattern_full)

    def get_random_state(self, path_list: list, pattern_full: str):
        if self.use_ad and self.book_reader_ad is not None:
            return self.book_reader_ad.get_random_state(path_list, pattern_full)
        else:
            return self._book_reader.get_random_state(path_list, pattern_full)

    def dispatch(self, path_list: list, pattern: str, target: str | int):
        try:
            target = int(target)
            if target >= 128:
                target = int(np.log2(target))
        except ValueError:
            return
        if not pattern or not target:
            return

        found = False
        for path, success_rate_dtype in path_list:
            if not os.path.exists(path):
                continue

            with os.scandir(path) as entries:
                for entry in entries:
                    # 检查条目名称是否以新算法文件后缀结尾
                    for rank in (1, 0.75, 0.5, 0.25):
                        if entry.name.endswith(f'_{str(int(2 ** target * rank))}b'):
                            found = True
                            break
                    if found:
                        break
        if not found:
            self.use_ad = False
        else:
            self.set_book_reader_ad(pattern, target)
            if self.book_reader_ad is not None:
                self.use_ad = True


if __name__ == "__main__":
    _result, _ = BookReader.move_on_dic(np.array([[8, 8, 8, 4],
                                               [64, 32, 2, 4],
                                               [256, 128, 32768, 32768],
                                               [32768, 32768, 32768, 32768]]),
                                     '442t', '512', '442t_512')
    print(_result, _)

    # br = BookReaderDispatcher()
    # br.dispatch(SingletonConfig().config['filepath_map']['L3_512_0']
    #                           , 'L3', '512')
    # _result = br.move_on_dic(np.array([[0, 2, 2, 4],
    #                                            [2, 8, 32, 64],
    #                                            [2, 32768, 32768, 32768],
    #                                            [2, 32768, 32768, 32768]]),
    #                                  'L3', '512', 'L3_512_0')
    # print(_result)
