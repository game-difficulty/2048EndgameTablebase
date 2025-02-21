import os
import struct
from typing import Callable, Dict, Tuple, Union, List, Optional

import numpy as np

from BoardMover import SingletonBoardMover, BoardMover
from Variants.vBoardMover import VBoardMover
from Config import SingletonConfig, formation_info
from TrieCompressor import trie_decompress_search
from Calculator import re_self
from BookReaderAD import BookReaderAD

PatternCheckFunc = Callable[[np.uint64], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]


class BookReader:
    bm: BoardMover = SingletonBoardMover(1)
    vbm: VBoardMover = SingletonBoardMover(3)

    @staticmethod
    def move_on_dic(board: np.typing.NDArray, pattern: str, target: str, pattern_full: str, pos: str = '0'
                    ) -> Dict[str, Union[str, float, int]]:
        bm = BookReader.bm if pattern not in ('2x4', '3x3', '3x4') else BookReader.vbm
        nums_adjust, pattern_check_func, to_find_func, success_check_func, _ = \
            formation_info.get(pattern, [0, None, re_self, None, None])
        path = SingletonConfig().config['filepath_map'].get(pattern_full, '')
        nums = (board.sum() + nums_adjust) / 2

        if pattern[:4] == 'free' and pattern[-1] != 'w':
            nums -= int(target) / 2
        if pattern == 'LL' and pos == 1:
            to_find_func = re_self
        if not path or not pattern_check_func:
            return {'?': '?'}
        if nums < 0:
            return {'down': '', 'right': '', 'left': '', 'up': ''}

        for rotation, flip, t_board in BookReader.gen_all_mirror(board, pattern):
            encoded = np.uint64(bm.encode_board(t_board))
            if pattern_check_func(encoded):
                results = BookReader.get_best_move(path, f'{pattern_full}_{int(nums)}.book', encoded,
                                                   pattern_check_func, bm, to_find_func)
                adjusted = {BookReader.adjust_direction(flip, rotation, direction): success_rate
                            for direction, success_rate in results.items()}
                float_items = {k: round(v, 10) for k, v in adjusted.items() if isinstance(v, (int, float))}
                non_float_items = {k: v for k, v in adjusted.items() if not isinstance(v, (int, float))}
                sorted_float_items = dict(sorted(float_items.items(), key=lambda item: item[1], reverse=True))
                sorted_results = {**sorted_float_items, **non_float_items}
                return sorted_results

        return {'down': '', 'right': '', 'left': '', 'up': ''}

    @staticmethod
    def gen_all_mirror(board: np.typing.NDArray, pattern: str) -> List[Tuple[str, str, np.typing.NDArray]]:
        if pattern in ('2x4', '3x3', '3x4'):
            return [('none', 'none', board)]
        operations = [
            ('none', 'none', board),
            ('rotate_90', 'none', np.rot90(board)),
            ('rotate_180', 'none', np.rot90(board, k=2)),
            ('rotate_270', 'none', np.rot90(board, k=3)),
            ('none', 'horizontal', np.flip(board, axis=1)),
            ('rotate_90', 'horizontal', np.flip(np.rot90(board), axis=1)),
            ('rotate_180', 'horizontal', np.flip(np.rot90(board, k=2), axis=1)),
            ('rotate_270', 'horizontal', np.flip(np.rot90(board, k=3), axis=1)),
        ]
        return operations if pattern != 'LL' else operations[:4]

    @staticmethod
    def get_best_move(pathname: str, filename: str, board: np.uint64, pattern_check_func: PatternCheckFunc,
                      bm: BoardMover, to_find_func: ToFindFunc) -> Dict[str, Optional[float]]:
        result = {'down': None, 'right': None, 'left': None, 'up': None}
        fullpath = os.path.join(pathname, filename.replace('.book', '.z'))
        if os.path.exists(fullpath):
            path = os.path.join(fullpath, filename.replace('.book', '.'))
            ind = np.fromfile(path + 'i', dtype='uint8,uint32')
            segments = np.fromfile(path + 's', dtype='uint32,uint64')
        else:
            ind, segments = None, None

        for newt, d in zip(bm.move_all_dir(board), ('left', 'right', 'up', 'down')):
            newt = np.uint64(newt)
            if newt != board and pattern_check_func(newt):
                result[d] = BookReader.find_value(pathname, filename, to_find_func(newt), ind, segments)
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
    def find_value(pathname: str, filename: str, search_key: np.uint64, ind: np.typing.NDArray = None,
                   segments: np.typing.NDArray = None) -> Union[int, float, str, None]:
        search_key = np.uint64(search_key)
        fullpath = os.path.join(pathname, filename)
        if os.path.exists(fullpath):
            return BookReader.find_value_in_binary(pathname, filename, search_key)
        elif ind is not None and segments is not None:
            path = os.path.join(fullpath.replace('.book', '.z'), filename.replace('.book', '.'))
            return trie_decompress_search(path, search_key, ind, segments)
        elif os.path.exists(fullpath.replace('.book', '.z')):
            path = os.path.join(fullpath.replace('.book', '.z'), filename.replace('.book', '.'))
            ind = np.fromfile(path + 'i', dtype='uint8,uint32')
            segments = np.fromfile(path + 's', dtype='uint32,uint64')
            return trie_decompress_search(path, search_key, ind, segments)

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
                key, value_int = struct.unpack('QI', f.read(record_size))
                if np.uint64(key) == search_key:
                    return value_int / 4000000000
                elif np.uint64(key) < search_key:
                    left = mid + 1
                else:
                    right = mid - 1
        # 没有找到局面
        return 0

    @staticmethod
    def get_random_state(pathname: str, pattern_full: str) -> np.uint64:
        book_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        while len(book_index) > 0:
            book_id = np.random.choice(book_index)
            book_index.remove(book_id)
            filepath = os.path.join(pathname, pattern_full + f'_{book_id}.book')
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


class BookReaderDispatcher:
    _book_reader: BookReader = BookReader
    bm: BoardMover = SingletonBoardMover(1)

    def __init__(self):
        self.book_reader_ad: BookReaderAD | None = None
        self.use_ad = False

    def set_book_reader_ad(self, pattern: str, target: int):
        if self.book_reader_ad is not None:
            if pattern == self.book_reader_ad.pattern and target == self.book_reader_ad.target:
                return
        self.book_reader_ad = BookReaderAD(pattern, target)

    def move_on_dic(self, board: np.typing.NDArray, pattern: str, target: str, pattern_full: str, pos: str = '0'
                    ) -> Dict[str, Union[str, float, int]]:
        if self.use_ad and self.book_reader_ad is not None:
            return self.book_reader_ad.move_on_dic(board, pattern_full, pos)
        else:
            return self._book_reader.move_on_dic(board, pattern, target, pattern_full, pos)

    def get_random_state(self, pathname: str, pattern_full: str):
        if self.use_ad and self.book_reader_ad is not None:
            return self.book_reader_ad.get_random_state(pathname, pattern_full)
        else:
            return self._book_reader.get_random_state(pathname, pattern_full)

    def dispatch(self, pathname: str, pattern: str, target: str | int):
        try:
            target = int(target)
            if target >= 128:
                target = int(np.log2(target))
        except ValueError:
            return
        if not pattern or not target:
            return
        if not os.path.exists(pathname):
            return
        found = False
        with os.scandir(pathname) as entries:
            for entry in entries:
                # 检查条目名称是否以新算法文件后缀结尾
                if entry.name.endswith(f'_{str(2 ** target // 2)}b'):
                    found = True
                    break
                if entry.name.endswith(f'_{str(2 ** target // 4)}b'):
                    found = True
                    break
        if not found:
            self.use_ad = False
        else:
            self.set_book_reader_ad(pattern, target)
            if self.book_reader_ad is not None:
                self.use_ad = True



if __name__ == "__main__":
    _result = BookReader.move_on_dic(np.array([[2, 8, 4, 2],
                                               [64, 8, 4, 2],
                                               [32768, 32768, 32768, 32768],
                                               [32768, 32768, 32768, 32768]]),
                                     'free8w', '128', 'free8w_128')
    print(_result)
