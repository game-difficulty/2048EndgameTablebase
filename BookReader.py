import os
import struct

import numpy as np

from BoardMover import BoardMoverWithScore
from Config import SingletonConfig
from TrieCompressor import trie_decompress_search
import Calculator


class BookReader:
    bm = BoardMoverWithScore()

    pattern_map = {
        '': [0, Calculator.re_self],
        '24': [-262144 - 2, Calculator.is_24_pattern, Calculator.min24],
        '34': [-131072 - 2, Calculator.is_34_pattern, Calculator.min34],
        '33': [-229376 - 2, Calculator.is_33_pattern, Calculator.min33],
    }

    @staticmethod
    def move_on_dic(board, pattern):
        nums_adjust, pattern_check_func, to_find_func = BookReader.pattern_map[pattern]
        path = SingletonConfig().config['filepath_map'].get(pattern, '')
        nums = (board.sum() + nums_adjust) / 2
        if not path or not pattern:
            return {'?':'?'}
        if nums < 0:
            return {'down': '', 'right': '', 'left': '', 'up': ''}
        for rotation, flip, t_board in BookReader.gen_all_mirror(board):
            encoded = np.uint64(BookReader.bm.encode_board(t_board))
            if pattern_check_func(encoded):
                results = BookReader.get_best_move(path, f'{pattern}_{int(nums)}.book', encoded, pattern_check_func,
                                                   BookReader.bm, to_find_func)
                adjusted = {BookReader.adjust_direction(flip, rotation, direction): success_rate
                            for direction, success_rate in results.items()}
                float_items = {k: round(v, 12) for k, v in adjusted.items() if isinstance(v, (int, float))}
                non_float_items = {k: v for k, v in adjusted.items() if not isinstance(v, (int, float))}
                sorted_float_items = dict(sorted(float_items.items(), key=lambda item: item[1], reverse=True))
                sorted_results = {**sorted_float_items, **non_float_items}
                return sorted_results

        return {'down': '', 'right': '', 'left': '', 'up': ''}

    @staticmethod
    def gen_all_mirror(board):
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
        return operations

    @staticmethod
    def get_best_move(pathname, filename, board, pattern_check_func, bm, to_find_func):
        result = {'down': None, 'right': None, 'left': None, 'up': None}
        fullpath = os.path.join(pathname, filename.replace('.book', '.z'))
        if os.path.exists(fullpath):
            path = os.path.join(fullpath, filename.replace('.book', '.'))
            ind = np.fromfile(path + 'i', dtype='uint8,uint32')
            segments = np.fromfile(path + 's', dtype='uint32')
        else:
            ind = segments = None

        for (newt, new_score), d in zip(bm.move_all_dir(board), ('down', 'right', 'left', 'up')):
            newt = np.uint64(newt)
            if newt != board and pattern_check_func(newt):
                result[d] = BookReader.find_value(pathname, filename, to_find_func(newt), ind, segments) + new_score
        return result

    @staticmethod
    def adjust_direction(flip, rotation, direction):
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
    def find_value(pathname, filename, search_key, ind=None, segments=None):
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
            segments = np.fromfile(path + 's', dtype='uint32')
            return trie_decompress_search(path, search_key, ind, segments)

    @staticmethod
    def find_value_in_binary(pathname, filename, search_key):
        """
        从二进制文件中读取数据，并根据给定的键查找对应的值。
        """
        if not os.path.exists(os.path.join(pathname, filename)):
            return '?'
        with open(os.path.join(pathname, filename), 'rb') as f:
            record_size = struct.calcsize('QQ')
            f.seek(0, 2)
            file_size = f.tell()
            num_records = file_size // record_size
            # 二分查找
            left, right = 0, num_records - 1
            while left <= right:
                mid = (left + right) // 2
                f.seek(mid * record_size)
                key, value_int = struct.unpack('QQ', f.read(record_size))
                if np.uint64(key) == search_key:
                    return value_int / 2 ** 42
                elif np.uint64(key) < search_key:
                    left = mid + 1
                else:
                    right = mid - 1
        print(f'not found {search_key}')
        # 没有找到局面
        return 0

    @staticmethod
    def get_random_state(pathname, pattern_full):
        book_id = np.random.randint(0, 1)
        filepath = os.path.join(pathname, pattern_full + f'_{book_id}.Book')
        if not os.path.exists(filepath):
            return np.uint64(0)
        with open(filepath, 'rb')as file:
            record_size = struct.calcsize('QQ')
            file.seek(0, 2)
            file_size = file.tell()
            num_records = file_size // record_size
            random_record_index = np.random.randint(0, num_records - 1)
            offset = random_record_index * record_size
            file.seek(offset)
            state, _ = struct.unpack('QQ', file.read(record_size))
            state = np.uint64(BookReader.bm.gen_new_num(np.uint64(state))[0])
            return state
