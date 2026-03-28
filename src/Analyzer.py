import multiprocessing
import os
import threading
from collections import defaultdict
from pathlib import Path
from queue import Queue
import re

import numpy as np
from PyQt5.QtCore import QObject
from numba import njit
from PyQt5 import QtCore, QtWidgets, QtGui
import BoardMover as bm
import VBoardMover as vbm
from BookReader import BookReaderDispatcher
from Settings import TwoLevelComboBox, SingleLevelComboBox
import Config
from Config import category_info, SingletonConfig, DTYPE_CONFIG


logger = Config.logger
is_zh = (SingletonConfig().config['language'] == 'zh')
direction_map = defaultdict(lambda: "？")

direction_map.update({
    'u': "上",
    'd': "下",
    'l': "左",
    'r': "右",
    '?': "？",
})


# 旧版单字符映射表
PNG_MAP_DICT = {
    ' ': 0, '!': 1, '"': 2, '#': 3, '$': 4, '%': 5, '&': 6, "'": 7, '(': 8, ')': 9, '*': 10,
    '+': 11, ',': 12, '-': 13, '.': 14, '/': 15, '0': 16, '1': 17, '2': 18, '3': 19, '4': 20,
    '5': 21, '6': 22, '7': 23, '8': 24, '9': 25, ':': 26, ';': 27, '<': 28, '=': 29, '>': 30,
    '?': 31, '@': 32, 'A': 33, 'B': 34, 'C': 35, 'D': 36, 'E': 37, 'F': 38, 'G': 39, 'H': 40,
    'I': 41, 'J': 42, 'K': 43, 'L': 44, 'M': 45, 'N': 46, 'O': 47, 'P': 48, 'Q': 49, 'R': 50,
    'S': 51, 'T': 52, 'U': 53, 'V': 54, 'W': 55, 'X': 56, 'Y': 57, 'Z': 58, '[': 59, '\\': 60,
    ']': 61, '^': 62, '_': 63, '`': 64, 'a': 65, 'b': 66, 'c': 67, 'd': 68, 'e': 69, 'f': 70,
    'g': 71, 'h': 72, 'i': 73, 'j': 74, 'k': 75, 'l': 76, 'm': 77, 'n': 78, 'o': 79, 'p': 80,
    'q': 81, 'r': 82, 's': 83, 't': 84, 'u': 85, 'v': 86, 'w': 87, 'x': 88, 'y': 89, 'z': 90,
    '{': 91, '|': 92, '}': 93, '~': 94, 'Ç': 95, 'ü': 96, 'é': 97, 'â': 98, 'ä': 99, 'à': 100,
    'å': 101, 'ç': 102, 'ê': 103, 'ë': 104, 'è': 105, 'ï': 106, 'î': 107, 'ì': 108, 'Ä': 109,
    'Å': 110, 'É': 111, 'æ': 112, 'Æ': 113, 'ô': 114, 'ö': 115, 'ò': 116, 'û': 117, 'ù': 118,
    'ÿ': 119, 'Ö': 120, 'Ü': 121, 'ø': 122, '£': 123, 'Ø': 124, '×': 125, 'ƒ': 126, 'á': 127
}

# 新版 3字符 Base-128 映射表
NEW_CHARS_LIST = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "À", "Á", "Â", "Ã", "Ä", "Å", "Æ",
    "Ç", "È", "É", "Ê", "Ë", "Ì", "Í", "Î", "Ï", "Ð", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "×", "Ø", "Ù", "Ú", "Û", "Ü", "Ý",
    "Þ", "ß", "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï", "ð", "ñ", "ò", "ó", "ô",
    "õ", "ö", "÷", "ø", "ù", "ú", "û", "ü", "ý", "þ", "ÿ", "¤", "¾"
]
NEW_CHAR_MAP = {char: idx for idx, char in enumerate(NEW_CHARS_LIST)}

# 新版移动方向映射到原 Python 代码方向:
# 新版: 0=up, 1=down, 2=left, 3=right
# 原代码推导: 1,2代表水平操作(Left/Right), 3,4代表垂直操作(Up/Down)
NEW_TO_OLD_MOVE = {0: 3, 1: 4, 2: 1, 3: 2}


class ReplayDecoder:
    def __init__(self, filepath, bm, vbm):
        self.filepath = filepath
        self.bm = bm
        self.vbm = vbm
        self.variant = "4x4"
        self.record_list: np.typing.NDArray = np.empty(0, dtype='uint64,uint8,uint8,uint8')

    def read_replay(self):
        """兼容读取所有编码格式的文件"""
        with open(self.filepath, 'rb') as f:
            raw_data = f.read()

        # --- 判断是否为 AItest 回放的 numpy 结构化数组 ---
        dt = np.dtype([('f0', 'uint64'), ('f1', 'uint32'), ('f2', 'uint8')])
        item_size = dt.itemsize  # 13 字节

        # 1. 长度必须是 13 的倍数且不为空
        if len(raw_data) > 0 and len(raw_data) % item_size == 0:
            try:
                arr = np.frombuffer(raw_data, dtype=dt)

                # 2. 验证第一个元素的 uint32 (f1) 是否从 0 开始
                if arr['f1'][0] == 0:
                    # 3. 验证单调不减特性
                    f1_vals = arr['f1']
                    if len(f1_vals) == 1 or np.all(np.diff(f1_vals.astype(np.int64)) >= 0):
                        return arr
            except Exception:
                # 如果中间任何环节报错（比如数据过短），回退到文本处理
                pass

        common_encodings = ['utf-8', 'gb18030', 'big5', 'shift_jis', 'euc-kr', 'utf-16', 'utf-16le', 'utf-16be']

        if raw_data.startswith(b'\xff\xfe'):
            return raw_data.decode('utf-16le')
        elif raw_data.startswith(b'\xfe\xff'):
            return raw_data.decode('utf-16be')
        elif raw_data.startswith(b'\xef\xbb\xbf'):
            return raw_data.decode('utf-8-sig')

        for enc in common_encodings:
            try:
                return raw_data.decode(enc)
            except UnicodeDecodeError:
                continue

        return raw_data.decode('latin-1')

    def decode(self):
        """自动判断格式并调用对应的解码逻辑"""
        replay_text = self.read_replay()

        if isinstance(replay_text, str):
            # 尝试匹配新版格式正则表达式: 例如 "4x4-0000000000000000_abc..."
            new_format_match = re.match(r'^(\d+x\d+)-([^_]*)_(.*)$', replay_text)

            if new_format_match:
                variant_str = new_format_match.group(1)
                moves_str = new_format_match.group(3)
                self._decode_new_format(variant_str, moves_str)
            else:
                self._decode_old_format(replay_text)
        else:
            self._decode_test_replay(replay_text)

    @staticmethod
    def _apply_special_32k_rule(board, replay_move, current_score, mover):
        """抽象出 65k 合并特殊判定规则"""
        board_decoded = mover.decode_board(board)
        positions = np.where(board_decoded == 32768)
        first_position = (positions[0][0], positions[1][0])
        second_position = (positions[0][1], positions[1][1])

        if (positions[0][0] == positions[0][1] and abs(positions[1][0] - positions[1][1]) == 1 and
            replay_move in (1, 2)) or (positions[1][0] == positions[1][1] and
                                       abs(positions[0][0] - positions[0][1]) == 1 and replay_move in (3, 4)):
            board_decoded[first_position] = 16384
            board_decoded[second_position] = 16384
            board = np.uint64(mover.encode_board(board_decoded))
            current_score += 32768

        return board, current_score

    def _decode_new_format(self, variant_str: str, moves_str: str):
        """解析新版 21-bit / 3-chars 格式，但输出旧版 Numpy 数组结构"""
        if variant_str == "2x4":
            board, total_space = np.uint64(0xffff00000000ffff), 11
            mover = self.vbm
            self.variant = "2x4"
        elif variant_str == "3x3":
            board, total_space = np.uint64(0x000f000f000fffff), 15
            mover = self.vbm
            self.variant = "3x3"
        elif variant_str == "3x4":
            board, total_space = np.uint64(0x000000000000ffff), 15
            mover = self.vbm
            self.variant = "3x4"
        elif variant_str == "4x4":
            board, total_space = np.uint64(0), 15
            mover = self.bm
            self.variant = "4x4"
        else:
            logger.warning(f"Invalid variant {variant_str}")
            return

        num_moves = len(moves_str) // 3
        self.record_list = np.empty(max(0, num_moves - 2), dtype='uint64,uint32,uint8,uint8,uint8')
        record = np.empty(max(0, num_moves - 2), dtype='uint64,uint32,uint8')

        moves_made = 0
        current_score = 0

        for i in range(0, len(moves_str), 3):
            chunk = moves_str[i:i + 3]
            if len(chunk) < 3:
                break

            try:
                # 3 个字符转换为 21-bit 二进制整数
                binary = (NEW_CHAR_MAP[chunk[0]] << 14) + \
                         (NEW_CHAR_MAP[chunk[1]] << 7) + \
                         NEW_CHAR_MAP[chunk[2]]
            except KeyError:
                logger.warning(f"Invalid character in new format replay chunk: {chunk}")
                return

            # 位运算解析
            move_val = binary & 0b11  # 第 0-1 位
            spawn_val_bit = (binary >> 2) & 0b11  # 第 2-3 位 (0代表生成2, 1代表生成4)
            spawn_x = (binary >> 4) & 0b111  # 第 4-6 位
            spawn_y = (binary >> 7) & 0b111  # 第 7-9 位
            # 时间分辨率在 (binary >> 10) & 0b111, 时间乘数在 (binary >> 13) & 0b11111111，不需要提取

            # 转换为原代码变量
            replay_tile = spawn_val_bit + 1  # 0->1(数值2), 1->2(数值4)
            replay_move = NEW_TO_OLD_MOVE.get(move_val, 1)

            # 将二维坐标转回旧代码的 一维棋盘位置偏移 (适配原有 bitboard 插入规则)
            replay_position = total_space - (spawn_y << 2) - spawn_x

            # 状态推进逻辑
            if moves_made >= 2:
                self.record_list[moves_made - 2] = (board, current_score, replay_move, replay_tile,
                                                    15 - replay_position)

                if moves_made > 27000 and count_32ks(board) == 2:
                    board, current_score = self._apply_special_32k_rule(board, replay_move, current_score, mover)

                board, new_score = mover.s_move_board(board, replay_move)
                board = np.uint64(board)
                current_score += new_score

            # 位棋盘插入方块
            board |= np.uint64(replay_tile) << np.uint64(replay_position << 2)
            moves_made += 1

            if moves_made >= 3:
                record[moves_made - 3] = (board, current_score, 0)

    def _decode_old_format(self, replay_text: str):
        """老版逻辑"""
        if "2x4" in replay_text[:12]:
            board, total_space, header = np.uint64(0xffff00000000ffff), 11, 10
            mover = self.vbm
            self.variant = "2x4"
        elif "3x3" in replay_text[:12]:
            board, total_space, header = np.uint64(0x000f000f000fffff), 15, 10
            mover = self.vbm
            self.variant = "3x3"
        elif "3x4" in replay_text[:12]:
            board, total_space, header = np.uint64(0x000000000000ffff), 15, 10
            mover = self.vbm
            self.variant = "3x4"
        else:
            board, total_space, header = np.uint64(0), 15, 7
            mover = self.bm
            self.variant = "4x4"

        replay_text = replay_text[header:]
        self.record_list: np.typing.NDArray = np.empty(len(replay_text) - 2, dtype='uint64,uint32,uint8,uint8,uint8')
        record = np.empty(len(replay_text) - 2, dtype='uint64,uint32,uint8')

        move_map = [3, 2, 4, 1]
        moves_made = 0
        current_score = 0

        for i in replay_text:
            try:
                index_i = PNG_MAP_DICT[i]
            except KeyError:
                logger.warning(f"Character '{i}' not found in png_map_dict.")
                return

            replay_tile = ((index_i >> 4) & 1) + 1
            replay_move = move_map[index_i >> 5]
            replay_position = total_space - ((index_i & 3) << 2) - ((index_i & 15) >> 2)

            if moves_made >= 2:
                self.record_list[moves_made - 2] = (board, current_score, replay_move, replay_tile,
                                                    15 - replay_position)

                if moves_made > 27000 and count_32ks(board) == 2:
                    board, current_score = self._apply_special_32k_rule(board, replay_move, current_score, mover)

                board, new_score = mover.s_move_board(board, replay_move)
                board = np.uint64(board)
                current_score += new_score

            board |= np.uint64(replay_tile) << np.uint64(replay_position << 2)
            moves_made += 1

            if moves_made >= 3:
                record[moves_made - 3] = (board, current_score, 0)

    def _decode_test_replay(self, arr):
        self.record_list: np.typing.NDArray = np.full(len(arr), 1, dtype='uint64,uint32,uint8,uint8,uint8')
        self.record_list['f0'] = arr['f0']
        self.record_list['f1'] = arr['f1']
        self.record_list['f2'][:-1] = arr['f2'][1:]
        for i in range(1, len(arr) - 1):
            moved = bm.move_board(arr['f0'][i], arr['f2'][i + 1])
            diff = moved ^ arr['f0'][i + 1]
            pos = (int(diff).bit_length() - 1) // 4
            value = diff >> (pos * 4)
            self.record_list['f3'][i] = value
            self.record_list['f4'][i] = 15 - pos


class Analyzer(QObject):
    pattern_map = Config.pattern_32k_tiles_map

    def __init__(self, file_path: str, pattern: str, target: int, full_pattern: str, target_path: str):
        super().__init__()
        self.full_pattern = full_pattern
        self.pattern = pattern
        self.variant = ''
        self.target = target
        self.n_large_tiles = self.pattern_map[pattern][0]
        self.large_tile_sum = 0

        self.bm = bm
        self.vbm = vbm
        self.book_reader: BookReaderDispatcher = BookReaderDispatcher()
        spawn_rate4 = SingletonConfig().config['4_spawn_rate']
        bookfile_path_list = Config.SingletonConfig().config['filepath_map'].get((full_pattern, spawn_rate4), [])
        self.book_reader.dispatch(bookfile_path_list, pattern, target)

        self.filepath = file_path
        decoder = ReplayDecoder(self.filepath, self.bm, self.vbm)
        decoder.decode()
        self.record_list = decoder.record_list

        self.target_path = target_path

        self.text_list = []
        self.combo = 0
        self.result = dict()
        self.goodness_of_fit = 1
        self.max_combo = 0
        self.maximum_single_step_loss_relative = 0.0
        self.maximum_single_step_loss_absolute = 0.0
        self.step_count = 0
        self.performance_stats = {
            "**Perfect!**": 0,
            "**Excellent!**": 0,
            "**Nice try!**": 0,
            "**Not bad!**": 0,
            "**Mistake!**": 0,
            "**Blunder!**": 0,
            "**Terrible!**": 0,
        }
        self.record = np.empty(4000, dtype='uint64,uint8,uint32,uint32,uint32,uint32')
        self.rec_step_count = 0
        self.log_difficulty = 0.0  # 使用对数累加
        self.prev_expected_success_rate = None

    def clear_analysis(self):
        self.text_list = []
        self.combo = 0
        self.result = dict()
        self.goodness_of_fit = 1
        self.max_combo = 0
        self.maximum_single_step_loss_relative = 0.0
        self.maximum_single_step_loss_absolute = 0.0
        self.step_count = 0
        self.performance_stats = {
            "**Perfect!**": 0,
            "**Excellent!**": 0,
            "**Nice try!**": 0,
            "**Not bad!**": 0,
            "**Mistake!**": 0,
            "**Blunder!**": 0,
            "**Terrible!**": 0,
        }
        self.log_difficulty = 0.0
        self.prev_expected_success_rate = None

    def check_nth_largest(self, board_encoded: np.uint64) -> bool:
        # 初始化一个大小为16的列表，记录0-15的出现次数
        count = [0] * 16

        # 统计每个数字的出现次数
        for i in range(16):
            digit = (board_encoded >> np.uint64(i * 4)) & np.uint64(0xF)
            count[digit] += 1

        num_largest_found = 0
        for i in range(15, -1, -1):
            if count[i] > 0:
                # 检查是否有多个相同大数
                if count[i] > 1:
                    return False
                num_largest_found += 1
                if num_largest_found == self.n_large_tiles:
                    return i == self.target

        return False

    def mask_large_tiles(self, board: np.typing.NDArray) -> np.typing.NDArray:
        target_value = 1 << self.target

        # 创建条件掩码
        mask = (board >= target_value)

        # 批量应用掩码
        np.copyto(board, 32768, where=mask)
        return board

    def generate_reports(self):
        """循环调用analyze_one_step，找到连续的定式范围内的局面，将分析结果写入报告"""
        for i in range(len(self.record_list)):
            is_endgame, large_tile_changed = self.analyze_one_step(i)
            if is_endgame is None:
                self.write_error('''Table path not found, please make sure you have calculated the required table
                and that it is working properly in the practice module.''')
                break

            elif large_tile_changed:
                if len(self.text_list) > 100:
                    self.write_analysis(i)
                    self.save_rec_to_file(i)
                self.clear_analysis()

        if len(self.text_list) > 100:
            self.write_analysis(len(self.record_list))
            self.save_rec_to_file(len(self.record_list))
        self.clear_analysis()

    def print_board(self, board: np.typing.NDArray):
        rows = {'2x4':(1,3), '3x3':(0,3), '3x4':(0,3)}.get(self.variant, (0,4))
        items = 4 if self.variant != '3x3' else 3

        for row in board[rows[0]: rows[1]]:
            # 格式化每一行
            formatted_row = []
            for item in row[:items]:
                if item == 0:
                    formatted_element = '_'
                elif item >= 1024:
                    formatted_element = f"{item // 1024}k"
                else:
                    formatted_element = str(item)
                formatted_row.append(formatted_element.rjust(4, ' '))
            self.text_list.append(' '.join(formatted_row))

    def _analyze_one_step(self, board: np.typing.NDArray, masked_board: np.typing.NDArray, move: str,
                          new_tile: int, spawn_position:int) -> bool | None:
        if not move:
            return False
        target = str(int(2 ** self.target))
        self.result, success_rate_dtype = self.book_reader.move_on_dic(masked_board, self.pattern, target, self.full_pattern)
        _, _, _, zero_val = DTYPE_CONFIG.get(success_rate_dtype, DTYPE_CONFIG['uint32'])
        self.result = {key: formatting(value, zero_val) for key, value in self.result.items()}
        self.record_replay(board, move, new_tile, spawn_position)

        best_move = list(self.result.keys())[0]

        # 配置里没找到表路径
        if best_move == '?':
            return None
        # 超出定式范围、没查到、死亡、成功等情况
        if not self.result[best_move] or self.result[best_move] == 1 or self.result[best_move] == '?':
            return False

        # --- 计算游戏难度 ---
        if self.prev_expected_success_rate:
            diff_step = np.log(self.prev_expected_success_rate) - np.log(self.result[best_move])
            self.log_difficulty += diff_step
        # -----------------------------------------------

        self.print_board(board)
        self.text_list.append('')
        for key, value in self.result.items():
            self.text_list.append(f"{key[0].upper()}: {value}")
        self.text_list.append('')

        # 移动有效但是形成超出定式范围的局面
        if not isinstance(self.result[move.lower()], (int, float, np.integer, np.floating)):
            self.text_list.append(self.tr("The game goes beyond this formation"))
            self.text_list.append('--------------------------------------------------')
            self.text_list.append('')
            self.prev_expected_success_rate = None
            return False

        self.step_count += 1
        # 残局一开始的局面小概率不在tables中，故不分析; 后面rec_step_count = self.step_count - 5也是这里造成的
        if self.step_count < 5:
            return True

        if self.result[move.lower()] is not None and self.result[best_move] - self.result[move.lower()] <= 3e-10:
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.performance_stats["**Perfect!**"] += 1
            self.text_list.append(f"**Perfect! Combo: {self.combo}x**")
            if is_zh:
                self.text_list.append(f"你走的是 {direction_map[move[0].lower()]} ，"
                                 f"最优解正是 **{direction_map[best_move[0].lower()]}**")
                self.text_list.append(f'吻合度: {self.goodness_of_fit:.4f}, 游戏难度: {self.log_difficulty:.4f}')
            else:
                self.text_list.append(f"You pressed {move}. And the best move is **{best_move.capitalize()}**")
                self.text_list.append(f'total goodness of fit: {self.goodness_of_fit:.4f}, game difficulty: {self.log_difficulty:.4f}')
        else:
            self.combo = 0
            loss = self.result[move.lower()] / self.result[best_move]
            self.maximum_single_step_loss_relative = max(self.maximum_single_step_loss_relative, 1 - loss)
            loss_abs = self.result[best_move] - self.result[move.lower()]
            self.maximum_single_step_loss_absolute = max(self.maximum_single_step_loss_absolute, loss_abs)
            if loss != 0:
                self.goodness_of_fit *= loss
            # 根据 loss 值提供不同级别的评价
            evaluation = self.evaluation_of_performance(loss)
            self.performance_stats[evaluation] += 1

            self.text_list.append(evaluation)
            if is_zh:
                self.text_list.append(f"你走的是 {direction_map[move[0].lower()]} ，"
                                 f"但最优解是 **{direction_map[best_move[0].lower()]}**")
                self.text_list.append(f'单步相对损失: {1 - loss:.4f}, 单步绝对损失: {loss_abs:.4f}pt')
                self.text_list.append(f'吻合度: {self.goodness_of_fit:.4f}, 游戏难度: {self.log_difficulty:.4f}')
            else:
                self.text_list.append(f"You pressed {move}. But the best move is **{best_move.capitalize()}**")
                self.text_list.append(f'relative one-step loss: {1 - loss:.4f}, absolute one-step loss: {loss_abs:.4f}pt')
                self.text_list.append(f'total goodness of fit: {self.goodness_of_fit:.4f}, game difficulty: {self.log_difficulty:.4f}')

        self.text_list.append('--------------------------------------------------')
        self.text_list.append('')
        self.prev_expected_success_rate = self.result[move.lower()]
        return True

    def analyze_one_step(self, i):
        board_encoded, current_score, move_encoded, new_tile, spawn_position = self.record_list[i]
        board = self.bm.decode_board(board_encoded)

        # 把大数字用32768代替，返回能够进行查找的board
        if self.pattern in Config.category_info.get('variant', []):
            masked_board = board.copy()
        elif self.check_nth_largest(board_encoded):
            masked_board = self.mask_large_tiles(board.copy())
        else:
            return False, True
        large_tile_sum = masked_board.sum() - board.sum()
        large_tile_changed = large_tile_sum != self.large_tile_sum
        self.large_tile_sum = large_tile_sum

        move = ('', 'Left', 'Right', 'Up', 'Down')[move_encoded]
        is_endgame = self._analyze_one_step(board, masked_board, move, new_tile, spawn_position)
        return is_endgame, large_tile_changed

    def write_error(self, text: str):
        filename = self.full_pattern + '_' + 'error'
        target_file_path = os.path.join(self.target_path, filename)
        with open(target_file_path, 'a', encoding='utf-8') as file:
            file.write(text + '\n')

    def write_analysis(self, step: int):
        filename = self.full_pattern + '_' + Path(self.filepath).stem + '_' + str(step) + f'_{self.goodness_of_fit:.4f}' + '.txt'
        target_file_path = os.path.join(self.target_path, filename)
        with open(target_file_path, 'w', encoding='utf-8') as file:
            for line in self.text_list:
                file.write(line.replace('**', '') + '\n')

            # 最后输出统计信息
            file.write('======================Stats======================' + '\n')
            file.write('\n')

            file.write(
                self.tr('in ') + self.full_pattern +
                self.tr(' endgame in ') + str(self.step_count) +
                self.tr(' moves') + '\n'
            )
            file.write(
                self.tr('Total Goodness of Fit: ') +
                f'{self.goodness_of_fit:.4f}' + '\n'
            )
            file.write(
                self.tr('Game Difficulty: ') +
                f'{self.log_difficulty:.4f}' + '\n'
            )
            file.write(
                self.tr('Maximum Combo: ') +
                str(self.max_combo) + '\n'
            )
            file.write(
                self.tr('Maximum Single Step Loss (Relative, %): ') +
                f'{self.maximum_single_step_loss_relative:.4f}' + '\n'
            )
            file.write(
                self.tr('Maximum Single Step Loss (Absolute, pt): ') +
                f'{self.maximum_single_step_loss_absolute:.4f}' + '\n'
            )
            # 添加评价统计信息
            for evaluation, count in self.performance_stats.items():
                file.write(f'{evaluation}: {count}\n')
            file.write(self.tr('End of analysis'))

    @staticmethod
    def evaluation_of_performance(loss):
        if loss >= 0.999:
            text = "**Excellent!**"
        elif loss >= 0.99:
            text = "**Nice try!**"
        elif loss >= 0.975:
            text = "**Not bad!**"
        elif loss >= 0.9:
            text = "**Mistake!**"
        elif loss >= 0.75:
            text = "**Blunder!**"
        else:
            text = "**Terrible!**"
        return text

    def record_replay(self, board, direction: str, new_tile: int, spawn_position: int):
        rec_step_count = self.step_count - 5
        direct = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3}[direction.capitalize()]
        encoded = self.encode(direct, spawn_position, new_tile - 1)
        success_rates = []
        for d in ('left', 'right', 'up', 'down'):
            rate = self.result.get(d, None)
            if isinstance(rate, (int, float, np.integer, np.floating)):
                success_rates.append(np.uint32(rate * 4e9))
            else:
                success_rates.append(np.uint32(0))
        self.record[rec_step_count] = (np.uint64(self.bm.encode_board(board)), encoded, *success_rates)

    @staticmethod
    def encode(a, b, c):
        return np.uint8(((a << 5) | (b << 1) | c) & 0xFF)

    def save_rec_to_file(self, step: int):
        rec_step_count = self.step_count - 5
        if self.full_pattern is None or rec_step_count < 2:
            return

        filename = self.full_pattern + '_' + Path(self.filepath).stem + '_' + str(step) + f'_{self.goodness_of_fit:.4f}' + '.rpl'
        target_file_path = os.path.join(self.target_path, filename)
        self.record[rec_step_count] = (
            0, 88, 666666666, 233333333, 314159265, 987654321)
        self.record[:rec_step_count + 1].tofile(target_file_path)


@njit(cache=True)
def count_32ks(board):
    count = 0
    for i in range(16):
        tile = (board >> np.uint64(i * 4)) & np.uint64(0xf)
        count += tile == 0xf
    return count


def formatting(value, zero_val):
    if zero_val >= 0 or not isinstance(value, (int, float, np.integer, np.floating)):
        return value
    else:
        return abs(zero_val) + value


# noinspection PyAttributeOutsideInit
class AnalyzeWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.selected_filepaths = []

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QtGui.QIcon(r"pic\2048.ico"))
        self.resize(840, 240)
        color_mgr = Config.ColorManager()
        self.setStyleSheet("QMainWindow{\n"
                           f"    background-color: {color_mgr.get_css_color(1)};\n"
                           "}")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.selfLayout = QtWidgets.QGridLayout()
        self.selfLayout.setContentsMargins(25, 25, 25, 25)
        self.selfLayout.setHorizontalSpacing(15)
        self.selfLayout.setVerticalSpacing(15)
        self.selfLayout.setObjectName("selfLayout")

        self.pattern_text = QtWidgets.QLabel(self.centralwidget)
        self.pattern_text.setObjectName("pattern_text")
        self.pattern_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.pattern_text, 0, 0, 1, 1)
        self.filepath_text = QtWidgets.QLabel(self.centralwidget)
        self.filepath_text.setObjectName("filepath_text")
        self.filepath_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.filepath_text, 1, 0, 1, 1)
        self.filepath_edit = QtWidgets.QTextEdit(self.centralwidget)
        self.filepath_edit.setObjectName("filepath_edit")
        self.filepath_edit.setMaximumSize(QtCore.QSize(540, 80))
        self.selfLayout.addWidget(self.filepath_edit, 1, 1, 1, 1)
        self.set_filepath_bt = QtWidgets.QPushButton(self.centralwidget)
        self.set_filepath_bt.setObjectName("set_filepath_bt")
        self.set_filepath_bt.clicked.connect(self.filepath_changed)  # type: ignore
        self.selfLayout.addWidget(self.set_filepath_bt, 1, 2, 1, 1)

        self.target_combo = SingleLevelComboBox(self.tr("Target Tile"), self.centralwidget)
        self.target_combo.setObjectName("target_combo")
        self.target_combo.add_items(["64", "128", "256", "512", "1024", "2048", "4096", "8192"])
        self.selfLayout.addWidget(self.target_combo, 0, 2, 1, 1)
        self.pattern_combo = TwoLevelComboBox(self.tr("Select Formation"), self.centralwidget)
        self.pattern_combo.setObjectName("pattern_combo")
        for category, items in category_info.items():
            self.pattern_combo.add_category(category, items)
        self.selfLayout.addWidget(self.pattern_combo, 0, 1, 1, 1)

        self.analyze_bt = QtWidgets.QPushButton(self.centralwidget)
        self.analyze_bt.setObjectName("analyze_bt")
        self.analyze_bt.clicked.connect(self.start_analyze)  # type: ignore
        self.analyze_bt.setMinimumWidth(135)
        self.selfLayout.addWidget(self.analyze_bt, 1, 3, 1, 1)
        self.notes_text = QtWidgets.QLabel(self.centralwidget)
        self.notes_text.setObjectName("notes_text")
        self.notes_text.setStyleSheet("font: 360 9pt \"Cambria\";")
        self.notes_text.setMaximumSize(QtCore.QSize(720, 54))
        self.selfLayout.addWidget(self.notes_text, 2, 0, 1, 3)

        self.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(self.selfLayout)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    # noinspection PyTypeChecker
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Analysis", "Analysis"))
        self.pattern_text.setText(_translate("Analysis", "Pattern:"))
        self.filepath_text.setText(_translate("Analysis", "Load Replay"))
        self.analyze_bt.setText(_translate("Analysis", "Analyze"))
        self.set_filepath_bt.setText(_translate("Analysis", "SET..."))
        self.notes_text.setText(_translate("Analysis", "*https://2048verse.com/ supports save game replay\n"
                                                       "*The analysis results will be saved to the same path"))
        self.pattern_combo.retranslateUi(self.tr("Select Formation"))
        self.target_combo.retranslateUi(self.tr("Target Tile"))

    def filepath_changed(self):
        options = QtWidgets.QFileDialog.Options()
        # 打开文件或文件夹选择窗口
        filepaths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            self.tr("Select .txt or .vrs files"),
            "",
            "Supported Files (*.txt *.vrs);;Text Files (*.txt);;VRS Files (*.vrs);;All Files (*)",
            options=options | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        if filepaths:
            filepath_str = "\n".join(filepaths)  # 用换行符分隔
            self.filepath_edit.setText(filepath_str)
            self.selected_filepaths = filepaths

    def start_analyze(self):
        pattern = self.pattern_combo.currentText
        target = self.target_combo.currentText

        if pattern and target:
            if self.selected_filepaths:
                file_list = [f for f in self.selected_filepaths
                            if os.path.exists(f) and
                            (f.lower().endswith('.txt') or f.lower().endswith('.vrs') or '.' not in f)]
            else:
                # 从文本框中解析文件路径
                # 文件路径用换行符分隔
                pathname_text = self.filepath_edit.toPlainText()
                paths = [p.strip() for p in pathname_text.split('\n') if p.strip()]
                file_list = []
                for path in paths:
                    if os.path.exists(path):
                        if os.path.isdir(path):
                            # 如果是文件夹，遍历其中的txt和vrs文件
                            folder_files = [os.path.join(path, f) for f in os.listdir(path)
                                            if (f.lower().endswith('.txt') or f.lower().endswith('.vrs')) and
                                            os.path.isfile(os.path.join(path, f))]
                            file_list.extend(folder_files)
                        elif os.path.isfile(path) and (path.lower().endswith('.txt') or path.lower().endswith('.vrs') or '.' not in path):
                            # 如果是单个文件
                            file_list.append(path)

            if not file_list:
                QtWidgets.QMessageBox.warning(self, "warning", "No valid .txt or .vrs file found!")
                return

            # 计算公共参数
            ptn = pattern + '_' + target

            target_value = int(np.log2(int(target)))

            # 更新UI状态
            self.analyze_bt.setText(self.tr('Analyzing...'))
            self.analyze_bt.setEnabled(False)

            # 创建批量分析管理器
            self.analyze_manager = BatchAnalyzeManager(self)
            self.analyze_manager.pattern = pattern
            self.analyze_manager.target_value = target_value
            self.analyze_manager.ptn = ptn

            # 启动批量分析
            self.analyze_manager.start_batch_analyze(file_list)

    def on_analyze_finished(self):
        self.analyze_bt.setText(self.tr('Analyze'))
        self.analyze_bt.setEnabled(True)

        for thread in self.analyze_manager.analyze_threads:
            thread.wait()
            thread.deleteLater()

        # 安全清理完毕后再清空 Python 列表
        self.analyze_manager.analyze_threads.clear()


class AnalyzeThread(QtCore.QThread):
    def __init__(self, file_path: str, pattern: str, target: int, full_pattern: str, target_path: str):
        super().__init__()
        self.pattern = pattern
        self.target = target
        self.ptn = full_pattern
        self.file_path = file_path
        self.target_path = target_path

    def run(self):
        anlz = Analyzer(self.file_path, self.pattern, self.target, self.ptn, self.target_path)
        anlz.generate_reports()


class BatchAnalyzeManager(QtCore.QObject):
    def __init__(self, parent:AnalyzeWindow):
        super().__init__()
        self.parent = parent
        self.file_queue = Queue()
        self.analyze_threads = []
        self.active_threads = 0
        self.completed_count = 0
        self.total_count = 0
        self.max_threads = max(1, multiprocessing.cpu_count() - 1)  # 留一个核心给系统
        self.lock = threading.Lock()
        self.pattern = None
        self.target_value = None
        self.ptn = None

    def start_batch_analyze(self, file_list):
        """启动批量分析"""
        self.total_count = len(file_list)
        self.completed_count = 0

        # 将文件加入队列
        for file_path in file_list:
            self.file_queue.put(file_path)

        # 启动初始线程
        self._start_threads()

    def _start_threads(self):
        """启动线程，不超过最大限制"""
        threads_to_start = []

        with self.lock:
            # 在锁内收集需要启动的线程信息
            while (self.active_threads < self.max_threads and
                   not self.file_queue.empty()):
                file_path = self.file_queue.get()
                self.active_threads += 1

                # 只收集文件路径，不创建线程
                threads_to_start.append(file_path)

        # 在锁外创建和启动线程
        for file_path in threads_to_start:
            analyze_thread = AnalyzeThread(
                file_path, self.pattern, self.target_value, self.ptn,
                os.path.dirname(file_path)
            )
            analyze_thread.finished.connect(self._on_thread_finished)
            self.analyze_threads.append(analyze_thread)
            analyze_thread.start()

    def _on_thread_finished(self):
        """单个线程完成回调"""
        with self.lock:
            self.active_threads -= 1
            self.completed_count += 1

            # print(f"{self.completed_count}/{self.total_count}")
            if self.completed_count == self.total_count:
                all_finished = True
            else:
                all_finished = False

        if all_finished:
            self.parent.on_analyze_finished()
        else:
            self._start_threads()


if __name__ == "__main__":
    pass
    #
    # anlz = Analyzer(r"D:\2048calculates\test\analysis\vgame.txt", '2x4',
    #                 8, '2x4_256', r"D:\2048calculates\test\analysis", '0')
    # anlz.generate_reports()

    import sys
    app = QtWidgets.QApplication(sys.argv)
    main = AnalyzeWindow()
    main.show()
    sys.exit(app.exec_())
