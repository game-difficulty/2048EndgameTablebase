from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import njit

import egtb_core.BoardMover as bm
import Config
import egtb_core.VBoardMover as vbm
from egtb_core.BookReader import BookReaderDispatcher
from Config import DTYPE_CONFIG, SingletonConfig, category_info


logger = Config.logger
is_zh = SingletonConfig().config.get("language") == "zh"
direction_map = defaultdict(lambda: "?")
direction_map.update(
    {
        "u": "上",
        "d": "下",
        "l": "左",
        "r": "右",
        "?": "?",
    }
)


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

NEW_CHARS_LIST = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "À", "Á", "Â", "Ã", "Ä", "Å", "Æ",
    "Ç", "È", "É", "Ê", "Ë", "Ì", "Í", "Î", "Ï", "Ð", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "×", "Ø", "Ù", "Ú", "Û", "Ü", "Ý",
    "Þ", "ß", "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï", "ð", "ñ", "ò", "ó", "ô",
    "õ", "ö", "÷", "ø", "ù", "ú", "û", "ü", "ý", "þ", "ÿ", "¤", "¾"
]
NEW_CHAR_MAP = {char: idx for idx, char in enumerate(NEW_CHARS_LIST)}
NEW_TO_OLD_MOVE = {0: 3, 1: 4, 2: 1, 3: 2}


class ReplayDecoder:
    def __init__(self, filepath: str, board_mover, variant_mover) -> None:
        self.filepath = filepath
        self.bm = board_mover
        self.vbm = variant_mover
        self.variant = "4x4"
        self.record_list: np.typing.NDArray = np.empty(
            0, dtype="uint64,uint8,uint8,uint8"
        )

    def read_replay(self):
        with open(self.filepath, "rb") as file:
            raw_data = file.read()

        dt = np.dtype([("f0", "uint64"), ("f1", "uint32"), ("f2", "uint8")])
        item_size = dt.itemsize

        # 1. 长度必须是 13 的倍数且不为空
        if len(raw_data) > 0 and len(raw_data) % item_size == 0:
            try:
                arr = np.frombuffer(raw_data, dtype=dt)
                
                # 2. 验证第一个元素的 uint32 (f1) 是否从 0 开始
                if arr["f1"][0] == 0:
                    # 3. 验证单调不减特性
                    f1_vals = arr["f1"]
                    if len(f1_vals) == 1 or np.all(
                        np.diff(f1_vals.astype(np.int64)) >= 0
                    ):
                        return arr
            except Exception:
                # 如果中间任何环节报错（比如数据过短），回退到文本处理
                pass

        common_encodings = [
            "utf-8",
            "gb18030",
            "big5",
            "shift_jis",
            "euc-kr",
            "utf-16",
            "utf-16le",
            "utf-16be",
        ]

        if raw_data.startswith(b"\xff\xfe"):
            return raw_data.decode("utf-16le")
        if raw_data.startswith(b"\xfe\xff"):
            return raw_data.decode("utf-16be")
        if raw_data.startswith(b"\xef\xbb\xbf"):
            return raw_data.decode("utf-8-sig")

        for encoding in common_encodings:
            try:
                return raw_data.decode(encoding)
            except UnicodeDecodeError:
                continue

        return raw_data.decode("latin-1")

    def decode(self) -> None:
        replay_text = self.read_replay()
        if isinstance(replay_text, str):
            new_format_match = re.match(r"^(\d+x\d+)-([^_]*)_(.*)$", replay_text)
            if new_format_match:
                self._decode_new_format(
                    new_format_match.group(1), new_format_match.group(3)
                )
            else:
                self._decode_old_format(replay_text)
            return

        self._decode_test_replay(replay_text)

    @staticmethod
    def _apply_special_32k_rule(
        board: np.uint64, replay_move: int, current_score: int, mover
    ) -> tuple[np.uint64, int]:
        board_decoded = mover.decode_board(board)
        positions = np.where(board_decoded == 32768)
        first_position = (positions[0][0], positions[1][0])
        second_position = (positions[0][1], positions[1][1])

        is_horizontal_pair = (
            positions[0][0] == positions[0][1]
            and abs(positions[1][0] - positions[1][1]) == 1
            and replay_move in (1, 2)
        )
        is_vertical_pair = (
            positions[1][0] == positions[1][1]
            and abs(positions[0][0] - positions[0][1]) == 1
            and replay_move in (3, 4)
        )
        if is_horizontal_pair or is_vertical_pair:
            board_decoded[first_position] = 16384
            board_decoded[second_position] = 16384
            board = np.uint64(mover.encode_board(board_decoded))
            current_score += 32768

        return board, current_score

    def _decode_new_format(self, variant_str: str, moves_str: str) -> None:
        if variant_str == "2x4":
            board, total_space, mover = np.uint64(0xFFFF00000000FFFF), 11, self.vbm
            self.variant = "2x4"
        elif variant_str == "3x3":
            board, total_space, mover = np.uint64(0x000F000F000FFFFF), 15, self.vbm
            self.variant = "3x3"
        elif variant_str == "3x4":
            board, total_space, mover = np.uint64(0x000000000000FFFF), 15, self.vbm
            self.variant = "3x4"
        elif variant_str == "4x4":
            board, total_space, mover = np.uint64(0), 15, self.bm
            self.variant = "4x4"
        else:
            logger.warning("Invalid variant %s", variant_str)
            return

        num_moves = len(moves_str) // 3
        self.record_list = np.empty(
            max(0, num_moves - 2), dtype="uint64,uint32,uint8,uint8,uint8"
        )
        moves_made = 0
        current_score = 0

        for index in range(0, len(moves_str), 3):
            chunk = moves_str[index : index + 3]
            if len(chunk) < 3:
                break

            try:
                binary = (
                    (NEW_CHAR_MAP[chunk[0]] << 14)
                    + (NEW_CHAR_MAP[chunk[1]] << 7)
                    + NEW_CHAR_MAP[chunk[2]]
                )
            except KeyError:
                logger.warning("Invalid character in new format replay chunk: %s", chunk)
                return

            move_val = binary & 0b11
            spawn_val_bit = (binary >> 2) & 0b11
            spawn_x = (binary >> 4) & 0b111
            spawn_y = (binary >> 7) & 0b111

            replay_tile = spawn_val_bit + 1
            replay_move = NEW_TO_OLD_MOVE.get(move_val, 1)
            replay_position = total_space - (spawn_y << 2) - spawn_x

            if moves_made >= 2:
                self.record_list[moves_made - 2] = (
                    board,
                    current_score,
                    replay_move,
                    replay_tile,
                    15 - replay_position,
                )
                if moves_made > 27000 and count_32ks(board) == 2:
                    board, current_score = self._apply_special_32k_rule(
                        board, replay_move, current_score, mover
                    )

                board, new_score = mover.s_move_board(board, replay_move)
                board = np.uint64(board)
                current_score += new_score

            board |= np.uint64(replay_tile) << np.uint64(replay_position << 2)
            moves_made += 1

    def _decode_old_format(self, replay_text: str) -> None:
        if "2x4" in replay_text[:12]:
            board, total_space, header, mover = (
                np.uint64(0xFFFF00000000FFFF),
                11,
                10,
                self.vbm,
            )
            self.variant = "2x4"
        elif "3x3" in replay_text[:12]:
            board, total_space, header, mover = (
                np.uint64(0x000F000F000FFFFF),
                15,
                10,
                self.vbm,
            )
            self.variant = "3x3"
        elif "3x4" in replay_text[:12]:
            board, total_space, header, mover = (
                np.uint64(0x000000000000FFFF),
                15,
                10,
                self.vbm,
            )
            self.variant = "3x4"
        else:
            board, total_space, header, mover = np.uint64(0), 15, 7, self.bm
            self.variant = "4x4"

        replay_text = replay_text[header:]
        self.record_list = np.empty(
            len(replay_text) - 2, dtype="uint64,uint32,uint8,uint8,uint8"
        )
        move_map = [3, 2, 4, 1]
        moves_made = 0
        current_score = 0

        for char in replay_text:
            try:
                index_i = PNG_MAP_DICT[char]
            except KeyError:
                logger.warning("Character %r not found in png_map_dict.", char)
                return

            replay_tile = ((index_i >> 4) & 1) + 1
            replay_move = move_map[index_i >> 5]
            replay_position = total_space - ((index_i & 3) << 2) - ((index_i & 15) >> 2)

            if moves_made >= 2:
                self.record_list[moves_made - 2] = (
                    board,
                    current_score,
                    replay_move,
                    replay_tile,
                    15 - replay_position,
                )
                if moves_made > 27000 and count_32ks(board) == 2:
                    board, current_score = self._apply_special_32k_rule(
                        board, replay_move, current_score, mover
                    )

                board, new_score = mover.s_move_board(board, replay_move)
                board = np.uint64(board)
                current_score += new_score

            board |= np.uint64(replay_tile) << np.uint64(replay_position << 2)
            moves_made += 1

    def _decode_test_replay(self, arr) -> None:
        self.record_list = np.full(
            len(arr), 1, dtype="uint64,uint32,uint8,uint8,uint8"
        )
        self.record_list["f0"] = arr["f0"]
        self.record_list["f1"] = arr["f1"]
        self.record_list["f2"][:-1] = arr["f2"][1:]
        for i in range(1, len(arr) - 1):
            moved = bm.move_board(arr["f0"][i], arr["f2"][i + 1])
            diff = moved ^ arr["f0"][i + 1]
            pos = (int(diff).bit_length() - 1) // 4
            value = diff >> (pos * 4)
            self.record_list["f3"][i] = value
            self.record_list["f4"][i] = 15 - pos


class Analyzer:
    pattern_map = Config.pattern_32k_tiles_map

    def __init__(
        self, file_path: str, pattern: str, target: int, full_pattern: str, target_path: str
    ) -> None:
        self.full_pattern = full_pattern
        self.pattern = pattern
        self.variant = ""
        self.target = target
        self.n_large_tiles = self.pattern_map[pattern][0]
        self.large_tile_sum = 0

        self.bm = bm
        self.vbm = vbm
        self.book_reader: BookReaderDispatcher = BookReaderDispatcher()
        spawn_rate4 = SingletonConfig().config["4_spawn_rate"]
        bookfile_path_list = SingletonConfig().config["filepath_map"].get(
            (full_pattern, spawn_rate4), []
        )
        self.book_reader.dispatch(bookfile_path_list, pattern, target)

        self.filepath = file_path
        decoder = ReplayDecoder(self.filepath, self.bm, self.vbm)
        decoder.decode()
        self.record_list = decoder.record_list

        self.target_path = target_path
        self.text_list: list[str] = []
        self.combo = 0
        self.result: dict[str, float | int | None | str] = {}
        self.goodness_of_fit = 1.0
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
        self.record = np.empty(4000, dtype="uint64,uint8,uint32,uint32,uint32,uint32")
        self.rec_step_count = 0
        self.log_difficulty = 0.0
        self.prev_expected_success_rate = None

    @staticmethod
    def tr(text: str) -> str:
        return text

    def clear_analysis(self) -> None:
        self.text_list = []
        self.combo = 0
        self.result = {}
        self.goodness_of_fit = 1.0
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
        count = [0] * 16
        for i in range(16):
            digit = (board_encoded >> np.uint64(i * 4)) & np.uint64(0xF)
            count[digit] += 1

        num_largest_found = 0
        for i in range(15, -1, -1):
            if count[i] > 0:
                if count[i] > 1:
                    return False
                num_largest_found += 1
                if num_largest_found == self.n_large_tiles:
                    return i == self.target

        return False

    def mask_large_tiles(self, board: np.typing.NDArray) -> np.typing.NDArray:
        target_value = 1 << self.target
        mask = board >= target_value
        np.copyto(board, 32768, where=mask)
        return board

    def generate_reports(self) -> None:
        for i in range(len(self.record_list)):
            is_endgame, large_tile_changed = self.analyze_one_step(i)
            if is_endgame is None:
                self.write_error(
                    "Table path not found, please make sure you have calculated the required table\n"
                    "and that it is working properly in the practice module."
                )
                break
            if large_tile_changed:
                if len(self.text_list) > 100:
                    self.write_analysis(i)
                    self.save_rec_to_file(i)
                self.clear_analysis()

        if len(self.text_list) > 100:
            self.write_analysis(len(self.record_list))
            self.save_rec_to_file(len(self.record_list))
        self.clear_analysis()

    def print_board(self, board: np.typing.NDArray) -> None:
        rows = {"2x4": (1, 3), "3x3": (0, 3), "3x4": (0, 3)}.get(
            self.variant, (0, 4)
        )
        items = 4 if self.variant != "3x3" else 3

        for row in board[rows[0] : rows[1]]:
            formatted_row = []
            for item in row[:items]:
                if item == 0:
                    formatted = "_"
                elif item >= 1024:
                    formatted = f"{item // 1024}k"
                else:
                    formatted = str(item)
                formatted_row.append(formatted.rjust(4, " "))
            self.text_list.append(" ".join(formatted_row))

    def _analyze_one_step(
        self,
        board: np.typing.NDArray,
        masked_board: np.typing.NDArray,
        move: str,
        new_tile: int,
        spawn_position: int,
    ) -> bool | None:
        if not move:
            return False

        target = str(2**self.target)
        self.result, success_rate_dtype = self.book_reader.move_on_dic(
            masked_board, self.pattern, target, self.full_pattern
        )
        _, _, _, zero_val = DTYPE_CONFIG.get(success_rate_dtype, DTYPE_CONFIG["uint32"])
        self.result = {
            key: formatting(value, zero_val) for key, value in self.result.items()
        }
        self.record_replay(board, move, new_tile, spawn_position)

        best_move = list(self.result.keys())[0]
        if best_move == "?":
            return None
        if (
            not self.result[best_move]
            or self.result[best_move] == 1
            or self.result[best_move] == "?"
        ):
            return False

        if self.prev_expected_success_rate:
            diff_step = np.log(self.prev_expected_success_rate) - np.log(
                self.result[best_move]
            )
            self.log_difficulty += diff_step

        self.print_board(board)
        self.text_list.append("")
        for key, value in self.result.items():
            self.text_list.append(f"{key[0].upper()}: {value}")
        self.text_list.append("")

        if not isinstance(
            self.result[move.lower()], (int, float, np.integer, np.floating)
        ):
            self.text_list.append(self.tr("The game goes beyond this formation"))
            self.text_list.append("--------------------------------------------------")
            self.text_list.append("")
            self.prev_expected_success_rate = None
            return False

        self.step_count += 1
        if self.step_count < 5:
            return True

        move_result = self.result[move.lower()]
        best_result = self.result[best_move]
        if move_result is not None and best_result - move_result <= 3e-10:
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.performance_stats["**Perfect!**"] += 1
            self.text_list.append(f"**Perfect! Combo: {self.combo}x**")
            if is_zh:
                self.text_list.append(
                    f"你走的是 {direction_map[move[0].lower()]}，最优解正是 **{direction_map[best_move[0].lower()]}**"
                )
                self.text_list.append(
                    f"吻合度: {self.goodness_of_fit:.4f}, 游戏难度: {self.log_difficulty:.4f}"
                )
            else:
                self.text_list.append(
                    f"You pressed {move}. And the best move is **{best_move.capitalize()}**"
                )
                self.text_list.append(
                    f"total goodness of fit: {self.goodness_of_fit:.4f}, game difficulty: {self.log_difficulty:.4f}"
                )
        else:
            self.combo = 0
            loss = move_result / best_result
            self.maximum_single_step_loss_relative = max(
                self.maximum_single_step_loss_relative, 1 - loss
            )
            loss_abs = best_result - move_result
            self.maximum_single_step_loss_absolute = max(
                self.maximum_single_step_loss_absolute, loss_abs
            )
            if loss != 0:
                self.goodness_of_fit *= loss
            evaluation = self.evaluation_of_performance(loss)
            self.performance_stats[evaluation] += 1

            self.text_list.append(evaluation)
            if is_zh:
                self.text_list.append(
                    f"你走的是 {direction_map[move[0].lower()]}，但最优解是 **{direction_map[best_move[0].lower()]}**"
                )
                self.text_list.append(
                    f"单步相对损失: {1 - loss:.4f}, 单步绝对损失: {loss_abs:.4f}pt"
                )
                self.text_list.append(
                    f"吻合度: {self.goodness_of_fit:.4f}, 游戏难度: {self.log_difficulty:.4f}"
                )
            else:
                self.text_list.append(
                    f"You pressed {move}. But the best move is **{best_move.capitalize()}**"
                )
                self.text_list.append(
                    f"relative one-step loss: {1 - loss:.4f}, absolute one-step loss: {loss_abs:.4f}pt"
                )
                self.text_list.append(
                    f"total goodness of fit: {self.goodness_of_fit:.4f}, game difficulty: {self.log_difficulty:.4f}"
                )

        self.text_list.append("--------------------------------------------------")
        self.text_list.append("")
        self.prev_expected_success_rate = move_result
        return True

    def analyze_one_step(self, i: int) -> tuple[bool | None, bool]:
        board_encoded, _, move_encoded, new_tile, spawn_position = self.record_list[i]
        board = self.bm.decode_board(board_encoded)

        if self.pattern in category_info.get("variant", []):
            masked_board = board.copy()
        elif self.check_nth_largest(board_encoded):
            masked_board = self.mask_large_tiles(board.copy())
        else:
            return False, True

        large_tile_sum = masked_board.sum() - board.sum()
        large_tile_changed = large_tile_sum != self.large_tile_sum
        self.large_tile_sum = large_tile_sum

        move = ("", "Left", "Right", "Up", "Down")[move_encoded]
        is_endgame = self._analyze_one_step(
            board, masked_board, move, new_tile, spawn_position
        )
        return is_endgame, large_tile_changed

    def write_error(self, text: str) -> None:
        filename = self.full_pattern + "_error"
        target_file_path = os.path.join(self.target_path, filename)
        with open(target_file_path, "a", encoding="utf-8") as file:
            file.write(text + "\n")

    def write_analysis(self, step: int) -> None:
        filename = (
            self.full_pattern
            + "_"
            + Path(self.filepath).stem
            + "_"
            + str(step)
            + f"_{self.goodness_of_fit:.4f}.txt"
        )
        target_file_path = os.path.join(self.target_path, filename)
        with open(target_file_path, "w", encoding="utf-8") as file:
            for line in self.text_list:
                file.write(line.replace("**", "") + "\n")

            file.write("======================Stats======================\n\n")
            file.write(
                self.tr("in ")
                + self.full_pattern
                + self.tr(" endgame in ")
                + str(self.step_count)
                + self.tr(" moves")
                + "\n"
            )
            file.write(
                self.tr("Total Goodness of Fit: ")
                + f"{self.goodness_of_fit:.4f}\n"
            )
            file.write(self.tr("Game Difficulty: ") + f"{self.log_difficulty:.4f}\n")
            file.write(self.tr("Maximum Combo: ") + str(self.max_combo) + "\n")
            file.write(
                self.tr("Maximum Single Step Loss (Relative, %): ")
                + f"{self.maximum_single_step_loss_relative:.4f}\n"
            )
            file.write(
                self.tr("Maximum Single Step Loss (Absolute, pt): ")
                + f"{self.maximum_single_step_loss_absolute:.4f}\n"
            )
            for evaluation, count in self.performance_stats.items():
                file.write(f"{evaluation}: {count}\n")
            file.write(self.tr("End of analysis"))

    @staticmethod
    def evaluation_of_performance(loss) -> str:
        if loss >= 0.999:
            return "**Excellent!**"
        if loss >= 0.99:
            return "**Nice try!**"
        if loss >= 0.975:
            return "**Not bad!**"
        if loss >= 0.9:
            return "**Mistake!**"
        if loss >= 0.75:
            return "**Blunder!**"
        return "**Terrible!**"

    def record_replay(
        self, board, direction: str, new_tile: int, spawn_position: int
    ) -> None:
        rec_step_count = self.step_count - 5
        direct = {"Left": 0, "Right": 1, "Up": 2, "Down": 3}[direction.capitalize()]
        encoded = self.encode(direct, spawn_position, new_tile - 1)
        success_rates = []
        for direction_key in ("left", "right", "up", "down"):
            rate = self.result.get(direction_key, None)
            if isinstance(rate, (int, float, np.integer, np.floating)):
                success_rates.append(np.uint32(rate * 4e9))
            else:
                success_rates.append(np.uint32(0))
        self.record[rec_step_count] = (
            np.uint64(self.bm.encode_board(board)),
            encoded,
            *success_rates,
        )

    @staticmethod
    def encode(a, b, c):
        return np.uint8(((a << 5) | (b << 1) | c) & 0xFF)

    def save_rec_to_file(self, step: int) -> None:
        rec_step_count = self.step_count - 5
        if self.full_pattern is None or rec_step_count < 2:
            return

        filename = (
            self.full_pattern
            + "_"
            + Path(self.filepath).stem
            + "_"
            + str(step)
            + f"_{self.goodness_of_fit:.4f}.rpl"
        )
        target_file_path = os.path.join(self.target_path, filename)
        self.record[rec_step_count] = (
            0,
            88,
            666666666,
            233333333,
            314159265,
            987654321,
        )
        self.record[: rec_step_count + 1].tofile(target_file_path)


@njit(cache=True)
def count_32ks(board):
    count = 0
    for i in range(16):
        tile = (board >> np.uint64(i * 4)) & np.uint64(0xF)
        count += tile == 0xF
    return count


def formatting(value, zero_val):
    if zero_val >= 0 or not isinstance(value, (int, float, np.integer, np.floating)):
        return value
    return abs(zero_val) + value
