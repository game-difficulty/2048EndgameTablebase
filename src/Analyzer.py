import os
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from PyQt5 import QtCore, QtWidgets, QtGui

from BoardMover import SingletonBoardMover
from BookReader import BookReaderDispatcher
from Settings import TwoLevelComboBox, SingleLevelComboBox
import Config
from Config import category_info, SingletonConfig


logger = Config.logger
translate_ = QtCore.QCoreApplication.translate
is_zh = (SingletonConfig().config['language'] == 'zh')
direction_map = defaultdict(lambda: "？")

direction_map.update({
    'u': "上",
    'd': "下",
    'l': "左",
    'r': "右",
    '?': "？",
})


class Analyzer:
    pattern_map = Config.pattern_32k_tiles_map

    def __init__(self, file_path: str, pattern: str, target: int, full_pattern: str, target_path: str, position: str):
        self.full_pattern = full_pattern
        self.position = position
        self.pattern = pattern
        self.target = target
        self.n_large_tiles = self.pattern_map[pattern][0]

        self.bm = SingletonBoardMover(1)
        self.vbm = SingletonBoardMover(3)
        self.book_reader: BookReaderDispatcher = BookReaderDispatcher()
        bookfile_path_list = Config.SingletonConfig().config['filepath_map'].get(full_pattern, [])
        self.book_reader.dispatch(bookfile_path_list, pattern, target)

        replay_text = self.read_replay(file_path)
        self.record_list: np.typing.NDArray = np.empty(0, dtype='uint64,uint8,uint8,uint8')
        self.decode_replay(replay_text)

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

    @staticmethod
    def read_replay(path):
        with open(path, encoding="utf-8") as replay:
            replay_text = replay.read()
        return replay_text

    def decode_replay(self, replay_text: str):
        if self.pattern not in ('2x4', '3x3', '3x4'):
            board = np.uint64(0)
            total_space = 15
            header = 7
            bm = self.bm
        else:
            board, total_space, header = {
                "2x4": (np.uint64(0xffff00000000ffff), 11, 10),
                "3x3": (np.uint64(0x000f000f000fffff), 15, 10),
                "3x4": (np.uint64(0x00000000000fffff), 15, 10),
            }[self.pattern]
            bm = self.vbm

        replay_text = replay_text[header:]
        self.record_list: np.typing.NDArray = np.empty(len(replay_text) - 2, dtype='uint64,uint8,uint8,uint8')

        png_map_dict = {
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
        move_map = [3, 2, 4, 1]
        moves_made = 0

        for i in replay_text:
            try:
                index_i = png_map_dict[i]
            except KeyError:
                index_i = 0
                logger.warning(f"Character '{i}' not found in png_map_dict, defaulting to index 0."
                               f"May cause errors in analysis.")

            replay_tile = ((index_i >> 4) & 1) + 1
            replay_move = move_map[index_i >> 5]
            replay_position = total_space - ((index_i & 3) << 2) - ((index_i & 15) >> 2)

            if moves_made >= 2:
                self.record_list[moves_made - 2] = (board, replay_move, replay_tile, 15 - replay_position)
                board = np.uint64(bm.move_board(board, replay_move))

            board |= np.uint64(replay_tile) << np.uint64(replay_position << 2)
            moves_made += 1

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

    def check_nth_largest(self, board_encoded: np.uint64) -> bool:
        is_free = (self.pattern[:4] == 'free' and self.pattern[-1] != 'w')
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
                if (num_largest_found == self.n_large_tiles and not is_free) or \
                        (is_free and num_largest_found == self.n_large_tiles + 1):
                    return i == self.target

        return False

    def mask_large_tiles(self, board: np.typing.NDArray) -> np.typing.NDArray:
        is_free = (self.pattern[:4] == 'free' and self.pattern[-1] != 'w')
        target = 2 ** self.target
        for i in range(4):
            for j in range(4):
                if (board[i][j] >= target and not is_free) or (is_free and board[i][j] > target):
                    board[i][j] = 32768
        return board

    def generate_reports(self):
        """循环调用analyze_one_step，找到连续的定式范围内的局面，将分析结果写入报告"""
        for i in range(len(self.record_list)):
            is_endgame = self.analyze_one_step(i)
            if is_endgame is None:
                self.write_error('''Table path not found, please make sure you have calculated the required table
                and that it is working properly in the practise module.''')
                break

            elif not is_endgame:
                if len(self.text_list) > 100:
                    self.write_analysis(i)
                    self.save_rec_to_file(i)
                self.clear_analysis()

        if len(self.text_list) > 100:
            self.write_analysis(len(self.record_list))
            self.save_rec_to_file(len(self.record_list))
        self.clear_analysis()

    def print_board(self, board: np.typing.NDArray):
        rows = {'2x4':(1,3), '3x3':(0,3), '3x4':(0,3)}.get(self.pattern, (0,4))
        items = 4 if self.pattern != '3x3' else 3

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
        target = str(int(2 ** self.target))
        self.result = self.book_reader.move_on_dic(masked_board, self.pattern, target, self.full_pattern, self.position)

        # 超出定式范围、没查到、死亡、成功等情况
        best_move = list(self.result.keys())[0]
        if not self.result[best_move] or self.result[best_move] == 1:
            return False

        # 配置里没找到表路径
        if best_move == '?' or self.result[best_move] == '?':
            return None

        self.print_board(board)
        self.text_list.append('')
        for key, value in self.result.items():
            self.text_list.append(f"{key[0].upper()}: {value}")
        self.text_list.append('')

        # 移动有效但是形成超出定式范围的局面
        if self.result[move.lower()] is None:
            self.text_list.append(translate_('Analyzer', "The game goes beyond this formation"))
            self.text_list.append('--------------------------------------------------')
            self.text_list.append('')
            return False

        self.step_count += 1
        # 残局一开始的局面小概率不在tables中，故不分析; 后面rec_step_count = self.step_count - 5也是这里造成的
        if self.step_count < 5:
            return True

        if self.result[move.lower()] is not None and self.result[move.lower()] / self.result[best_move] == 1:
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.performance_stats["**Perfect!**"] += 1
            self.text_list.append(f"**Perfect! Combo: {self.combo}x**")
            if is_zh:
                self.text_list.append(f"你走的是 {direction_map[move[0].lower()]} ，"
                                 f"最优解正是 **{direction_map[best_move[0].lower()]}**")
            else:
                self.text_list.append(f"You pressed {move}. And the best move is **{best_move.capitalize()}**")
        else:
            self.combo = 0
            loss = self.result[move.lower()] / self.result[best_move]
            self.maximum_single_step_loss_relative = max(self.maximum_single_step_loss_relative, 1 - loss)
            loss_abs = self.result[best_move] - self.result[move.lower()]
            self.maximum_single_step_loss_absolute = max(self.maximum_single_step_loss_absolute, loss_abs)
            self.goodness_of_fit *= loss
            # 根据 loss 值提供不同级别的评价
            evaluation = self.evaluation_of_performance(loss)
            self.performance_stats[evaluation] += 1

            self.text_list.append(evaluation)
            if is_zh:
                self.text_list.append(f"你走的是 {direction_map[move[0].lower()]} ，"
                                 f"但最优解是 **{direction_map[best_move[0].lower()]}**")
                self.text_list.append(f'单步相对损失: {1 - loss:.4f}, 单步绝对损失: {loss_abs:.4f}pt')
                self.text_list.append(f'吻合度: {self.goodness_of_fit:.4f}')
            else:
                self.text_list.append(f"You pressed {move}. But the best move is **{best_move.capitalize()}**")
                self.text_list.append(f'relative one-step loss: {1 - loss:.4f}, absolute one-step loss: {loss_abs:.4f}pt')
                self.text_list.append(f'total goodness of fit: {self.goodness_of_fit:.4f}')

        self.text_list.append('--------------------------------------------------')
        self.text_list.append('')
        self.record_replay(board, move, new_tile, spawn_position)
        return True

    def analyze_one_step(self, i):
        board_encoded, move_encoded, new_tile, spawn_position = self.record_list[i]
        board = self.bm.decode_board(board_encoded)

        # 把大数字用32768代替，返回能够进行查找的board
        if self.pattern in ('2x4', '3x3', '3x4'):
            masked_board = board.copy()
        elif self.check_nth_largest(board_encoded):
            masked_board = self.mask_large_tiles(board.copy())
        else:
            return False

        move = ('Left', 'Right', 'Up', 'Down')[move_encoded - 1]
        is_endgame = self._analyze_one_step(board, masked_board, move, new_tile, spawn_position)
        return is_endgame

    def write_error(self, text: str):
        filename = self.full_pattern + '_' + 'error'
        target_file_path = os.path.join(self.target_path, filename)
        with open(target_file_path, 'a', encoding='utf-8') as file:
            file.write(text + '\n')

    def write_analysis(self, step: int):
        filename = self.full_pattern + '_' + str(step) + f'_{self.goodness_of_fit:.4f}' + '.txt'
        target_file_path = os.path.join(self.target_path, filename)
        with open(target_file_path, 'w', encoding='utf-8') as file:
            for line in self.text_list:
                file.write(line.replace('**', '') + '\n')

            # 最后输出统计信息
            file.write('======================Stats======================' + '\n')
            file.write('\n')

            file.write(
                translate_('Analyzer', 'in ') + self.full_pattern +
                translate_('Analyzer', ' endgame in ') + str(self.step_count) +
                translate_('Analyzer', ' moves') + '\n'
            )
            file.write(
                translate_('Analyzer', 'Total Goodness of Fit: ') +
                f'{self.goodness_of_fit:.4f}' + '\n'
            )
            file.write(
                translate_('Analyzer', 'Maximum Combo: ') +
                str(self.max_combo) + '\n'
            )
            file.write(
                translate_('Analyzer', 'Maximum Single Step Loss (Relative, %): ') +
                f'{self.maximum_single_step_loss_relative:.4f}' + '\n'
            )
            file.write(
                translate_('Analyzer', 'Maximum Single Step Loss (Absolute, pt): ') +
                f'{self.maximum_single_step_loss_absolute:.4f}' + '\n'
            )
            # 添加评价统计信息
            for evaluation, count in self.performance_stats.items():
                file.write(f'{evaluation}: {count}\n')
            file.write(translate_('Analyzer', 'End of analysis'))

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
            rate = self.result[d]
            if isinstance(rate, (int, float)):
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

        filename = self.full_pattern + '_' + str(step) + f'_{self.goodness_of_fit:.4f}' + '.rpl'
        target_file_path = os.path.join(self.target_path, filename)
        self.record[rec_step_count] = (
            0, 88, 666666666, 233333333, 314159265, 987654321)
        self.record[:rec_step_count + 1].tofile(target_file_path)


# noinspection PyAttributeOutsideInit
class AnalyzeWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QtGui.QIcon(r"pic\2048.ico"))
        self.resize(840, 240)
        self.setStyleSheet("QMainWindow{\n"
                           "    background-color: rgb(245, 245, 247);\n"
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
        self.target_combo.add_items(["128", "256", "512", "1024", "2048", "4096", "8192"])
        self.selfLayout.addWidget(self.target_combo, 0, 2, 1, 1)
        self.pattern_combo = TwoLevelComboBox(self.tr("Select Formation"), self.centralwidget)
        self.pattern_combo.setObjectName("pattern_combo")
        for category, items in category_info.items():
            self.pattern_combo.add_category(category, items)
        self.selfLayout.addWidget(self.pattern_combo, 0, 1, 1, 1)
        self.pattern_combo.currentTextChanged.connect(self.update_pos_combo_visibility)
        self.pos_combo = SingleLevelComboBox(self.tr("Target Position"), self.centralwidget)
        self.pos_combo.setObjectName("pos_combo")
        self.pos_combo.add_items(["0", "1", "2"])
        self.selfLayout.addWidget(self.pos_combo, 0, 3, 1, 1)
        self.pos_combo.hide()

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
        self.pos_combo.retranslateUi(self.tr("Target Position"))

    def filepath_changed(self):
        options = QtWidgets.QFileDialog.Options()
        # 打开文件选择窗口，只能选择 .txt 文件
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a .txt file"),
            "",
            "Text Files (*.txt);;All Files (*)",
            options=options
        )
        if filepath:
            self.filepath_edit.setText(filepath)

    def start_analyze(self):
        position = self.pos_combo.currentText
        pattern = self.pattern_combo.currentText
        target = self.target_combo.currentText
        pathname = self.filepath_edit.toPlainText()
        position = '0' if not position else position
        if pattern and target and pathname and position and os.path.exists(pathname):
            if pattern in ['444', 'LL', 'L3']:
                ptn = pattern + '_' + target + '_' + position
            else:
                ptn = pattern + '_' + target
            target = int(np.log2(int(target)))

            self.analyze_bt.setText(self.tr('Analyzing...'))
            self.analyze_bt.setEnabled(False)
            self.Analyze_thread = AnalyzeThread(pathname, pattern, target, ptn, os.path.dirname(pathname), position)
            self.Analyze_thread.finished.connect(self.on_analyze_finished)
            self.Analyze_thread.start()  # 启动线程

    def on_analyze_finished(self):
        self.analyze_bt.setText(self.tr('Analyze'))
        self.analyze_bt.setEnabled(True)

    def update_pos_combo_visibility(self, pattern):
        if pattern in ['444', 'LL']:
            if '2' in self.pos_combo.items:
                self.pos_combo.remove_item('2')
            self.pos_combo.show()
        elif pattern == 'L3':
            if '2' not in self.pos_combo.items:
                self.pos_combo.add_item('2')
            self.pos_combo.show()
        else:
            self.pos_combo.hide()


class AnalyzeThread(QtCore.QThread):
    finished = QtCore.pyqtSignal()

    def __init__(self, file_path: str, pattern: str, target: int, full_pattern: str, target_path: str, position: str):
        super().__init__()
        self.pattern = pattern
        self.target = target
        self.position = position
        self.ptn = full_pattern
        self.file_path = file_path
        self.target_path = target_path

    def run(self):
        anlz = Analyzer(self.file_path, self.pattern, self.target, self.ptn, self.target_path, self.position)
        anlz.generate_reports()
        self.finished.emit()


if __name__ == "__main__":
    pass
    # anlz = Analyzer(r"C:\Users\Administrator\Downloads\message.txt", 'free12w',
    #                 12, 'free12w_2048', r"C:\Users\Administrator\Downloads", '0')
    # anlz.generate_reports()
    #
    # anlz = Analyzer(r"D:\2048calculates\test\analysis\vgame.txt", '2x4',
    #                 8, '2x4_256', r"D:\2048calculates\test\analysis", '0')
    # anlz.generate_reports()

    import sys
    app = QtWidgets.QApplication(sys.argv)
    main = AnalyzeWindow()
    main.show()
    sys.exit(app.exec_())
