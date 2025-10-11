import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import random
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit
from PyQt5.QtGui import QIcon

import BoardMover as bm
from BookReader import BookReaderDispatcher
from Config import SingletonConfig, category_info
from Gamer import BaseBoardFrame
from Analyzer import AnalyzeWindow
from RecordPlayer import ReplayWindow
from MistakesBook import mistakes_book
from MistakesBook import MistakeTrainingWindow
from SignalHub import practise_signal


direction_map = defaultdict(lambda: "？")

direction_map.update({
    'u': "上",
    'd': "下",
    'l': "左",
    'r': "右",
    '?': "？",
})


# noinspection PyAttributeOutsideInit
class ScrollTextDisplay(QWidget):
    def __init__(self, max_lines=100, parent=None):
        super(ScrollTextDisplay, self).__init__(parent)
        self.max_lines = max_lines  # 最大行数
        self.setupUi()
        self.lines = []
        self.show_text = True

    def setupUi(self):
        self.layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        self.text_edit.setFont(font)
        self.layout.addWidget(self.text_edit)
        self.setLayout(self.layout)

        self.scroll_bar = self.text_edit.verticalScrollBar()

    def clear_text(self):
        self.lines = []
        self.text_edit.clear()

    def update_text(self):
        if self.show_text:
            text = '\n\n'.join(self.lines[-self.max_lines:])
            if len(self.lines) > self.max_lines:
                text = self.lines[0] + '\n\n' + text
            self.text_edit.setMarkdown(text)
            # 保持滚动条在最下方
            self.scroll_bar.setValue(self.scroll_bar.maximum())
        else:
            self.text_edit.clear()

    def print_board(self, board):
        for row in board:
            # 格式化每一行
            formatted_row = []
            for item in row:
                if item == 0:
                    formatted_element = '_'  # 转义下划线
                elif item == 32768:
                    formatted_element = 'x'
                elif item >= 1024:
                    formatted_element = f"{item // 1024}k"
                else:
                    formatted_element = str(item)
                formatted_row.append(formatted_element.rjust(4, ' ').replace(' ', '&nbsp;'))
            self.lines.append(' '.join(formatted_row))
        self.update_text()

    def add_text(self, text_list):
        if isinstance(text_list, str):
            self.lines.append(text_list)
        else:
            for text in text_list:
                self.lines.append(text)


class TestFrame(BaseBoardFrame):
    v_inits = {
        '2x4': np.array([np.uint64(0xffff00000000ffff)], dtype=np.uint64),
        '3x3': np.array([np.uint64(0x000f000f000fffff)], dtype=np.uint64),
        '3x4': np.array([np.uint64(0x000000000000ffff)], dtype=np.uint64),
    }

    def __init__(self, centralwidget=None):
        super(TestFrame, self).__init__(centralwidget)
        self.combo = 0
        self.goodness_of_fit = 1
        self.max_combo = 0
        self.performance_stats = {
            "**Perfect!**": 0,
            "**Excellent!**": 0,
            "**Nice try!**": 0,
            "**Not bad!**": 0,
            "**Mistake!**": 0,
            "**Blunder!**": 0,
            "**Terrible!**": 0,
        }

    def update_frame(self, value, row, col):
        """重写方法以配合不显示32k格子数字的设置"""
        super().update_frame(value, row, col)
        if value == 32768 and not SingletonConfig().config.get('dis_32k', False):
            self.game_square.labels[row][col].setText('')

    def mousePressEvent(self, event):
        self.setFocus()

    def set_to_variant(self, pattern: str):
        self.set_use_variant(pattern)
        self.board_encoded = self.v_inits[pattern][0]
        self.board = bm.decode_board(self.board_encoded)
        self.update_all_frame(self.board)

    def set_to_44(self):
        if self.use_variant_mover != 0:
            self.set_use_variant('')
            self.board = np.zeros((4, 4), dtype=np.int32)
            self.board_encoded = bm.encode_board(self.board)
            self.update_all_frame(self.board)


class TestWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.isProcessing = False
        self.result = {}
        self.pattern = ['?', '?', '?']
        self.full_pattern = None
        # 分析verse replay的窗口
        self.analyze_window = None
        self.replay_window = None
        self.notebook_window = None
        self.book_reader: BookReaderDispatcher = BookReaderDispatcher()
        # 保存回放和相关信息
        self.record = np.empty(4000, dtype='uint64,uint8,uint32,uint32,uint32,uint32')
        self.step_count = 0

        self.reader_thread = ReaderWorker(self.book_reader, np.zeros((4, 4)), ['?', '?', '?'], '')
        self.reader_thread.result_ready.connect(self._show_results)

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QIcon(r"pic\2048.ico"))
        self.setMinimumSize(450, 320)
        self.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(6, 6, 6, 6)
        self.gridLayout.setVerticalSpacing(8)
        self.gridLayout.setHorizontalSpacing(8)
        self.gridLayout.setObjectName("gridLayout")

        self.setboard_Layout = QtWidgets.QHBoxLayout()
        self.setboard_Layout.setObjectName("setboard_Layout")
        self.board_state = QtWidgets.QLineEdit(self.centralwidget)
        self.board_state.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.board_state.setObjectName("board_state")
        self.board_state.setText('0000000000000000')
        self.setboard_Layout.addWidget(self.board_state)

        self.set_board_bt = QtWidgets.QPushButton(self.centralwidget)
        self.set_board_bt.setMaximumSize(QtCore.QSize(90, 16777215))
        self.set_board_bt.setObjectName("set_board_bt")
        self.set_board_bt.clicked.connect(self.set_board_init)  # type: ignore
        self.setboard_Layout.addWidget(self.set_board_bt)
        self.gridLayout.addLayout(self.setboard_Layout, 0, 0, 1, 1)

        self.gameframe = TestFrame(self.centralwidget)
        self.gridLayout.addWidget(self.gameframe, 1, 0, 1, 1)
        self.gameframe.setMinimumSize(240, 240)

        self.dis_text_checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.dis_text_checkBox.setStyleSheet("font: 360 10pt \"Cambria\";")
        self.dis_text_checkBox.setObjectName("dis_text_checkBox")
        self.gridLayout.addWidget(self.dis_text_checkBox, 0, 1, 1, 1)
        if SingletonConfig().config.get('dis_text', True):
            self.dis_text_checkBox.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.dis_text_checkBox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.dis_text_checkBox.stateChanged.connect(self.dis_text_state_change)  # type: ignore

        self.text_display = ScrollTextDisplay(max_lines=200)
        self.gridLayout.addWidget(self.text_display, 1, 1, 1, 1)
        self.text_display.setMinimumSize(120, 240)

        self.bts_Layout = QtWidgets.QHBoxLayout()
        self.practise_button = QtWidgets.QPushButton(self.centralwidget)
        self.bts_Layout.addWidget(self.practise_button)
        self.bts_Layout.setAlignment(self.practise_button, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.practise_button.setMaximumSize(300, 36)
        self.practise_button.setMinimumSize(80, 30)
        self.practise_button.clicked.connect(self.jump_to_practise)  # type: ignore

        self.replay_bt = QtWidgets.QPushButton(self.centralwidget)
        self.bts_Layout.addWidget(self.replay_bt)
        self.bts_Layout.setAlignment(self.replay_bt, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.replay_bt.setMaximumSize(300, 36)
        self.replay_bt.setMinimumSize(80, 30)
        self.replay_bt.clicked.connect(self.open_replay)  # type: ignore

        self.analyzer_bt = QtWidgets.QPushButton(self.centralwidget)
        self.bts_Layout.addWidget(self.analyzer_bt)
        self.bts_Layout.setAlignment(self.analyzer_bt, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.analyzer_bt.setMaximumSize(300, 36)
        self.analyzer_bt.setMinimumSize(80, 30)
        self.analyzer_bt.clicked.connect(self.open_analyzer)  # type: ignore

        self.notebook_bt = QtWidgets.QPushButton(self.centralwidget)
        self.bts_Layout.addWidget(self.notebook_bt)
        self.bts_Layout.setAlignment(self.notebook_bt, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.notebook_bt.setMaximumSize(300, 36)
        self.notebook_bt.setMinimumSize(80, 30)
        self.notebook_bt.clicked.connect(self.open_notebook)  # type: ignore

        self.save_log_bt = QtWidgets.QPushButton(self.centralwidget)
        self.bts_Layout.addWidget(self.save_log_bt)
        self.bts_Layout.setAlignment(self.save_log_bt, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.save_log_bt.setMaximumSize(300, 36)
        self.save_log_bt.setMinimumSize(80, 30)
        self.save_log_bt.clicked.connect(self.save_logs_to_file)  # type: ignore

        self.save_rec_bt = QtWidgets.QPushButton(self.centralwidget)
        self.bts_Layout.addWidget(self.save_rec_bt)
        self.bts_Layout.setAlignment(self.save_rec_bt, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.save_rec_bt.setMaximumSize(300, 36)
        self.save_rec_bt.setMinimumSize(80, 30)
        self.save_rec_bt.clicked.connect(self.save_rec_to_file)  # type: ignore
        self.gridLayout.addLayout(self.bts_Layout, 2, 0, 1, 2)

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)

        category_info_t = category_info | {'?': ['?']}
        self.menu_ptn = QtWidgets.QMenu(self.menubar)
        self.menu_ptn.setObjectName("menuMENU")
        # 遍历分类字典创建二级菜单
        for category, patterns in category_info_t.items():
            submenu = QtWidgets.QMenu(category, self.menu_ptn)
            submenu.setObjectName(f"submenu_{category}")

            for ptn in patterns:
                action = QtWidgets.QAction(ptn, self)
                action.triggered.connect(lambda: self.menu_selected(0))
                submenu.addAction(action)

            self.menu_ptn.addMenu(submenu)
        self.menubar.addAction(self.menu_ptn.menuAction())

        self.menu_tgt = QtWidgets.QMenu(self.menubar)
        self.menu_tgt.setObjectName("menuMENU")
        for ptn in ["128", "256", "512", "1024", "2048", "4096", "8192", '?']:
            m = QtWidgets.QAction(ptn, self)
            m.triggered.connect(lambda: self.menu_selected(1))  # type: ignore
            self.menu_tgt.addAction(m)
        self.menubar.addAction(self.menu_tgt.menuAction())
        self.menu_pos = QtWidgets.QMenu(self.menubar)
        self.menu_pos.setObjectName("menuMENU")
        for ptn in ["0", "1", "2", '?']:
            m = QtWidgets.QAction(ptn, self)
            m.triggered.connect(lambda: self.menu_selected(2))  # type: ignore
            self.menu_pos.addAction(m)
        self.menubar.addAction(self.menu_pos.menuAction())
        QtCore.QMetaObject.connectSlotsByName(self)
        self.setCentralWidget(self.centralwidget)

        self.retranslateUi()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Tester", "Tester"))
        self.menu_ptn.setTitle(_translate("Tester", "Pattern"))
        self.menu_pos.setTitle(_translate("Tester", "Position"))
        self.menu_tgt.setTitle(_translate("Tester", "Target"))
        self.set_board_bt.setText(_translate("Tester", "SET"))
        self.save_log_bt.setText(_translate("Tester", "Save Logs"))
        self.save_rec_bt.setText(_translate("Tester", "Save Replay"))
        self.analyzer_bt.setText(_translate("Tester", "Analyze Verse Replay"))
        self.notebook_bt.setText(_translate("Tester", "Mistakes Notebook"))
        self.replay_bt.setText(_translate("Tester", "Review Replay"))
        self.practise_button.setText(_translate("Tester", "Practise"))
        self.dis_text_checkBox.setText(_translate("Tester", "Display Text"))

    def menu_selected(self, i):
        self.isProcessing = False
        self.pattern[i] = self.sender().text()
        self.text_display.add_text('_'.join(self.pattern))
        self.text_display.update_text()

        if '?' not in self.pattern[:2]:
            if self.pattern[0] in ['2x4', '3x3', '3x4']:
                self.gameframe.set_to_variant(self.pattern[0])
            else:
                self.gameframe.set_to_44()

            if self.pattern[0] in ("444", "LL", "L3") and self.pattern[2] != '?':
                self.full_pattern = '_'.join(self.pattern)
                path_found, path_list = self.init()
                if path_found:
                    self.book_reader.dispatch(path_list, self.pattern[0], self.pattern[1])
                    self.init_random_state()
            elif self.pattern[0] not in ("444", "LL", "L3"):
                self.full_pattern = '_'.join(self.pattern[:2])
                path_found, path_list = self.init()
                if path_found:
                    self.book_reader.dispatch(path_list, self.pattern[0], self.pattern[1])
                    self.init_random_state()

            self.reader_thread.current_pattern = self.full_pattern
            self.reader_thread.pattern_settings = self.pattern

    def init(self):
        self.text_display.clear_text()
        self.gameframe.score = 0
        self.gameframe.combo = 0
        self.gameframe.goodness_of_fit = 1
        self.gameframe.max_combo = 0
        self.gameframe.performance_stats = {
            "**Perfect!**": 0,
            "**Excellent!**": 0,
            "**Nice try!**": 0,
            "**Not bad!**": 0,
            "**Mistake!**": 0,
            "**Blunder!**": 0,
            "**Terrible!**": 0,
        }

        path = SingletonConfig().config['filepath_map'].get(self.full_pattern, [])
        if not path:
            self.text_display.add_text(self.tr('Table file path not found!'))
            self.text_display.update_text()
            return False, path
        else:
            self.text_display.add_text(
                self.tr("You have selected: ") +
                "**" + self.full_pattern + "**. " +
                self.tr("Loading...")
            )
            self.text_display.update_text()
            QApplication.processEvents()
            return True, path

    def init_random_state(self):
        path_list = SingletonConfig().config['filepath_map'].get(self.full_pattern, [])
        self.gameframe.board_encoded = self.book_reader.get_random_state(path_list, self.full_pattern)
        self.gameframe.board_encoded = self.do_random_rotate(self.gameframe.board_encoded)
        self.gameframe.board = bm.decode_board(self.gameframe.board_encoded)
        self.result = self.book_reader.move_on_dic(
            self.gameframe.board, self.pattern[0], self.pattern[1], self.full_pattern)
        self.text_display.lines[0] = self.text_display.lines[0].replace(self.tr(' Loading...'), '')
        self.text_display.add_text(self.tr("We'll start from:"))
        self.text_display.print_board(self.gameframe.board)
        self.gameframe.update_all_frame(self.gameframe.board)
        self.gameframe.setFocus()

        self.step_count = 0
        self.record[0][0] = self.gameframe.board_encoded

    def set_board_init(self):
        self.isProcessing = False
        path_found, path_list = self.init()
        if path_found:
            self.gameframe.board_encoded = np.uint64(int(self.board_state.text(), 16))
            self.gameframe.board = bm.decode_board(self.gameframe.board_encoded)
            self.result = self.book_reader.move_on_dic(self.gameframe.board, self.pattern[0], self.pattern[1],
                                                 self.full_pattern)
            self.text_display.lines[0] = self.text_display.lines[0].replace(self.tr(' Loading...'), '')
            self.text_display.add_text(self.tr("We'll start from:"))
            self.text_display.print_board(self.gameframe.board)
            self.gameframe.update_all_frame(self.gameframe.board)
            self.gameframe.setFocus()

            self.step_count = 0
            self.record[0][0] = self.gameframe.board_encoded
        self.gameframe.setFocus()

    def do_random_rotate(self, board_encoded):
        operation_func = random.choice(self.book_reader._book_reader.gen_all_mirror(self.pattern[0]))[-1]
        return np.uint64(bm.encode_board(operation_func(bm.decode_board(board_encoded))))

    def save_logs_to_file(self):
        if self.full_pattern is None or len(self.text_display.lines) < 1:
            return
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        default_filename = f"{self.full_pattern}_{current_time}_{self.gameframe.goodness_of_fit:.4f}_log.txt"
        # 打开文件保存对话框
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Logs",  # 对话框标题
            default_filename,  # 默认文件名
            "Text Files (*.txt);;All Files (*)",  # 文件过滤器
        )
        if file_path:
            # 如果用户选择了文件路径，写入文件
            with open(file_path, 'w', encoding='utf-8') as file:
                for line in self.text_display.lines:
                    file.write(line.replace('**', '').replace('&nbsp;', ' ') + '\n')

    def save_rec_to_file(self):
        if self.full_pattern is None or len(self.text_display.lines) < 1:
            return
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        default_filename = f"{self.full_pattern}_{current_time}_{self.gameframe.goodness_of_fit:.4f}_rec.rpl"
        # 打开文件保存对话框
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Logs",  # 对话框标题
            default_filename,  # 默认文件名
            "Text Files (*.rpl);;All Files (*)",  # 文件过滤器
        )
        if file_path:
            # 添加验证数据
            self.record[self.step_count] = (
                self.record[self.step_count][0], 88, 666666666, 233333333, 314159265, 987654321)
            self.record[:self.step_count + 1].tofile(file_path)

    def open_analyzer(self):
        if self.analyze_window is None:
            self.analyze_window = AnalyzeWindow()

        if self.analyze_window.windowState() & QtCore.Qt.WindowState.WindowMinimized:
            self.analyze_window.setWindowState(
                self.analyze_window.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                | QtCore.Qt.WindowState.WindowActive)
        self.analyze_window.show()
        self.analyze_window.activateWindow()
        self.analyze_window.raise_()

    def open_notebook(self):
        if self.notebook_window is None:
            self.notebook_window = MistakeTrainingWindow(mistakes_book)

        if self.notebook_window.windowState() & QtCore.Qt.WindowState.WindowMinimized:
            self.notebook_window.setWindowState(
                self.notebook_window.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                | QtCore.Qt.WindowState.WindowActive)
        self.notebook_window.show()
        self.notebook_window.activateWindow()
        self.notebook_window.raise_()

    def open_replay(self):
        # 添加验证数据
        self.record[self.step_count] = (
            self.record[self.step_count][0], 88, 666666666, 233333333, 314159265, 987654321)
        record = self.record[:self.step_count + 1].copy()

        if self.replay_window is None:
            self.replay_window = ReplayWindow()
            self.replay_window.reset_record(record, self.full_pattern)
        elif len(record) > 1 or len(self.replay_window.gameframe.record) == 0:
            self.replay_window.reset_record(record, self.full_pattern)

        if self.replay_window.windowState() & QtCore.Qt.WindowState.WindowMinimized:
            self.replay_window.setWindowState(
                self.replay_window.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                | QtCore.Qt.WindowState.WindowActive)
        self.replay_window.show()
        self.replay_window.activateWindow()
        self.replay_window.raise_()

        self.replay_window.gameframe.set_use_variant(self.pattern[0])

    def keyPressEvent(self, event):
        if self.isProcessing:
            return
        if event.key() in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_W):
            self.process_input('Up')
            event.accept()
        elif event.key() in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_S):
            self.process_input('Down')
            event.accept()
        elif event.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_A):
            self.process_input('Left')
            event.accept()
        elif event.key() in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_D):
            self.process_input('Right')
            event.accept()
        else:
            super().keyPressEvent(event)  # 其他键交给父类处理

    def process_input(self, direction:str):
        if self.result.get(direction.lower(), None) is None:
            return
        self.isProcessing = True  # 设置标志防止进一步的输入
        board_to_log = np.uint64(self.gameframe.board_encoded)
        self.gameframe.do_move(direction)
        self.one_step(direction, board_to_log)

        self.reader_thread.board_state = self.gameframe.board
        self.reader_thread.start()  # type:ignore

        self.gameframe.setFocus()

    def _show_results(self, result: dict):
        """ 绑定reader_thread信号 """
        self.result = result

        self.text_display.add_text('--------------------------------------------------')
        self.text_display.print_board(self.gameframe.board)
        self.text_display.add_text('')

        best_move = list(self.result.keys())[0]
        if not self.result[best_move]:
            self.text_display.add_text(self.tr("**Game Over**: NO possible moves left."))
            for evaluation, count in self.gameframe.performance_stats.items():
                self.text_display.add_text(f'{evaluation}: {count}\n')
            mistakes_book.save_to_file()

        elif self.result[best_move] == 1:
            self.text_display.add_text(self.tr("**Congratulations!** You're about to reach the target tile."))
            for evaluation, count in self.gameframe.performance_stats.items():
                self.text_display.add_text(f'{evaluation}: {count}\n')
            mistakes_book.save_to_file()

        self.text_display.add_text(self.tr('Total Goodness of Fit: ') + f'{self.gameframe.goodness_of_fit:.4f}' + '\n')
        self.text_display.add_text(self.tr('Maximum Combo: ') + f'{self.gameframe.max_combo}' + '\n')
        self.text_display.update_text()
        QtCore.QTimer.singleShot(10, lambda: setattr(self, 'isProcessing', False))

    def one_step(self, move: str, board_to_log):
        text_list = []
        is_zh = (SingletonConfig().config['language'] == 'zh')
        for key, value in self.result.items():
            if is_zh:
                text_list.append(f"{direction_map[key[0].lower()]}: {value}")
            else:
                text_list.append(f"{key[0].upper()}: {value}")
        text_list.append('')
        best_move = list(self.result.keys())[0]
        self.record_replay(move)

        if not self.result[best_move] or isinstance(self.result[best_move], str):
            return
        if isinstance(self.result[move.lower()], str):
            return
        if self.result[move.lower()] is not None and self.result[best_move] - self.result[move.lower()] <= 3e-10:
            self.gameframe.combo += 1
            self.gameframe.max_combo = max(self.gameframe.max_combo, self.gameframe.combo)
            self.gameframe.performance_stats["**Perfect!**"] += 1
            text_list.append(f"**Perfect! Combo: {self.gameframe.combo}x**")
            if is_zh:
                text_list.append(f"你走的是 {direction_map[move[0].lower()]} ，"
                                 f"最优解正是 **{direction_map[best_move[0].lower()]}**")
            else:
                text_list.append(f"You pressed {move}. And the best move is **{best_move.capitalize()}**")
        else:
            self.gameframe.combo = 0
            loss = self.result[move.lower()] / self.result[best_move] if self.result[best_move] else 1
            self.gameframe.goodness_of_fit *= loss
            # 根据 loss 值提供不同级别的评价
            evaluation = self.evaluation_of_performance(loss)
            self.gameframe.performance_stats[evaluation] += 1

            mistakes_book.add_mistake(self.full_pattern, board_to_log, loss, best_move)

            text_list.append(evaluation)
            if is_zh:
                text_list.append(f'单步损失: {1 - loss:.4f}, 吻合度: {self.gameframe.goodness_of_fit:.4f}')
                text_list.append(f"你走的是 {direction_map[move[0].lower()]} ，"
                                 f"但最优解是 **{direction_map[best_move[0].lower()]}**")
            else:
                text_list.append(f'one-step loss: {1 - loss:.4f}, goodness of fit: {self.gameframe.goodness_of_fit:.4f}')
                text_list.append(f"You pressed {move}. But the best move is **{best_move.capitalize()}**")
        self.text_display.add_text(text_list)

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

    def record_replay(self, direction: str):
        direct = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3}[direction.capitalize()]
        encoded = self.encode(direct, self.gameframe.newtile_pos, self.gameframe.newtile - 1)
        success_rates = []
        for d in ('left', 'right', 'up', 'down'):
            rate = self.result[d]
            if isinstance(rate, (int, float)):
                success_rates.append(np.uint32(rate * 4e9))
            else:
                success_rates.append(np.uint32(0))
        self.record[self.step_count] = (self.record[self.step_count][0], encoded, *success_rates)
        self.step_count += 1
        self.record[self.step_count][0] = self.gameframe.board_encoded

    def jump_to_practise(self):
        practise_signal.board_update.emit(self.gameframe.board_encoded, self.full_pattern)

    def dis_text_state_change(self):
        self.text_display.show_text = self.dis_text_checkBox.isChecked()
        self.text_display.update_text()
        SingletonConfig().config['dis_text'] = self.dis_text_checkBox.isChecked()

    @staticmethod
    def encode(a, b, c):
        return np.uint8(((a << 5) | (b << 1) | c) & 0xFF)

    def closeEvent(self, event):
        mistakes_book.save_to_file()
        event.accept()


class ReaderWorker(QtCore.QThread):
    """ 常驻工作线程，负责查表 """
    result_ready = QtCore.pyqtSignal(dict)

    def __init__(self, reader: BookReaderDispatcher, board_state:np.uint64,
                 pattern_settings: list, current_pattern: str):
        super(ReaderWorker, self).__init__()
        self.book_reader = reader
        self.board_state = board_state
        self.pattern_settings = pattern_settings
        self.current_pattern = current_pattern

    def run(self):
        result = self.book_reader.move_on_dic(self.board_state, self.pattern_settings[0], self.pattern_settings[1],
                                              self.current_pattern, self.pattern_settings[2])
        self.result_ready.emit(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())
