import sys

import numpy as np
import random
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit
from PyQt5.QtGui import QIcon

from BookReader import BookReader
from Config import SingletonConfig
from Gamer import BaseBoardFrame
from Analyzer import AnalyzeWindow


# noinspection PyAttributeOutsideInit
class ScrollTextDisplay(QWidget):
    def __init__(self, max_lines=100, parent=None):
        super(ScrollTextDisplay, self).__init__(parent)
        self.max_lines = max_lines  # 最大行数
        self.setupUi()
        self.lines = []

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
        text = '\n\n'.join(self.lines[-self.max_lines:])
        if len(self.lines) > self.max_lines:
            text = self.lines[0] + '\n\n' + text
        self.text_edit.setMarkdown(text)
        # 保持滚动条在最下方
        self.scroll_bar.setValue(self.scroll_bar.maximum())

    def print_board(self, board):
        for row in board:
            # 格式化每一行
            formatted_row = []
            for item in row:
                if item == 0:
                    formatted_element = '_'
                elif item == 32768:
                    formatted_element = 'x'
                elif item >= 1024:
                    formatted_element = f"{item // 1024}k"
                else:
                    formatted_element = str(item)
                formatted_row.append(formatted_element.rjust(4, ' '))
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

    def update_frame(self, value, row, col, anim=False):
        """重写方法以配合不显示32k格子数字的设置"""
        super().update_frame(value, row, col, anim)
        if value == 32768 and not SingletonConfig().config.get('dis_32k', False):
            self.game_square.labels[row][col].setText('')

    def mousePressEvent(self, event):
        self.setFocus()

    def set_to_variant(self, pattern: str):
        self.set_use_variant(pattern)
        self.board_encoded = self.v_inits[pattern][0]
        self.board = self.mover.decode_board(self.board_encoded)
        self.update_all_frame(self.board)

    def set_to_44(self):
        if self.use_variant_mover != 0:
            self.set_use_variant('')
            self.board = np.zeros((4, 4), dtype=np.int32)
            self.board_encoded = self.mover.encode_board(self.board)
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

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QIcon(r"pic\2048.ico"))
        self.setMinimumSize(450, 320)
        self.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(6, 6, 6, 6)
        self.gridLayout.setVerticalSpacing(15)
        self.gridLayout.setHorizontalSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self.gameframe = TestFrame(self.centralwidget)
        self.gridLayout.addWidget(self.gameframe, 0, 0, 1, 1)
        self.gameframe.setMinimumSize(240, 240)
        self.text_display = ScrollTextDisplay(max_lines=200)
        self.gridLayout.addWidget(self.text_display, 0, 1, 1, 1)
        self.text_display.setMinimumSize(120, 240)

        self.save_bt = QtWidgets.QPushButton(self.centralwidget)
        self.gridLayout.addWidget(self.save_bt, 1, 1, 1, 1)
        self.gridLayout.setAlignment(self.save_bt, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.save_bt.setMaximumSize(300, 36)
        self.save_bt.setMinimumSize(80, 30)
        self.save_bt.clicked.connect(self.save_logs_to_file)  # type: ignore

        self.analyzer_bt = QtWidgets.QPushButton(self.centralwidget)
        self.gridLayout.addWidget(self.analyzer_bt, 1, 0, 1, 1)
        self.gridLayout.setAlignment(self.analyzer_bt, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.analyzer_bt.setMaximumSize(300, 36)
        self.analyzer_bt.setMinimumSize(80, 30)
        self.analyzer_bt.clicked.connect(self.open_analyzer)  # type: ignore

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 522, 22))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)

        self.menu_ptn = QtWidgets.QMenu(self.menubar)
        self.menu_ptn.setObjectName("menuMENU")
        for ptn in ['t', 'L3', '442', 'LL', '444', '4431', "4441", "4432", '4442', 'free8', 'free9', 'free10',
                    "3433", "3442", "3432", "2433", "movingLL",
                    'free8w', 'free9w', 'free10w', "free11w", '2x4', '3x3', '3x4', '?']:
            m = QtWidgets.QAction(ptn, self)
            m.triggered.connect(lambda: self.menu_selected(0))  # type: ignore
            self.menu_ptn.addAction(m)
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

    def resizeEvent(self, event):
        super(TestWindow, self).resizeEvent(event)
        self.text_display.setMaximumWidth(self.width() - self.height() + 50)
        self.gameframe.resize(self.height(), self.height())

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Tester", "Tester"))
        self.menu_ptn.setTitle(_translate("Tester", "Pattern"))
        self.menu_pos.setTitle(_translate("Tester", "Position"))
        self.menu_tgt.setTitle(_translate("Tester", "Target"))
        self.save_bt.setText(_translate("Tester", "Save Logs"))
        self.analyzer_bt.setText(_translate("Tester", "Analyze Verse Replay"))

    def menu_selected(self, i):
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
                self.init()
            elif self.pattern[0] not in ("444", "LL", "L3"):
                self.full_pattern = '_'.join(self.pattern[:2])
                self.init()

    def init(self):
        self.text_display.clear_text()
        self.gameframe.score = 0
        self.gameframe.combo = 0
        self.gameframe.goodness_of_fit = 1

        path = SingletonConfig().config['filepath_map'].get(self.full_pattern, '')
        if not path:
            self.text_display.add_text('Table file path not found!')
            self.text_display.update_text()
            return
        else:
            self.text_display.add_text(f"You have selected: **{self.full_pattern}**. Loading...")
            self.text_display.update_text()
            QApplication.processEvents()
        self.gameframe.board_encoded = BookReader.get_random_state(path, self.full_pattern)
        self.gameframe.board_encoded = self.do_random_rotate(self.gameframe.board_encoded)
        self.gameframe.board = BookReader.bm.decode_board(self.gameframe.board_encoded)
        self.result = BookReader.move_on_dic(self.gameframe.board, self.pattern[0], self.pattern[1], self.full_pattern)
        self.text_display.lines[0] = self.text_display.lines[0].replace(' Loading...', '')
        self.text_display.add_text("We'll start from:")
        self.text_display.print_board(self.gameframe.board)
        self.text_display.add_text('--------------------------------------------------')
        self.gameframe.update_all_frame(self.gameframe.board)
        self.gameframe.setFocus()

    def do_random_rotate(self, board_encoded):
        return np.uint64(BookReader.bm.encode_board(random.choice(BookReader.gen_all_mirror(
            BookReader.bm.decode_board(board_encoded), self.pattern[0]))[-1]))

    def save_logs_to_file(self):
        if self.full_pattern is None or len(self.text_display.lines) < 1:
            return
        # 打开文件保存对话框
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Logs",  # 对话框标题
            self.full_pattern + " log",  # 默认文件名
            "Text Files (*.txt);;All Files (*)",  # 文件过滤器
        )
        if file_path:
            # 如果用户选择了文件路径，写入文件
            with open(file_path, 'w', encoding='utf-8') as file:
                for line in self.text_display.lines:
                    file.write(line.replace('**', '') + '\n')

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

    def keyPressEvent(self, event):
        if self.isProcessing:
            return
        if event.key() in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_W):
            self.process_input('Up')
        elif event.key() in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_S):
            self.process_input('Down')
        elif event.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_A):
            self.process_input('Left')
        elif event.key() in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_D):
            self.process_input('Right')
        else:
            super().keyPressEvent(event)  # 其他键交给父类处理

    def process_input(self, direction):
        if self.result.get(direction.lower(), None) is None:
            return
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.one_step(direction)
        self.result = BookReader.move_on_dic(self.gameframe.board, self.pattern[0], self.pattern[1], self.full_pattern)
        self.text_display.print_board(self.gameframe.board)
        self.text_display.add_text('--------------------------------------------------')
        if not self.result[list(self.result.keys())[0]]:
            self.text_display.add_text("**Game Over**: NO possible moves left.")
        self.text_display.update_text()
        self.gameframe.update_all_frame(self.gameframe.board)
        self.gameframe.setFocus()
        self.isProcessing = False

    def one_step(self, move):
        text_list = []
        for key, value in self.result.items():
            text_list.append(f"{key.ljust(5,' ')}: {value}")
        text_list.append('')
        best_move = list(self.result.keys())[0]
        if move == best_move.capitalize() or self.result[move.lower()] == 1:
            self.gameframe.combo += 1
            text_list.append(f"**Perfect! Combo: {self.gameframe.combo}x**")
            text_list.append(f"You pressed {move}. And the best move is **{best_move.capitalize()}**")
        else:
            self.gameframe.combo = 0
            loss = self.result[move.lower()] / self.result[best_move] if self.result[best_move] else 1
            self.gameframe.goodness_of_fit *= loss
            # 根据 loss 值提供不同级别的评价
            text_list.append(self.evaluation_of_performance(loss))
            text_list.append(f'one-step loss: {1 - loss:.4f}, goodness of fit: {self.gameframe.goodness_of_fit:.4f}')
            text_list.append(f"You pressed {move}. But the best move is **{best_move.capitalize()}**")
            text_list.append('')
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())
