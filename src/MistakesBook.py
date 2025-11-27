import os
import pickle
from collections import defaultdict

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer

import BoardMover as bm
from Config import SingletonConfig
from Gamer import BaseBoardFrame
from SignalHub import practice_signal
import Calculator


direction_map = defaultdict(lambda: "？")
direction_map.update({
    'Up': "上",
    'Down': "下",
    'Left': "左",
    'Right': "右",
    '?': "？",
})


class MistakesBook:
    def __init__(self):
        # 数据结构：{full_pattern: {board_encoded: (count, total_loss)}}
        self.mistakes = defaultdict(dict)
        self.filename = 'mistakes_book.pkl'
        self.load_from_file()

    def remove_mistake(self, full_pattern: str, board_encoded: np.uint64):
        """从错题本中删除一个局面"""
        if full_pattern in self.mistakes:
            pattern_dict = self.mistakes[full_pattern]
            if board_encoded in pattern_dict:
                del pattern_dict[board_encoded]
                # 如果该pattern下没有错误了，删除整个pattern
                if not pattern_dict:
                    del self.mistakes[full_pattern]

    def get_mistakes_for_pattern(self, full_pattern: str):
        """获取指定pattern的所有错误局面"""
        return self.mistakes.get(full_pattern, {})

    def get_all_patterns(self):
        """获取所有有错误记录的pattern"""
        return list(self.mistakes.keys())

    def save_to_file(self):
        """将错题本保存到文件"""
        with open(self.filename, 'wb') as f:
            pickle.dump(dict(self.mistakes), f)

    def load_from_file(self):
        """从文件加载错题本"""
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                data = pickle.load(f)
                self.mistakes = defaultdict(dict, data)

    def add_mistake(self, full_pattern: str, board_encoded: np.uint64, loss: float,
                   best_move):
        if loss and loss > SingletonConfig().config.get('notebook_threshold', 0.999):
            return
        pattern_dict = self.mistakes[full_pattern]
        all_symm = [board_encoded, Calculator.ReverseLR(board_encoded), Calculator.ReverseUD(board_encoded),
                    Calculator.ReverseUL(board_encoded), Calculator.ReverseUR(board_encoded),
                    Calculator.Rotate180(board_encoded), Calculator.RotateL(board_encoded),
                    Calculator.RotateR(board_encoded)]

        for bd in all_symm:
            if bd in pattern_dict:
                count, total_loss, move = pattern_dict[bd]
                pattern_dict[bd] = (count + 1, total_loss + (1 - loss), move)
                break
        else:
            pattern_dict[board_encoded] = (1, 1 - loss, best_move)


mistakes_book = MistakesBook()


class NotebookFrame(BaseBoardFrame):
    def update_board(self, board_encoded):
        self.board_encoded = np.uint64(board_encoded)
        self.board = bm.decode_board(self.board_encoded)
        self.update_all_frame(self.board)

    def update_frame(self, value, row, col):
        """重写方法以配合不显示32k格子数字的设置"""
        super().update_frame(value, row, col)
        if value == 32768 and not SingletonConfig().config.get('dis_32k', False):
            self.game_square.labels[row][col].setText('')


# noinspection PyAttributeOutsideInit
class MistakeTrainingWindow(QtWidgets.QMainWindow):
    def __init__(self, mistakes_book, parent=None):
        super().__init__(parent)
        self.mistakes_book = mistakes_book
        self.current_pattern = None
        self.current_board = None
        self.current_best_move = None
        self.current_count = None
        self.combo = 0
        self.correct = 0
        self.incorrect = 0
        self.unseen_boards = list()  # 存储尚未出现的局面
        self.answered = 0

        self.setupUi()
        self.setWindowTitle(self.tr("Notebook"))
        self.setMinimumSize(960, 600)

        self.next_problem_timer = QTimer()

    def setupUi(self):
        # 创建菜单栏
        self.menu_bar = self.menuBar()
        self.pattern_menu = self.menu_bar.addMenu(self.tr("Pattern"))

        # 创建主布局
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        main_layout.setObjectName('main_layout')

        # 左侧：棋盘显示
        self.gameframe = NotebookFrame(self.centralwidget)
        main_layout.addWidget(self.gameframe, 3)

        # 右侧：控制面板
        control_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_layout, 2)

        lang = SingletonConfig().config['language']
        is_zh = lang == 'zh'

        # 当前定式
        self.pattern_text = QtWidgets.QLabel(self.centralwidget)
        self.pattern_text.setStyleSheet("font: 2400 32pt \"Cambria\";")
        self.pattern_text.setObjectName("pattern_text")
        self.pattern_text.setMaximumSize(QtCore.QSize(640, 80))
        control_layout.addWidget(self.pattern_text)
        self.pattern_text.setText('')

        # 方向按钮
        self.direction_buttons = {}
        for direction in ['Up', 'Down', 'Left', 'Right']:
            if is_zh:
                btn = QtWidgets.QPushButton(direction_map[direction])
            else:
                btn = QtWidgets.QPushButton(direction)
            btn.setObjectName(direction + '_btn')
            btn.setMinimumHeight(60)
            btn.setStyleSheet("font-size: 24px; font-weight: bold;")
            btn.clicked.connect(lambda _, d=direction: self.check_answer(d))
            self.direction_buttons[direction] = btn
            control_layout.addWidget(btn)

        # 下一题、练习、删除按钮
        button_row = QtWidgets.QHBoxLayout()
        control_layout.addLayout(button_row)

        self.next_btn = QtWidgets.QPushButton(self.tr("Next"))
        self.next_btn.setObjectName('next_btn')
        self.next_btn.setMinimumHeight(60)
        self.next_btn.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.next_btn.clicked.connect(self.next_problem)
        button_row.addWidget(self.next_btn)

        self.practice_btn = QtWidgets.QPushButton(self.tr("Practice"))
        self.practice_btn.setObjectName('practice_btn')
        self.practice_btn.setMinimumHeight(60)
        self.practice_btn.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.practice_btn.clicked.connect(self.jump_to_practice)
        button_row.addWidget(self.practice_btn)

        self.delete_btn = QtWidgets.QPushButton(self.tr("Delete"))
        self.delete_btn.setObjectName('delete_btn')
        self.delete_btn.setMinimumHeight(60)
        self.delete_btn.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.delete_btn.clicked.connect(self.delete_current)
        button_row.addWidget(self.delete_btn)

        # 权重模式选择
        self.weight_combo = QtWidgets.QComboBox(self.centralwidget)
        self.weight_combo.setMinimumHeight(40)
        self.weight_combo.addItems([
            self.tr("Mistake Count Priority"),
            self.tr("Total Loss Priority"),
            self.tr("Combined Weighted")
        ])
        self.weight_combo.currentIndexChanged.connect(self.on_weight_mode_changed)
        control_layout.addWidget(self.weight_combo)

        self.setCentralWidget(self.centralwidget)

        # 状态栏
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        # 初始化模式菜单
        self.update_pattern_menu()

        # 初始状态
        self.set_button_colors(None)
        self.next_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)

    def update_pattern_menu(self):
        """更新模式菜单"""
        self.pattern_menu.clear()
        patterns = self.mistakes_book.get_all_patterns()

        if not patterns:
            no_pattern_action = QtWidgets.QAction(self.tr("No available patterns"), self)
            no_pattern_action.setEnabled(False)
            self.pattern_menu.addAction(no_pattern_action)
            return

        for pattern in patterns:
            action = QtWidgets.QAction(pattern, self)
            action.triggered.connect(lambda _, p=pattern: self.select_pattern(p))
            self.pattern_menu.addAction(action)

    def select_pattern(self, pattern):
        """选择训练模式"""
        self.current_pattern = pattern
        self.pattern_text.setText(pattern)
        self.reset_unseen_boards()  # 重置未出现局面集合
        self.next_problem()
        self.next_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)

    def reset_unseen_boards(self):
        """重置未出现局面集合，并根据权重进行随机排序"""
        if not self.current_pattern:
            return

        mistakes = self.mistakes_book.get_mistakes_for_pattern(self.current_pattern)
        if not mistakes:
            return

        # 获取当前权重模式
        weight_mode = self.weight_combo.currentIndex()

        # 计算所有局面的权重
        boards = list(mistakes.keys())
        weights = self.calculate_weights(boards, mistakes, weight_mode)

        # 根据权重进行随机排序
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        shuffled_indices = np.random.choice(
            len(boards),
            size=min(len(boards), 1000),
            replace=False,
            p=normalized_weights
        )
        self.unseen_boards = [np.uint64(boards[i]) for i in shuffled_indices]
        self.correct = 0
        self.incorrect = 0

    def next_problem(self):
        """随机选择下一个问题"""
        # 停止任何待处理的计时器
        if self.next_problem_timer.isActive():
            self.next_problem_timer.stop()

        if not self.current_pattern:
            return

        mistakes = self.mistakes_book.get_mistakes_for_pattern(self.current_pattern)

        # 如果没有错误记录
        if not mistakes:
            QtWidgets.QMessageBox.information(self, self.tr("Information"),
                                              self.tr("No mistakes recorded for this pattern"))
            return

        # 如果所有局面都已出现，重置未出现局面集合
        if not self.unseen_boards:
            self.reset_unseen_boards()

        # 从列表中取出下一个问题
        self.current_board = self.unseen_boards.pop(0)  # 从列表开头取出

        # 获取最佳移动方向
        self.current_count, _, self.current_best_move = mistakes[self.current_board]

        # 更新棋盘显示
        self.gameframe.update_board(self.current_board)

        # 重置按钮颜色和状态
        self.set_button_colors(None)

        # 重新启用所有方向按钮
        self.answered = 0
        self.setFocus()

        self.statusbar.showMessage(self.tr('Correct: ') + str(self.correct) + self.tr('  Incorrect: ') +
                                   str(self.incorrect) + self.tr('  Remaining: ') + str(len(self.unseen_boards))
                                   , 3000)

    @staticmethod
    def calculate_weights(boards, mistakes, mode):
        """根据权重模式计算选择权重"""
        weights = []

        for board in boards:
            count, total_loss, _ = mistakes[board]

            if mode == 0:  # 错误次数优先
                weights.append(count ** 2)
            elif mode == 1:  # 总损失优先
                weights.append(total_loss ** 2)
            else:  # 综合
                # 简单实现：错误次数和总损失的乘积
                weights.append(count * total_loss)

        if sum(weights) == 0:
            weights = [1] * len(weights)

        return weights

    def check_answer(self, selected_direction):
        """检查用户选择的答案是否正确"""
        if not self.current_board or not self.current_best_move or self.answered == 1:
            return

        # 获取最佳移动方向（字符串格式）
        best_move_str = self.current_best_move.lower()

        # 检查答案
        is_correct = (selected_direction.lower() == best_move_str)

        # 禁用所有方向按钮，防止重复作答
        self.answered = 1

        # 更新按钮颜色
        if is_correct:
            # 正确：选中的按钮变绿
            self.direction_buttons[selected_direction].setStyleSheet("background-color: green;")

            # 自动下一题
            self.next_problem_timer.singleShot(1500, self.next_problem)

            # 减少错误次数（最小为1）
            # self.update_mistake_count(-1)
            self.combo += 1
            self.correct += 1
        else:
            # 错误：选中的按钮变红
            self.direction_buttons[selected_direction].setStyleSheet("background-color: red;")

            # 正确答案的按钮变绿
            self.direction_buttons[self.current_best_move.capitalize()].setStyleSheet("background-color: green;")

            # 增加错误次数
            self.update_mistake_count(1)
            self.combo = 0
            self.incorrect += 1

        if self.combo:
            self.statusbar.showMessage(self.tr('Combo: ') + str(self.combo) + 'x', 1500)
        else:
            self.statusbar.showMessage(self.tr('Mistake Count: ') + str(self.current_count) + 'x', 1500)

    def update_mistake_count(self, delta):
        """更新错误次数"""
        if not self.current_board or not self.current_pattern:
            return

        # 获取当前错误记录
        mistakes = self.mistakes_book.get_mistakes_for_pattern(self.current_pattern)
        if self.current_board not in mistakes:
            return

        count, total_loss, best_move = mistakes[self.current_board]

        # 更新错误次数（最小为1）
        new_count = max(1, count + delta)

        # 更新错题本
        self.mistakes_book.mistakes[self.current_pattern][self.current_board] = (
            new_count, total_loss, best_move
        )

    def set_button_colors(self, correct_direction):
        """设置按钮颜色"""
        for direction, btn in self.direction_buttons.items():
            if direction == correct_direction:
                btn.setStyleSheet("background-color: green;")
            else:
                btn.setStyleSheet("")

    def delete_current(self):
        """删除当前局面"""
        if not self.current_board or not self.current_pattern:
            return

        if self.next_problem_timer.isActive():
            self.next_problem_timer.stop()

        # 确认删除
        reply = QtWidgets.QMessageBox.question(
            self, self.tr("Confirm deletion"),
            self.tr("Are you sure you want to delete this position forever?"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            # 从错题本中删除
            self.mistakes_book.remove_mistake(self.current_pattern, self.current_board)

            # 选择下一个问题
            self.next_problem()

    def jump_to_practice(self):
        practice_signal.board_update.emit(self.gameframe.board_encoded, self.current_pattern)

    def on_weight_mode_changed(self):
        """权重模式变化时的处理"""
        if self.current_pattern:
            self.reset_unseen_boards()
            self.next_problem()

    def keyPressEvent(self, event):
        # 如果没有当前局面或最佳移动方向，不处理按键
        if not self.current_board or not self.current_best_move:
            super().keyPressEvent(event)
            return

        # 检查用户是否可以作答
        if self.answered == 1:
            super().keyPressEvent(event)
            return

        key = event.key()
        if key in (QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete):
            self.delete_current()
        elif key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.next_problem()

        direction = None

        # 方向键映射
        if key == QtCore.Qt.Key.Key_Up:
            direction = 'Up'
        elif key == QtCore.Qt.Key.Key_Down:
            direction = 'Down'
        elif key == QtCore.Qt.Key.Key_Left:
            direction = 'Left'
        elif key == QtCore.Qt.Key.Key_Right:
            direction = 'Right'
        # WASD 键映射
        elif key == QtCore.Qt.Key.Key_W:
            direction = 'Up'
        elif key == QtCore.Qt.Key.Key_S:
            direction = 'Down'
        elif key == QtCore.Qt.Key.Key_A:
            direction = 'Left'
        elif key == QtCore.Qt.Key.Key_D:
            direction = 'Right'

        # 如果检测到有效方向键，模拟按钮点击
        if direction:
            self.check_answer(direction)
            event.accept()
        else:
            super().keyPressEvent(event)  # 其他按键交给父类处理

    def closeEvent(self, event):
        """关闭窗口时保存错题本"""
        if self.next_problem_timer.isActive():
            self.next_problem_timer.stop()
        self.mistakes_book.save_to_file()
        super().closeEvent(event)

    def showEvent(self, event):
        self.update_pattern_menu()  # 更新模式菜单
        super().showEvent(event)

    def retranslateUi(self):
        self.setWindowTitle(self.tr("Notebook"))
        self.pattern_menu.setTitle(self.tr("Pattern"))
        lang = SingletonConfig().config['language']
        is_zh = lang == 'zh'
        for direction, btn in self.direction_buttons.items():
            if is_zh:
                btn.setText(direction_map[direction])
            else:
                btn.setText(direction)
        self.next_btn.setText(self.tr("Next"))
        self.practice_btn.setText(self.tr("Practice"))
        self.delete_btn.setText(self.tr("Delete"))
        self.weight_combo.clear()
        self.weight_combo.addItems([
            self.tr("Mistake Count Priority"),
            self.tr("Total Loss Priority"),
            self.tr("Combined Weighted")
        ])
