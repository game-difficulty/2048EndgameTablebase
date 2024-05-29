import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QPropertyAnimation, QRect, QEasingCurve, QSize, QPoint

from BoardMover import BoardMoverWithScore
from Config import SingletonConfig


def simulate_move_and_merge(line):
    """模拟一行的移动和合并过程，返回新的行和合并发生的位置。"""
    # 移除所有的0，保留非0元素
    non_zero = [value for value in line if value != 0]
    merged = [0] * len(line)  # 合并标记
    new_line = []
    skip = False

    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1] and non_zero[i] != 32768:
            # 发生合并
            new_line.append(2 * non_zero[i])
            merged[len(new_line) - 1] = 1  # 标记合并发生的位置
            skip = True
        else:
            new_line.append(non_zero[i])

    # 用0填充剩余的空间
    new_line.extend([0] * (len(line) - len(new_line)))
    return new_line, merged


def find_merge_positions(current_board, move_direction):
    # 初始化合并位置数组
    merge_positions = np.zeros_like(current_board)
    move_direction = move_direction.lower()

    for i in range(len(current_board)):
        if move_direction in ['left', 'right']:
            line = current_board[i, :]
        else:
            line = current_board[:, i]
        line_to_process = line[::-1] if move_direction in ['down', 'right'] else line
        processed_line, merge_line = simulate_move_and_merge(line_to_process)
        if move_direction in ['right', 'down']:
            merge_line = merge_line[::-1]

        if move_direction in ['left', 'right']:
            merge_positions[i, :] = merge_line
        else:
            merge_positions[:, i] = merge_line

    return merge_positions


# noinspection PyAttributeOutsideInit
class SquareFrame(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.base_font_size = int(self.width() / 14.4 * SingletonConfig().config.get('font_size_factor', 100) / 100)
        self.setupUi()
        self.colors = SingletonConfig().config['colors']
        '''['#043c24', '#06643d', '#1b955b', '#20c175', '#fc56a0', '#e4317f', '#e900ad', '#bf009c',
            '#94008a', '#6a0079', '#3f0067', '#00406b', '#006b9a', '#0095c8', '#00c0f7', '#00c0f7'] + [
            '#ffffff'] * 20'''
        self.anims = [[None for _ in range(self.cols)] for __ in range(self.rows)]

    def resizeEvent(self, event):
        new_size = min(self.width(), self.height())
        self.resize(new_size, new_size)
        super().resizeEvent(event)
        # 更新位置以居中
        self.move(
            (self.parent().width() - new_size) // 2,
            (self.parent().height() - new_size) // 2
        )
        self.base_font_size = int(self.width() / 14.4 * SingletonConfig().config.get('font_size_factor', 100) / 100)
        layout = self.layout()
        if layout:
            margin = int(new_size / 32)
            layout.setContentsMargins(margin, margin, margin, margin)
            layout.setHorizontalSpacing(int(margin / 1.2))
            layout.setVerticalSpacing(int(margin / 1.2))

    def setupUi(self, num_rows=4, num_cols=4):
        self.setMaximumSize(1000,1000)
        self.rows = num_rows
        self.cols = num_cols
        self.setStyleSheet("border-radius: 5px; background-color: rgb(209, 209, 209);")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setObjectName("game_square")
        self.game_grid = QtWidgets.QGridLayout(self)
        self.game_grid.setObjectName("game_grid")
        self.frames = []  # 存储QFrame对象
        self.labels = []  # 存储QLabel对象

        for i in range(num_rows):
            row_frames = []
            row_labels = []
            for j in range(num_cols):
                frame = QtWidgets.QFrame(self)
                frame.setStyleSheet("border-radius: 3px; background-color: rgba(229, 229, 229, 1);")
                frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
                frame.setFrameShadow(QtWidgets.QFrame.Raised)
                frame.setObjectName(f"f{i * num_cols + j}")

                layout = QtWidgets.QGridLayout(frame)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setAlignment(QtCore.Qt.AlignCenter)

                label = QtWidgets.QLabel(frame)
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setStyleSheet(
                    f"font: {self.base_font_size}pt 'Calibri'; font-weight: bold; color: white;")
                label.setText("")
                layout.addWidget(label)

                self.game_grid.addWidget(frame, i, j, 1, 1)
                row_frames.append(frame)
                row_labels.append(label)
            self.frames.append(row_frames)
            self.labels.append(row_labels)

    def animate_appear(self, r, c):
        frame = self.frames[r][c]
        if self.anims[r][c] is not None:
            return
        opacity_effect = QtWidgets.QGraphicsOpacityEffect()
        frame.setGraphicsEffect(opacity_effect)

        opacity_animation = QPropertyAnimation(opacity_effect, b"opacity")
        opacity_animation.setDuration(240)
        opacity_animation.setStartValue(0)
        opacity_animation.setEndValue(1)
        opacity_animation.setEasingCurve(QEasingCurve.OutCubic)
        opacity_animation.finished.connect(lambda: self.cleanup_animation(r, c))
        opacity_animation.start()

        scale_animation = QPropertyAnimation(frame, b"geometry")
        scale_animation.setDuration(240)
        original_rect = frame.geometry()
        start_rect = QRect(frame.geometry().center(), QSize(0, 0))
        scale_animation.setStartValue(start_rect)
        scale_animation.setEndValue(original_rect)
        scale_animation.setEasingCurve(QEasingCurve.OutCubic)
        # scale_animation.finished.connect(lambda: self.cleanup_animation(r, c))
        scale_animation.start()

        self.anims[r][c] = (opacity_animation, scale_animation)

    def animate_pop(self, r, c):
        frame = self.frames[r][c]
        if self.anims[r][c] is not None:
            return
        original_rect = frame.geometry()
        center = original_rect.center()
        big_scale_size = QSize(int(original_rect.width() * 1.2), int(original_rect.height() * 1.2))
        small_scale_size = QSize(int(original_rect.width() * 0), int(original_rect.height() * 0))
        big_scale_rect = QRect(QPoint(center.x() - big_scale_size.width() // 2,
                                      center.y() - big_scale_size.height() // 2), big_scale_size)
        small_scale_rect = QRect(QPoint(center.x() - small_scale_size.width() // 2,
                                        center.y() - small_scale_size.height() // 2), small_scale_size)

        pop_animation = QPropertyAnimation(frame, b"geometry")
        pop_animation.setDuration(240)
        pop_animation.setEasingCurve(QEasingCurve.InOutQuad)
        pop_animation.setKeyValueAt(0, small_scale_rect)
        pop_animation.setKeyValueAt(0.5, big_scale_rect)
        pop_animation.setKeyValueAt(1.0, original_rect)
        pop_animation.finished.connect(lambda: self.cleanup_animation(r, c))

        pop_animation.start()
        self.anims[r][c] = pop_animation

    def cleanup_animation(self, r, c):
        if isinstance(self.anims[r][c], tuple):
            self.anims[r][c][0].deleteLater()
            self.anims[r][c][1].deleteLater()
        else:
            self.anims[r][c].deleteLater()
        self.anims[r][c] = None


# noinspection PyAttributeOutsideInit
class BaseBoardFrame(QtWidgets.QFrame):
    def __init__(self, centralwidget=None):
        super(BaseBoardFrame, self).__init__(centralwidget)
        self.setupUi()
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.board_encoded = np.uint64(0)
        self.score = 0
        self.mover = BoardMoverWithScore()
        self.history = []

        self.newtile_pos = 0
        self.newtile = 1

    def setupUi(self):
        self.setMaximumSize(QtCore.QSize(100000, 100000))
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: rgb(222, 222, 222);")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setObjectName("gameframe")
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setObjectName("gameframe_grid")
        self.game_square = SquareFrame(self)

        self.grid.addWidget(self.game_square, 0, 0, 1, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_all_frame(self.board)

    def update_frame(self, value, row, col, anim=False):
        label = self.game_square.labels[row][col]
        frame = self.game_square.frames[row][col]
        if value != 0:
            label.setText(str(value))
            color = self.game_square.colors[int(np.log2(value)) - 1]
        else:
            label.setText('')
            color = 'rgba(229, 229, 229, 1)'
        fontsize = self.game_square.base_font_size if len(str(value)) < 3 else int(
            self.game_square.base_font_size * 3 / (0.5 + len(str(value))))
        frame.setStyleSheet(f"background-color: {color};")
        label.setStyleSheet(f"font: {fontsize}pt 'Calibri'; font-weight: bold; color: white;")

        if anim:
            self.game_square.animate_appear(row, col)

    def update_all_frame(self, values):
        for i in range(self.game_square.rows):
            for j in range(self.game_square.cols):
                self.update_frame(values[i][j], i, j, anim=False)

    def gen_new_num(self, do_anim=True):
        self.board_encoded, _, new_tile_pos, val = self.mover.gen_new_num(self.board_encoded)
        self.board_encoded = np.uint64(self.board_encoded)
        self.board = self.mover.decode_board(self.board_encoded)
        self.update_frame(2 ** val, new_tile_pos // 4, new_tile_pos % 4, anim=do_anim)
        self.history.append((self.board_encoded, self.score))
        self.newtile_pos, self.newtile = new_tile_pos, val

    def do_move(self, direction, do_gen=True):
        do_anim = SingletonConfig().config['do_animation']
        board_encoded_new, new_score = self.mover.move_board(self.board_encoded, direction)
        board_encoded_new = np.uint64(board_encoded_new)
        if board_encoded_new != self.board_encoded:
            if do_anim[1]:
                self.pop_merged(self.board, direction)
            self.board_encoded = board_encoded_new
            self.board = self.mover.decode_board(self.board_encoded)
            self.update_all_frame(self.board)
            self.score += new_score
            if do_gen:
                self.gen_new_num(do_anim[0])

    def pop_merged(self, board, direction):
        merged_pos = find_merge_positions(board, direction)
        for row in range(self.game_square.rows):
            for col in range(self.game_square.cols):
                if merged_pos[row][col] == 1:
                    self.game_square.animate_pop(row, col)

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.board_encoded, self.score = self.history[-1]
            self.board_encoded = np.uint64(self.board_encoded)
            self.board = self.mover.decode_board(self.board_encoded)
            self.update_all_frame(self.board)
