import sys
from typing import List, Union
import random

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QPropertyAnimation, QRect, QEasingCurve, QSize, QPoint

from MinigameMover import MinigameBoardMover, MinigameBoardMover_mxn
from Config import SingletonConfig
from Calculator import find_merge_positions


# noinspection PyAttributeOutsideInit
class MinigameSquareFrame(QtWidgets.QFrame):
    def __init__(self, parent=None, shape=(4, 4)):
        super().__init__(parent)
        self.base_font_size = int(
            self.width() / (3.6 * shape[1]) * SingletonConfig().config.get('font_size_factor', 100) / 100)
        self.setupUi(shape)
        self.colors = SingletonConfig().config['colors']
        '''['#043c24', '#06643d', '#1b955b', '#20c175', '#fc56a0', '#e4317f', '#e900ad', '#bf009c',
            '#94008a', '#6a0079', '#3f0067', '#00406b', '#006b9a', '#0095c8', '#00c0f7', '#00c0f7'] + [
            '#ffffff'] * 20'''
        self.anims: List[List[Union[None, QtCore.QAbstractAnimation]]] = [
            [None for _ in range(self.cols)] for __ in range(self.rows)
        ]

    def setupUi(self, shape):
        self.setMaximumSize(1000, 1000)
        self.rows, self.cols = shape
        self.setStyleSheet("border-radius: 5px; background-color: rgb(209, 209, 209);")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setObjectName("game_square")
        self.game_grid = QtWidgets.QGridLayout(self)
        self.game_grid.setObjectName("game_grid")
        self.frames = []  # 存储QFrame对象
        self.labels = []  # 存储QLabel对象
        self.small_labels = []  # 存储右下角较小的QLabel对象

        for i in range(self.rows):
            row_frames = []
            row_labels = []
            row_small_labels = []
            for j in range(self.cols):
                frame = QtWidgets.QFrame(self)
                frame.setStyleSheet("border-radius: 3px; background-color: rgba(229, 229, 229, 1);")
                frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
                frame.setFrameShadow(QtWidgets.QFrame.Raised)
                frame.setObjectName(f"f{i * self.cols + j}")

                layout = QtWidgets.QGridLayout(frame)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

                # 主label
                label = QtWidgets.QLabel(frame)
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet(f"""
                            font: {int(self.width() / (3.6 * self.cols))}pt 'Calibri'; 
                            font-weight: bold; color: white; 
                            background-color: transparent;
                        """)
                label.setText("")
                layout.addWidget(label)

                # 右下角较小的label
                small_label = QtWidgets.QLabel(frame)
                small_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom)
                small_label.setStyleSheet(f"""
                    font: {self.base_font_size // 4}pt 'Calibri'; 
                    font-weight: bold; color: white; 
                    background-color: transparent;
                """)
                small_label.setText("")

                self.game_grid.addWidget(frame, i, j, 1, 1)
                row_frames.append(frame)
                row_labels.append(label)
                row_small_labels.append(small_label)

            self.frames.append(row_frames)
            self.labels.append(row_labels)
            self.small_labels.append(row_small_labels)

    def updateGeometry(self):
        # 保持正方形尺寸并居中显示
        parent_size = self.parent().size()
        new_tile_size = int(min(parent_size.width() // self.cols, parent_size.height() // self.rows) * 0.95)
        new_size_x = int(new_tile_size * (self.cols + 0.05))
        new_size_y = int(new_tile_size * (self.rows + 0.05))
        new_x = (parent_size.width() - new_size_x) // 2
        new_y = (parent_size.height() - new_size_y) // 2
        self.setGeometry(new_x, new_y, new_size_x, new_size_y)

        self.base_font_size = int(self.width() / (3.6 * self.cols) *
                                  SingletonConfig().config.get('font_size_factor', 100) / 100)

        margin = int(min(new_size_x, new_size_y) / (8 * min(self.rows, self.cols)))
        self.game_grid.setContentsMargins(margin, margin, margin, margin)
        self.game_grid.setHorizontalSpacing(int(margin / 1.2))
        self.game_grid.setVerticalSpacing(int(margin / 1.2))
        self.update_small_labels_position()

    def update_small_labels_position(self):
        for row_small_labels in self.small_labels:
            for small_label in row_small_labels:
                # 计算位置
                new_x = small_label.parent().width() - small_label.width()
                new_y = small_label.parent().height() - small_label.height()
                small_label.move(new_x, new_y)

                # 更新样式表，只在字体大小改变时更新
                new_font_size = self.base_font_size // 4
                if small_label.font().pointSize() != new_font_size:
                    small_label.setStyleSheet(f"""
                            font: {new_font_size}pt 'Calibri'; 
                            font-weight: bold; color: white; 
                            background-color: transparent;
                        """)
                    small_label.adjustSize()  # 确保调整后大小正确

    def animate_appear(self, r, c):
        frame = self.frames[r][c]
        if self.anims[r][c] is not None:
            return
        opacity_effect = QtWidgets.QGraphicsOpacityEffect()
        frame.setGraphicsEffect(opacity_effect)

        opacity_animation = QPropertyAnimation(opacity_effect, b"opacity")
        opacity_animation.setDuration(150)
        opacity_animation.setStartValue(0)
        opacity_animation.setEndValue(1)
        opacity_animation.setEasingCurve(QEasingCurve.OutCubic)
        opacity_animation.finished.connect(lambda: self.cleanup_animation(r, c))  # type: ignore

        scale_animation = QPropertyAnimation(frame, b"geometry")
        scale_animation.setDuration(150)
        original_rect = frame.geometry()
        start_rect = QRect(frame.geometry().center(), QSize(0, 0))
        scale_animation.setStartValue(start_rect)
        scale_animation.setEndValue(original_rect)
        scale_animation.setEasingCurve(QEasingCurve.OutCubic)
        # scale_animation.finished.connect(lambda: self.cleanup_animation(r, c))

        animation_group = QtCore.QParallelAnimationGroup()
        animation_group.addAnimation(opacity_animation)
        animation_group.addAnimation(scale_animation)
        animation_group.finished.connect(lambda: self.cleanup_animation(r, c))  # type: ignore
        animation_group.start()

        self.anims[r][c] = animation_group

    def animate_pop(self, r, c):
        frame = self.frames[r][c]
        if self.anims[r][c] is not None:
            return
        original_rect = frame.geometry()
        center = original_rect.center()
        big_scale_size = QSize(int(original_rect.width() * 1.15), int(original_rect.height() * 1.15))
        small_scale_size = QSize(int(original_rect.width() * 1), int(original_rect.height() * 1))
        big_scale_rect = QRect(QPoint(center.x() - big_scale_size.width() // 2,
                                      center.y() - big_scale_size.height() // 2), big_scale_size)
        small_scale_rect = QRect(QPoint(center.x() - small_scale_size.width() // 2,
                                        center.y() - small_scale_size.height() // 2), small_scale_size)

        pop_animation = QPropertyAnimation(frame, b"geometry")
        pop_animation.setDuration(120)
        pop_animation.setEasingCurve(QEasingCurve.InOutCubic)
        pop_animation.setKeyValueAt(0, small_scale_rect)
        pop_animation.setKeyValueAt(0.5, big_scale_rect)
        pop_animation.setKeyValueAt(1.0, original_rect)
        pop_animation.finished.connect(lambda: self.cleanup_animation(r, c))  # type: ignore

        pop_animation.start()
        self.anims[r][c] = pop_animation

    def cleanup_animation(self, r, c):
        if self.anims[r][c]:
            self.anims[r][c].deleteLater()
            self.anims[r][c] = None


# minigame系列的棋盘基类，比BaseBoardFrame提供更多接口，以支持不同minigame的逻辑
# noinspection PyAttributeOutsideInit
class MinigameFrame(QtWidgets.QFrame):
    new_game_signal = QtCore.pyqtSignal()

    def __init__(self, centralwidget=None, minigame_type='minigame', shape=(4, 4)):
        super().__init__(centralwidget)
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        self.rows, self.cols = shape
        self.minigame = minigame_type
        self.setupUi()
        try:
            self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos = \
                SingletonConfig().config['minigame_state'][self.difficulty][self.minigame][0]
        except KeyError:
            self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos = \
                np.zeros((self.rows, self.cols), dtype=np.int32), 0, 0, 0, False, 0
        if shape == (4, 4):
            self.mover = MinigameBoardMover()
        else:
            self.mover = MinigameBoardMover_mxn(shape=shape)
        self.current_max_num = self.board.max()
        if np.all(self.board <= 0):
            self.setup_new_game()
            self.current_max_num = self.board.max()
        self.newtile = self.board[self.newtile_pos // self.cols][self.newtile_pos % self.cols]

    def setup_new_game(self):
        self.board, _, self.newtile_pos, self.newtile = self.mover.gen_new_num(
            self.mover.gen_new_num(np.zeros((self.rows, self.cols), dtype=np.int32),
                                   SingletonConfig().config['4_spawn_rate'])[0],
            SingletonConfig().config['4_spawn_rate'])
        self.update_all_frame(self.board)
        self.score = 0
        self.current_max_num = self.board.max()

    def new_game(self):
        #  仅在需要发射新游戏信号时使用此方法
        self.setup_new_game()
        self.new_game_signal.emit()

    def setupUi(self):
        self.setMaximumSize(QtCore.QSize(100000, 100000))
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: rgb(222, 222, 222);")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setObjectName("gameframe")
        self.game_square = MinigameSquareFrame(self, shape=(self.rows, self.cols))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.game_square.updateGeometry()
        self.update_all_frame(self.board)

    def update_frame(self, value, row, col, anim=False):
        label = self.game_square.labels[row][col]
        frame = self.game_square.frames[row][col]
        if value == -1:
            label.setText('')
            frame.setStyleSheet(f"""
                        QFrame#f{row * self.cols + col} {{
                            border-image: url(pic//stop.png) 2 2 2 2 stretch stretch;
                        }}
                        """)
        elif value != 0:
            label.setText(str(2 ** value))
            color = self.game_square.colors[value - 1]
            frame.setStyleSheet(f"background-color: {color};")
        else:
            label.setText('')
            color = 'rgba(229, 229, 229, 1)'
            frame.setStyleSheet(f"background-color: {color};")
        fontsize = self.game_square.base_font_size if (value == -1 or len(str(2 ** value)) < 3) else int(
            self.game_square.base_font_size * 3 / (0.5 + len(str(2 ** value))))
        label.setStyleSheet(
            f"font: {fontsize}pt 'Calibri'; font-weight: bold; color: white; background-color: transparent;")
        self.update_frame_small_label(row, col)

        if anim:
            self.game_square.animate_appear(row, col)

    def update_frame_small_label(self, row, col):
        pass

    def update_all_frame(self, values=None):
        if values is None:
            values = self.board
        for i in range(self.game_square.rows):
            for j in range(self.game_square.cols):
                self.update_frame(values[i][j], i, j, anim=False)

    def gen_new_num(self, do_anim=True):
        self.board, _, new_tile_pos, val = self.mover.gen_new_num(
            self.board, SingletonConfig().config['4_spawn_rate'])
        self.newtile_pos, self.newtile = new_tile_pos, val
        self.update_all_frame(self.board)
        self.update_frame(val, new_tile_pos // self.cols, new_tile_pos % self.cols, anim=do_anim)

    def before_move(self, direct):
        pass

    def before_gen_num(self, direct):
        pass

    def after_gen_num(self):
        pass

    def check_game_passed(self):  # 同时维护self.max_num, self.current_max_num
        if self.board.max() <= 9:
            self.current_max_num = max(self.current_max_num, self.board.max())
            self.max_num = max(self.max_num, self.current_max_num)
            return
        if self.board.max() > self.current_max_num:
            self.current_max_num = self.board.max()
            if self.current_max_num > self.max_num:
                self.max_num = self.current_max_num
                self.is_passed = {12: 3, 11: 2, 10: 1}.get(self.max_num, 4)
                level = {12: 'golden', 11: 'silver', 10: 'bronze'}.get(self.current_max_num, 'golden')
                message = f'You achieved {2 ** self.max_num}!\n You get a {level} trophy!'
                self.show_trophy(f'pic/{level}.png', message)
            else:
                level = {12: 'golden', 11: 'silver', 10: 'bronze'}.get(self.current_max_num, 'golden')
                message = f'You achieved {2 ** self.current_max_num}!\n Take it further!'
                self.show_trophy(f'pic/{level}.png', message)

    def show_trophy(self, image_path, message):
        self.update_all_frame(self.board)
        dialog = TrophyDialog(self, image_path, message)
        dialog.newGameSignal.connect(self.new_game)
        dialog.exec_()

    def check_game_over(self):
        if self.has_possible_move():
            pass
        else:
            QtCore.QTimer().singleShot(500, self.game_over)

    def has_possible_move(self):
        if (self.board == 0).sum() > 0:
            return True
        else:
            for direct in (1, 2, 3, 4):
                board_new, _, is_valid_move = self.move_and_check_validity(direct)
                if is_valid_move:
                    return True
            return False

    def do_move(self, direction, do_gen=True):
        do_anim = SingletonConfig().config['do_animation']
        direct = {'Left': 1, 'Right': 2, 'Up': 3, 'Down': 4}[direction.capitalize()]
        self.before_move(direct)
        board_new, new_score, is_valid_move = self.move_and_check_validity(direct)
        if is_valid_move:
            if do_anim[1]:
                self.pop_merged(self.board, direction)
            self.board = board_new
            self.score += new_score
            self.max_score = max(self.max_score, self.score)
            self.before_gen_num(direct)
            if do_gen:
                self.gen_new_num(do_anim[0])
            else:
                self.update_all_frame(self.board)
            self.after_gen_num()
            self.check_game_passed()
            self.check_game_over()

    def move_and_check_validity(self, direct):
        board_new, new_score = self.mover.move_board(self.board, direct)
        is_valid_move = np.any(board_new != self.board)
        return board_new, new_score, is_valid_move

    def pop_merged(self, board, direction):
        merged_pos = find_merge_positions(board, direction)
        for row in range(self.game_square.rows):
            for col in range(self.game_square.cols):
                if merged_pos[row][col] == 1:
                    self.game_square.animate_pop(row, col)

    def game_over(self):
        dialog = GameOverDialog(self)
        dialog.newGameSignal.connect(self.new_game)
        dialog.exec_()

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            ([self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [])
        event.accept()


# noinspection PyAttributeOutsideInit
class MinigameWindow(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal()  # 关闭更新菜单页面奖杯展示

    def __init__(self, minigame='minigame', frame_type=None):
        super().__init__()
        self.score_anims = []
        self.minigame = minigame
        self.frame_type = frame_type
        self.setupUi()
        self.best_points.setText(str(self.gameframe.max_score))
        self.update_score()
        self.gameframe.setFocus()
        self.isProcessing = False
        self.retranslateUi()

        init_power_ups = 1 if self.gameframe.difficulty == 1 else 5
        power_up_counts = SingletonConfig().config['power_ups_state'][self.gameframe.difficulty].get(
            self.minigame, [init_power_ups, init_power_ups, init_power_ups])
        self.powerup_grid.set_power_ups_counts(power_up_counts)

    def setupUi(self):
        self.frame_type = MinigameFrame if self.frame_type is None else self.frame_type

        self.setObjectName("self")
        self.setWindowIcon(QtGui.QIcon(r"../pic/2048.ico"))
        self.resize(540, 880)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.gameframe = self.frame_type(self.centralwidget, self.minigame)
        self.gameframe.new_game_signal.connect(self.handleNewGame)
        self.gridLayout.addWidget(self.gameframe, 2, 0, 1, 1)

        self.operate_frame = QtWidgets.QFrame(self.centralwidget)
        self.operate_frame.setMaximumSize(QtCore.QSize(16777215, 180))
        self.operate_frame.setMinimumSize(QtCore.QSize(120, 150))
        self.operate_frame.setStyleSheet("QFrame{\n"
                                         "    border-color: rgb(167, 167, 167);\n"
                                         "    background-color: rgb(236, 236, 236);\n"
                                         "}")
        self.operate_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.operate_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.operate_frame.setObjectName("operate")
        self.grid1 = QtWidgets.QGridLayout(self.operate_frame)
        self.grid1.setObjectName("grid1")
        self.scores = QtWidgets.QHBoxLayout()
        self.scores.setObjectName("scores")
        self.score_frame = QtWidgets.QFrame(self.operate_frame)
        self.score_frame.setStyleSheet("border-radius: 12px; \n"
                                       "background-color: rgb(244, 241, 232);")
        self.score_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.score_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.score_frame.setObjectName("score_frame")
        self.score_grid = QtWidgets.QGridLayout(self.score_frame)
        self.score_grid.setObjectName("score_grid")
        self.score_group = QtWidgets.QGridLayout()
        self.score_group.setObjectName("score_group")
        self.score_text = QtWidgets.QLabel(self.score_frame)
        self.score_text.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.score_text.setScaledContents(False)
        self.score_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.score_text.setWordWrap(False)
        self.score_text.setObjectName("score_text")
        self.score_group.addWidget(self.score_text, 0, 0, 1, 1)
        self.score_points = QtWidgets.QLabel(self.score_frame)
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setBold(True)
        font.setWeight(75)
        font.setPointSize(16)
        self.score_points.setFont(font)
        self.score_points.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.score_points.setObjectName("score_points")
        self.score_group.addWidget(self.score_points, 1, 0, 1, 1)
        self.score_grid.addLayout(self.score_group, 0, 0, 1, 1)
        self.scores.addWidget(self.score_frame)
        self.best_frame = QtWidgets.QFrame(self.operate_frame)
        self.best_frame.setStyleSheet("border-radius: 12px; \n"
                                      "background-color: rgb(244, 241, 232);")
        self.best_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.best_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.best_frame.setObjectName("best_frame")
        self.best_grid = QtWidgets.QGridLayout(self.best_frame)
        self.best_grid.setObjectName("best_grid")
        self.best_group = QtWidgets.QGridLayout()
        self.best_group.setObjectName("best_group")
        self.best_text = QtWidgets.QLabel(self.best_frame)
        self.best_text.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.best_text.setScaledContents(False)
        self.best_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.best_text.setWordWrap(False)
        self.best_text.setObjectName("best_text")
        self.best_group.addWidget(self.best_text, 0, 0, 1, 1)
        self.best_points = QtWidgets.QLabel(self.best_frame)
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setBold(True)
        font.setWeight(75)
        font.setPointSize(16)
        self.best_points.setFont(font)
        self.best_points.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.best_points.setObjectName("best_points")
        self.best_group.addWidget(self.best_points, 1, 0, 1, 1)
        self.best_grid.addLayout(self.best_group, 0, 0, 1, 1)
        self.scores.addWidget(self.best_frame)
        self.grid1.addLayout(self.scores, 0, 0, 1, 1)

        self.buttons = QtWidgets.QGridLayout()
        self.buttons.setObjectName("buttons")
        self.new_game = QtWidgets.QPushButton(self.operate_frame)
        self.new_game.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.new_game.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.new_game.setObjectName("new_game")
        self.new_game.clicked.connect(self.handleNewGame)  # type: ignore
        self.buttons.addWidget(self.new_game, 0, 0, 1, 1)
        self.grid1.addLayout(self.buttons, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.operate_frame, 1, 0, 1, 1)

        self.powerup_frame = QtWidgets.QFrame(self)
        self.powerup_grid = PowerUpGrid(self.powerup_frame, self)
        self.powerup_frame.setMaximumSize(16777215, 120)
        self.gridLayout.addWidget(self.powerup_frame, 3, 0, 1, 1)

        self.infoButton = QtWidgets.QPushButton()
        self.infoButton.setIcon(QtGui.QIcon(r'pic\OQM.png'))
        self.infoButton.setIconSize(QtCore.QSize(30, 30))
        self.infoButton.setFlat(True)
        self.infoButton.clicked.connect(self.show_message)  # type: ignore
        self.gridLayout.addWidget(self.infoButton, 4, 0, 1, 1)

        self.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        QtCore.QMetaObject.connectSlotsByName(self)

    def resizeEvent(self, event):
        self.powerup_frame.setMaximumSize(16777215, min(120, self.height() // 8))
        super().resizeEvent(event)
        self.powerup_grid.updateGeometry()

    def show(self):
        self.gameframe.update_all_frame(self.gameframe.board)
        self.score_points.setText(str(self.gameframe.score))
        self.best_points.setText(str(self.gameframe.max_score))
        super().show()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Minigame", self.minigame))
        self.score_text.setText(_translate("Minigame", "SCORE"))
        self.score_points.setText(_translate("Minigame", "0"))
        self.best_text.setText(_translate("Minigame", "BEST"))
        self.best_points.setText(_translate("Minigame", "0"))
        self.new_game.setText(_translate("Minigame", "New Game"))

    def show_message(self):
        QtWidgets.QMessageBox.information(self, 'Information', 'More minigames.')
        self.gameframe.setFocus()

    def keyPressEvent(self, event, ):
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
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.update_score()
        self.isProcessing = False

    def handleNewGame(self):
        init_power_ups = 1 if self.gameframe.difficulty == 1 else 5
        self.powerup_grid.set_power_ups_counts([init_power_ups, init_power_ups, init_power_ups])
        self.gameframe.setup_new_game()
        self.update_score()

    def update_score(self):
        score = self.gameframe.score
        previous_score = int(self.score_points.text()) if self.score_points.text() else 0
        if score > previous_score and not (previous_score == 0 and score > 8):
            self.show_score_animation(score - previous_score)
        self.score_points.setText(str(score))
        if score > int(self.best_points.text()):
            self.best_points.setText(str(score))
        # 概率获得道具
        if 1024 > score - previous_score >= 300:
            p = 0.05 if self.gameframe.difficulty == 1 else 0.25
            if random.random() < p:
                self.powerup_grid.add_power_ups_counts(random.randint(0, 2))

    def show_score_animation(self, increment):
        # 获取 score_points 的相对于主窗口的坐标
        score_rect = self.score_points.geometry()
        local_pos = self.score_points.mapTo(self, score_rect.topLeft())
        decoration_height = self.frameGeometry().height() - self.geometry().height()

        # 在主窗口的坐标系上创建一个新的 QLabel
        score_animation_label = QtWidgets.QLabel(f"+{increment}", self)
        score_animation_label.setStyleSheet("color: green; background-color: transparent;")
        score_animation_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        score_animation_label.setGeometry(local_pos.x(), local_pos.y() - decoration_height - 10,
                                          score_rect.width(), score_rect.height())
        font = QtGui.QFont()
        font.setPointSize(10)
        score_animation_label.setFont(font)
        score_animation_label.show()

        # 获取初始位置和结束位置
        start_pos = score_animation_label.pos()
        end_pos = QtCore.QPoint(start_pos.x(), start_pos.y() - 50)

        # 位置动画
        pos_anim = QtCore.QPropertyAnimation(score_animation_label, b"pos")
        pos_anim.setDuration(600)
        pos_anim.setStartValue(start_pos)
        pos_anim.setEndValue(end_pos)
        pos_anim.setEasingCurve(QtCore.QEasingCurve.InQuad)

        # 透明度动画
        opacity_effect = QtWidgets.QGraphicsOpacityEffect(score_animation_label)
        score_animation_label.setGraphicsEffect(opacity_effect)
        opacity_anim = QtCore.QPropertyAnimation(opacity_effect, b"opacity")
        opacity_anim.setDuration(600)
        opacity_anim.setStartValue(1)
        opacity_anim.setEndValue(0)
        opacity_anim.setEasingCurve(QtCore.QEasingCurve.InQuad)

        # 动画组
        anim_group = QtCore.QParallelAnimationGroup()
        anim_group.addAnimation(pos_anim)
        anim_group.addAnimation(opacity_anim)
        anim_group.finished.connect(score_animation_label.deleteLater)

        self.score_anims.append(anim_group)
        if len(self.score_anims) >= 200:
            self.score_anims = self.score_anims[100:]
        anim_group.start()

    def closeEvent(self, event):
        self.gameframe.close()  # 触发保存游戏状态的关闭事件
        difficulty = self.gameframe.difficulty
        SingletonConfig().config['power_ups_state'][difficulty][self.minigame] = \
            tuple(self.powerup_grid.power_ups.values())
        # 恢复所有被覆盖的光标
        while QtWidgets.QApplication.overrideCursor() is not None:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.closed.emit()  # 发出信号
        event.accept()


class GameOverDialog(QtWidgets.QDialog):
    newGameSignal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        overlay_width = int(min(max(parent.width() // 1.3, 270), 520))
        overlay_height = int(min(max(parent.width() // 1.5, 160), 300))
        self.setFixedSize(overlay_width, overlay_height)
        self.setStyleSheet("QFrame{\n"
                           "    border-color: rgb(160, 160, 160);\n"
                           "    background-color: rgba(160, 160, 160, 160);\n"
                           "}")

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Game Over", self)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font: 24pt 'Cambria'; font-weight: bold; color: white; background-color: transparent;")
        layout.addWidget(title)

        button_layout = QtWidgets.QHBoxLayout()

        new_game_button = QtWidgets.QPushButton("New Game", self)
        new_game_button.setStyleSheet("font: 750 12pt \"Cambria\";")
        new_game_button.clicked.connect(self.new_game)  # type: ignore
        button_layout.addWidget(new_game_button)

        return_button = QtWidgets.QPushButton("Return", self)
        return_button.setStyleSheet("font: 750 12pt \"Cambria\";")
        return_button.clicked.connect(self.close)  # type: ignore
        button_layout.addWidget(return_button)

        layout.addLayout(button_layout)

    def new_game(self):
        self.close()
        self.newGameSignal.emit()

    def showEvent(self, event):
        super().showEvent(event)
        parent_rect = self.parent().frameGeometry()
        parent_pos = self.parent().mapToGlobal(QtCore.QPoint(0, 0))
        x = parent_pos.x() + (parent_rect.width() - self.width()) // 2
        y = parent_pos.y() + (parent_rect.height() - self.height()) // 2
        self.move(x, y)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(160, 160, 160, 160)))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())


class TrophyDialog(QtWidgets.QDialog):
    newGameSignal = QtCore.pyqtSignal()

    def __init__(self, parent=None, image_path="", message=""):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        overlay_width = int(min(max(parent.width() // 1.3, 270), 520))
        overlay_height = int(min(max(parent.width() // 1.5, 200), 400))
        self.setFixedSize(overlay_width, overlay_height)
        self.setStyleSheet("background-color: rgba(160, 160, 160, 160);")

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Congratulations!", self)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font: 24pt 'Cambria'; font-weight: bold; color: white; background-color: transparent;")
        layout.addWidget(title)

        message_label = QtWidgets.QLabel(message, self)
        message_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        message_label.setStyleSheet("font: 14pt 'Cambria'; color: white; background-color: transparent;")
        layout.addWidget(message_label)

        # 添加奖杯图像
        trophy_label = QtWidgets.QLabel(self)
        trophy_pixmap = QtGui.QPixmap(image_path)
        scale = max(int(overlay_height // 2) - 50, 50)
        trophy_label.setPixmap(trophy_pixmap.scaled(scale, scale, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                    QtCore.Qt.TransformationMode.SmoothTransformation))
        trophy_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        trophy_label.setStyleSheet("background-color: transparent;")
        layout.addWidget(trophy_label)

        button_layout = QtWidgets.QHBoxLayout()

        return_button = QtWidgets.QPushButton("Continue", self)
        return_button.setStyleSheet("font: 750 12pt \"Cambria\";")
        return_button.clicked.connect(self.close)  # type: ignore
        button_layout.addWidget(return_button)

        new_game_button = QtWidgets.QPushButton("New Game", self)
        new_game_button.setStyleSheet("font: 750 12pt \"Cambria\";")
        new_game_button.clicked.connect(self.new_game)  # type: ignore
        button_layout.addWidget(new_game_button)

        layout.addLayout(button_layout)

    def new_game(self):
        self.close()
        self.newGameSignal.emit()

    def showEvent(self, event):
        super().showEvent(event)
        parent_rect = self.parent().frameGeometry()
        parent_pos = self.parent().mapToGlobal(QtCore.QPoint(0, 0))
        x = parent_pos.x() + (parent_rect.width() - self.width()) // 2
        y = parent_pos.y() + (parent_rect.height() - self.height()) // 2
        self.move(x, y)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(160, 160, 160, 160)))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())


class PowerUpGrid(QtWidgets.QFrame):
    def __init__(self, parent, parent_window: MinigameWindow):
        self.parent_window: MinigameWindow = parent_window
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.grid_layout = QtWidgets.QGridLayout(self)
        self.power_up_frames: List[QtWidgets.QFrame] = []
        self.animation_group: QtCore.QParallelAnimationGroup | QtCore.QSequentialAnimationGroup | None = None
        self.first_click_pos = None  # 移动道具第一次点击位置
        # 存储 power-up 的剩余次数
        init_power_ups = 1 if SingletonConfig().config['minigame_difficulty'] else 5
        self.power_ups = {
            'bomb': ("pic//bomb2.png", init_power_ups),
            'glove': ("pic//glove.png", init_power_ups),
            'twist': ("pic//twist.png", init_power_ups),
        }

        self.add_power_ups()

    def resizeEvent(self, event):
        self.updateGeometry()
        super().resizeEvent(event)

    def updateGeometry(self):
        parent_size = self.parent().size()
        new_size = int(min(parent_size.width() // 2.85, parent_size.height()))
        new_x = (parent_size.width() - int(new_size * 2.85)) // 2
        new_y = (parent_size.height() - new_size) // 2
        self.setGeometry(new_x, new_y, int(new_size * 2.85), new_size)

        margin = int(new_size / 32)
        self.grid_layout.setContentsMargins(margin, margin, margin, margin)
        self.grid_layout.setHorizontalSpacing(int(margin / 1.2))
        self.grid_layout.setVerticalSpacing(int(margin / 1.2))

    def add_power_ups(self):
        for idx, (power_up_name, (image_name, count)) in enumerate(self.power_ups.items()):
            row = idx // 3
            col = idx % 3

            frame = QtWidgets.QFrame(self)
            layout = QtWidgets.QVBoxLayout(frame)
            layout.setContentsMargins(0, 0, 0, 0)

            # 主label
            label = QtWidgets.QLabel(frame)
            pixmap = QtGui.QPixmap(image_name).scaled(100, 100, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            layout.addWidget(label)

            # 右上角的较小label
            count_label = QtWidgets.QLabel(label)
            count_label.setStyleSheet("color: black; font: bold 16pt; background-color: transparent;")
            count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
            count_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

            # 存储数据到标签
            frame.power_up_name = power_up_name
            frame.image_name = image_name
            frame.count_label = count_label
            frame.count = count
            frame.label = label  # 将 label 存储在 frame 中以便后续使用
            self.update_frame_state(frame)

            frame.mousePressEvent = self.create_mouse_press_event(frame)

            self.grid_layout.addWidget(frame, row, col)
            self.power_up_frames.append(frame)

    def update_frame_state(self, frame: QtWidgets.QFrame):
        count = frame.count
        if count == 0:
            frame.setEnabled(False)
            frame.count_label.setStyleSheet("color: red; font: bold 16pt; background-color: transparent;")
            frame.setStyleSheet("background-color: rgba(0, 0, 0, 0.5);")
        else:
            frame.setEnabled(True)
            frame.count_label.setStyleSheet("color: black; font: bold 16pt; background-color: transparent;")
            frame.setStyleSheet("background-color: rgba(0, 0, 0, 0.02);")
        self.power_ups[frame.power_up_name] = count
        frame.count_label.setText(str(count))

    def create_mouse_press_event(self, frame):
        def mousePressEvent(_):
            if frame.count == 0 or self.parent_window.isProcessing or self.first_click_pos:
                return
            self.parent_window.isProcessing = True
            frame.setStyleSheet("background-color: rgba(0, 0, 0, 0.2);")
            scale = self.parent_window.gameframe.game_square.labels[0][0].height()
            scale *= 2 if frame.image_name == "pic//twist.png" else 1
            # 更新鼠标图案
            cursor_pixmap = QtGui.QPixmap(frame.image_name).scaled(
                scale, scale, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            cursor = QtGui.QCursor(cursor_pixmap)
            QtWidgets.QApplication.setOverrideCursor(cursor)

            # 连接一次性的点击事件
            self.parent_window.mousePressEvent = self.create_single_click_event(frame)

        return mousePressEvent

    def create_single_click_event(self, frame):
        def single_click_event(event):
            # 这里调用执行某个方法
            executed = self.execute_power_up_action(frame.image_name, event)
            if executed:
                # 更新剩余次数
                frame.count -= 1
            self.update_frame_state(frame)

            if self.first_click_pos is None:
                # 恢复鼠标指针
                QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor())
                # 解除绑定
                self.parent_window.mousePressEvent = None
                frame.setStyleSheet("background-color: rgba(0, 0, 0, 0.02);")
                self.parent_window.isProcessing = False

        return single_click_event

    def execute_power_up_action(self, image_name, event):
        if image_name == "pic//bomb2.png":
            return self.execute_bomb_action(event)
        elif image_name == "pic//glove.png":
            return self.execute_glove_action(event)
        elif image_name == "pic//twist.png":
            return self.execute_twist_action(event)
        else:
            print(f"No action defined for: {image_name}")
            return False

    def execute_bomb_action(self, event):
        local_pos = self.parent_window.gameframe.mapFromParent(event.pos())
        local_pos = self.parent_window.gameframe.game_square.mapFromParent(local_pos)
        for i, row in enumerate(self.parent_window.gameframe.game_square.frames):
            for j, frame in enumerate(row):
                if frame.geometry().contains(local_pos):
                    if self.parent_window.gameframe.board[i][j] <= 0:
                        return False
                    self.parent_window.gameframe.board[i][j] = 0
                    self.parent_window.gameframe.update_frame(0, i, j, False)
                    self.trigger_powerup_explosion(i, j)
                    return True
        return False

    def execute_glove_action(self, event):
        local_pos = self.parent_window.gameframe.mapFromParent(event.pos())
        local_pos = self.parent_window.gameframe.game_square.mapFromParent(local_pos)
        for i, row in enumerate(self.parent_window.gameframe.game_square.frames):
            for j, frame in enumerate(row):
                if frame.geometry().contains(local_pos):
                    if self.first_click_pos is None:
                        # 第一次点击，记录位置
                        if self.parent_window.gameframe.board[i][j] > 0:
                            self.first_click_pos = ((i, j), self.parent_window.gameframe.board[i][j])
                            pixmap = QtGui.QPixmap(frame.size())
                            frame.render(pixmap)
                            cursor = QtGui.QCursor(pixmap.scaled(int(frame.width() // 1.8), int(frame.height() // 1.8),
                                                                 QtCore.Qt.AspectRatioMode.KeepAspectRatio))
                            QtWidgets.QApplication.setOverrideCursor(cursor)
                            self.parent_window.gameframe.update_frame(0, i, j, False)
                            return False
                    else:
                        (src_i, src_j), num = self.first_click_pos
                        # 第二次点击，执行移动
                        if self.parent_window.gameframe.board[i][j] == 0:
                            self.parent_window.gameframe.board[i][j] = num
                            self.parent_window.gameframe.board[src_i][src_j] = 0
                            self.parent_window.gameframe.update_frame(0, src_i, src_j, False)
                            self.parent_window.gameframe.update_frame(num, i, j, True)
                            self.first_click_pos = None
                            return True
                        else:
                            self.parent_window.gameframe.update_frame(num, src_i, src_j, True)
                            # 第二次点击的目标位置不为空，重置第一次点击
                            self.first_click_pos = None
                            return False
        return False

    def find_2x2_subgrid_indices(self, click_pos):
        frame_height = self.parent_window.gameframe.game_square.frames[0][0].height()
        margin = self.parent_window.gameframe.game_square.game_grid.getContentsMargins()[0]
        spacing = self.parent_window.gameframe.game_square.game_grid.horizontalSpacing()
        i0, j0 = None, None
        for i in range(self.parent_window.gameframe.rows):
            if frame_height * (i + 0.66) < click_pos.y() - margin < frame_height * (i + 1.34) + spacing * i:
                i0 = i
                break
        for j in range(self.parent_window.gameframe.cols):
            if frame_height * (j + 0.66) < click_pos.x() - margin < frame_height * (j + 1.34) + spacing * j:
                j0 = j
                break
        return i0, j0

    def execute_twist_action(self, event):
        local_pos = self.parent_window.gameframe.mapFromParent(event.pos())
        local_pos = self.parent_window.gameframe.game_square.mapFromParent(local_pos)
        i, j = self.find_2x2_subgrid_indices(local_pos)
        if i is None or j is None:
            return False
        elif i >= self.parent_window.gameframe.rows - 1 or j >= self.parent_window.gameframe.cols - 1:
            return False
        elif np.all(self.parent_window.gameframe.board[i: i + 2, j: j + 2] == 0) or \
                np.any(self.parent_window.gameframe.board[i: i + 2, j: j + 2] == -1):
            return False
        else:
            self.trigger_powerup_twist(i, j)
            return True

    def trigger_powerup_explosion(self, row, col):
        # 获取目标 QFrame
        target_frame = self.parent_window.gameframe.game_square.frames[row][col]

        # 获取 target_frame 相对于 self.parent_window.gameframe 的位置
        target_global_pos = target_frame.mapToGlobal(QtCore.QPoint(0, 0))
        gameframe_global_pos = self.parent_window.gameframe.mapToGlobal(QtCore.QPoint(0, 0))
        target_relative_pos = target_global_pos - gameframe_global_pos

        target_size = target_frame.size()

        # 创建一个临时 QFrame 作为爆炸效果
        explosion_frame = QtWidgets.QFrame(self.parent_window.gameframe)
        explosion_frame.setStyleSheet("background: transparent;")
        explosion_frame.setGeometry(QtCore.QRect(target_relative_pos, target_size))

        # 创建一个 QLabel 用于显示爆炸图案
        explosion_label = QtWidgets.QLabel(explosion_frame)
        explosion_label.setPixmap(QtGui.QPixmap('pic/explode.png'))
        explosion_label.setScaledContents(True)
        explosion_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        explosion_label.setGeometry(0, 0, target_frame.width(), target_frame.height())

        # 创建放大动画
        size_animation = QtCore.QPropertyAnimation(explosion_label, b"geometry")
        size_animation.setDuration(400)
        size_animation.setStartValue(QtCore.QRect(target_frame.width() // 4, target_frame.height() // 4,
                                                  target_frame.width() // 2, target_frame.height() // 2))
        size_animation.setEndValue(QtCore.QRect(-target_frame.width() // 10, -target_frame.height() // 10,
                                                int(target_frame.width() * 1.2), int(target_frame.height() * 1.2)))
        size_animation.setEasingCurve(QtCore.QEasingCurve.OutQuart)

        # 创建透明度动画
        opacity_effect = QtWidgets.QGraphicsOpacityEffect(explosion_label)
        explosion_label.setGraphicsEffect(opacity_effect)
        opacity_animation = QtCore.QPropertyAnimation(opacity_effect, b"opacity")
        opacity_animation.setDuration(400)
        opacity_animation.setStartValue(1)
        opacity_animation.setEndValue(0)
        opacity_animation.setEasingCurve(QtCore.QEasingCurve.InExpo)

        # 创建动画组
        self.animation_group = QtCore.QParallelAnimationGroup()
        self.animation_group.addAnimation(size_animation)
        self.animation_group.addAnimation(opacity_animation)

        # 在动画结束后移除临时 QFrame
        self.animation_group.finished.connect(explosion_frame.deleteLater)

        # 显示爆炸效果
        explosion_frame.show()
        self.animation_group.start()

    def trigger_powerup_twist(self, i, j):
        game_square = self.parent_window.gameframe.game_square
        frames = game_square.frames

        self.animation_group = QtCore.QParallelAnimationGroup()

        def add_animation(tile_widget, start_rect, end_rect):
            animation = QtCore.QPropertyAnimation(tile_widget, b"geometry")
            animation.setDuration(240)
            animation.setStartValue(start_rect)
            animation.setEndValue(end_rect)
            self.animation_group.addAnimation(animation)

        start_positions = [
            frames[i][j].geometry(),
            frames[i][j + 1].geometry(),
            frames[i + 1][j].geometry(),
            frames[i + 1][j + 1].geometry()
        ]
        end_positions = [start_positions[1], start_positions[3], start_positions[0], start_positions[2]]

        # 创建动画
        for widget, start_pos, end_pos in zip(
                [frames[i][j], frames[i][j + 1], frames[i + 1][j], frames[i + 1][j + 1]], start_positions,
                end_positions):
            add_animation(widget, start_pos, end_pos)

        # 在动画结束后更新棋盘并移除动画
        def on_animation_finished():
            sub_board = self.parent_window.gameframe.board[i:i + 2, j:j + 2].copy()
            rotated_sub_board = [
                [sub_board[1, 0], sub_board[0, 0]],
                [sub_board[1, 1], sub_board[0, 1]]
            ]
            self.parent_window.gameframe.board[i:i + 2, j:j + 2] = rotated_sub_board
            self.parent_window.gameframe.update_all_frame()

        self.animation_group.finished.connect(on_animation_finished)
        self.animation_group.start()

    def set_power_ups_counts(self, counts_list):
        for i, power_up_name in enumerate(self.power_ups.keys()):
            self.power_up_frames[i].count = counts_list[i]
            self.update_frame_state(self.power_up_frames[i])

    def add_power_ups_counts(self, i):
        frame = self.power_up_frames[i]
        frame.count += 1
        self.update_frame_state(frame)

        original_geometry = frame.geometry()
        expanded_geometry = original_geometry.adjusted(-10, -10, 10, 10)  # 调整大小变大

        # 变大的动画
        grow_animation = QtCore.QPropertyAnimation(frame, b"geometry")
        grow_animation.setDuration(160)  # 持续时间
        grow_animation.setStartValue(original_geometry)
        grow_animation.setEndValue(expanded_geometry)
        grow_animation.setEasingCurve(QEasingCurve.OutSine)

        # 变回的动画
        shrink_animation = QtCore.QPropertyAnimation(frame, b"geometry")
        shrink_animation.setDuration(160)  # 持续时间
        shrink_animation.setStartValue(expanded_geometry)
        shrink_animation.setEndValue(original_geometry)
        shrink_animation.setEasingCurve(QEasingCurve.InSine)

        self.animation_group = QtCore.QSequentialAnimationGroup()
        self.animation_group.addAnimation(grow_animation)
        self.animation_group.addAnimation(shrink_animation)

        # 开始动画
        self.animation_group.start()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = MinigameWindow()
    main.gameframe.board = np.array([[0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]])
    main.show()
    sys.exit(app.exec_())
