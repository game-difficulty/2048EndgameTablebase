import sys
from typing import Union, List

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QEasingCurve, QTimer, QSize

from AIPlayer import AIPlayer, Dispatcher, EvilGen
from BoardMover import SingletonBoardMover, BoardMoverWithScore
from Calculator import find_merge_positions
from Config import SingletonConfig


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
        self.anims: List[List[Union[None, QtCore.QAbstractAnimation]]] = [
            [None for _ in range(self.cols)] for __ in range(self.rows)]
        self.animation_config = {
            'appear': {'duration': 150, 'curve': QtCore.QEasingCurve.OutCubic},
            'pop': {'duration': 120, 'curve': QtCore.QEasingCurve.InOutCubic},}

    def updateGeometry(self):
        # 保持正方形尺寸并居中显示
        parent_size = self.parent().size()
        new_size = int(min(parent_size.width(), parent_size.height()) * 0.95)
        new_x = (parent_size.width() - new_size) // 2
        new_y = (parent_size.height() - new_size) // 2
        self.setGeometry(new_x, new_y, new_size, new_size)

        self.base_font_size = int(self.width() / 14.4 * SingletonConfig().config.get('font_size_factor', 100) / 100)

        margin = int(new_size / 32)
        self.game_grid.setContentsMargins(margin, margin, margin, margin)
        self.game_grid.setHorizontalSpacing(int(margin / 1.2))
        self.game_grid.setVerticalSpacing(int(margin / 1.2))

    def setupUi(self, num_rows=4, num_cols=4):
        self.setMaximumSize(1000, 1000)
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
                layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

                label = QtWidgets.QLabel(frame)
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet(f"""
                    font: {self.base_font_size}pt 'Calibri'; 
                    font-weight: bold; color: white; 
                    background-color: transparent;
                """)
                label.setText("")
                layout.addWidget(label)

                self.game_grid.addWidget(frame, i, j, 1, 1)
                row_frames.append(frame)
                row_labels.append(label)
            self.frames.append(row_frames)
            self.labels.append(row_labels)

    def animate_appear(self, r, c):
        if self._check_animation_running(r, c):
            return

        frame = self.frames[r][c]
        opacity_effect = QtWidgets.QGraphicsOpacityEffect(frame)
        frame.setGraphicsEffect(opacity_effect)

        anim_group = QtCore.QParallelAnimationGroup()

        # 透明度动画
        opacity_anim = QtCore.QPropertyAnimation(opacity_effect, b"opacity")
        opacity_anim.setDuration(self.animation_config['appear']['duration'])
        opacity_anim.setStartValue(0.0)
        opacity_anim.setEndValue(1.0)

        # 缩放动画
        scale_anim = QtCore.QPropertyAnimation(frame, b"geometry")
        scale_anim.setDuration(self.animation_config['appear']['duration'])
        scale_anim.setEasingCurve(self.animation_config['appear']['curve'])
        start_size = QtCore.QSize(10, 10)
        scale_anim.setStartValue(QtCore.QRect(frame.geometry().center(), start_size))
        scale_anim.setEndValue(frame.geometry())

        anim_group.addAnimation(opacity_anim)
        anim_group.addAnimation(scale_anim)
        anim_group.finished.connect(lambda: self._finalize_appear(r, c))

        self.anims[r][c] = anim_group
        anim_group.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _finalize_appear(self, r, c):
        """专用完成处理"""
        self.frames[r][c].setGraphicsEffect(None)  # 必须解除效果关联
        self.anims[r][c] = None

    def animate_pop(self, r, c):
        """优化后的合并动画（脉冲缩放效果）"""
        if self._check_animation_running(r, c):
            return

        frame = self.frames[r][c]
        anim = QtCore.QSequentialAnimationGroup()

        # 第一阶段：放大
        enlarge = QtCore.QPropertyAnimation(frame, b"geometry")
        enlarge.setDuration(self.animation_config['pop']['duration'] // 2)
        enlarge.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        enlarge.setEndValue(self._scaled_rect(frame, 1.2))

        # 第二阶段：恢复
        shrink = QtCore.QPropertyAnimation(frame, b"geometry")
        shrink.setDuration(self.animation_config['pop']['duration'] // 2)
        shrink.setEasingCurve(QtCore.QEasingCurve.InCubic)
        shrink.setEndValue(frame.geometry())

        anim.addAnimation(enlarge)
        anim.addAnimation(shrink)
        anim.finished.connect(lambda: self._finalize_animation(r, c))

        self.anims[r][c] = anim
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _check_animation_running(self, r, c):
        """检查当前单元格是否有未完成动画"""
        return self.anims[r][c] and self.anims[r][c].state() == QtCore.QAbstractAnimation.State.Running

    def _finalize_animation(self, r, c):
        """统一动画完成处理"""
        self.anims[r][c] = None
        self.frames[r][c].setGraphicsEffect(None)

    @staticmethod
    def _scaled_rect(frame, factor):
        """计算缩放后的几何矩形"""
        original = frame.geometry()
        center = original.center()
        new_width = int(original.width() * factor)
        new_height = int(original.height() * factor)
        return QtCore.QRect(
            center.x() - new_width // 2,
            center.y() - new_height // 2,
            new_width,
            new_height
        )


# noinspection PyAttributeOutsideInit
class BaseBoardFrame(QtWidgets.QFrame):
    def __init__(self, centralwidget=None):
        super(BaseBoardFrame, self).__init__(centralwidget)
        self.setupUi()
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.board_encoded = np.uint64(0)
        self.score = 0
        self.mover: BoardMoverWithScore = SingletonBoardMover(2)
        self.v_mover = SingletonBoardMover(4)
        self.history = []

        self.newtile_pos = 0
        self.newtile = 1

        self.use_variant_mover = 0

    def setupUi(self):
        self.setMaximumSize(QtCore.QSize(100000, 100000))
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: rgb(222, 222, 222);")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setObjectName("gameframe")
        self.game_square = SquareFrame(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.game_square.updateGeometry()
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
        self.board_encoded, _, new_tile_pos, val = self.mover.gen_new_num(
            self.board_encoded, SingletonConfig().config['4_spawn_rate'])
        self.board_encoded = np.uint64(self.board_encoded)
        self.board = self.mover.decode_board(self.board_encoded)
        self.history.append((self.board_encoded, self.score))
        self.newtile_pos, self.newtile = new_tile_pos, val
        self.update_all_frame(self.board)
        if do_anim:
            self.update_frame(2 ** val, new_tile_pos // 4, new_tile_pos % 4, anim=do_anim)

    def do_move(self, direction, do_gen=True):
        mover = self.mover if self.use_variant_mover == 0 else self.v_mover
        do_anim = SingletonConfig().config['do_animation']
        direct = {'Left': 1, 'Right': 2, 'Up': 3, 'Down': 4}[direction.capitalize()]
        board_encoded_new, new_score = mover.move_board(self.board_encoded, direct)
        board_encoded_new = np.uint64(board_encoded_new)
        if board_encoded_new != self.board_encoded:
            if do_anim[1]:
                self.pop_merged(self.board, direction)
            self.board_encoded = board_encoded_new
            self.board = self.mover.decode_board(self.board_encoded)
            self.score += new_score
            if do_gen:
                self.gen_new_num(do_anim[0])
            else:
                self.update_all_frame(self.board)

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

    def set_use_variant(self, pattern: str = ''):
        self.use_variant_mover = {'2x4': 1, '3x3': 2, '3x4': 3}.get(pattern, 0)


# noinspection PyAttributeOutsideInit
class GameFrame(BaseBoardFrame):
    AIMoveDone = QtCore.pyqtSignal(bool)  # 传递下一步操作

    def __init__(self, centralwidget=None):
        super(GameFrame, self).__init__(centralwidget)
        self.board_encoded, self.score, _ = SingletonConfig().config['game_state']
        self.board = self.mover.decode_board(self.board_encoded)
        self.died_when_ai_state = False
        self.ai_processing = False

        # 初始化 AI 线程
        self.ai_thread = AIThread(self.board)
        self.ai_thread.updateBoard.connect(self.do_ai_move)

        # 困难模式
        self.evil_gen = EvilGen(self.board)
        self.difficulty = 0

        if self.board_encoded == 0:
            self.setup_new_game()

    def setup_new_game(self):
        self.board_encoded = np.uint64(self.mover.gen_new_num(
            self.mover.gen_new_num(np.uint64(0), SingletonConfig().config['4_spawn_rate'])[0],
            SingletonConfig().config['4_spawn_rate'])[0])
        self.board = self.mover.decode_board(self.board_encoded)
        self.ai_thread.ai_player.board = self.board
        self.evil_gen.reset_board(self.board)
        self.update_all_frame(self.board)
        self.score = 0
        self.history = [(self.board_encoded, self.score)]

    def ai_step(self):
        self.ai_processing = True
        self.ai_thread.ai_player.spawn_rate4 = SingletonConfig().config['4_spawn_rate']
        self.ai_thread.ai_player.board = self.board
        self.ai_thread.start()

    def do_ai_move(self, direction):
        if not direction:
            self.died_when_ai_state = True
            self.AIMoveDone.emit(False)
        else:
            self.died_when_ai_state = False
            self.do_move(direction)
            self.AIMoveDone.emit(True)
        self.ai_processing = False

    def gen_new_num(self, do_anim=True):
        if np.random.rand() > self.difficulty:
            self.board_encoded, _, new_tile_pos, val = self.mover.gen_new_num(self.board_encoded,
                                                                              SingletonConfig().config['4_spawn_rate'])
        else:
            self.evil_gen.reset_board(self.board)
            self.board_encoded, new_tile_pos, val = self.evil_gen.gen_new_num(5)
        self.board_encoded = np.uint64(self.board_encoded)
        self.board = self.mover.decode_board(self.board_encoded)
        self.history.append((self.board_encoded, self.score))
        self.newtile_pos, self.newtile = new_tile_pos, val
        self.update_all_frame(self.board)
        if do_anim:
            self.update_frame(2 ** val, new_tile_pos // 4, new_tile_pos % 4, anim=do_anim)


class AIThread(QtCore.QThread):
    updateBoard = QtCore.pyqtSignal(str)  # 传递下一步操作

    def __init__(self, board):
        super(AIThread, self).__init__()
        self.ai_player = AIPlayer(board)

    def is_mess(self):
        """检查是否乱阵"""
        board = self.ai_player.board
        large_tiles = (board > 64).sum()
        board_flatten = board.flatten()
        if large_tiles < 3:
            return False
        elif large_tiles > 3:
            top4_pos = np.argpartition(board_flatten, -4)[-4:]
            if len(np.unique(board_flatten[top4_pos])) < 4:
                return False
            top4_pos = tuple(sorted(top4_pos))
            return top4_pos not in ((0, 1, 2, 3), (0, 4, 8, 12), (12, 13, 14, 15), (3, 7, 11, 15),
                                    (0, 1, 2, 4), (4, 8, 12, 13), (11, 13, 14, 15), (2, 3, 7, 11),
                                    (0, 1, 4, 8), (8, 12, 13, 14), (7, 11, 14, 15), (1, 2, 3, 7),
                                    (0, 1, 4, 5), (8, 9, 12, 13), (10, 11, 14, 15), (2, 3, 6, 7),
                                    (2, 3, 14, 15), (0, 1, 12, 13), (8, 11, 12, 15), (0, 3, 4, 7))
        else:
            top3_pos = np.argpartition(board_flatten, -3)[-3:]
            if len(np.unique(board_flatten[top3_pos])) < 3:
                return False
            top3_pos = tuple(sorted(top3_pos))
            return top3_pos not in (
                (0, 1, 2), (1, 2, 3), (3, 7, 11), (7, 11, 15), (13, 14, 15), (12, 13, 14), (4, 8, 12), (0, 4, 8),
                (0, 1, 3), (0, 2, 3), (3, 7, 15), (3, 11, 15), (12, 14, 15), (12, 13, 15), (0, 8, 12), (0, 4, 12),
                (0, 1, 12), (0, 3, 4), (0, 3, 7), (2, 3, 15), (3, 14, 15), (11, 12, 15), (8, 12 ,15), (0, 12, 13),
                (0, 1, 4), (2, 3, 7), (11, 14, 15), (8, 12, 13))

    def run(self):
        # 根据局面设定搜索深度
        empty_slots = np.sum(self.ai_player.board == 0)
        big_nums = (self.ai_player.board > 128).sum()
        if self.is_mess():
            big_nums2 = (self.ai_player.board > 256).sum()
            depth = 5
            if self.ai_player.check_corner(np.uint64(self.ai_player.bm.encode_board(self.ai_player.board))):
                depth = 8
            self.ai_player.start_search(depth)
            while self.ai_player.node < 200000 * big_nums2 ** 2 and depth < 9:
                depth += 1
                self.ai_player.start_search(depth)
        elif empty_slots > 9 or big_nums < 1:
            self.ai_player.start_search(1)
        elif empty_slots > 4 and big_nums < 2:
            self.ai_player.start_search(2)
        elif (empty_slots > 3 > big_nums) or (big_nums < 2):
            self.ai_player.start_search(4)
        else:
            depth = 4 if big_nums < 4 else 5
            self.ai_player.start_search(depth)
            while self.ai_player.node < 24000 * depth * big_nums ** 1.25 and depth < 9:
                depth += 1
                self.ai_player.start_search(depth)
            # print(depth, self.ai_player.node)
        self.updateBoard.emit({1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}.get(self.ai_player.best_operation, ''))


# noinspection PyAttributeOutsideInit
class GameWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.isProcessing = False
        self.ai_state = False

        self.best_points.setText(str(SingletonConfig().config['game_state'][2]))
        self.score_anims = []

        self.ai_timer = QTimer(self)
        self.ai_dispatcher = Dispatcher(SingletonBoardMover(2).decode_board(np.uint64(0)), np.uint64(0))

        self.statusbar.showMessage("All features may be slow when used for the first time. Please be patient.", 8000)
        self.update_score()
        self.gameframe.setFocus()
        self.gameframe.AIMoveDone.connect(self.ai_move_done)

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QtGui.QIcon(r"pic\2048.ico"))
        self.resize(800, 940)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self.gameframe = GameFrame(self.centralwidget)
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
        self.one_step = QtWidgets.QPushButton(self.operate_frame)
        self.one_step.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # 禁用按钮的键盘焦点
        self.one_step.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.one_step.setObjectName("one_step")
        self.buttons.addWidget(self.one_step, 0, 3, 1, 1)
        self.undo = QtWidgets.QPushButton(self.operate_frame)
        self.undo.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.undo.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.undo.setObjectName("undo")
        self.buttons.addWidget(self.undo, 0, 2, 1, 1)
        self.new_game = QtWidgets.QPushButton(self.operate_frame)
        self.new_game.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.new_game.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.new_game.setObjectName("new_game")
        self.buttons.addWidget(self.new_game, 0, 1, 1, 1)
        self.ai = QtWidgets.QPushButton(self.operate_frame)
        self.ai.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.ai.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.ai.setObjectName("ai")
        self.buttons.addWidget(self.ai, 0, 0, 1, 1)
        self.grid1.addLayout(self.buttons, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.operate_frame, 1, 0, 1, 1)

        self.one_step.clicked.connect(self.handleOneStep)  # type: ignore
        self.undo.clicked.connect(self.handleUndo)  # type: ignore
        self.new_game.clicked.connect(self.handleNewGame)  # type: ignore
        self.ai.clicked.connect(self.toggleAI)  # type: ignore

        self.difficulty_frame = QtWidgets.QFrame(self.centralwidget)
        self.difficulty_frame.setMaximumSize(QtCore.QSize(16777215, 30))
        self.difficulty_frame.setMinimumSize(QtCore.QSize(120, 20))
        self.difficulty_frame.setStyleSheet("QFrame{\n"
                                            "    border-color: rgb(167, 167, 167);\n"
                                            "    background-color: rgb(236, 236, 236);\n"
                                            "}")
        self.difficulty_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.difficulty_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.difficulty_frame.setObjectName("difficulty")
        self.difficulty_layout = QtWidgets.QGridLayout(self.difficulty_frame)
        self.difficulty_layout.setObjectName("difficulty_layout")
        self.difficulty_frame.setMaximumSize(QtCore.QSize(16777215, 60))
        self.difficulty_frame.setMinimumSize(QtCore.QSize(120, 45))
        self.difficulty_text = QtWidgets.QLabel(self.centralwidget)
        self.difficulty_text.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.difficulty_text.setScaledContents(False)
        self.difficulty_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.difficulty_text.setWordWrap(False)
        self.difficulty_text.setObjectName("difficulty_text")
        self.difficulty_layout.addWidget(self.difficulty_text, 0, 0, 1, 3)
        self.difficulty_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.centralwidget)
        self.difficulty_slider.setMinimum(0)
        self.difficulty_slider.setMaximum(100)
        self.difficulty_slider.setValue(0)
        self.difficulty_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.difficulty_slider.setTickInterval(1)
        self.difficulty_slider.valueChanged.connect(self.difficulty_changed)  # type: ignore
        self.difficulty_slider.setObjectName("difficulty_slider")
        self.difficulty_layout.addWidget(self.difficulty_slider, 0, 3, 1, 8)
        self.infoButton = QtWidgets.QPushButton()
        self.infoButton.setIcon(QtGui.QIcon(r'pic\OQM.png'))
        self.infoButton.setIconSize(QSize(30, 30))
        self.infoButton.setFlat(True)
        self.difficulty_layout.addWidget(self.infoButton, 0, 11, 1, 1)
        self.infoButton.clicked.connect(self.show_message)  # type: ignore
        self.gridLayout.addWidget(self.difficulty_frame, 3, 0, 1, 1)

        self.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)
        self.setTabOrder(self.ai, self.new_game)
        self.setTabOrder(self.new_game, self.undo)
        self.setTabOrder(self.undo, self.one_step)

    def show(self):
        self.gameframe.update_all_frame(self.gameframe.board)
        super().show()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Game", "Game"))
        self.score_text.setText(_translate("Game", "SCORE"))
        self.score_points.setText(_translate("Game", "0"))
        self.best_text.setText(_translate("Game", "BEST"))
        self.best_points.setText(_translate("Game", "0"))
        self.one_step.setText(_translate("Game", "One Step"))
        self.undo.setText(_translate("Game", "Undo"))
        self.new_game.setText(_translate("Game", "New Game"))
        self.ai.setText(_translate("Game", "AI: ON"))
        self.difficulty_text.setText(_translate("Game", "Difficulty"))

    def difficulty_changed(self):
        self.gameframe.difficulty = self.difficulty_slider.value() / 100
        self.gameframe.setFocus()

    def show_message(self):
        QtWidgets.QMessageBox.information(self, 'Information', '''Probability of generating an EVIL number, default 0.
Only effective for players''')
        self.gameframe.setFocus()

    def update_score(self):
        score = self.gameframe.score
        previous_score = int(self.score_points.text()) if self.score_points.text() else 0
        if score > previous_score and not (previous_score == 0 and score > 8):
            self.show_score_animation(score - previous_score)
        self.score_points.setText(str(score))
        if score > int(self.best_points.text()):
            self.best_points.setText(str(score))

    def show_score_animation(self, increment):
        # 获取 score_points 的相对于主窗口的坐标
        score_rect = self.score_points.geometry()
        local_pos = self.score_points.mapTo(self, score_rect.topLeft())
        decoration_width = self.frameGeometry().height() - self.geometry().height()

        # 在主窗口的坐标系上创建一个新的 QLabel
        score_animation_label = QtWidgets.QLabel(f"+{increment}", self)
        score_animation_label.setStyleSheet("font 750 12pt; color: green;")
        score_animation_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        score_animation_label.setGeometry(local_pos.x(), local_pos.y() - decoration_width - 10,
                                          score_rect.width(), score_rect.height())
        score_animation_label.show()

        # 获取初始位置和结束位置
        start_pos = score_animation_label.pos()
        end_pos = QtCore.QPoint(start_pos.x(), start_pos.y() - 50)

        # 位置动画
        pos_anim = QtCore.QPropertyAnimation(score_animation_label, b"pos")
        pos_anim.setDuration(600)
        pos_anim.setStartValue(start_pos)
        pos_anim.setEndValue(end_pos)
        pos_anim.setEasingCurve(QEasingCurve.InQuad)

        # 透明度动画
        opacity_effect = QtWidgets.QGraphicsOpacityEffect(score_animation_label)
        score_animation_label.setGraphicsEffect(opacity_effect)
        opacity_anim = QtCore.QPropertyAnimation(opacity_effect, b"opacity")
        opacity_anim.setDuration(600)
        opacity_anim.setStartValue(1)
        opacity_anim.setEndValue(0)
        opacity_anim.setEasingCurve(QEasingCurve.InQuad)

        # 动画组
        anim_group = QtCore.QParallelAnimationGroup()
        anim_group.addAnimation(pos_anim)
        anim_group.addAnimation(opacity_anim)
        anim_group.finished.connect(score_animation_label.deleteLater)

        self.score_anims.append(anim_group)
        if len(self.score_anims) >= 200:
            self.score_anims = self.score_anims[100:]
        anim_group.start()

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

    def handleOneStep(self):
        if self.isProcessing or self.gameframe.ai_processing:
            return
        self.isProcessing = True
        self.ai_dispatcher.reset(self.gameframe.board, self.gameframe.board_encoded)
        best_move = self.ai_dispatcher.dispatcher()
        if best_move == 'AI':
            self.gameframe.ai_step()
        else:
            self.gameframe.do_move(best_move.capitalize())
        self.update_score()
        if self.ai_state:
            if self.gameframe.died_when_ai_state:
                self.ai_state = False
                self.ai.setText("AI: ON")
                self.gameframe.died_when_ai_state = False
            elif best_move != 'AI':
                self.ai_timer.singleShot(20, self.handleOneStep)
        # print(self.ai_dispatcher.last_operator)
        self.isProcessing, self.gameframe.ai_processing = False, False

    def ai_move_done(self, is_done):
        if is_done and self.ai_state:
            self.ai_timer.singleShot(20, self.handleOneStep)

    def handleUndo(self):
        self.gameframe.undo()
        self.update_score()
        self.gameframe.died_when_ai_state = False

    def handleNewGame(self):
        self.gameframe.setup_new_game()
        self.update_score()
        self.gameframe.died_when_ai_state = False

    def toggleAI(self):
        if not self.ai_state:
            self.ai.setText("STOP")
            self.ai_state = True
            self.ai_timer.singleShot(20, self.handleOneStep)
            if not SingletonConfig().config['filepath_map'].get('4431_2048_0', []) or \
                    not SingletonConfig().config['filepath_map'].get('LL_2048_0', []):
                self.statusbar.showMessage(
                    "Run free12w-2k free11w-2k 4442f-2k free11w-512 for best performance.", 3000)
        else:
            self.ai.setText("AI: ON")
            self.ai_state = False

    def closeEvent(self, event):
        self.ai.setText("AI: ON")
        self.ai_state = False
        SingletonConfig().config['game_state'] = [self.gameframe.board_encoded, self.gameframe.score,
                                                  int(self.best_points.text())]
        SingletonConfig.save_config(SingletonConfig().config)
        event.accept()  # 确认关闭事件


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = GameWindow()
    main.show()
    sys.exit(app.exec_())
