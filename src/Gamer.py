import sys
import time
from typing import List, Tuple, Dict

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QEasingCurve, QTimer, QSize

from AIPlayer import AIPlayer, Dispatcher, EvilGen
import Variants.vBoardMover as vbm
import BoardMover as bm
from Calculator import find_merge_positions, slide_distance
from Config import SingletonConfig


# noinspection PyAttributeOutsideInit
class SquareFrame(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.animation_config = {
            'appear': {'duration': 120, 'curve': QtCore.QEasingCurve.OutCubic},
            'pop': {'duration': 120, 'curve': QtCore.QEasingCurve.InOutCubic},
            'slide': {'duration': 120, 'curve1': QtCore.QEasingCurve.Linear},}
        self.active_animations: Dict[Tuple, QtCore.QAbstractAnimation] = dict()  # 跟踪所有活动的动画对象

    def setupUi(self, num_rows=4, num_cols=4):
        self.setMaximumSize(1200, 1000)
        self.setMinimumSize(120, 120)
        self.rows = num_rows
        self.cols = num_cols
        self.setStyleSheet("border-radius: 8px; background-color: #bbada0;")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setObjectName("game_square")
        self.game_grid = QtWidgets.QGridLayout(self)
        self.game_grid.setObjectName("game_grid")
        self.grids = []  # 存储QFrame对象
        self.frames: List[List[None | QtWidgets.QFrame]] = [[None for _0 in range(self.cols)] for _1 in range(self.rows)]  # 存储QFrame对象
        self.labels: List[List[None | QtWidgets.QLabel]] = [[None for _0 in range(self.cols)] for _1 in range(self.rows)]  # 存储QLabel对象

        for i in range(num_rows):
            row_grids = []
            for j in range(num_cols):
                grid = QtWidgets.QFrame(self)
                grid.setStyleSheet("border-radius: 8px; background-color: #cdc1b4;")
                grid.setFrameShape(QtWidgets.QFrame.StyledPanel)
                grid.setFrameShadow(QtWidgets.QFrame.Raised)
                grid.setObjectName(f"grid{i * num_cols + j}")

                self.game_grid.addWidget(grid, i, j, 1, 1)
                row_grids.append(grid)
            self.grids.append(row_grids)

    def update_tile_frame(self, value, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"无效的位置 ({row}, {col})")
        if value == 0:
            if self.frames[row][col]:
                self.labels[row][col].deleteLater()
                self.frames[row][col].deleteLater()
                self.labels[row][col] = None
                self.frames[row][col] = None
            return

        if self.frames[row][col] is None:
            self._create_tile_components(row, col)
        self._update_tile_style(value, row, col)
        self.frames[row][col].show()

    def _create_tile_components(self, row, col):
        grid = self.grids[row][col]
        grid_rect = grid.geometry()

        frame = QtWidgets.QFrame(self)
        frame.setGeometry(grid_rect)
        frame.setObjectName(f"f{row * 4 + col}")
        frame.setStyleSheet("border-radius: 8px;")

        label = QtWidgets.QLabel(frame)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setObjectName(f"l{row * 4 + col}")
        label.setGeometry(0, 0, grid_rect.width(), grid_rect.height())

        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        self.frames[row][col] = frame
        self.labels[row][col] = label

    def _update_tile_style(self, value, row, col):
        frame = self.frames[row][col]
        label = self.labels[row][col]

        color_index = int(np.log2(value) - 1) if value > 1 else -1
        color = self.colors[color_index]

        # 设置数字块颜色
        frame.setStyleSheet(f"""
            background-color: {color};
            border-radius: 8px;
        """)
        fontsize = self.base_font_size if len(str(value)) < 3 else int(
            self.base_font_size * 3 / (0.5 + len(str(value))))
        font_style = self.get_label_style(fontsize, value)
        label.setStyleSheet(font_style)
        label.setText(str(value))

    @property
    def colors(self):
        """
        默认主题
        ['#043c24', '#06643d', '#1b955b', '#20c175', '#fc56a0', '#e4317f', '#e900ad', '#bf009c',
        '#94008a', '#6a0079', '#3f0067', '#00406b', '#006b9a', '#0095c8', '#00c0f7', '#00c0f7'] + [
        '#000000'] * 20
        """
        return SingletonConfig().config['colors']

    @property
    def base_font_size(self):
        return int(self.width() / 14.4 * SingletonConfig().config.get('font_size_factor', 100) / 100)

    def updateGeometry(self):
        # 保持正方形尺寸并居中显示
        parent_size = self.parent().size()
        new_size = int(min(parent_size.width(), parent_size.height()) * 0.95)
        new_x = (parent_size.width() - new_size) // 2
        new_y = (parent_size.height() - new_size) // 2
        self.setGeometry(new_x, new_y, new_size, new_size)

        margin = int(new_size / 32)
        self.game_grid.setContentsMargins(margin, margin, margin, margin)
        self.game_grid.setHorizontalSpacing(int(margin / 1.2))
        self.game_grid.setVerticalSpacing(int(margin / 1.2))

        # 更新所有已创建的tile位置
        self._adjust_tile_positions()

    def _adjust_tile_positions(self):
        """调整所有tile的位置到对应grid的位置"""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.frames[row][col] is not None:
                    # 获取对应grid的新位置
                    grid_rect = self.grids[row][col].geometry()

                    # 设置frame到grid位置
                    self.frames[row][col].setGeometry(grid_rect)

                    # 调整内部label大小匹配grid
                    label = self.labels[row][col]
                    if label:
                        value = label.text()
                        label.setGeometry(0, 0, grid_rect.width(), grid_rect.height())
                        fontsize = self.base_font_size if len(value) < 3 else int(
                            self.base_font_size * 3 / (0.5 + len(value)))
                        font_style = self.get_label_style(fontsize, value)
                        label.setStyleSheet(font_style)

    @staticmethod
    def get_label_style(fontsize, value):
        try:
            value = int(value)
            value = 32768 if value <= 1 else value
        except (TypeError, ValueError):
            value = 32768

        color = '#776e65' if not SingletonConfig.font_colors[int(np.log2(value)) - 1] else '#f9f6f2'

        return f"""
            font: {fontsize}pt 'Clear Sans';
            font-weight: bold;
            color: {color};
            background-color: transparent;
        """

    def animate_appear(self, r, c, value):
        """数字块出现动画"""
        if self._check_animation_running(r, c):
            for anim_id in (2, 1, 0):
                anim = self.active_animations.pop((r, c, anim_id), None)
                if anim:
                    try:
                        anim.setCurrentTime(anim.duration())
                        anim.deleteLater()
                    except RuntimeError:  # wrapped C/C++ object of type QParallelAnimationGroup has been deleted
                        continue

        if self.frames[r][c]:
            self.frames[r][c].deleteLater()
        self._create_tile_components(r, c)
        self._update_tile_style(value, r, c)
        frame = self.frames[r][c]
        frame.show()

        opacity_effect = QtWidgets.QGraphicsOpacityEffect(frame)
        frame.setGraphicsEffect(opacity_effect)

        anim_group = QtCore.QParallelAnimationGroup()

        opacity_anim = QtCore.QPropertyAnimation(opacity_effect, b"opacity")
        opacity_anim.setDuration(self.animation_config['appear']['duration'])
        opacity_anim.setStartValue(0.0)
        opacity_anim.setEndValue(1.0)

        scale_anim = QtCore.QPropertyAnimation(frame, b"geometry")
        scale_anim.setDuration(self.animation_config['appear']['duration'])
        scale_anim.setEasingCurve(self.animation_config['appear']['curve'])

        start_size = QtCore.QSize(10, 10)
        scale_anim.setStartValue(
            QtCore.QRect(frame.geometry().center(), start_size)
        )
        scale_anim.setEndValue(frame.geometry())

        anim_group.addAnimation(opacity_anim)
        anim_group.addAnimation(scale_anim)

        anim_group.finished.connect(lambda: self._finalize_animation(r, c, 0))
        anim_group.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

        self.active_animations[(r, c, 0)] = anim_group

    def animate_pop(self, r, c):
        if self._check_animation_running(r, c):
            return

        frame = self.frames[r][c]
        if not frame:
            return
        anim = QtCore.QSequentialAnimationGroup()

        # 第一阶段：放大
        enlarge = QtCore.QPropertyAnimation(frame, b"geometry")
        enlarge.setDuration(self.animation_config['pop']['duration'] // 2)
        enlarge.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        enlarge.setEndValue(self._scaled_rect(frame, 1.15))

        # 第二阶段：恢复
        shrink = QtCore.QPropertyAnimation(frame, b"geometry")
        shrink.setDuration(self.animation_config['pop']['duration'] // 2)
        shrink.setEasingCurve(QtCore.QEasingCurve.InCubic)
        shrink.setEndValue(frame.geometry())

        anim.addAnimation(enlarge)
        anim.addAnimation(shrink)
        anim.finished.connect(lambda: self._finalize_animation(r, c, 1))
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

        self.active_animations[(r, c, 1)] = anim

    def _check_animation_running(self, r, c):
        """检查当前单元格是否有未完成动画"""
        return (r, c, 0) in self.active_animations or (r, c, 1) in self.active_animations

    def _finalize_animation(self, r, c, anim_id):
        anim = self.active_animations.pop((r, c, anim_id), None)
        if anim:
            anim.deleteLater()

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

    def animate_slide(self, r, c, direction: str, distance):
        if distance <= 0:
            return

        current_anim = self.active_animations.pop((r, c, 2), None)
        if current_anim:
            current_anim.setCurrentTime(current_anim.duration())
            current_anim.deleteLater()

        # 获取动画对象
        frame = self.frames[r][c]
        label = self.labels[r][c]
        if not frame or not label:
            return
        self.frames[r][c] = None
        self.labels[r][c] = None

        new_r, new_c = self._calculate_new_position(r, c, direction, distance)

        # 创建移动动画
        anim = QtCore.QPropertyAnimation(frame, b"pos")
        anim.setDuration(self.animation_config['slide']['duration'])
        anim.setEasingCurve(self.animation_config['slide']['curve1'])

        start_pos = frame.pos()
        end_pos = self._get_grid_pos(new_r, new_c)
        anim.setStartValue(start_pos)
        anim.setEndValue(end_pos)

        anim.finished.connect(lambda: self._finalize_slide(frame, label, r, c, 2))

        self.active_animations[(r, c, 2)] = anim
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    @staticmethod
    def _calculate_new_position(r: int, c: int, direction: str, distance: int) -> tuple:
        """计算移动后的新坐标"""
        direction_map = {
            'Up': (-distance, 0),
            'Down': (distance, 0),
            'Left': (0, -distance),
            'Right': (0, distance)
        }
        dr, dc = direction_map[direction.capitalize()]
        return r + dr, c + dc

    def _get_grid_pos(self, r: int, c: int) -> QtCore.QPoint:
        """获取网格的坐标位置"""
        grid = self.grids[r][c]
        return QtCore.QPoint(grid.x(), grid.y())

    def _finalize_slide(self, frame, label, r, c, anim_id):
        label.deleteLater()
        frame.deleteLater()
        anim = self.active_animations.pop((r, c, anim_id), None)
        if anim:
            anim.deleteLater()

    def cancel_all_animations(self):
        """立即终止所有正在运行的动画"""
        for anim in list(self.active_animations.values()):
            try:
                anim.setCurrentTime(anim.duration())
                anim.deleteLater()
            except RuntimeError:  # wrapped C/C++ object of type QParallelAnimationGroup has been deleted
                continue
        self.active_animations.clear()


# noinspection PyAttributeOutsideInit
class BaseBoardFrame(QtWidgets.QFrame):
    def __init__(self, centralwidget=None):
        super(BaseBoardFrame, self).__init__(centralwidget)
        self.setupUi()
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.board_encoded = np.uint64(0)
        self.score = 0
        self.history = []

        self.newtile_pos = 0
        self.newtile = 1

        self.use_variant_mover = 0

        self.timer1, self.timer2, self.timer3, self.timer4 = QTimer(), QTimer(), QTimer(), QTimer()
        self._last_values = self.board.copy()
        self.last_move_time = time.time()

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

    def update_frame(self, value, row, col):
        self.game_square.update_tile_frame(value, row, col)

    def update_all_frame(self, values):
        for i in range(self.game_square.rows):
            for j in range(self.game_square.cols):
                self.update_frame(values[i][j], i, j)

    def gen_new_num(self, do_anim=True):
        self.board_encoded, _, new_tile_pos, val = bm.s_gen_new_num(
            self.board_encoded, SingletonConfig().config['4_spawn_rate'])
        self.board_encoded = np.uint64(self.board_encoded)
        self.board = bm.decode_board(self.board_encoded)
        self.history.append((self.board_encoded, self.score))
        self.newtile_pos, self.newtile = new_tile_pos, val
        if do_anim:
            self.timer1.singleShot(125, lambda: self.game_square.animate_appear(new_tile_pos // 4, new_tile_pos % 4, 2 ** val))  # 125

    def do_move(self, direction: str, do_gen=True):
        if self.game_square.active_animations:
            for timer in (self.timer1, self.timer2, self.timer3):
                timer.stop()
            self.game_square.cancel_all_animations()
            self.update_all_frame(self._last_values)
        mover = bm if self.use_variant_mover == 0 else vbm
        do_anim = SingletonConfig().config['do_animation']

        time_now = time.time()
        slow_move = (time_now - self.last_move_time > 0.08)
        self.last_move_time = time_now

        direct = {'Left': 1, 'Right': 2, 'Up': 3, 'Down': 4}[direction.capitalize()]
        board_encoded_new, new_score = mover.s_move_board(self.board_encoded, direct)
        board_encoded_new = np.uint64(board_encoded_new)
        if board_encoded_new != self.board_encoded:
            self.board_encoded = board_encoded_new
            self.score += new_score
            current_values = self.board
            if do_gen:
                self.gen_new_num(do_anim)
            self.board = bm.decode_board(self.board_encoded)
            if do_anim:
                self.slide_tiles(current_values, direction)
                self.timer2.singleShot(110, lambda: self.pop_merged(current_values, direction))  # 110
                # 生成新数字之前的局面
                self.timer3.singleShot(100, lambda: self.update_all_frame((bm.decode_board(board_encoded_new))))  # 100
                if not slow_move:
                    self.timer4.singleShot(250, lambda: self.update_all_frame(self.board))
            else:
                self.update_all_frame(self.board)
            self._last_values = self.board.copy()
        else:
            self.update_all_frame(self.board)

    def pop_merged(self, board, direction):
        merged_pos = find_merge_positions(board, direction)
        for row in range(self.game_square.rows):
            for col in range(self.game_square.cols):
                if merged_pos[row][col] == 1:
                    self.game_square.animate_pop(row, col)

    def slide_tiles(self, board, direction):
        slide_distances = slide_distance(board, direction)
        if self.use_variant_mover:  # variant 32k不移动
            slide_distances[board == 32768] = 0
        for row in range(self.game_square.rows):
            for col in range(self.game_square.cols):
                self.game_square.animate_slide(row, col, direction, slide_distances[row][col])

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.board_encoded, self.score = self.history[-1]
            self.board_encoded = np.uint64(self.board_encoded)
            self.board = bm.decode_board(self.board_encoded)
            self.update_all_frame(self.board)

    def set_use_variant(self, pattern: str = ''):
        self.use_variant_mover = {'2x4': 1, '3x3': 2, '3x4': 3}.get(pattern, 0)

    def showEvent(self, event):
        super().showEvent(event)
        self.game_square.updateGeometry()


# noinspection PyAttributeOutsideInit
class GameFrame(BaseBoardFrame):
    AIMoveDone = QtCore.pyqtSignal(bool)  # 传递下一步操作

    def __init__(self, centralwidget=None):
        super(GameFrame, self).__init__(centralwidget)
        self.board_encoded, self.score, _ = SingletonConfig().config['game_state']
        self.board = bm.decode_board(self.board_encoded)
        self.died_when_ai_state = False
        self.ai_processing = False

        self.has_65k = self.score > 960000

        # 初始化 AI 线程
        self.ai_thread = AIThread(self.board)
        self.ai_thread.updateBoard.connect(self.do_ai_move)

        # 困难模式
        self.evil_gen = EvilGen(self.board)
        self.difficulty = 0

        if self.board_encoded == 0:
            self.setup_new_game()

    def setup_new_game(self):
        self.board_encoded = np.uint64(bm.gen_new_num(
            bm.gen_new_num(np.uint64(0), SingletonConfig().config['4_spawn_rate'])[0],
            SingletonConfig().config['4_spawn_rate'])[0])
        self.board = bm.decode_board(self.board_encoded)
        self.ai_thread.ai_player.board = self.board
        self.evil_gen.reset_board(self.board)
        self.update_all_frame(self.board)
        self.score = 0
        self.has_65k = False
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
            self.board_encoded, _, new_tile_pos, val = bm.s_gen_new_num(self.board_encoded,
                                                                              SingletonConfig().config['4_spawn_rate'])
        else:
            self.evil_gen.reset_board(bm.decode_board(self.board_encoded))
            self.board_encoded, new_tile_pos, val = self.evil_gen.gen_new_num(5)
        self.board_encoded = np.uint64(self.board_encoded)
        self.board = bm.decode_board(self.board_encoded)
        self.history.append((self.board_encoded, self.score))
        self.newtile_pos, self.newtile = new_tile_pos, val
        if do_anim:
            self.timer1.singleShot(125, lambda: self.game_square.animate_appear(
                self.newtile_pos // 4, self.newtile_pos % 4, 2 ** self.newtile))

    def do_move(self, direction: str, do_gen=True):
        """ 支持65536 """
        if not self.has_65k and np.sum((self.board == 32768)) == 2:
            positions = np.where(self.board == 32768)
            first_position = (positions[0][0], positions[1][0])
            second_position = (positions[0][1], positions[1][1])

            if (positions[0][0] == positions[0][1] and abs(positions[1][0] - positions[1][1]) == 1 and
                direction.capitalize() in ('Right', 'Left')) or (positions[1][0] == positions[1][1] and
                abs(positions[0][0] - positions[0][1]) == 1 and direction.capitalize() in ('Up', 'Down')):
                    self.board[first_position] = 16384
                    self.board[second_position] = 16384
                    self.board_encoded = np.uint64(bm.encode_board(self.board))
                    self.score += 32768
                    self.has_65k = True

        super().do_move(direction, do_gen)

    def update_all_frame(self, values):
        """ 支持65536 """
        if self.has_65k:
            values = values.copy()
            values.flat[next((i for i, x in enumerate(values.flat) if x == 32768), None)] = 65536
        super().update_all_frame(values)


class AIThread(QtCore.QThread):
    updateBoard = QtCore.pyqtSignal(str)  # 传递下一步操作

    def __init__(self, board):
        super(AIThread, self).__init__()
        self.ai_player = AIPlayer(board)

    def is_mess(self):
        """检查是否乱阵"""
        board = self.ai_player.board
        if np.sum(board) % 512 < 12:
            return False

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
            if self.ai_player.check_corner(np.uint64(bm.encode_board(self.ai_player.board))):
                depth = 8
            self.ai_player.start_search(depth)
            while self.ai_player.node < 160000 * big_nums2 ** 2 and depth < 9:
                depth += 1
                self.ai_player.start_search(depth)
        elif empty_slots > 9 or big_nums < 1:
            self.ai_player.start_search(1)
        elif empty_slots > 4 and big_nums < 2:
            self.ai_player.start_search(2)
        elif (empty_slots > 3 > big_nums) or (big_nums < 2):
            self.ai_player.start_search(4)
        elif np.sum(self.ai_player.board) % 512 < 16:
            depth = 4 if big_nums < 4 else 5
            self.ai_player.start_search(depth)
        else:
            depth = 4 if big_nums < 4 else 5
            self.ai_player.start_search(depth)
            while self.ai_player.node < 20000 * depth * big_nums ** 1.25 and depth < 9:
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
        self.ai_dispatcher = Dispatcher(bm.decode_board(np.uint64(0)), np.uint64(0))
        self.last_table = 'AI'

        self.statusbar.showMessage(self.tr(
            "All features may be slow when used for the first time. Please be patient."), 8000)
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
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
        self.gridLayout.addWidget(self.gameframe, 3, 0, 1, 1)

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
        self.gridLayout.addWidget(self.difficulty_frame, 4, 0, 1, 1)

        self.setboard_frame = QtWidgets.QFrame(self.centralwidget)
        self.setboard_frame.setMaximumSize(QtCore.QSize(16777215, 30))
        self.setboard_frame.setMinimumSize(QtCore.QSize(120, 20))
        self.setboard_frame.setStyleSheet("QFrame{\n"
                                            "    border-color: rgb(167, 167, 167);\n"
                                            "    background-color: rgb(236, 236, 236);\n"
                                            "}")
        self.setboard_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setboard_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setboard_frame.setObjectName("setboard")
        self.setboard_frame.setMaximumSize(QtCore.QSize(16777215, 60))
        self.setboard_frame.setMinimumSize(QtCore.QSize(120, 45))
        self.setboard_Layout = QtWidgets.QHBoxLayout(self.setboard_frame)
        self.setboard_Layout.setObjectName("setboard_Layout")
        self.board_state = QtWidgets.QLineEdit(self.centralwidget)
        self.board_state.setStyleSheet("font: 600 12pt \"consolas\";")
        self.board_state.setObjectName("board_state")
        self.board_state.setText('0000000000000000')
        self.setboard_Layout.addWidget(self.board_state)
        self.set_board_bt = QtWidgets.QPushButton(self.centralwidget)
        self.set_board_bt.setMaximumSize(QtCore.QSize(90, 16777215))
        self.set_board_bt.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.set_board_bt.setObjectName("set_board_bt")
        self.set_board_bt.clicked.connect(self.textbox_reset_board)  # type: ignore
        self.setboard_Layout.addWidget(self.set_board_bt)
        self.gridLayout.addWidget(self.setboard_frame, 2, 0, 1, 1)

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
        self.set_board_bt.setText(_translate("Game", "SET"))

    def difficulty_changed(self):
        self.gameframe.difficulty = self.difficulty_slider.value() / 100
        self.gameframe.setFocus()

    def show_message(self):
        QtWidgets.QMessageBox.information(self, self.tr('Information'),
                                          self.tr('''Probability of generating an EVIL number, default 0. 
                                          Only effective for players'''))
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
        anim_group.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def keyPressEvent(self, event, ):
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
        elif event.key() in (QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete) and not self.ai_state:
            self.handleUndo()
            event.accept()
        elif event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) and not self.ai_state:
            self.handleOneStep()
            event.accept()
        else:
            super().keyPressEvent(event)  # 其他键交给父类处理

    def process_input(self, direction):
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.update_score()
        self.save_game_state()
        self.isProcessing = False

    def handleOneStep(self):
        if self.isProcessing or self.gameframe.ai_processing:
            return
        self.isProcessing = True
        self.ai_dispatcher.reset(self.gameframe.board, self.gameframe.board_encoded)
        best_move = self.ai_dispatcher.dispatcher()
        current_table = self.ai_dispatcher.current_table
        if current_table != self.last_table:
            self.statusbar.showMessage(self.tr("Using " + current_table), 1000)
            self.last_table = current_table

        if best_move == 'AI':
            self.gameframe.ai_step()
        else:
            self.gameframe.do_move(best_move.capitalize())
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.update_score()
        if self.ai_state:
            if self.gameframe.died_when_ai_state:
                self.ai_state = False
                self.ai.setText("AI: ON")
                self.gameframe.died_when_ai_state = False
            elif best_move != 'AI':
                self.ai_timer.singleShot(20, self.handleOneStep)
        # print(self.ai_dispatcher.last_operator)
        self.save_game_state()
        self.isProcessing, self.gameframe.ai_processing = False, False

    def ai_move_done(self, is_done):
        if is_done and self.ai_state:
            self.ai_timer.singleShot(20, self.handleOneStep)

    def handleUndo(self):
        self.gameframe.undo()
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.update_score()
        self.gameframe.died_when_ai_state = False

    def handleNewGame(self):
        self.gameframe.setup_new_game()
        self.update_score()
        self.gameframe.died_when_ai_state = False

    def toggleAI(self):
        if not self.ai_state:
            self.ai.setText(self.tr("STOP"))
            self.ai_state = True
            self.ai_timer.singleShot(20, self.handleOneStep)
            if (not SingletonConfig().config['filepath_map'].get('free12w_2048', []) or
                    not SingletonConfig().config['filepath_map'].get('4442f_2k', []) or
                not SingletonConfig().config['filepath_map'].get('free11w_512', []) or
                not SingletonConfig().config['filepath_map'].get('free11w_2k', [])):
                self.statusbar.showMessage(self.tr(
                    "Run free12w-2k free11w-2k 4442f-2k free11w-512 for best performance."), 5000)
        else:
            self.ai.setText("AI: ON")
            self.ai_state = False

    def textbox_reset_board(self):
        if not self.board_state.text():
            return
        if self.ai_state or self.isProcessing or self.gameframe.ai_processing:
            self.statusbar.showMessage(self.tr(
                "Please turn off the AI or wait for the previous process to finish."), 1000)
            return

        self.gameframe.board_encoded = np.uint64(int(self.board_state.text(), 16))
        self.gameframe.board = bm.decode_board(self.gameframe.board_encoded)
        self.gameframe._last_values = self.gameframe.board.copy()
        self.gameframe.update_all_frame(self.gameframe.board)
        self.gameframe.setFocus()

        # 重置历史记录
        self.gameframe.score = 0
        self.update_score()
        self.gameframe.history = []
        self.gameframe.history.append((self.gameframe.board_encoded, self.gameframe.score))
        self.gameframe.died_when_ai_state = False

    def save_game_state(self, save=False):
        current_time = int(time.time() * 100)
        if save or current_time % 32 == 0:
            SingletonConfig().config['game_state'] = [self.gameframe.board_encoded, self.gameframe.score,
                                                      int(self.best_points.text())]
            SingletonConfig.save_config(SingletonConfig().config)

    def closeEvent(self, event):
        self.ai.setText("AI: ON")
        self.ai_state = False
        self.save_game_state(True)
        event.accept()  # 确认关闭事件


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = GameWindow()
    main.show()
    sys.exit(app.exec_())
