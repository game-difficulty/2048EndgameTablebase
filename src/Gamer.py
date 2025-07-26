from typing import List, Tuple, Dict

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer

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
            for anim_id in (1,0):
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
        self.history = []

        self.newtile_pos = 0
        self.newtile = 1

        self.use_variant_mover = 0

        self.timer1, self.timer2, self.timer3 = QTimer(), QTimer(), QTimer()
        self._last_values = self.board.copy()

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
        self.history.append(self.board_encoded)
        self.newtile_pos, self.newtile = new_tile_pos, val
        if do_anim:
            self.timer1.singleShot(125, lambda: self.game_square.animate_appear(new_tile_pos // 4, new_tile_pos % 4, 2 ** val))

    def do_move(self, direction: str, do_gen=True):
        if self.game_square.active_animations:
            for timer in (self.timer1, self.timer2, self.timer3):
                timer.stop()
            self.game_square.cancel_all_animations()
            self.update_all_frame(self._last_values)
        mover = bm
        do_anim = SingletonConfig().config['do_animation']
        direct = {'Left': 1, 'Right': 2, 'Up': 3, 'Down': 4}[direction.capitalize()]
        board_encoded_new = mover.move_board(self.board_encoded, direct)
        board_encoded_new = np.uint64(board_encoded_new)
        if board_encoded_new != self.board_encoded:
            self.board_encoded = board_encoded_new

            current_values = self.board
            if do_gen:
                self.gen_new_num(do_anim)
            self.board = bm.decode_board(self.board_encoded)
            if do_anim:
                self.slide_tiles(current_values, direction)
                self.timer2.singleShot(110, lambda: self.pop_merged(current_values, direction))
                # 生成新数字之前的局面
                self.timer3.singleShot(100, lambda: self.update_all_frame((bm.decode_board(board_encoded_new))))
            else:
                self.update_all_frame(self.board)
            self._last_values = self.board.copy()

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
            self.board_encoded = self.history[-1]
            self.board_encoded = np.uint64(self.board_encoded)
            self.board = bm.decode_board(self.board_encoded)
            self.update_all_frame(self.board)

    def showEvent(self, event):
        super().showEvent(event)
        self.game_square.updateGeometry()
