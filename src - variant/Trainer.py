import os.path
import sys
from typing import Union, List

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon, QCursor

from Calculator import ReverseUD, ReverseLR, RotateR, RotateL
from Config import SingletonConfig, formation_info
from BoardMover import SingletonBoardMover
from Calculator import find_merge_positions
from BookReader import BookReader


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
        self.mover = SingletonBoardMover(2)
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


class TrainFrame(BaseBoardFrame):
    dis32k = SingletonConfig().config.get('dis_32k', False)
    v_inits = {
        '2x4': np.array([np.uint64(0xffff00000000ffff)], dtype=np.uint64),
        '3x3': np.array([np.uint64(0x000f000f000fffff)], dtype=np.uint64),
        '3x4': np.array([np.uint64(0x000000000000ffff)], dtype=np.uint64),
    }
    update_results = QtCore.pyqtSignal()  # 手动模式查表更新

    def __init__(self, centralwidget=None):
        super(TrainFrame, self).__init__(centralwidget)
        self.num_to_set = None
        self.manual_mode = False

    def gen_new_num(self, do_anim=True):
        if not self.manual_mode:
            super().gen_new_num(do_anim)
        else:
            self.history.append((self.board_encoded, self.score))
            self.update_all_frame(self.board)

    def mousePressEvent(self, event):
        if self.num_to_set is None and not self.manual_mode:
            self.setFocus()
            return

        self.score = 0
        local_pos = self.game_square.mapFromParent(event.pos())
        # 找到被点击的格子
        i, j = 0, 0
        is_click_on_tiles = False
        for i, row in enumerate(self.game_square.frames):
            for j, frame in enumerate(row):
                if frame.geometry().contains(local_pos):
                    is_click_on_tiles = True
                    break  # 跳出内层循环
            else:
                continue  # 如果没有遇到 break，继续外层循环
            break  # 如果内层循环遇到 break，这里就会跳出外层循环

        if not is_click_on_tiles:  # 点击到的是间隙（不是任何一个tiles）
            return

        if self.num_to_set is not None:
            if event.button() == Qt.LeftButton:
                num_to_set = self.num_to_set
            elif event.button() == Qt.RightButton:
                num_to_set = self.board[i][j] * 2 if self.board[i][j] > 0 else 2
                num_to_set = 0 if num_to_set > 32768 else num_to_set
            else:
                num_to_set = self.board[i][j] // 2 if self.board[i][j] > 0 else 32768
                num_to_set = 0 if num_to_set == 1 else num_to_set
            self.update_frame(num_to_set, i, j, False)
            self.board[i][j] = num_to_set
            self.board_encoded = np.uint64(self.mover.encode_board(self.board))
            return
        else:  # 手动模式
            if self.board[i][j] == 0:
                if event.button() == Qt.LeftButton:
                    num_to_set = 2
                    self.newtile_pos, self.newtile = i * 4 + j, 1
                elif event.button() == Qt.RightButton:
                    num_to_set = 4
                    self.newtile_pos, self.newtile = i * 4 + j, 2
                else:
                    return
                self.update_frame(num_to_set, i, j, False)
                self.board[i][j] = num_to_set
                self.board_encoded = np.uint64(self.mover.encode_board(self.board))
                self.history.append((self.board_encoded, self.score))
                self.update_results.emit()
                return

    def update_frame(self, value, row, col, anim=False):
        """重写方法以配合不显示32k格子数字的设置"""
        super().update_frame(value, row, col, anim)
        if value == 32768 and not self.dis32k:
            self.game_square.labels[row][col].setText('')

    def set_new_num(self, new_tile_pos, val, do_anim=True):
        self.board_encoded |= np.uint64(val) << np.uint64(4 * (15 - new_tile_pos))
        self.board = self.mover.decode_board(self.board_encoded)
        self.update_frame(2 ** val, new_tile_pos // 4, new_tile_pos % 4, anim=do_anim)
        self.history.append((self.board_encoded, self.score))
        self.newtile_pos, self.newtile = new_tile_pos, val

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


# noinspection PyAttributeOutsideInit
class TrainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.current_pattern = ''
        self.pattern_settings = ['', '', '0']
        self.isProcessing = False
        self.result = dict()  # 四个方向成功率字典

        self.ai_state = False  # demo状态
        self.ai_timer = QTimer(self)

        self.recording_state = False  # 录制状态
        self.records = np.empty(0, dtype='uint8,uint16,uint16,uint16,uint16')  # 录制的回放
        self.record_length = 0  # 已录制的长度
        self.playing_record_state = False  # 播放状态
        self.record_loaded = None  # 已加载的回放
        self.played_length = 0  # 已播放的长度
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self.replay_step)  # type: ignore # 回放状态定时自动走棋从

        self.book_reader = BookReader()

        self.statusbar.showMessage("All features may be slow when used for the first time. Please be patient.", 8000)

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QIcon(r'pic\2048.ico'))
        self.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(6, 6, 6, 6)
        self.gridLayout.setVerticalSpacing(8)
        self.gridLayout.setHorizontalSpacing(8)
        self.gridLayout.setObjectName("gridLayout")

        self.gameframe = TrainFrame(self.centralwidget)
        self.gameframe.setFocusPolicy(Qt.StrongFocus)
        self.gridLayout.addWidget(self.gameframe, 1, 0, 1, 2)
        self.gameframe.update_results.connect(self.manual_mode_update_results)

        self.operate = QtWidgets.QFrame(self.centralwidget)
        self.operate.setMaximumSize(QtCore.QSize(560, 720))
        self.operate.setStyleSheet("QFrame{\n"
                                   "    border-color: rgb(167, 167, 167);\n"
                                   "    background-color: rgb(236, 236, 236);\n"
                                   "}")
        self.operate.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.operate.setFrameShadow(QtWidgets.QFrame.Raised)
        self.operate.setObjectName("operate")
        self.gridLayout_operate = QtWidgets.QGridLayout(self.operate)
        self.gridLayout_operate.setContentsMargins(6, 6, 6, 6)
        self.gridLayout_operate.setVerticalSpacing(8)
        self.gridLayout_operate.setHorizontalSpacing(8)
        self.gridLayout_operate.setObjectName("gridLayout_operate")

        self.pattern_text = QtWidgets.QLabel(self.operate)
        self.pattern_text.setStyleSheet("font: 2400 32pt \"Cambria\";")
        self.pattern_text.setObjectName("pattern_text")
        self.pattern_text.setMaximumSize(QtCore.QSize(640, 80))
        self.gridLayout_operate.addWidget(self.pattern_text, 0, 0, 1, 1)
        self.setboard_Layout = QtWidgets.QHBoxLayout()
        self.setboard_Layout.setObjectName("setboard_Layout")
        self.board_state = QtWidgets.QLineEdit(self.operate)
        self.board_state.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.board_state.setObjectName("board_state")
        self.board_state.setText('0000000000000000')
        self.setboard_Layout.addWidget(self.board_state)
        self.set_board_bt = QtWidgets.QPushButton(self.operate)
        self.set_board_bt.setMaximumSize(QtCore.QSize(90, 16777215))
        self.set_board_bt.setObjectName("set_board_bt")
        self.set_board_bt.clicked.connect(self.handle_set_board)  # type: ignore
        self.setboard_Layout.addWidget(self.set_board_bt)
        self.dis32k_checkBox = QtWidgets.QCheckBox(self.operate)
        self.dis32k_checkBox.setStyleSheet("font: 360 10pt \"Cambria\";")
        self.dis32k_checkBox.setObjectName("dis32k_checkBox")
        self.setboard_Layout.addWidget(self.dis32k_checkBox)
        if SingletonConfig().config.get('dis_32k', True):
            self.dis32k_checkBox.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.dis32k_checkBox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.dis32k_checkBox.stateChanged.connect(self.dis32k_state_change)  # type: ignore
        self.gridLayout.addLayout(self.setboard_Layout, 0, 0, 1, 2)

        self.tiles_frame = QtWidgets.QFrame(self.operate)
        self.tiles_frame.setFixedSize(QtCore.QSize(544, 136))
        self.tiles_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tiles_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tiles_frame.setObjectName("tiles_frame")
        self.gridLayout_tiles = QtWidgets.QGridLayout(self.tiles_frame)
        self.gridLayout_tiles.setContentsMargins(1, 1, 1, 1)
        self.gridLayout_tiles.setHorizontalSpacing(2)
        self.gridLayout_tiles.setVerticalSpacing(2)
        self.gridLayout_tiles.setObjectName("gridLayout_tiles")

        self.tile_buttons = []
        button_numbers = ['0', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1k', '2k', '4k', '8k', '16k',
                          '32k']
        for index, num in enumerate(button_numbers):
            button = QtWidgets.QPushButton(self.tiles_frame)
            button.setObjectName(f"t{num}")
            button.setText(f"{num}")
            button.setCheckable(True)
            button.clicked.connect(self.tiles_bt_on_click)  # type: ignore
            button.setFixedSize(QtCore.QSize(66, 66))
            self.gridLayout_tiles.addWidget(button, index // 8, index % 8, 1, 1)
            self.tile_buttons.append(button)
        self.gridLayout_operate.addWidget(self.tiles_frame, 8, 0, 1, 1)

        self.gridLayout_bts = QtWidgets.QGridLayout()
        self.gridLayout_bts.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_bts.setObjectName("gridLayout_bts")
        self.RL = QtWidgets.QPushButton(self.operate)
        self.RL.setMaximumSize(QtCore.QSize(180, 16777215))
        self.RL.setObjectName("RL")
        self.RL.clicked.connect(lambda: self.handle_rotate('RL'))  # type: ignore
        self.gridLayout_bts.addWidget(self.RL, 1, 1, 1, 1)
        self.Demo = QtWidgets.QPushButton(self.operate)
        self.Demo.setMaximumSize(QtCore.QSize(180, 16777215))
        self.Demo.setObjectName("L90")
        self.Demo.clicked.connect(self.toggle_demo)  # type: ignore
        self.gridLayout_bts.addWidget(self.Demo, 0, 1, 1, 1)
        self.R90 = QtWidgets.QPushButton(self.operate)
        self.R90.setMaximumSize(QtCore.QSize(180, 16777215))
        self.R90.setObjectName("R90")
        self.R90.clicked.connect(lambda: self.handle_rotate('R90'))  # type: ignore
        self.gridLayout_bts.addWidget(self.R90, 1, 2, 1, 1)
        self.undo = QtWidgets.QPushButton(self.operate)
        self.undo.setMaximumSize(QtCore.QSize(180, 16777215))
        self.undo.setObjectName("undo")
        self.undo.clicked.connect(self.handleUndo)  # type: ignore
        self.gridLayout_bts.addWidget(self.undo, 0, 0, 1, 1)
        self.UD = QtWidgets.QPushButton(self.operate)
        self.UD.setMaximumSize(QtCore.QSize(180, 16777215))
        self.UD.setObjectName("UD")
        self.UD.clicked.connect(lambda: self.handle_rotate('UD'))  # type: ignore
        self.gridLayout_bts.addWidget(self.UD, 1, 0, 1, 1)
        self.step = QtWidgets.QPushButton(self.operate)
        self.step.setMaximumSize(QtCore.QSize(180, 16777215))
        self.step.setObjectName("step")
        self.step.clicked.connect(self.handle_step)  # type: ignore
        self.gridLayout_bts.addWidget(self.step, 0, 2, 1, 1)
        self.default = QtWidgets.QPushButton(self.operate)
        self.default.setMaximumSize(QtCore.QSize(180, 16777215))
        self.default.setObjectName("default")
        self.default.clicked.connect(self.handle_set_default)  # type: ignore
        self.gridLayout_bts.addWidget(self.default, 0, 3, 1, 1)
        self.L90 = QtWidgets.QPushButton(self.operate)
        self.L90.setMaximumSize(QtCore.QSize(180, 16777215))
        self.L90.setObjectName("L90")
        self.L90.clicked.connect(lambda: self.handle_rotate('L90'))  # type: ignore
        self.gridLayout_bts.addWidget(self.L90, 1, 3, 1, 1)
        self.gridLayout_operate.addLayout(self.gridLayout_bts, 5, 0, 1, 1)

        self.gridLayout_record = QtWidgets.QGridLayout()
        self.gridLayout_record.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_record.setObjectName("gridLayout_bts")
        self.record = QtWidgets.QPushButton(self.operate)
        self.record.setMaximumSize(QtCore.QSize(240, 16777215))
        self.record.setObjectName("record")
        self.record.clicked.connect(self.handle_record)  # type: ignore
        self.gridLayout_record.addWidget(self.record, 0, 0, 1, 1)
        self.load_record = QtWidgets.QPushButton(self.operate)
        self.load_record.setMaximumSize(QtCore.QSize(240, 16777215))
        self.load_record.setObjectName("record")
        self.load_record.clicked.connect(self.handle_load_record)  # type: ignore
        self.gridLayout_record.addWidget(self.load_record, 0, 1, 1, 1)
        self.play_record = QtWidgets.QPushButton(self.operate)
        self.play_record.setMaximumSize(QtCore.QSize(240, 16777215))
        self.play_record.setObjectName("record")
        self.play_record.clicked.connect(self.handle_play_record)  # type: ignore
        self.gridLayout_record.addWidget(self.play_record, 0, 2, 1, 1)
        self.manual_checkBox = QtWidgets.QCheckBox(self.operate)
        self.manual_checkBox.setStyleSheet("font: 360 10pt \"Cambria\";")
        self.manual_checkBox.setObjectName("manual_checkBox")
        self.gridLayout_record.addWidget(self.manual_checkBox, 0, 3, 1, 1)
        self.manual_checkBox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.manual_checkBox.stateChanged.connect(self.manual_state_change)  # type: ignore
        self.gridLayout_operate.addLayout(self.gridLayout_record, 6, 0, 1, 1)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.results_text = QtWidgets.QLabel(self.operate)
        self.results_text.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.results_text.setObjectName("results_text")
        self.horizontalLayout.addWidget(self.results_text)
        self.show_results_checkbox = QtWidgets.QCheckBox(self.operate)
        self.show_results_checkbox.setStyleSheet("font: 360 10pt \"Cambria\";")
        self.show_results_checkbox.setObjectName("show_results_checkbox")
        self.show_results_checkbox.stateChanged.connect(self.show_results)  # type: ignore
        self.horizontalLayout.addWidget(self.show_results_checkbox)
        self.gridLayout_operate.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.results_label = QtWidgets.QLabel(self.operate)
        self.results_label.setMinimumSize(QtCore.QSize(0, 180))
        self.results_label.setMaximumSize(QtCore.QSize(16777215, 220))
        self.results_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                         "border-color: rgb(0, 0, 0); font: 75 18pt \"Consolas\";")
        self.results_label.setText("")
        self.results_label.setObjectName("results_label")
        self.gridLayout_operate.addWidget(self.results_label, 2, 0, 1, 1)

        self.setpath_Layout = QtWidgets.QHBoxLayout()
        self.setpath_Layout.setObjectName("setpath_Layout")
        self.filepath_text = QtWidgets.QLabel(self.operate)
        self.filepath_text.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.filepath_text.setObjectName("filepath_text")
        self.setpath_Layout.addWidget(self.filepath_text)
        self.filepath = QtWidgets.QLineEdit(self.operate)
        self.filepath.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.filepath.setObjectName("filepath")
        self.setpath_Layout.addWidget(self.filepath)
        self.set_filepath_bt = QtWidgets.QPushButton(self.operate)
        self.set_filepath_bt.setMaximumSize(QtCore.QSize(120, 16777215))
        self.set_filepath_bt.setObjectName("set_filepath_bt")
        self.set_filepath_bt.clicked.connect(self.filepath_changed)  # type: ignore
        self.setpath_Layout.addWidget(self.set_filepath_bt)
        self.gridLayout_operate.addLayout(self.setpath_Layout, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.operate, 0, 2, 2, 1)
        self.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 522, 22))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.menu_ptn = QtWidgets.QMenu(self.menubar)
        self.menu_ptn.setObjectName("menuMENU")
        for ptn in list(formation_info.keys()) + ['?']:
            m = QtWidgets.QAction(ptn, self)
            m.triggered.connect(lambda: self.menu_selected(0))  # type: ignore
            self.menu_ptn.addAction(m)
        self.menubar.addAction(self.menu_ptn.menuAction())
        self.menu_tgt = QtWidgets.QMenu(self.menubar)
        self.menu_tgt.setObjectName("menuMENU")
        for ptn in ["128", "256", "512", "1024", "2048", "4096", "8192"]:
            m = QtWidgets.QAction(ptn, self)
            m.triggered.connect(lambda: self.menu_selected(1))  # type: ignore
            self.menu_tgt.addAction(m)
        self.menubar.addAction(self.menu_tgt.menuAction())
        self.menu_pos = QtWidgets.QMenu(self.menubar)
        self.menu_pos.setObjectName("menuMENU")

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def show(self):
        self.gameframe.update_all_frame(self.gameframe.board)
        button_numbers = ['0', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1k', '2k', '4k', '8k', '16k',
                          '32k']
        for index, num in enumerate(button_numbers):
            color = self.gameframe.game_square.colors[index - 1] if index != 0 else '#e5e5e5'
            button = self.tile_buttons[index]
            button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {color};
                            font: 600 12pt 'Calibri';
                            color: rgb(255, 255, 255);
                            border: 2px solid transparent;
                            width: 56px; height: 56px; 
                        }}
                        QPushButton:checked {{
                            background-color: {color};
                            font: 1350 16pt 'Calibri';
                            color: rgb(255, 255, 255);
                            border: 6px solid red;
                        }}
                    """)
        super().show()

    # noinspection PyTypeChecker
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Train", "Practise"))
        self.pattern_text.setText(_translate("Train", ""))
        self.set_board_bt.setText(_translate("Train", "SET"))
        self.RL.setText(_translate("Train", "RL"))
        self.Demo.setText(_translate("Train", "DEMO"))
        self.R90.setText(_translate("Train", "R90"))
        self.undo.setText(_translate("Train", "UNDO"))
        self.UD.setText(_translate("Train", "UD"))
        self.L90.setText(_translate("Train", "L90"))
        self.record.setText(_translate("Train", 'Record Demo'))
        self.play_record.setText(_translate("Train", 'Play Record'))
        self.load_record.setText(_translate("Train", 'Load Record'))
        self.manual_checkBox.setText(_translate("Train", "Manual"))
        self.step.setText(_translate("Train", "ONESTEP"))
        self.default.setText(_translate("Train", "Default"))
        self.results_text.setText(_translate("Train", "RESULTS:"))
        self.show_results_checkbox.setText(_translate("Train", "Show"))
        self.filepath_text.setText(_translate("Train", "FILEPATH:"))
        self.set_filepath_bt.setText(_translate("Train", "SET..."))
        self.dis32k_checkBox.setText(_translate("Train", "Display numbers for 32k tile"))
        self.menu_ptn.setTitle(_translate("Train", "Pattern"))
        self.menu_pos.setTitle(_translate("Train", "Position"))
        self.menu_tgt.setTitle(_translate("Train", "Target"))

    def dis32k_state_change(self):
        SingletonConfig().config['dis_32k'] = self.dis32k_checkBox.isChecked()
        self.gameframe.dis32k = SingletonConfig().config['dis_32k']
        self.gameframe.update_all_frame(self.gameframe.board)
        self.gameframe.setFocus()

    def manual_state_change(self):
        self.gameframe.manual_mode = self.manual_checkBox.isChecked()
        self.gameframe.setFocus()

    def menu_selected(self, i):
        self.pattern_settings[i] = self.sender().text()
        self.pattern_text.setText('_'.join(self.pattern_settings[:2]))
        if '' not in self.pattern_settings:
            self.current_pattern = '_'.join(self.pattern_settings[:2])

            path_list = SingletonConfig().config['filepath_map'].get(self.current_pattern, [])
            if path_list:
                self.filepath.setText(path_list[-1])
            else:
                self.filepath.setText(' ')
            self.pattern_text.setText(self.current_pattern)
            self.handle_set_default()
            self.show_results()

    def tiles_bt_on_click(self):
        sender = self.sender()
        self.ai_state = False
        self.Demo.setText('Demo')
        if sender.isChecked():
            self.isProcessing = True
            self.gameframe.num_to_set = \
                {'0': 0, '2': 2, '4': 4, '8': 8, '16': 16, '32': 32, '64': 64, '128': 128, '256': 256, '512': 512,
                 '1k': 1024, '2k': 2048, '4k': 4096, '8k': 8192, '16k': 16384, '32k': 32768}[sender.text()]
            for button in self.tile_buttons:
                if button != sender:
                    button.setChecked(False)
            QtWidgets.QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))
        else:
            QtWidgets.QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))
            self.isProcessing = False
            self.gameframe.num_to_set = None
            self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
            self.gameframe.history.append((self.gameframe.board_encoded, self.gameframe.score))
            self.show_results()

    def show_results(self):
        if self.show_results_checkbox.isChecked():
            board = self.gameframe.mover.decode_board(np.uint64(int('0x' + self.board_state.text().rjust(16, '0'), 16)))
            result = self.book_reader.move_on_dic(board, self.pattern_settings[0], self.pattern_settings[1],
                                            self.current_pattern)
            if isinstance(result, dict):
                results_text = "\n".join(f"  {key.capitalize()[0]}: {value}" for key, value in result.items())
                self.results_label.setText(results_text)
                self.result = result
                result0 = result[list(result.keys())[0]]
                if not result0 or not isinstance(result0, (float, int)):
                    self.statusbar.showMessage("State not found or 0 success rate", 1000)
            else:
                self.results_label.setText(str(result))
                self.result = dict()
        else:
            self.results_label.setText('')
            self.result = dict()
        self.gameframe.setFocus()

    def manual_mode_update_results(self):
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.show_results()

    def filepath_changed(self):
        options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filepath = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if filepath:
            self.filepath.setText(filepath)
            current_path_list = SingletonConfig().config['filepath_map'].get(self.current_pattern, [])

            # 清理无效路径
            for path in current_path_list:
                if not os.path.exists(path):
                    current_path_list.remove(path)

            if filepath not in current_path_list:
                SingletonConfig().config['filepath_map'][self.current_pattern] = (
                        current_path_list + [filepath])
                SingletonConfig().save_config(SingletonConfig().config)
            self.show_results()

    def handleUndo(self):
        if not self.playing_record_state:
            self.gameframe.undo()
            self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
            if self.record_loaded is not None and self.played_length > 1:
                self.played_length -= 1
                current_state = self.record_loaded[self.played_length]
                self.decode_result_from_record(current_state)
                results_text = "\n".join(f"  {key.capitalize()[0]}: {value}" for key, value in self.result.items())
                self.results_label.setText(results_text)
            else:
                self.show_results()
                
    def handle_set_board(self):
        self.textbox_reset_board()
        self.show_results()

    def textbox_reset_board(self):
        self.gameframe.board_encoded = np.uint64(int(self.board_state.text(), 16))
        self.gameframe.board = self.gameframe.mover.decode_board(self.gameframe.board_encoded)
        self.gameframe.update_all_frame(self.gameframe.board)
        self.gameframe.history.append((self.gameframe.board_encoded, self.gameframe.score))

    def handle_rotate(self, mode):
        rotate_func = {'UD': ReverseUD, 'RL': ReverseLR, 'R90': RotateR, 'L90': RotateL}[mode]
        self.board_state.setText(hex(rotate_func(self.gameframe.board_encoded))[2:].rjust(16, '0'))
        self.handle_set_board()

    def handle_set_default(self):
        path_list = SingletonConfig().config['filepath_map'].get(self.current_pattern, [])
        random_board = self.book_reader.get_random_state(path_list, self.current_pattern)
        self.board_state.setText(hex(random_board)[2:].rjust(16, '0'))
        self.handle_set_board()

        self.gameframe.score = 0
        self.gameframe.history = []
        self.gameframe.history.append((self.gameframe.board_encoded, self.gameframe.score))
        self.played_length = 0
        self.record_loaded = None

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
        elif event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            # 焦点在 board_state
            if self.focusWidget() == self.board_state:
                self.handle_set_board()
                return
            # 执行默认的 one_step
            self.one_step()
        else:
            super().keyPressEvent(event)  # 其他键交给父类处理

    def process_input(self, direction):
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.show_results()
        self.isProcessing = False

    def handle_step(self):
        if self.played_length == 0 or self.record_loaded is None:
            self.one_step()
        else:
            self.playing_record_state = True
            self.replay_step()
            self.playing_record_state = False

    def one_step(self):
        if not self.isProcessing and self.result:
            move = list(self.result.keys())[0]
            if self.ai_state and (self.result[move] == 0 or self.result[move] is None or self.result[move] == '?'):
                self.ai_state = False
                self.Demo.setText('Demo')
                self.ai_timer.stop()
                self.isProcessing = False
            if isinstance(self.result[move], (int, float)):
                self.isProcessing = True
                self.gameframe.do_move(move.capitalize())
                self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
                self.show_results()
                if self.recording_state:
                    self.record_current_state(move)
                if self.ai_state:
                    steps_per_second = SingletonConfig().config['demo_speed'] / 10
                    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
                    self.ai_timer.singleShot(int(1000 / steps_per_second), self.one_step)

                self.isProcessing = False

    def toggle_demo(self):
        self.ai_state = False
        if not self.ai_state and not self.playing_record_state:
            self.played_length = 0
            self.ai_state = True
            self.Demo.setText('STOP')
            self.one_step()
        else:
            self.ai_state = False
            self.Demo.setText('Demo')
            self.ai_timer.stop()

    def handle_record(self):
        if not self.recording_state:
            self.recording_state = True
            self.record.setText('Save')
            self.records = np.empty(10000, dtype='uint8,uint16,uint16,uint16,uint16')
            v = np.uint64(self.gameframe.board_encoded)
            self.records[0] = (np.uint8(0), np.uint16(v & np.uint64(0xFFFF)),
                               np.uint16((v >> np.uint64(16)) & np.uint64(0xFFFF)),
                               np.uint16((v >> np.uint64(32)) & np.uint64(0xFFFF)),
                               np.uint16((v >> np.uint64(48)) & np.uint64(0xFFFF)))
            self.record_length = 1
            self.statusbar.showMessage('Recording started', 3000)
        else:
            if self.ai_state:
                self.toggle_demo()
            if self.record_length > 2:
                file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Record", "",
                                                                     "Record Files (*.rec);;All Files (*)")
                if file_path:  # 用户选择了文件路径
                    self.records[:self.record_length].tofile(file_path)
            self.recording_state = False
            self.record_length = 0
            self.records = np.empty(0, dtype='uint8,uint16,uint16,uint16,uint16')
            self.record.setText('Record Demo')

    def record_current_state(self, move):
        gen_pos, gen_num = self.gameframe.newtile_pos, self.gameframe.newtile
        move = {'up': 0, 'down': 1, 'left': 2, 'right': 3}.get(move)
        changes = np.uint8((move & 0b11) | ((gen_pos & 0b1111) << 2) | (((gen_num - 1) & 0b1) << 6))
        success_rates = []
        for direction in ('up', 'down', 'left', 'right'):
            rate = self.result.get(direction, 0)
            rate = rate if isinstance(rate, (float, int)) else 0
            success_rates.append(np.uint16(rate * 16000))
        self.records[self.record_length] = (changes, *success_rates)
        self.record_length += 1

    def handle_load_record(self):
        if self.ai_state:
            self.toggle_demo()
        if self.playing_record_state:
            self.handle_play_record()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Record File", "",
                                                             "Record Files (*.rec);;All Files (*)")
        if file_path:  # 检查用户是否选择了文件
            try:
                self.record_loaded = np.fromfile(file_path, dtype='uint8,uint16,uint16,uint16,uint16')
                self.statusbar.showMessage("File loaded successfully")
                board = np.uint64((np.uint64(self.record_loaded[0][4]) << np.uint16(48)) |
                                  (np.uint64(self.record_loaded[0][3]) << np.uint16(32)) |
                                  (np.uint64(self.record_loaded[0][2]) << np.uint16(16)) |
                                  np.uint64(self.record_loaded[0][1]))
                self.board_state.setText(hex(board)[2:].rjust(16,'0'))
                self.results_label.setText('')
                self.textbox_reset_board()
                self.played_length = 1
            except Exception as e:
                self.statusbar.showMessage(f"Failed to load file: {e}")
                self.record_loaded = None

    def handle_play_record(self):
        if self.playing_record_state:
            self.playing_record_state = False
            self.play_record.setText('Play Record')
            self.replay_timer.stop()
            self.isProcessing = False
        elif self.record_loaded is None:
            return
        else:
            self.isProcessing = True
            self.playing_record_state = True
            self.play_record.setText('Stop')
            if self.played_length == 0:
                board = np.uint64((np.uint64(self.record_loaded[0][4]) << np.uint16(48)) |
                                  (np.uint64(self.record_loaded[0][3]) << np.uint16(32)) |
                                  (np.uint64(self.record_loaded[0][2]) << np.uint16(16)) |
                                  np.uint64(self.record_loaded[0][1]))
                self.board_state.setText(hex(board)[2:])
                self.textbox_reset_board()
                self.played_length += 1
            steps_per_second = SingletonConfig().config['demo_speed'] / 10
            self.replay_timer.start(int(1000 / steps_per_second))

    def replay_step(self):
        if self.record_loaded is None or self.played_length >= len(self.record_loaded):
            self.played_length = 0
            self.playing_record_state = False
            self.play_record.setText('Play Record')
            self.replay_timer.stop()
            self.isProcessing = False
        elif self.playing_record_state:
            current_state = self.record_loaded[self.played_length]
            move = current_state[0] & 0b11
            new_val_pos = (current_state[0] >> 2) & 0b1111
            new_val = ((current_state[0] >> 6) & 0b1) + 1
            move = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}[move]
            self.gameframe.do_move(move.capitalize(), False)
            self.gameframe.set_new_num(new_val_pos, new_val, SingletonConfig().config['do_animation'][0])
            self.decode_result_from_record(current_state)
            results_text = "\n".join(f"  {key.capitalize()[0]}: {value}" for key, value in self.result.items())
            self.results_label.setText(results_text)
            self.played_length += 1

    def decode_result_from_record(self, current_state):
        result = {}
        for i in range(4):
            direction = ('up', 'down', 'left', 'right')[i]
            result[direction] = round((current_state[i + 1] / 16000), 5)
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        self.result = result

    def closeEvent(self, event):
        self.ai_state = False
        self.Demo.setText('Demo')
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = TrainWindow()
    main.show()
    sys.exit(app.exec_())
