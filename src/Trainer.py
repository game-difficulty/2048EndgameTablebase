import os.path
import sys
import time
from typing import Dict, Tuple
from collections import defaultdict

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon, QCursor, QKeySequence
from PyQt5.QtWidgets import QShortcut, QApplication

import BoardMover as bm
from BookReader import BookReaderDispatcher
from Calculator import ReverseUD, ReverseLR, RotateR, RotateL
from Config import SingletonConfig, category_info, pattern_32k_tiles_map
from Gamer import BaseBoardFrame
from SignalHub import practise_signal


direction_map = defaultdict(lambda: "？")

direction_map.update({
    'u': "上",
    'd': "下",
    'l': "左",
    'r': "右",
    '?': "？",
})


class TrainFrame(BaseBoardFrame):
    dis32k = SingletonConfig().config.get('dis_32k', False)
    v_inits = {
        '2x4': np.array([np.uint64(0xffff00000000ffff)], dtype=np.uint64),
        '3x3': np.array([np.uint64(0x000f000f000fffff)], dtype=np.uint64),
        '3x4': np.array([np.uint64(0x000000000000ffff)], dtype=np.uint64),
    }
    update_results = QtCore.pyqtSignal()  # 手动模式查表更新

    def __init__(self, parents, centralwidget=None):
        super(TrainFrame, self).__init__(centralwidget)
        self.parents = parents
        self.num_to_set = None
        self.spawn_mode = 0  # 0:随机 1：最好 2：最坏 3:手动
        self.moved = 0  # 手动模式下上一操作是否为移动

    def do_move(self, direction: str, do_gen=True):
        # 手动模式下移动后必须先出数
        if self.spawn_mode == 3 and self.moved == 1:
            return
        current_board = self.board_encoded
        super().do_move(direction, do_gen)
        if self.board_encoded != current_board and self.spawn_mode == 3:
            self.moved = 1

    def gen_new_num(self, do_anim=True):
        if self.spawn_mode == 0:  # 0:随机
            self.moved = 0
            super().gen_new_num(do_anim)
        elif self.spawn_mode == 3:  # 3:手动
            self.history.append((self.board_encoded, self.score))  # 移动后摆数前局面是否加入history
        elif self.parents.book_reader:
            self.moved = 0
            results = self._spawns_success_rates()
            if not results:
                super().gen_new_num(do_anim)
            elif self.spawn_mode == 1:
                new_tile_pos, val = max(results, key=results.get)
                self.set_new_num(new_tile_pos, val, do_anim)
            elif self.spawn_mode == 2:
                new_tile_pos, val = min(results, key=results.get)
                self.set_new_num(new_tile_pos, val, do_anim)

    def _spawns_success_rates(self) -> Dict[Tuple[int, int], float | int]:
        results = {}
        for val in (1,2):
            for new_tile_pos in range(16):
                pos = 15 - new_tile_pos
                if ((self.board_encoded >> np.uint64(4 * pos)) & np.uint64(0xF)) == np.uint64(0):
                    board = self.board_encoded | np.uint64(val) << np.uint64(4 * pos)
                    result = self.parents.book_reader.move_on_dic(bm.decode_board(board),
                                                 self.parents.pattern_settings[0], self.parents.pattern_settings[1],
                                            self.parents.current_pattern, self.parents.pattern_settings[2])
                    if isinstance(result, dict):
                        result0 = result[list(result.keys())[0]]
                        if result0 is None or isinstance(result0, str):
                            result0 = 0
                        results[(new_tile_pos, val)] = result0
        return results

    def mousePressEvent(self, event):
        if self.num_to_set is None and self.spawn_mode != 3:
            self.setFocus()
            return

        self.score = 0
        local_pos = self.game_square.mapFromParent(event.pos())
        # 找到被点击的格子
        i, j = 0, 0
        is_click_on_tiles = False
        # 遍历网格布局的所有单元格
        for i in range(self.game_square.rows):
            for j in range(self.game_square.cols):
                # 获取背景网格的位置
                grid_rect = self.game_square.grids[i][j].geometry()

                # 检查点击位置是否在背景网格内
                if grid_rect.contains(local_pos):
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
            self.update_frame(num_to_set, i, j)
            self.board[i][j] = num_to_set
            self.board_encoded = np.uint64(bm.encode_board(self.board))
            return
        else:  # 手动模式
            if self.moved == 0:
                return

            if self.board[i][j] == 0:
                if event.button() == Qt.LeftButton:
                    num_to_set = 2
                    self.newtile_pos, self.newtile = i * 4 + j, 1
                elif event.button() == Qt.RightButton:
                    num_to_set = 4
                    self.newtile_pos, self.newtile = i * 4 + j, 2
                else:
                    return
                self.moved = 0
                self.update_frame(num_to_set, i, j)
                self.board[i][j] = num_to_set
                self.board_encoded = np.uint64(bm.encode_board(self.board))
                self.history.append((self.board_encoded, self.score))
                self.update_results.emit()
                return

    def update_frame(self, value, row, col):
        """重写方法以配合不显示32k格子数字的设置"""
        super().update_frame(value, row, col)
        if value == 32768 and not self.dis32k:
            self.game_square.labels[row][col].setText('')

    def set_new_num(self, new_tile_pos, val, do_anim=True):
        self.board_encoded |= np.uint64(val) << np.uint64(4 * (15 - new_tile_pos))
        self.board = bm.decode_board(self.board_encoded)
        self.history.append((self.board_encoded, self.score))
        self.newtile_pos, self.newtile = new_tile_pos, val
        if do_anim:
            self.timer1.singleShot(125, lambda: self.game_square.animate_appear(new_tile_pos // 4, new_tile_pos % 4, 2 ** val))

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

        self.book_reader: BookReaderDispatcher = BookReaderDispatcher()
        self.reader_thread = ReaderWorker(self.book_reader, np.uint64(0), self.pattern_settings, self.current_pattern)
        self.reader_thread.result_ready.connect(self._show_results)

        practise_signal.board_update.connect(self.handle_page_jump)

        self.statusbar.showMessage(self.tr("All features may be slow when used for the first time. Please be patient."), 8000)

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

        self.gameframe = TrainFrame(self, self.centralwidget)
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

        self.screenshot_btn = QtWidgets.QPushButton(self.operate)
        self.screenshot_btn.clicked.connect(self.capture_and_copy)  # type: ignore
        self.screenshot_btn.setToolTip("Ctrl+Z")
        self.shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut.activated.connect(self.capture_and_copy)  # type: ignore
        self.gridLayout_record.addWidget(self.screenshot_btn, 0, 3, 1, 1)

        self.manual_checkBox = QtWidgets.QCheckBox(self.operate)
        self.manual_checkBox.setStyleSheet("font: 360 10pt \"Cambria\";")
        self.manual_checkBox.setObjectName("manual_checkBox")
        self.gridLayout_record.addWidget(self.manual_checkBox, 1, 0, 1, 1)
        self.manual_checkBox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.manual_checkBox.stateChanged.connect(self.manual_state_change)  # type: ignore
        self.best_spawn_checkBox = QtWidgets.QCheckBox(self.operate)
        self.best_spawn_checkBox.setStyleSheet("font: 360 10pt \"Cambria\";")
        self.best_spawn_checkBox.setObjectName("best_spawn_checkBox")
        self.gridLayout_record.addWidget(self.best_spawn_checkBox, 1, 1, 1, 2)
        self.best_spawn_checkBox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.best_spawn_checkBox.stateChanged.connect(lambda: self.spawn_state_change(1))  # type: ignore
        self.worst_spawn_checkBox = QtWidgets.QCheckBox(self.operate)
        self.worst_spawn_checkBox.setStyleSheet("font: 360 10pt \"Cambria\";")
        self.worst_spawn_checkBox.setObjectName("worst_spawn_checkBox")
        self.gridLayout_record.addWidget(self.worst_spawn_checkBox, 1, 2, 1, 2)
        self.worst_spawn_checkBox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.worst_spawn_checkBox.stateChanged.connect(lambda: self.spawn_state_change(2))  # type: ignore
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
        self.show_results_checkbox.stateChanged.connect(self.read_results)  # type: ignore
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
        for ptn in ["128", "256", "512", "1024", "2048", "4096", "8192"]:
            m = QtWidgets.QAction(ptn, self)
            m.triggered.connect(lambda: self.menu_selected(1))  # type: ignore
            self.menu_tgt.addAction(m)
        self.menubar.addAction(self.menu_tgt.menuAction())
        self.menu_pos = QtWidgets.QMenu(self.menubar)
        self.menu_pos.setObjectName("menuMENU")
        for ptn in ["0", "1", "2"]:
            m = QtWidgets.QAction(ptn, self)
            m.triggered.connect(lambda: self.menu_selected(2))  # type: ignore
            self.menu_pos.addAction(m)
        self.menubar.addAction(self.menu_pos.menuAction())

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def show(self):
        self.gameframe.update_all_frame(self.gameframe.board)
        button_numbers = ['0', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1k', '2k', '4k', '8k', '16k',
                          '32k']
        for index, num in enumerate(button_numbers):
            bg_color = self.gameframe.game_square.colors[index - 1] if index != 0 else '#e5e5e5'
            color = '#776e65' if not SingletonConfig.font_colors[index - 1] else '#f9f6f2'
            button = self.tile_buttons[index]
            button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {bg_color};
                            font: 600 12pt 'Calibri';
                            color: {color};
                            border: 2px solid transparent;
                            width: 56px; height: 56px; 
                        }}
                        QPushButton:checked {{
                            background-color: {bg_color};
                            font: 1350 16pt 'Calibri';
                            color: {color};
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
        self.screenshot_btn.setText(_translate("Train", 'Screenshot(Z)'))
        self.manual_checkBox.setText(_translate("Train", "Manual"))
        self.best_spawn_checkBox.setText(_translate("Train", "Always Best Spawn"))
        self.worst_spawn_checkBox.setText(_translate("Train", "Always Worst Spawn"))
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
        if self.manual_checkBox.isChecked():
            self.worst_spawn_checkBox.setChecked(False)
            self.best_spawn_checkBox.setChecked(False)
            self.gameframe.spawn_mode = 3
        else:
            self.gameframe.spawn_mode = 0
        self.gameframe.setFocus()

    def menu_selected(self, i):
        if self.ai_state:
            self.toggle_demo()  # 停止
            time.sleep(0.05)
        self.pattern_settings[i] = self.sender().text()
        self.pattern_text.setText('_'.join(self.pattern_settings[:2]))
        self.set_to_new_pattern()

    def set_to_new_pattern(self):
        if '' in self.pattern_settings:
            return

        if self.pattern_settings[0] in ['444', 'LL', 'L3']:
            if self.pattern_settings[0] != 'L3' and self.pattern_settings[2] == '2':
                self.pattern_settings[2] = '0'
            self.current_pattern = '_'.join(self.pattern_settings)
        else:
            self.pattern_settings[2] = '0'
            self.current_pattern = '_'.join(self.pattern_settings[:2])

        if self.pattern_settings[0] in ['2x4', '3x3', '3x4']:
            self.gameframe.set_to_variant(self.pattern_settings[0])
        else:
            self.gameframe.set_to_44()

        path_list = SingletonConfig().config['filepath_map'].get(self.current_pattern, [])
        if path_list:
            self.filepath.setText(path_list[-1])
        else:
            self.filepath.setText(' ')
        self.pattern_text.setText(self.current_pattern)
        self.book_reader.dispatch(path_list, self.pattern_settings[0], self.pattern_settings[1])

        self.reader_thread.current_pattern = self.current_pattern
        self.reader_thread.pattern_settings = self.pattern_settings
        self.reader_thread._32ks = pattern_32k_tiles_map.get(self.pattern_settings[0], [0])[0]

        self.handle_set_default()
        self.read_results()

    def tiles_bt_on_click(self):
        sender = self.sender()
        self.ai_state = False
        self.Demo.setText(self.tr('Demo'))
        if sender.isChecked():
            self.isProcessing = True
            self.gameframe.num_to_set = \
                {'0': 0, '2': 2, '4': 4, '8': 8, '16': 16, '32': 32, '64': 64, '128': 128, '256': 256, '512': 512,
                 '1k': 1024, '2k': 2048, '4k': 4096, '8k': 8192, '16k': 16384, '32k': 32768}[sender.text()]
            for button in self.tile_buttons:
                if button != sender:
                    button.setChecked(False)
            QtWidgets.QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))
            if self.show_results_checkbox.isChecked():
                self.results_label.setText(self.tr('   Setting the board...'))
        else:
            QtWidgets.QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))
            self.isProcessing = False
            self.gameframe.num_to_set = None

            new_state = hex(self.gameframe.board_encoded)[2:].rjust(16, '0')
            current_state = self.board_state.text()
            self.board_state.setText(new_state)
            # 摆盘后局面变化过大时重置历史记录
            if count_match_positions(new_state, current_state) < 12:
                self.gameframe.score = 0
                self.gameframe.history = []
                self.gameframe.history.append((self.gameframe.board_encoded, self.gameframe.score))

            self.gameframe._last_values = self.gameframe.board.copy()
            self.gameframe.history.append((self.gameframe.board_encoded, self.gameframe.score))
            self.read_results()

    def read_results(self):
        if self.show_results_checkbox.isChecked():
            board = self.gameframe.board_encoded
            self.reader_thread.board_state = board
            self.reader_thread.start()
            QTimer.singleShot(1000, lambda: self.check_reader_thread_state(board))
        else:
            self.results_label.setText('')
            self.result = dict()
            self.isProcessing = False
        self.gameframe.setFocus()

    def check_reader_thread_state(self, board):
        if (self.reader_thread.board_state == board and self.reader_thread.state == '?' and
                self.show_results_checkbox.isChecked()):
            self.results_label.setText(self.tr('   Reading...'))

    def _show_results(self, result:dict, state:str):
        """ 绑定reader_thread信号 """
        if result:
            if self.reader_thread.board_state != self.gameframe.board_encoded:
                self.results_label.setText('')
                self.result = dict()
                self.read_results()
            else:
                if SingletonConfig().config['language'] == 'zh':
                    results_text = "\n".join(f"  {direction_map[key[0].lower()]}: {value}" for key, value in result.items())
                else:
                    results_text = "\n".join(f"  {key.capitalize()[0]}: {value}" for key, value in result.items())

                self.results_label.setText(results_text)
                self.result = result
                result0 = result[list(result.keys())[0]]
                if not result0 or not isinstance(result0, (float, int)):
                    self.statusbar.showMessage(self.tr("Table not found or 0 success rate"), 3000)
        else:
            self.results_label.setText(state)
            self.result = dict()

        if self.ai_state:
            steps_per_second = SingletonConfig().config['demo_speed'] / 10
            self.ai_timer.singleShot(int(1000 / steps_per_second), self.one_step)

        self.isProcessing = False

    def manual_mode_update_results(self):
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.read_results()

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

            self.book_reader.dispatch(SingletonConfig().config['filepath_map'][self.current_pattern]
                                      , self.pattern_settings[0], self.pattern_settings[1])
            self.handle_set_default()
            self.read_results()

    def spawn_state_change(self, state):
        if state == 1 and self.best_spawn_checkBox.isChecked():
            self.manual_checkBox.setChecked(False)
            self.worst_spawn_checkBox.setChecked(False)
            self.gameframe.spawn_mode = 1
        elif state == 2 and self.worst_spawn_checkBox.isChecked():
            self.manual_checkBox.setChecked(False)
            self.best_spawn_checkBox.setChecked(False)
            self.gameframe.spawn_mode = 2
        else:
            self.gameframe.spawn_mode = 0

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
                self.read_results()

            if self.gameframe.spawn_mode == 3:
                self.gameframe.moved = 1 - self.gameframe.moved

    def handle_set_board(self):
        self.textbox_reset_board()
        self.read_results()

    def textbox_reset_board(self):
        self.gameframe.board_encoded = np.uint64(int(self.board_state.text(), 16))
        self.gameframe.board = bm.decode_board(self.gameframe.board_encoded)
        self.gameframe._last_values = self.gameframe.board.copy()
        self.gameframe.update_all_frame(self.gameframe.board)

        # 重置历史记录
        self.gameframe.score = 0
        self.gameframe.history = []
        self.gameframe.history.append((self.gameframe.board_encoded, self.gameframe.score))

    def handle_rotate(self, mode):
        rotate_func = {'UD': ReverseUD, 'RL': ReverseLR, 'R90': RotateR, 'L90': RotateL}[mode]
        self.board_state.setText(hex(rotate_func(self.gameframe.board_encoded))[2:].rjust(16, '0'))
        self.handle_set_board()

    def handle_set_default(self):
        if self.ai_state:
            return
        path_list = SingletonConfig().config['filepath_map'].get(self.current_pattern, [])
        random_board = self.book_reader.get_random_state(path_list, self.current_pattern)
        self.board_state.setText(hex(random_board)[2:].rjust(16, '0'))
        self.handle_set_board()

        self.gameframe.score = 0
        self.gameframe.history = []
        self.gameframe.history.append((self.gameframe.board_encoded, self.gameframe.score))
        self.played_length = 0
        self.record_loaded = None
        self.isProcessing = False

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
        elif event.key() in (QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete):
            self.handleUndo()
            event.accept()
        elif event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            if self.ai_state:
                # 停止Demo
                self.toggle_demo()
            elif self.focusWidget() == self.board_state:
                # 焦点在 board_state
                self.handle_set_board()
                return
            else:
                # 默认执行最佳移动
                self.one_step()
            event.accept()
        else:
            super().keyPressEvent(event)  # 其他键交给父类处理

    def process_input(self, direction):
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.read_results()

    def handle_step(self):
        if self.played_length == 0 or self.record_loaded is None:
            self.one_step()
        else:
            self.playing_record_state = True
            self.replay_step()
            self.playing_record_state = False

    def one_step(self):
        if not self.isProcessing and self.result:
            self.isProcessing = True
            move = list(self.result.keys())[0]
            if self.ai_state and (self.result[move] == 0 or self.result[move] is None or self.result[move] == '?'):
                self.ai_state = False
                self.Demo.setText(self.tr('Demo'))
                self.ai_timer.stop()
                self.isProcessing = False
            if isinstance(self.result[move], (int, float)):
                self.gameframe.do_move(move.capitalize())
                self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
                if self.recording_state:
                    self.record_current_state(move)
                self.read_results()

    def toggle_demo(self):
        if not self.ai_state and not self.playing_record_state:
            self.Demo.setText(self.tr('Press Enter'))
            self.played_length = 0
            self.ai_state = True
            self.ai_timer.singleShot(20, self.one_step)
            self.statusbar.showMessage(self.tr("Press Enter to Stop"), 3000)
        else:
            self.Demo.setText(self.tr('Demo'))
            self.ai_state = False

    def handle_record(self):
        if not self.recording_state:
            self.recording_state = True
            self.record.setText(self.tr('Save'))
            self.records = np.empty(10000, dtype='uint8,uint16,uint16,uint16,uint16')
            v = np.uint64(self.gameframe.board_encoded)
            self.records[0] = (np.uint8(0), np.uint16(v & np.uint64(0xFFFF)),
                               np.uint16((v >> np.uint64(16)) & np.uint64(0xFFFF)),
                               np.uint16((v >> np.uint64(32)) & np.uint64(0xFFFF)),
                               np.uint16((v >> np.uint64(48)) & np.uint64(0xFFFF)))
            self.record_length = 1
            self.statusbar.showMessage(self.tr('Recording started'), 3000)
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
            self.record.setText(self.tr('Record Demo'))

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
                self.statusbar.showMessage(self.tr("File loaded successfully"), 1000)
                board = np.uint64((np.uint64(self.record_loaded[0][4]) << np.uint16(48)) |
                                  (np.uint64(self.record_loaded[0][3]) << np.uint16(32)) |
                                  (np.uint64(self.record_loaded[0][2]) << np.uint16(16)) |
                                  np.uint64(self.record_loaded[0][1]))
                self.board_state.setText(hex(board)[2:].rjust(16,'0'))
                self.results_label.setText('')
                self.textbox_reset_board()
                self.played_length = 1
            except Exception as e:
                self.statusbar.showMessage(self.tr("Failed to load file: ") + str(e))
                self.record_loaded = None

    def handle_play_record(self):
        if self.playing_record_state:
            self.playing_record_state = False
            self.play_record.setText(self.tr('Play Record'))
            self.replay_timer.stop()
            self.isProcessing = False
        elif self.record_loaded is None:
            return
        else:
            self.isProcessing = True
            self.playing_record_state = True
            self.play_record.setText(self.tr('Stop'))
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
            self.play_record.setText(self.tr('Play Record'))
            self.replay_timer.stop()
            self.isProcessing = False
        elif self.playing_record_state:
            current_state = self.record_loaded[self.played_length]
            move = current_state[0] & 0b11
            new_val_pos = (current_state[0] >> 2) & 0b1111
            new_val = ((current_state[0] >> 6) & 0b1) + 1
            move = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}[move]
            self.gameframe.do_move(move.capitalize(), False)
            self.gameframe.set_new_num(new_val_pos, new_val, SingletonConfig().config['do_animation'])
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

    def capture_and_copy(self):
        """捕获指定控件区域的截图并复制到剪贴板"""
        target_widget = self.gameframe.game_square  # 获取目标控件
        screenshot = target_widget.grab()
        scaled = screenshot.scaled(300, 300, Qt.KeepAspectRatio)
        clipboard = QApplication.clipboard()
        clipboard.setPixmap(scaled)
        self.statusbar.showMessage(self.tr("Screenshot saved to the clipboard"), 3000)

    def handle_page_jump(self, board_encoded, full_pattern:str):
        if not self.ai_state and not self.playing_record_state and not self.isProcessing:
            if full_pattern != '_         ' and ('' in self.pattern_settings or '?' in self.pattern_settings):
                splits = full_pattern.split('_')
                self.pattern_settings[:len(splits)] = splits
                self.set_to_new_pattern()

            self.board_state.setText(hex(board_encoded)[2:].rjust(16, '0'))
            self.handle_set_board()

            if self.windowState() & QtCore.Qt.WindowState.WindowMinimized:
                self.setWindowState(
                    self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
            self.show()
            self.activateWindow()
            self.raise_()

    def closeEvent(self, event):
        self.ai_state = False
        self.Demo.setText(self.tr('Demo'))
        try: np.array(self.gameframe.history, dtype='uint64, uint32').tofile(fr'C:\Users\Administrator\Desktop\record\0')
        except:pass
        self.gameframe.history = []
        event.accept()


class ReaderWorker(QtCore.QThread):
    """ 常驻工作线程，负责查表 """
    result_ready = QtCore.pyqtSignal(dict, str)  # 信号：结果+状态消息

    def __init__(self, reader: BookReaderDispatcher, board_state:np.uint64,
                 pattern_settings: list, current_pattern: str):
        super(ReaderWorker, self).__init__()
        self.book_reader = reader
        self.board_state = board_state
        self.pattern_settings = pattern_settings
        self.current_pattern = current_pattern
        self._32ks = pattern_32k_tiles_map.get(self.pattern_settings[0], [0])[0]
        self.state = ''

    def run(self):
        board_encoded = replace_largest_tiles(self.board_state, self._32ks)

        board = bm.decode_board(board_encoded)
        self.state = '?'
        result = self.book_reader.move_on_dic(board, self.pattern_settings[0], self.pattern_settings[1],
                                              self.current_pattern, self.pattern_settings[2])
        if not isinstance(result, dict):
            self.state = str(result)
            result = dict()
        else:
            self.state = ''
        self.result_ready.emit(result, self.state)


def count_match_positions(s1: str, s2: str) -> int:
    if len(s1) != len(s2):
        return 0
    return sum(1 for a, b in zip(s1, s2) if a == b)


def replace_largest_tiles(board_encoded, n):
    if n == 0:
        return board_encoded

    tiles = []
    for i in range(16):
        tile_value = board_encoded & 0xF
        tiles.append(tile_value)
        board_encoded >>= 4

    sorted_tiles = sorted(tiles, reverse=True)
    threshold = sorted_tiles[n - 1]
    count = 0
    for i in range(len(tiles)):
        if count < n and tiles[i] >= threshold:
            if tiles[i] > threshold or count < n:
                tiles[i] = 0xF
                count += 1
    result = 0
    for i in range(15, -1, -1):
        result <<= 4
        result |= tiles[i]

    return np.uint64(result)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = TrainWindow()
    main.show()
    sys.exit(app.exec_())
