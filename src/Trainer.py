import os.path
import sys
import time
from collections import defaultdict

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

import BoardMover as bm
from BookReader import BookReader
from Calculator import ReverseUD, ReverseLR, RotateR, RotateL
from Config import SingletonConfig, category_info
from Gamer import BaseBoardFrame

direction_map = defaultdict(lambda: "？")

direction_map.update({
    'u': "上",
    'd': "下",
    'l': "左",
    'r': "右",
    '?': "？",
})


class MatplotlibPlotWidget(QtWidgets.QWidget):
    # 定义方向颜色映射
    COLOR_MAP = {
        'down': 'blue',
        'right': 'red',
        'left': 'green',
        'up': 'purple'
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        # 创建Matplotlib图形和画布
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # 设置极小的边距
        self.figure.subplots_adjust(
            left=0.1,  # 左侧边距 (默认0.125)
            right=0.97,  # 右侧边距 (默认0.9)
            bottom=0.1,  # 底部边距 (默认0.1)
            top=0.95,  # 顶部边距 (默认0.9)
            wspace=0.0,  # 子图水平间距
            hspace=0.0  # 子图垂直间距
        )
        # 设置布局
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)


    def plot_data(self, data_dict):
        """
        参数:
            data_dict: 包含方向数据的字典，格式为 {'down': array, 'right': array, ...}
        """
        # 清除之前的绘图
        self.ax.clear()

        # 获取所有非空数组的长度
        valid_data = [arr for arr in data_dict.values() if isinstance(arr, np.ndarray)]
        if not valid_data:
            return

        counts = len(valid_data[0])

        # 创建横坐标 (0到1之间等分)
        x = np.arange(0, 1, 1 / counts)

        # 绘制每条折线
        for direction, data in data_dict.items():
            if isinstance(data, np.ndarray):
                color = self.COLOR_MAP.get(direction, 'black')
                self.ax.plot(
                    x, data,
                    color=color,
                    linewidth=0.8,
                    label=direction.capitalize()
                )

        # 设置网格（虚线灰色）
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.25, color='gray')

        # 添加图例
        self.ax.legend(loc='upper center', frameon=False, shadow=False)

        # 设置坐标轴范围
        self.ax.set_xlim(0, 1)

        # 自动调整Y轴范围，留出10%的边距
        y_min = min(arr.min() for arr in valid_data)
        y_max = max(arr.max() for arr in valid_data)
        margin = max((y_max - y_min) * 0.1, 0.05)
        self.ax.set_ylim(y_min - margin, y_max + margin)

        # 设置刻度
        self.ax.xaxis.set_major_locator(MultipleLocator(0.1))
        self.ax.yaxis.set_major_locator(MultipleLocator(round(margin * 2, 2)))

        # 设置坐标轴刻度标签字体大小
        self.ax.tick_params(axis='both', which='major', labelsize=10)

        # 绘制图形
        self.canvas.draw()


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

    def gen_new_num(self, do_anim=True):
        super().gen_new_num(do_anim)

    def mousePressEvent(self, event):
        if self.num_to_set is None:
            self.setFocus()
            return

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

    def update_frame(self, value, row, col):
        """重写方法以配合不显示32k格子数字的设置"""
        super().update_frame(value, row, col)
        if value == 32768 and not self.dis32k:
            self.game_square.labels[row][col].setText('')

    def set_new_num(self, new_tile_pos, val, do_anim=True):
        self.board_encoded |= np.uint64(val) << np.uint64(4 * (15 - new_tile_pos))
        self.board = bm.decode_board(self.board_encoded)
        self.history.append(self.board_encoded)
        self.newtile_pos, self.newtile = new_tile_pos, val
        if do_anim:
            self.timer1.singleShot(125, lambda: self.game_square.animate_appear(new_tile_pos // 4, new_tile_pos % 4, 2 ** val))


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

        self.book_reader: BookReader = BookReader()
        self.reader_thread = ReaderWorker(self.book_reader, np.uint64(0), self.pattern_settings, self.current_pattern)
        self.reader_thread.result_ready.connect(self._show_results)

        self.statusbar.showMessage(self.tr("All features may be slow when used for the first time. Please be patient."), 8000)

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QIcon(r'pic\2048.ico'))
        self.resize(1440, 720)
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

        self.operate = QtWidgets.QFrame(self.centralwidget)
        self.operate.setMaximumSize(QtCore.QSize(800, 800))
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

        self.step = QtWidgets.QPushButton(self.operate)
        self.step.setMaximumSize(QtCore.QSize(180, 16777215))
        self.step.setObjectName("step")
        self.step.clicked.connect(self.handle_step)  # type: ignore
        self.gridLayout_bts.addWidget(self.step, 0, 2, 1, 1)

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

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.results_text = QtWidgets.QLabel(self.operate)
        self.results_text.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.results_text.setObjectName("results_text")
        self.horizontalLayout.addWidget(self.results_text)

        self.gridLayout_operate.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.plot_widget = MatplotlibPlotWidget(self.operate)
        self.plot_widget.setMinimumSize(QtCore.QSize(320, 240))
        self.plot_widget.setMaximumSize(QtCore.QSize(16777215, 800))
        self.plot_widget.setObjectName("plot_widget")
        self.gridLayout_operate.addWidget(self.plot_widget, 2, 0, 1, 1)

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

        self.R90.setText(_translate("Train", "R90"))
        self.undo.setText(_translate("Train", "UNDO"))
        self.UD.setText(_translate("Train", "UD"))
        self.L90.setText(_translate("Train", "L90"))

        self.default.setText(_translate("Train", "Default"))
        self.results_text.setText(_translate("Train", "RESULTS:"))
        self.filepath_text.setText(_translate("Train", "FILEPATH:"))
        self.set_filepath_bt.setText(_translate("Train", "SET..."))
        self.dis32k_checkBox.setText(_translate("Train", "Display numbers for 32k tile"))
        self.menu_ptn.setTitle(_translate("Train", "Pattern"))
        self.menu_pos.setTitle(_translate("Train", "Position"))
        self.menu_tgt.setTitle(_translate("Train", "Target"))

        self.Demo.setText(_translate("Train", "DEMO"))
        self.step.setText(_translate("Train", "ONESTEP"))

    def dis32k_state_change(self):
        SingletonConfig().config['dis_32k'] = self.dis32k_checkBox.isChecked()
        self.gameframe.dis32k = SingletonConfig().config['dis_32k']
        self.gameframe.update_all_frame(self.gameframe.board)
        self.gameframe.setFocus()

    def menu_selected(self, i):
        self.pattern_settings[i] = self.sender().text()
        self.pattern_text.setText('_'.join(self.pattern_settings[:2]))
        if '' not in self.pattern_settings:
            if self.pattern_settings[0] in ['444', 'LL', 'L3']:
                if self.pattern_settings[0] != 'L3' and self.pattern_settings[2] == '2':
                    self.pattern_settings[2] = '0'
                self.current_pattern = '_'.join(self.pattern_settings)
            else:
                self.pattern_settings[2] = '0'
                self.current_pattern = '_'.join(self.pattern_settings[:2])

            path_list = SingletonConfig().config['filepath_map'].get(self.current_pattern, [])
            if path_list:
                self.filepath.setText(path_list[-1])
            else:
                self.filepath.setText(' ')
            self.pattern_text.setText(self.current_pattern)

            self.reader_thread.current_pattern = self.current_pattern
            self.reader_thread.pattern_settings = self.pattern_settings

            self.handle_set_default()
            self.read_results()

    def tiles_bt_on_click(self):
        sender = self.sender()
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
            self.gameframe._last_values = self.gameframe.board.copy()
            self.gameframe.history.append((self.gameframe.board_encoded))
            self.read_results()

    def read_results(self):
        board = np.uint64(int('0x' + self.board_state.text().rjust(16, '0'), 16))
        self.reader_thread.board_state = board
        self.reader_thread.start()
        self.gameframe.setFocus()

    def _show_results(self, result:dict, state:str):
        """ 绑定reader_thread信号 """
        if result:
            self.plot_widget.plot_data(result)
            self.result = result
            if self.ai_state:
                steps_per_second = SingletonConfig().config['demo_speed'] / 10
                self.ai_timer.singleShot(int(1000 / steps_per_second), self.one_step)
        else:
            self.plot_widget.ax.clear()
            self.result = dict()

        self.isProcessing = False

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

            self.handle_set_default()
            self.read_results()

    def handleUndo(self):
        self.gameframe.undo()
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.read_results()
                
    def handle_set_board(self):
        self.textbox_reset_board()
        self.read_results()

    def textbox_reset_board(self):
        self.gameframe.board_encoded = np.uint64(int(self.board_state.text(), 16))
        self.gameframe.board = bm.decode_board(self.gameframe.board_encoded)
        self.gameframe._last_values = self.gameframe.board.copy()
        self.gameframe.update_all_frame(self.gameframe.board)
        self.gameframe.history.append(self.gameframe.board_encoded)

    def handle_rotate(self, mode):
        rotate_func = {'UD': ReverseUD, 'RL': ReverseLR, 'R90': RotateR, 'L90': RotateL}[mode]
        self.board_state.setText(hex(rotate_func(self.gameframe.board_encoded))[2:].rjust(16, '0'))
        self.handle_set_board()

    def handle_set_default(self):
        path_list = SingletonConfig().config['filepath_map'].get(self.current_pattern, [])
        random_board = self.book_reader.get_random_state(path_list, self.current_pattern)
        self.board_state.setText(hex(random_board)[2:].rjust(16, '0'))
        self.handle_set_board()

        self.gameframe.history = []
        self.gameframe.history.append(self.gameframe.board_encoded)
        self.played_length = 0
        self.isProcessing = False

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
        elif event.key() in (QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete):
            self.handleUndo()
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
        else:
            super().keyPressEvent(event)  # 其他键交给父类处理

    def process_input(self, direction):
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.read_results()

    def handle_step(self):
        if self.played_length == 0:
            self.one_step()

    def one_step(self):
        if not self.isProcessing and self.result:
            self.isProcessing = True
            results = {k: v for k, v in self.result.items() if isinstance(v, np.ndarray)}
            if not results:
                if self.ai_state:
                    self.ai_state = False
                    self.Demo.setText(self.tr('Demo'))
                    self.ai_timer.stop()
                    self.isProcessing = False
                return

            points = len(results[list(results.keys())[0]])

            # 找到最小差值的索引
            sr4_arr = np.arange(0, 1, 1 / points, dtype=np.float32)
            sr4 = SingletonConfig().config['4_spawn_rate']
            abs_diff = np.abs(sr4_arr - sr4)
            index = np.argmin(abs_diff)

            max_value = 0
            max_key = None

            for key, array in results.items():
                value = array[index]
                if value > max_value:
                    max_value = value
                    max_key = key

            self.gameframe.do_move(max_key.capitalize())
            self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))

            self.read_results()

    def toggle_demo(self):
        if not self.ai_state:
            self.Demo.setText(self.tr('Press Enter'))
            self.played_length = 0
            self.ai_state = True
            self.ai_timer.singleShot(20, self.one_step)
            self.statusbar.showMessage(self.tr("Press Enter to Stop"), 3000)
        else:
            self.Demo.setText(self.tr('Demo'))
            self.ai_state = False

    def capture_and_copy(self):
        """捕获指定控件区域的截图并复制到剪贴板"""
        target_widget = self.gameframe.game_square  # 获取目标控件
        screenshot = target_widget.grab()
        scaled = screenshot.scaled(300, 300, Qt.KeepAspectRatio)
        clipboard = QApplication.clipboard()
        clipboard.setPixmap(scaled)
        self.statusbar.showMessage(self.tr("Screenshot saved to the clipboard"), 3000)

    def closeEvent(self, event):
        self.gameframe.history = []
        event.accept()


class ReaderWorker(QtCore.QThread):
    """ 常驻工作线程，负责查表 """
    result_ready = QtCore.pyqtSignal(dict, str)  # 信号：结果+状态消息

    def __init__(self, reader: BookReader, board_state:np.uint64,
                 pattern_settings: list, current_pattern: str):
        super(ReaderWorker, self).__init__()
        self.book_reader = reader
        self.board_state = board_state
        self.pattern_settings = pattern_settings
        self.current_pattern = current_pattern

    def run(self):
        board = bm.decode_board(self.board_state)
        result = self.book_reader.move_on_dic(board, self.pattern_settings[0], self.pattern_settings[1],
                                              self.current_pattern, self.pattern_settings[2])
        state = ''
        if not isinstance(result, dict):
            state = str(result)
            result = dict()
        self.result_ready.emit(result, state)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = TrainWindow()
    main.show()
    sys.exit(app.exec_())
