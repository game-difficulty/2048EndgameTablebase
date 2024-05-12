import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon

from BookReader import BookReader
from Calculator import ReverseUD, ReverseLR, RotateR, RotateL
from Gamer import BaseBoardFrame
from Config import SingletonConfig


class TrainFrame(BaseBoardFrame):
    dis32k = SingletonConfig().config.get('dis_32k', False)

    def __init__(self, centralwidget=None):
        super(TrainFrame, self).__init__(centralwidget)
        self.num_to_set = None

    def mousePressEvent(self, event):
        if self.num_to_set is None:
            return
        self.score = 0
        local_pos = self.game_square.mapFromParent(event.pos())
        for i, row in enumerate(self.game_square.frames):
            for j, frame in enumerate(row):
                if frame.geometry().contains(local_pos):
                    self.update_frame(self.num_to_set, i, j, False)
                    self.board[i][j] = self.num_to_set
                    self.board_encoded = np.uint64(self.mover.encode_board(self.board))
                    return

    def update_frame(self, value, row, col, anim=False):
        """重写方法以配合不显示32k格子数字的设置"""
        super().update_frame(value, row, col, anim)
        if value == 32768 and not self.dis32k:
            self.game_square.labels[row][col].setText('')


# noinspection PyAttributeOutsideInit
class TrainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.current_pattern = ''
        self.pattern_settings = ['', '', '0']
        self.isProcessing = False
        self.result = dict()
        self.ai_state = False

        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self.one_step)

        self.statusbar.showMessage("All features may be slow when used for the first time. Please be patient.", 8000)

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QIcon(r'pic\2048.ico'))
        self.resize(988, 1200)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self.gameframe = TrainFrame(self.centralwidget)
        self.gameframe.setFocusPolicy(Qt.StrongFocus)
        self.gridLayout.addWidget(self.gameframe, 2, 0, 1, 1)

        self.operate = QtWidgets.QFrame(self.centralwidget)
        self.operate.setMaximumSize(QtCore.QSize(16777215, 360))
        self.operate.setStyleSheet("QFrame{\n"
                                   "    border-color: rgb(167, 167, 167);\n"
                                   "    background-color: rgb(236, 236, 236);\n"
                                   "}")
        self.operate.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.operate.setFrameShadow(QtWidgets.QFrame.Raised)
        self.operate.setObjectName("operate")
        self.gridLayout_upper = QtWidgets.QGridLayout(self.operate)
        self.gridLayout_upper.setContentsMargins(9, 9, 9, 9)
        self.gridLayout_upper.setObjectName("gridLayout_upper")
        self.left_Layout = QtWidgets.QVBoxLayout()
        self.left_Layout.setSpacing(12)
        self.left_Layout.setObjectName("left_Layout")
        self.pattern_text = QtWidgets.QLabel(self.operate)
        self.pattern_text.setStyleSheet("font: 2400 32pt \"Cambria\";")
        self.pattern_text.setObjectName("pattern_text")
        self.left_Layout.addWidget(self.pattern_text)
        self.setboard_Layout = QtWidgets.QHBoxLayout()
        self.setboard_Layout.setObjectName("setboard_Layout")
        self.board_state = QtWidgets.QLineEdit(self.operate)
        self.board_state.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.board_state.setObjectName("board_state")
        self.board_state.setText('0000000000000000')
        self.setboard_Layout.addWidget(self.board_state)
        self.set_board_bt = QtWidgets.QPushButton(self.operate)
        self.set_board_bt.setMaximumSize(QtCore.QSize(50, 16777215))
        self.set_board_bt.setObjectName("set_board_bt")
        self.set_board_bt.clicked.connect(self.handle_set_board)
        self.setboard_Layout.addWidget(self.set_board_bt)
        self.left_Layout.addLayout(self.setboard_Layout)

        self.tiles_frame = QtWidgets.QFrame(self.operate)
        self.tiles_frame.setMaximumSize(QtCore.QSize(480, 120))
        self.tiles_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tiles_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tiles_frame.setObjectName("tiles_frame")
        self.gridLayout_tiles = QtWidgets.QGridLayout(self.tiles_frame)
        self.gridLayout_tiles.setContentsMargins(1, 1, 1, 1)
        self.gridLayout_tiles.setHorizontalSpacing(2)
        self.gridLayout_tiles.setVerticalSpacing(2)
        self.gridLayout_tiles.setObjectName("gridLayout_2")

        self.tile_buttons = []
        button_numbers = ['0', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1k', '2k', '4k', '8k', '16k',
                          '32k']
        for index, num in enumerate(button_numbers):
            button = QtWidgets.QPushButton(self.tiles_frame)
            button.setObjectName(f"t{num}")
            button.setText(f"{num}")
            button.setCheckable(True)
            button.clicked.connect(self.tiles_bt_on_click)
            self.gridLayout_tiles.addWidget(button, index // 8, index % 8, 1, 1)
            self.tile_buttons.append(button)
        self.left_Layout.addWidget(self.tiles_frame)

        self.gridLayout_bts = QtWidgets.QGridLayout()
        self.gridLayout_bts.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_bts.setObjectName("gridLayout_bts")
        self.RL = QtWidgets.QPushButton(self.operate)
        self.RL.setMaximumSize(QtCore.QSize(120, 16777215))
        self.RL.setObjectName("RL")
        self.RL.clicked.connect(lambda: self.handle_rotate('RL'))
        self.gridLayout_bts.addWidget(self.RL, 1, 1, 1, 1)
        self.Demo = QtWidgets.QPushButton(self.operate)
        self.Demo.setMaximumSize(QtCore.QSize(120, 16777215))
        self.Demo.setObjectName("L90")
        self.Demo.clicked.connect(self.toggle_demo)
        self.gridLayout_bts.addWidget(self.Demo, 0, 1, 1, 1)
        self.R90 = QtWidgets.QPushButton(self.operate)
        self.R90.setMaximumSize(QtCore.QSize(120, 16777215))
        self.R90.setObjectName("R90")
        self.R90.clicked.connect(lambda: self.handle_rotate('R90'))
        self.gridLayout_bts.addWidget(self.R90, 1, 2, 1, 1)
        self.undo = QtWidgets.QPushButton(self.operate)
        self.undo.setMaximumSize(QtCore.QSize(120, 16777215))
        self.undo.setObjectName("undo")
        self.undo.clicked.connect(self.handleUndo)
        self.gridLayout_bts.addWidget(self.undo, 0, 0, 1, 1)
        self.UD = QtWidgets.QPushButton(self.operate)
        self.UD.setMaximumSize(QtCore.QSize(120, 16777215))
        self.UD.setObjectName("UD")
        self.UD.clicked.connect(lambda: self.handle_rotate('UD'))
        self.gridLayout_bts.addWidget(self.UD, 1, 0, 1, 1)
        self.step = QtWidgets.QPushButton(self.operate)
        self.step.setMaximumSize(QtCore.QSize(120, 16777215))
        self.step.setObjectName("step")
        self.step.clicked.connect(self.one_step)
        self.gridLayout_bts.addWidget(self.step, 0, 2, 1, 1)
        self.default = QtWidgets.QPushButton(self.operate)
        self.default.setMaximumSize(QtCore.QSize(120, 16777215))
        self.default.setObjectName("default")
        self.default.clicked.connect(self.handle_set_default)
        self.gridLayout_bts.addWidget(self.default, 0, 3, 1, 1)
        self.L90 = QtWidgets.QPushButton(self.operate)
        self.L90.setMaximumSize(QtCore.QSize(120, 16777215))
        self.L90.setObjectName("L90")
        self.L90.clicked.connect(lambda: self.handle_rotate('L90'))
        self.gridLayout_bts.addWidget(self.L90, 1, 3, 1, 1)
        self.left_Layout.addLayout(self.gridLayout_bts)

        self.gridLayout_upper.addLayout(self.left_Layout, 0, 0, 1, 1)
        self.right_Layout = QtWidgets.QVBoxLayout()
        self.right_Layout.setSpacing(12)
        self.right_Layout.setObjectName("right_Layout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.results_text = QtWidgets.QLabel(self.operate)
        self.results_text.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.results_text.setObjectName("results_text")
        self.horizontalLayout.addWidget(self.results_text)
        self.show_results_checkbox = QtWidgets.QCheckBox(self.operate)
        self.show_results_checkbox.setObjectName("show_results_checkbox")
        self.show_results_checkbox.stateChanged.connect(self.show_results)
        self.horizontalLayout.addWidget(self.show_results_checkbox)
        self.right_Layout.addLayout(self.horizontalLayout)
        self.results_label = QtWidgets.QLabel(self.operate)
        self.results_label.setMinimumSize(QtCore.QSize(0, 200))
        self.results_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                         "border-color: rgb(0, 0, 0); font: 75 16pt \"Consolas\";")
        self.results_label.setText("")
        self.results_label.setObjectName("results_label")
        self.right_Layout.addWidget(self.results_label)
        self.filepath_text = QtWidgets.QLabel(self.operate)
        self.filepath_text.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.filepath_text.setObjectName("filepath_text")
        self.right_Layout.addWidget(self.filepath_text)
        self.setpath_Layout = QtWidgets.QHBoxLayout()
        self.setpath_Layout.setObjectName("setpath_Layout")
        self.filepath = QtWidgets.QLineEdit(self.operate)
        self.filepath.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.filepath.setObjectName("filepath")
        self.setpath_Layout.addWidget(self.filepath)
        self.set_filepath_bt = QtWidgets.QPushButton(self.operate)
        self.set_filepath_bt.setMaximumSize(QtCore.QSize(100, 16777215))
        self.set_filepath_bt.setObjectName("set_filepath_bt")
        self.set_filepath_bt.clicked.connect(self.filepath_changed)
        self.setpath_Layout.addWidget(self.set_filepath_bt)
        self.right_Layout.addLayout(self.setpath_Layout)
        self.gridLayout_upper.addLayout(self.right_Layout, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.operate, 0, 0, 1, 1)
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
        for ptn in ['444', '4431', 'LL', 'L3', 'free8', 'free9', 'free10', "4441", "4432"]:
            m = QtWidgets.QAction(ptn, self)
            m.triggered.connect(lambda: self.menu_selected(0))
            self.menu_ptn.addAction(m)
        self.menubar.addAction(self.menu_ptn.menuAction())
        self.menu_tgt = QtWidgets.QMenu(self.menubar)
        self.menu_tgt.setObjectName("menuMENU")
        for ptn in ["128", "256", "512", "1024", "2048", "4096", "8192"]:
            m = QtWidgets.QAction(ptn, self)
            m.triggered.connect(lambda: self.menu_selected(1))
            self.menu_tgt.addAction(m)
        self.menubar.addAction(self.menu_tgt.menuAction())
        self.menu_pos = QtWidgets.QMenu(self.menubar)
        self.menu_pos.setObjectName("menuMENU")
        for ptn in ["0", "1", "2"]:
            m = QtWidgets.QAction(ptn, self)
            m.triggered.connect(lambda: self.menu_selected(2))
            self.menu_pos.addAction(m)
        self.menubar.addAction(self.menu_pos.menuAction())

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
        self.step.setText(_translate("Train", "ONESTEP"))
        self.default.setText(_translate("Train", "Default"))
        self.results_text.setText(_translate("Train", "RESULTS:"))
        self.show_results_checkbox.setText(_translate("Train", "Show"))
        self.filepath_text.setText(_translate("Train", "FILEPATH:"))
        self.set_filepath_bt.setText(_translate("Train", "SET..."))
        self.menu_ptn.setTitle(_translate("Train", "Pattern"))
        self.menu_pos.setTitle(_translate("Train", "Position"))
        self.menu_tgt.setTitle(_translate("Train", "Target"))

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
            self.filepath.setText(SingletonConfig().config['filepath_map'].get(self.current_pattern, ''))
            self.show_results()
            self.pattern_text.setText(self.current_pattern)

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
        else:
            self.isProcessing = False
            self.gameframe.num_to_set = None
            self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
            self.show_results()

    def show_results(self):
        if self.show_results_checkbox.isChecked():
            board = self.gameframe.mover.decode_board(np.uint64(int('0x' + self.board_state.text().ljust(16, '0'), 16)))
            result = BookReader.move_on_dic(board, self.pattern_settings[0], self.pattern_settings[1],
                                            self.current_pattern, self.pattern_settings[2])
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

    def filepath_changed(self):
        options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filepath = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if filepath:
            self.filepath.setText(filepath)
            SingletonConfig().config['filepath_map'][self.current_pattern] = filepath
            SingletonConfig().save_config(SingletonConfig().config)
            self.show_results()

    def handleUndo(self):
        self.gameframe.undo()
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.show_results()

    def handle_set_board(self):
        self.gameframe.board_encoded = np.uint64(int(self.board_state.text(), 16))
        self.gameframe.board = self.gameframe.mover.decode_board(self.gameframe.board_encoded)
        self.gameframe.update_all_frame(self.gameframe.board)
        self.show_results()

    def handle_rotate(self, mode):
        rotate_func = {'UD': ReverseUD, 'RL': ReverseLR, 'R90': RotateR, 'L90': RotateL}[mode]
        self.board_state.setText(hex(rotate_func(self.gameframe.board_encoded))[2:].rjust(16, '0'))
        self.handle_set_board()

    def handle_set_default(self):
        random_board = BookReader.get_random_state(self.filepath.text(), self.current_pattern)
        self.board_state.setText(hex(random_board)[2:].rjust(16, '0'))
        self.handle_set_board()

    def keyPressEvent(self, event):
        if self.isProcessing:
            return
        if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_W):
            self.process_input('Up')
        elif event.key() in (QtCore.Qt.Key_Down, QtCore.Qt.Key_S):
            self.process_input('Down')
        elif event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
            self.process_input('Left')
        elif event.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
            self.process_input('Right')
        else:
            super().keyPressEvent(event)  # 其他键交给父类处理

    def process_input(self, direction):
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))
        self.show_results()
        self.isProcessing = False

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
                self.isProcessing = False

    def toggle_demo(self):
        if not self.ai_state:
            self.ai_state = True
            self.Demo.setText('STOP')
            steps_per_second = SingletonConfig().config['demo_speed'] / 10
            self.ai_timer.start(int(1000 / steps_per_second))
        else:
            self.ai_state = False
            self.Demo.setText('Demo')
            self.ai_timer.stop()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = TrainWindow()
    main.show()
    sys.exit(app.exec_())
