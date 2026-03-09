import os
import sys
import time
from datetime import datetime

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QEasingCurve, QTimer, QSize
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

import BoardMover as bm
from AIPlayer import Dispatcher, CoreAILogic
from BoardFrame import BaseBoardFrame
from Config import SingletonConfig, ColorManager
from ai_and_sort import ai_core


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

        # verse回放编码
        self.index_to_char = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜø£Ø×ƒá'
        self.verse_code = 'replay_'
        self.last_direc = 1

        # 初始化 AI 线程
        self.ai_thread = AIThread(self.board_encoded, np.zeros(16, dtype=np.uint32))
        self.ai_thread.updateBoard.connect(self.do_ai_move)

        # 困难模式
        self.evil_gen = ai_core.EvilGen(self.board_encoded)
        self.difficulty = 0

        if self.board_encoded == 0:
            self.setup_new_game()

    def setup_new_game(self):
        self.has_65k = False
        self.verse_code = 'replay_'
        self.board_encoded, _, new_tile_pos, val = bm.s_gen_new_num(np.uint64(0),
                                                                    SingletonConfig().config['4_spawn_rate'])
        self.verse_code += self.decode_to_character(val, self.last_direc, 15 - new_tile_pos)
        self.board_encoded, _, new_tile_pos, val = bm.s_gen_new_num(self.board_encoded,
                                                                    SingletonConfig().config['4_spawn_rate'])
        self.verse_code += self.decode_to_character(val, self.last_direc, 15 - new_tile_pos)
        self.board = bm.decode_board(self.board_encoded)

        self.evil_gen.reset_board(self.board_encoded)
        self.update_all_frame(self.board)
        self.score = 0
        self.history = [(self.board_encoded, self.score)]

    def ai_step(self, counts):
        if self.ai_thread.isRunning():
            return

        self.ai_processing = True
        self.ai_thread.prepare(
            board=self.board,
            board_encoded=self.board_encoded,
            counts=counts,
            spawn_rate4=SingletonConfig().config['4_spawn_rate']
        )
        self.ai_thread.start()

    def do_ai_move(self, direction):
        if not direction:
            self.died_when_ai_state = True
            self.AIMoveDone.emit(True)
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
            self.evil_gen.reset_board(self.board_encoded)
            self.board_encoded, new_tile_pos, val = self.get_hardest_tile()
        self.board_encoded = np.uint64(self.board_encoded)
        self.board = bm.decode_board(self.board_encoded)
        self.history.append((self.board_encoded, self.score))
        self.newtile_pos, self.newtile = new_tile_pos, val
        self.verse_code += self.decode_to_character(self.newtile, self.last_direc, 15 - self.newtile_pos)
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
        self.last_direc = {'Left': 1, 'Right': 2, 'Up': 3, 'Down': 4}[direction.capitalize()]
        super().do_move(direction, do_gen)

    def update_all_frame(self, values):
        """ 支持65536 """
        if self.has_65k:
            values = values.copy()
            values.flat[next((i for i, x in enumerate(values.flat) if x == 32768), None)] = 65536
        super().update_all_frame(values)

    def decode_to_character(self, replay_tile, replay_move, replay_position, total_space=15):
        right_side = total_space - replay_position
        low4 = ((right_side & 0b11) << 2) | ((right_side >> 2) & 0b11)
        index_i = (([3, 2, 4, 1].index(replay_move)) << 5) | ((replay_tile - 1) << 4) | low4
        return self.index_to_char[index_i]

    def undo(self):
        if len(self.history) > 1:
            self.verse_code = self.verse_code[:-1]
        super().undo()

    def get_hardest_tile(self, depth=5):
        result = self.evil_gen.gen_new_num(depth)
        return result


class AIThread(QtCore.QThread):
    updateBoard = QtCore.pyqtSignal(str)
    move_map = {1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}

    def __init__(self, board_encoded, counts):
        super(AIThread, self).__init__()
        self.ai_player = ai_core.AIPlayer(board_encoded)
        self.ai_player.max_threads = 8 if os.cpu_count() >= 16 else 4
        self.logic = CoreAILogic()
        self.board = None
        self.board_encoded = board_encoded
        self.counts = counts

    def prepare(self, board, board_encoded, counts, spawn_rate4):
        """
        统一的数据注入接口，避免在外部直接操作成员变量
        """
        self.board = board
        self.board_encoded = board_encoded
        self.counts = counts
        self.ai_player.board = board_encoded
        self.ai_player.spawn_rate4 = spawn_rate4

    def run(self):
        time_start = time.time()
        best_move_code = self.logic.calculate_step(
            self.ai_player,
            self.board,
            self.counts
        )
        time_end = time.time()
        time_cost = time_end - time_start
        if self.logic.last_move == 'search':
            print(f"Depth: {self.logic.last_depth}, Time: {time_cost:.6f}")

        move_str = self.move_map.get(best_move_code, '')
        self.updateBoard.emit(move_str)


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
        color_mgr = ColorManager()
        self.operate_frame.setStyleSheet("QFrame{\n"
                                         f"    border-color: {color_mgr.get_css_color(8)};\n"
                                         f"    background-color: {color_mgr.get_css_color(3)};\n"
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
                                       f"background-color: {color_mgr.get_css_color(2)};")
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
                                      f"background-color: {color_mgr.get_css_color(2)};")
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
                                            f"    border-color: {color_mgr.get_css_color(8)};\n"
                                            f"    background-color: {color_mgr.get_css_color(3)};\n"
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
        self.difficulty_slider.setTickInterval(10)
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

        self.speed_frame = QtWidgets.QFrame(self.centralwidget)
        self.speed_frame.setMaximumSize(QtCore.QSize(16777215, 60))
        self.speed_frame.setMinimumSize(QtCore.QSize(120, 45))
        self.speed_frame.setStyleSheet("QFrame{\n"
                                       f"    border-color: {color_mgr.get_css_color(8)};\n"
                                       f"    background-color: {color_mgr.get_css_color(3)};\n"
                                       "}")
        self.speed_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.speed_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.speed_frame.setObjectName("speed_frame")

        self.speed_layout = QtWidgets.QGridLayout(self.speed_frame)
        self.speed_layout.setObjectName("speed_layout")

        self.speed_text = QtWidgets.QLabel("AI 思考速度")
        self.speed_text.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.speed_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.speed_text.setObjectName("speed_text")
        self.speed_layout.addWidget(self.speed_text, 0, 0, 1, 3)
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.centralwidget)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.speed_slider.setTickInterval(20)
        self.speed_slider.setObjectName("speed_slider")
        self.speed_layout.addWidget(self.speed_slider, 0, 3, 1, 8)
        self.empty_label = QtWidgets.QLabel()
        self.speed_layout.addWidget(self.empty_label, 0, 11, 1, 1)
        self.speed_slider.valueChanged.connect(self.speed_changed)
        self.gridLayout.addWidget(self.speed_frame, 5, 0, 1, 1)

        self.setboard_frame = QtWidgets.QFrame(self.centralwidget)
        self.setboard_frame.setMaximumSize(QtCore.QSize(16777215, 30))
        self.setboard_frame.setMinimumSize(QtCore.QSize(120, 20))
        self.setboard_frame.setStyleSheet("QFrame{\n"
                                            f"    border-color: {color_mgr.get_css_color(8)};\n"
                                            f"    background-color: {color_mgr.get_css_color(3)};\n"
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

        self.shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut.activated.connect(self.save_verse_replay)  # type: ignore

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
        self.ai.setText("AI: ON")  # 暂不翻译
        self.difficulty_text.setText(_translate("Game", "Difficulty"))
        self.speed_text.setText(_translate("Game", "AI Speed"))
        self.set_board_bt.setText(_translate("Game", "SET"))

    def difficulty_changed(self):
        self.gameframe.difficulty = self.difficulty_slider.value() / 100
        self.gameframe.setFocus()

    def speed_changed(self, value):
        """ 将滑块值 [0, 100, 200] 映射为 [10, 1, 0.1] """
        exponent = (100 - value) / 100.0
        ratio = 10 ** exponent
        self.gameframe.ai_thread.logic.time_limit_ratio = ratio

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
        # 1. 获取 score_points 左上角相对于 self 的准确位置
        # 使用 QPoint(0, 0) 映射，避免了 geometry().topLeft() 带来的二次偏移
        local_pos = self.score_points.mapTo(self, QtCore.QPoint(0, 0))
        score_rect = self.score_points.geometry()

        # 2. 创建 Label
        score_animation_label = QtWidgets.QLabel(f"+{increment}", self)
        score_animation_label.setStyleSheet("font-weight: 750; font-size: 12pt; color: #3EB489;")
        score_animation_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # 3. 解决点击遮挡问题：设置鼠标穿透
        score_animation_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # 4. 设置初始位置（不再计算标题栏高度 decoration_width）
        # 稍微向上偏移 10 像素以美观
        score_animation_label.setGeometry(local_pos.x(), local_pos.y() - 10,
                                          score_rect.width(), score_rect.height())
        score_animation_label.show()

        # 动画逻辑
        start_pos = score_animation_label.pos()
        end_pos = QtCore.QPoint(start_pos.x(), start_pos.y() - 60)

        # 位置动画
        pos_anim = QtCore.QPropertyAnimation(score_animation_label, b"pos")
        pos_anim.setDuration(400)
        pos_anim.setStartValue(start_pos)
        pos_anim.setEndValue(end_pos)
        pos_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutQuad)  # 修改为 OutQuad 更有漂浮感

        # 透明度动画
        opacity_effect = QtWidgets.QGraphicsOpacityEffect(score_animation_label)
        score_animation_label.setGraphicsEffect(opacity_effect)
        opacity_anim = QtCore.QPropertyAnimation(opacity_effect, b"opacity")
        opacity_anim.setDuration(400)
        opacity_anim.setStartValue(1.0)
        opacity_anim.setEndValue(0.0)

        # 动画组管理
        anim_group = QtCore.QParallelAnimationGroup()
        anim_group.addAnimation(pos_anim)
        anim_group.addAnimation(opacity_anim)

        # 自动清理：动画完成后删除 Label 并从列表中移除自己
        anim_group.finished.connect(score_animation_label.deleteLater)
        anim_group.finished.connect(
            lambda: self.score_anims.remove(anim_group) if anim_group in self.score_anims else None)

        self.score_anims.append(anim_group)
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
        elif event.key() in (QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete):
            if self.ai_state:
                self.toggleAI()
            else:
                self.handleUndo()
            event.accept()
        elif event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            if self.ai_state:
                self.toggleAI()
            else:
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
            self.statusbar.showMessage(self.tr("Using " + current_table), 3000)
            self.last_table = current_table

        if best_move == 'AI':
            self.gameframe.ai_step(self.ai_dispatcher.counts)
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
                self.ai_timer.singleShot(1, self.handleOneStep)
        # print(self.ai_dispatcher.last_operator)
        self.save_game_state()
        self.isProcessing, self.gameframe.ai_processing = False, False

    def ai_move_done(self, is_done):
        if is_done and self.ai_state:
            self.ai_timer.singleShot(1, self.handleOneStep)

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
            self.ai_timer.singleShot(1, self.handleOneStep)
            self.statusbar.showMessage(self.tr(
                "Run larger tables for better performance."), 5000)
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
        self.gameframe.ai_thread.logic.last_depth = 4

        # 重置历史记录
        self.gameframe.score = 0
        self.update_score()
        self.gameframe.history = []
        self.gameframe.history.append((self.gameframe.board_encoded, self.gameframe.score))
        self.gameframe.died_when_ai_state = False

    def save_verse_replay(self):
        current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{self.gameframe.score}_{current_date}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.gameframe.verse_code)

        self.statusBar().showMessage(f"replay has been saved to: {filename}", 3000)

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
