import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QPropertyAnimation, QRect, QEasingCurve, QTimer, QSize, QPoint
from PyQt5.QtGui import QIcon

from AIPlayer import AutoplayS
from BoardMover import BoardMoverWithScore
from BookReader import BookReader
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
        self.base_font_size = int(self.width() / 14.4)
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
        self.base_font_size = int(self.width() / 14.4)
        layout = self.layout()
        if layout:
            margin = int(new_size / 32)
            layout.setContentsMargins(margin, margin, margin, margin)
            layout.setHorizontalSpacing(int(margin / 1.2))
            layout.setVerticalSpacing(int(margin / 1.2))

    def setupUi(self, num_rows=4, num_cols=4):
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

    def do_move(self, direction):
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
            self.gen_new_num(do_anim[0])
            # print(f'move: {direction} done')
            # print('')

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


# noinspection PyAttributeOutsideInit
class GameFrame(BaseBoardFrame):
    def __init__(self, centralwidget=None):
        super(GameFrame, self).__init__(centralwidget)
        self.board_encoded, self.score, _ = SingletonConfig().config['game_state']
        self.board = self.mover.decode_board(self.board_encoded)
        self.died_when_ai_state = False
        self.ai_processing = False

        # 初始化 AI 线程
        self.ai_thread = AIThread(self.board)
        self.ai_thread.updateBoard.connect(self.do_ai_move)

        if self.board_encoded == 0:
            self.setup_new_game()

    def setup_new_game(self):
        self.board_encoded = np.uint64(self.mover.gen_new_num(self.mover.gen_new_num(np.uint64(0))[0])[0])
        self.board = self.mover.decode_board(self.board_encoded)
        self.ai_thread.ai_player.board = self.board
        self.update_all_frame(self.board)
        self.score = 0
        self.history = [(self.board_encoded, self.score)]

    def ai_step(self):
        self.ai_processing = True
        self.ai_thread.start()
        self.ai_thread.ai_player.board = self.board

    def do_ai_move(self, direction):
        if direction == '':
            self.died_when_ai_state = True
        else:
            self.died_when_ai_state = False
            self.do_move(direction)
        self.ai_processing = False


class AIThread(QtCore.QThread):
    updateBoard = QtCore.pyqtSignal(str)  # 传递更新的棋盘和得分

    def __init__(self, board):
        super(AIThread, self).__init__()
        self.ai_player = AutoplayS(board)

    def run(self):
        # AI 计算步骤
        empty_slots = (self.ai_player.board == 0).sum()
        big_nums = (self.ai_player.board > 256).sum()
        if empty_slots > 10:
            self.ai_player.start_search(1)
        elif empty_slots > 4:
            self.ai_player.start_search(2)
        elif empty_slots > 2 or big_nums < 3:
            self.ai_player.start_search(3)
        else:
            depth = 4
            self.ai_player.start_search(depth)
            while self.ai_player.node < 6000 * depth * big_nums ** 2 and depth < 7:
                depth += 1
                self.ai_player.start_search(depth)
        self.updateBoard.emit(self.ai_player.best_operation)


# noinspection PyAttributeOutsideInit
class GameWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.isProcessing = False
        self.ai_state = False

        self.best_points.setText(str(SingletonConfig().config['game_state'][2]))

        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self.handleOneStep)

        self.statusbar.showMessage("All features may be slow when used for the first time. Please be patient.", 8000)

        self.update_score()

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QIcon(r"pic\2048.ico"))
        self.resize(815, 888)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self.gameframe = GameFrame(self.centralwidget)
        self.gridLayout.addWidget(self.gameframe, 2, 0, 1, 1)

        self.operate = QtWidgets.QFrame(self.centralwidget)
        self.operate.setMaximumSize(QtCore.QSize(16777215, 160))
        self.operate.setStyleSheet("QFrame{\n"
                                   "    border-color: rgb(167, 167, 167);\n"
                                   "    background-color: rgb(236, 236, 236);\n"
                                   "}")
        self.operate.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.operate.setFrameShadow(QtWidgets.QFrame.Raised)
        self.operate.setObjectName("operate")
        self.grid1 = QtWidgets.QGridLayout(self.operate)
        self.grid1.setObjectName("grid1")
        self.scores = QtWidgets.QHBoxLayout()
        self.scores.setObjectName("scores")
        self.score_frame = QtWidgets.QFrame(self.operate)
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
        self.score_text.setAlignment(QtCore.Qt.AlignCenter)
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
        self.score_points.setAlignment(QtCore.Qt.AlignCenter)
        self.score_points.setObjectName("score_points")
        self.score_group.addWidget(self.score_points, 1, 0, 1, 1)
        self.score_grid.addLayout(self.score_group, 0, 0, 1, 1)
        self.scores.addWidget(self.score_frame)
        self.best_frame = QtWidgets.QFrame(self.operate)
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
        self.best_text.setAlignment(QtCore.Qt.AlignCenter)
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
        self.best_points.setAlignment(QtCore.Qt.AlignCenter)
        self.best_points.setObjectName("best_points")
        self.best_group.addWidget(self.best_points, 1, 0, 1, 1)
        self.best_grid.addLayout(self.best_group, 0, 0, 1, 1)
        self.scores.addWidget(self.best_frame)
        self.grid1.addLayout(self.scores, 0, 0, 1, 1)

        self.buttons = QtWidgets.QGridLayout()
        self.buttons.setObjectName("buttons")
        self.one_step = QtWidgets.QPushButton(self.operate)
        self.one_step.setFocusPolicy(QtCore.Qt.NoFocus)  # 禁用按钮的键盘焦点
        self.one_step.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.one_step.setObjectName("one_step")
        self.buttons.addWidget(self.one_step, 0, 3, 1, 1)
        self.undo = QtWidgets.QPushButton(self.operate)
        self.undo.setFocusPolicy(QtCore.Qt.NoFocus)
        self.undo.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.undo.setObjectName("undo")
        self.buttons.addWidget(self.undo, 0, 2, 1, 1)
        self.new_game = QtWidgets.QPushButton(self.operate)
        self.new_game.setFocusPolicy(QtCore.Qt.NoFocus)
        self.new_game.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.new_game.setObjectName("new_game")
        self.buttons.addWidget(self.new_game, 0, 1, 1, 1)
        self.ai = QtWidgets.QPushButton(self.operate)
        self.ai.setFocusPolicy(QtCore.Qt.NoFocus)
        self.ai.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.ai.setObjectName("ai")
        self.buttons.addWidget(self.ai, 0, 0, 1, 1)
        self.grid1.addLayout(self.buttons, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.operate, 1, 0, 1, 1)
        self.setCentralWidget(self.centralwidget)

        self.one_step.clicked.connect(self.handleOneStep)
        self.undo.clicked.connect(self.handleUndo)
        self.new_game.clicked.connect(self.handleNewGame)
        self.ai.clicked.connect(self.toggleAI)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.action_2 = QtWidgets.QAction(self)
        self.action_2.setObjectName("action_2")

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)
        self.setTabOrder(self.ai, self.new_game)
        self.setTabOrder(self.new_game, self.undo)
        self.setTabOrder(self.undo, self.one_step)

    def show(self):
        self.gameframe.update_all_frame(self.gameframe.board)
        super().show()

    # noinspection PyTypeChecker
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
        self.action_2.setText(_translate("Game", "Settings"))

    def update_score(self):
        score = self.gameframe.score
        self.score_points.setText(str(score))
        if score > int(self.best_points.text()):
            self.best_points.setText(str(score))

    def keyPressEvent(self, event, ):
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
        self.update_score()
        self.isProcessing = False

    def handleOneStep(self):
        if self.isProcessing or self.gameframe.ai_processing:
            return
        self.isProcessing = True
        if best_move := self.check_reader_4(self.gameframe.board):
            self.gameframe.do_move(best_move.capitalize())
        else:
            self.gameframe.ai_step()
        self.update_score()
        if self.ai_state:
            if self.gameframe.died_when_ai_state:
                self.ai_state = False
                self.ai.setText("AI: ON")
                self.ai_timer.stop()
        self.isProcessing = False

    @staticmethod
    def check_reader_4(board):
        if (board >= 512).sum() < 4:
            return ''
        flat_arr = board.flatten()
        if (board == np.partition(flat_arr, -4)[-4]).sum() > 1:
            return ''
        max_indices = np.argpartition(flat_arr, -4)[-4:]
        flat_arr[max_indices] = 32768
        masked_board = flat_arr.reshape(4, 4)
        if 'LL_2048_0' in SingletonConfig().config['filepath_map'].keys():
            r1 = BookReader.move_on_dic(masked_board, 'LL', '2048', 'LL_2048_0')
            move = list(r1.keys())[0]
            if isinstance(r1[move], float) and r1[move] > 0.2:
                return move
        if '4431_2048' in SingletonConfig().config['filepath_map'].keys():
            r2 = BookReader.move_on_dic(masked_board, '4431', '2048', '4431_2048')
            move = list(r2.keys())[0]
            if isinstance(r2[move], float) and r2[move] > 0.2:
                return move
        return ''

    def handleUndo(self):
        self.gameframe.undo()
        self.update_score()

    def handleNewGame(self):
        self.gameframe.setup_new_game()
        self.update_score()

    def toggleAI(self):
        if not self.ai_state:
            self.ai.setText("STOP")
            self.ai_state = True
            self.ai_timer.start(20)
            if not SingletonConfig().config['filepath_map'].get('4431_2048_0', '') or \
                    not SingletonConfig().config['filepath_map'].get('LL_2048_0', ''):
                self.statusbar.showMessage("Requires 4431_2048 and LL_2048_0 for best results", 3000)
        else:
            self.ai.setText("AI: ON")
            self.ai_state = False
            self.ai_timer.stop()

    def closeEvent(self, event):
        self.ai.setText("AI: ON")
        self.ai_state = False
        if self.ai_timer.isActive():
            self.ai_timer.stop()
        SingletonConfig().config['game_state'] = [self.gameframe.board_encoded, self.gameframe.score,
                                                  int(self.best_points.text())]
        event.accept()  # 确认关闭事件


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = GameWindow()
    main.show()
    sys.exit(app.exec_())
