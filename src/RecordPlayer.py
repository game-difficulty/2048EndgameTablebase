import os.path
from collections import defaultdict

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QColor, QPolygonF
from numpy.typing import NDArray

import BoardMover as bm
from Gamer import BaseBoardFrame
from Config import SingletonConfig


direction_map = defaultdict(lambda: "？")

direction_map.update({
    'u': "上",
    'd': "下",
    'l': "左",
    'r': "右",
    '?': "？",
})


class ColoredMarkSlider(QtWidgets.QSlider):
    def __init__(self, data_points: NDArray, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.data_points = data_points
        self.draw_values = np.empty(0, dtype='float')
        self.points_rank = np.empty(0, dtype='int')

        self.setMinimum(0)
        self.setMaximum(len(self.data_points) - 1)
        self.setSingleStep(1)
        self.setPageStep(56)
        self.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.setTickInterval(56)
        self.valueChanged.connect(self._on_value_changed)

        # 当前值
        self._current_value = 0

        # 确定需要绘制的点和颜色信息
        if len(data_points) > 0:
            self._prepare_drawing_data()

    def _on_value_changed(self, value):
        self._current_value = value

    def _prepare_drawing_data(self):
        threshold = np.quantile(self.data_points,0.1)
        self.points_rank = np.where((self.data_points < threshold) & (self.data_points < 1))[0]
        self.draw_values = self.data_points[self.points_rank]

        min_val = np.min(self.data_points) - 0.00001
        self.point_colors = []
        normalized_vals = (self.draw_values - min_val) / (1 - min_val)
        normalized_vals = np.clip(normalized_vals, 0.0, 1.0)

        reds = np.where(normalized_vals <= 0.5, 255, 510 * (1 - normalized_vals))
        greens = np.where(normalized_vals <= 0.5, 510 * normalized_vals, 255)
        blues = np.zeros_like(reds)

        for r, g, b in zip(reds, greens, blues):
            color = QColor(int(r), int(g), int(b), 160)
            self.point_colors.append(color)

    def paintEvent(self, event):
        # 绘制默认滑块
        super().paintEvent(event)
        if len(self.draw_values) == 0:
            return

        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)

        groove_rect = self.rect()
        groove_x_start = groove_rect.x()
        groove_width = groove_rect.width()
        groove_y = groove_rect.y() + groove_rect.height() // 2

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        size = 10

        positions = groove_x_start + (
                groove_width - int(size * 0.75)) * self.points_rank / len(self.data_points) + size // 2
        for idx, x_pos in enumerate(positions):
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.point_colors[idx])

            # 三角形标志
            polygon = QPolygonF([
                QPointF(x_pos - size // 2, groove_y),  # 左下角
                QPointF(x_pos + size // 2, groove_y),  # 右下角
                QPointF(x_pos, groove_y - size)  # 上顶点
            ])
            painter.drawPolygon(polygon)


# noinspection PyAttributeOutsideInit
class ReplayFrame(BaseBoardFrame):
    def __init__(self, centralwidget=None):
        super().__init__(centralwidget)
        self.record = np.empty(0, dtype='uint64,uint8,uint32,uint32,uint32,uint32')
        self.losses = np.empty(0, dtype='float')
        self.moves = np.empty(0, dtype='uint8')
        self.goodness_of_fit = np.empty(0, dtype='float')
        self.combo = np.empty(0, dtype=np.uint16)
        self.current_step = 0

    def load_record(self, record: NDArray | str):
        if isinstance(record, str):
            if os.path.exists(record):
                self.record = np.fromfile(record, dtype='uint64,uint8,uint32,uint32,uint32,uint32')
        elif isinstance(record, np.ndarray):
            self.record = record
        else:
            self.record = np.empty(0, dtype='uint64,uint8,uint32,uint32,uint32,uint32')

        is_valid_record = self.validate_record()
        if not is_valid_record:
            self.record = np.empty(0, dtype='uint64,uint8,uint32,uint32,uint32,uint32')
            return
        else:
            self.record = self.record[:-1]
            self.analyze_record()
            self.init_frame()

    def validate_record(self):
        if not len(self.record):
            return False
        self.record[-1][0] = 0
        record_tuple = tuple(self.record[-1])
        return record_tuple == (0, 88, 666666666, 233333333, 314159265, 987654321)

    def analyze_record(self):
        if not len(self.record):
            return
        self.moves = ((self.record['f1'] >> 5) & 3).astype(np.uint8)
        arr_rates = np.vstack((self.record['f2'], self.record['f3'], self.record['f4'], self.record['f5'])).T
        optimal_sr = np.max(arr_rates, axis=1)
        player_sr = arr_rates[np.arange(len(self.moves)), self.moves]
        self.losses = player_sr / optimal_sr
        self.goodness_of_fit = np.cumprod(self.losses)

        self.combo = np.empty(len(self.losses), dtype=np.uint16)
        count = 0
        for i in range(len(self.losses)):
            if self.losses[i] == 1:
                count += 1
            else:
                count = 0
            self.combo[i] = count

    def gen_new_num(self, do_anim=True):
        # 根据record记录的情况出数
        _, self.newtile_pos, newtile = self.decode(self.record[self.current_step][1])
        self.newtile = newtile +1
        if self.current_step < len(self.record) - 1:
            self.board_encoded = self.record[self.current_step + 1][0]
        else:
            self.board_encoded = self.board_encoded | np.uint64(self.newtile) << np.uint64(4 * self.newtile_pos)
        self.board = bm.decode_board(self.board_encoded)
        self.history.append((self.board_encoded, self.score))
        if do_anim:
            self.timer1.singleShot(125, lambda: self.game_square.animate_appear(
                self.newtile_pos // 4, self.newtile_pos % 4, 2 ** self.newtile))

    @staticmethod
    def decode(x):
        a = (x >> 5) & 3
        b = (x >> 1) & 15
        c = x & 1
        return a, b, c

    def get_current_move(self):
        move = self.moves[self.current_step]
        return ('Left', 'Right', 'Up', 'Down')[move]

    def get_current_results(self):
        if self.current_step == len(self.record):
            return None
        values = list(self.record[['f2','f3','f4','f5']][self.current_step])
        values = [sr / 4e9 for sr in values]
        keys = ('left', 'right', 'up', 'down')
        results = dict(sorted(zip(keys, values), key=lambda item: item[1], reverse=True))
        return results

    def update_frame(self, value, row, col):
        """重写方法以配合不显示32k格子数字的设置"""
        super().update_frame(value, row, col)
        if value == 32768 and not SingletonConfig().config.get('dis_32k', False):
            self.game_square.labels[row][col].setText('')

    def init_frame(self):
        if 0 < len(self.record):
            self.current_step = 0
            self.board_encoded = self.record[self.current_step][0]
            self.board = bm.decode_board(self.board_encoded)
            self.update_all_frame(self.board)


# noinspection PyAttributeOutsideInit
class ReplayWindow(QtWidgets.QMainWindow):
    def __init__(self, record: NDArray | str):
        super().__init__()
        self.setupUi(record)
        self.create_menu_bar()
        self.ai_state = False
        self.ai_timer = QtCore.QTimer(self)
        self.isProcessing = False
        self.gameframe.setFocus()

        if 0 < len(self.gameframe.record):
            self.update_results()

    def setupUi(self, record):
        self.setObjectName("self")
        self.setWindowIcon(QtGui.QIcon(r"pic\2048.ico"))
        self.resize(1200, 720)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(6, 6, 6, 6)
        self.gridLayout.setHorizontalSpacing(0)
        self.gridLayout.setVerticalSpacing(6)
        self.gridLayout.setObjectName("gridLayout")

        self.gameframe = ReplayFrame(self.centralwidget)
        self.gameframe.load_record(record)
        self.gridLayout.addWidget(self.gameframe, 1, 0, 1, 1)
        self.slider = ColoredMarkSlider(self.gameframe.losses)
        self.slider.setMinimumSize(QtCore.QSize(400, 60))
        self.slider.valueChanged.connect(self.handleSliderMove)  # type: ignore
        self.gridLayout.addWidget(self.slider, 0, 0, 1, 1)

        self.operate = QtWidgets.QFrame(self.centralwidget)
        self.operate.setMaximumSize(QtCore.QSize(16777215, 1200))
        self.operate.setMinimumSize(QtCore.QSize(300, 600))
        self.operate.setStyleSheet("QFrame{\n"
                                         "    border-color: rgb(167, 167, 167);\n"
                                         "    background-color: rgb(236, 236, 236);\n"
                                         "}")
        self.operate.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.operate.setFrameShadow(QtWidgets.QFrame.Raised)
        self.operate.setObjectName("operate")
        self.grid1 = QtWidgets.QGridLayout(self.operate)
        self.grid1.setContentsMargins(6, 6, 6, 6)
        self.grid1.setSpacing(6)
        self.grid1.setObjectName("grid1")

        self.board_state = QtWidgets.QLineEdit(self.operate)
        self.board_state.setReadOnly(True)
        self.board_state.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.board_state.setObjectName("board_state")
        self.board_state.setText('0000000000000000')
        self.grid1.addWidget(self.board_state, 0, 0, 1, 1)

        self.results_label = QtWidgets.QTextBrowser(self.operate)
        self.results_label.setReadOnly(True)
        self.results_label.setMinimumSize(QtCore.QSize(500, 400))
        self.results_label.setMaximumSize(QtCore.QSize(16777215, 800))
        self.results_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                         "border-color: rgb(0, 0, 0); font: 75 18pt \"Consolas\";")
        self.results_label.setText("")
        self.results_label.setObjectName("results_label")
        self.grid1.addWidget(self.results_label, 2, 0, 1, 1)

        self.buttons = QtWidgets.QGridLayout()
        self.buttons.setObjectName("buttons")

        self.Demo = QtWidgets.QPushButton(self.operate)
        self.Demo.setMaximumSize(QtCore.QSize(180, 16777215))
        self.Demo.setObjectName("Demo")
        self.Demo.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.buttons.addWidget(self.Demo, 0, 0, 1, 1)
        self.next_point = QtWidgets.QPushButton(self.operate)
        self.next_point.setMaximumSize(QtCore.QSize(180, 16777215))
        self.next_point.setObjectName("next_point")
        self.next_point.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.buttons.addWidget(self.next_point, 1, 0, 1, 1)
        self.one_step = QtWidgets.QPushButton(self.operate)
        self.one_step.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # 禁用按钮的键盘焦点
        self.one_step.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.one_step.setObjectName("one_step")
        self.buttons.addWidget(self.one_step, 0, 1, 1, 1)
        self.f_10_step = QtWidgets.QPushButton(self.operate)
        self.f_10_step.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # 禁用按钮的键盘焦点
        self.f_10_step.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.f_10_step.setObjectName("f_10_step")
        self.buttons.addWidget(self.f_10_step, 1, 1, 1, 1)
        self.undo = QtWidgets.QPushButton(self.operate)
        self.undo.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.undo.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.undo.setObjectName("undo")
        self.buttons.addWidget(self.undo, 0, 2, 1, 1)
        self.b_10_step = QtWidgets.QPushButton(self.operate)
        self.b_10_step.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.b_10_step.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.b_10_step.setObjectName("b_10_step")
        self.buttons.addWidget(self.b_10_step, 1, 2, 1, 1)

        self.grid1.addLayout(self.buttons, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.operate, 0, 1, 2, 1)

        self.one_step.clicked.connect(lambda: self.handle_one_step(1))  # type: ignore
        self.f_10_step.clicked.connect(lambda: self.handle_one_step(10))   # type: ignore
        self.undo.clicked.connect(lambda: self.handle_undo(1))  # type: ignore
        self.b_10_step.clicked.connect(lambda: self.handle_undo(10))  # type: ignore
        self.next_point.clicked.connect(self.handleNextPoint)  # type: ignore
        self.Demo.clicked.connect(self.toggle_demo)  # type: ignore

        self.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.Demo.setText(_translate("Replay", "DEMO"))
        self.next_point.setText(_translate("Replay", "Next Inaccuracy"))
        self.undo.setText(_translate("Replay", "UNDO"))
        self.b_10_step.setText(_translate("Replay", "-10 Step"))
        self.one_step.setText(_translate("Replay", "ONESTEP"))
        self.f_10_step.setText(_translate("Replay", "+10 Step"))

    def toggle_demo(self):
        if not self.ai_state:
            self.Demo.setText(self.tr('Stop'))
            self.played_length = 0
            self.ai_state = True
            self.ai_timer.singleShot(20, lambda: self.handle_one_step(1))
        else:
            self.Demo.setText(self.tr('Demo'))
            self.ai_state = False

    def keyPressEvent(self, event, ):
        if self.isProcessing:
            return
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.handle_one_step(1)
        elif event.key() in (QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete):
            self.handle_undo(1)
        else:
            super().keyPressEvent(event)

    def handle_undo(self, times):
        if not self.isProcessing:
            self.isProcessing = True
            do_anim = SingletonConfig().config['do_animation']
            if times > 1:
                SingletonConfig().config['do_animation'] = False
            for _ in range(times):
                if self.gameframe.current_step > 0:
                    self.gameframe.current_step -= 1
                    self.gameframe.undo()
            self.update_results()
            self.syncSliderPosition()
            self.isProcessing = False
            SingletonConfig().config['do_animation'] = do_anim

    def handle_one_step(self, times):
        if not self.isProcessing:
            self.isProcessing = True

            do_anim = SingletonConfig().config['do_animation']
            if times > 1:
                SingletonConfig().config['do_animation'] = False

            for _ in range(times):
                if self.gameframe.current_step < len(self.gameframe.record):
                    self.gameframe.board_encoded = self.gameframe.record[self.gameframe.current_step][0]
                    move = self.gameframe.get_current_move()
                    self.gameframe.do_move(move.capitalize())
                    self.gameframe.current_step += 1
            self.update_results()
            self.syncSliderPosition()
            self.isProcessing = False
            SingletonConfig().config['do_animation'] = do_anim

    def handleNextPoint(self):
        if len(self.slider.points_rank) == 0 or self.gameframe.current_step >= self.slider.points_rank[-1]:
            return
        self.gameframe.current_step = self.slider.points_rank[
            np.searchsorted(self.slider.points_rank, self.gameframe.current_step, side='right')]
        self.syncSliderPosition()
        self.handleSliderMove(self.gameframe.current_step)

    def update_results(self):
        result = self.gameframe.get_current_results()
        if result:
            best_move = list(result.keys())[0]
            move = self.gameframe.get_current_move()
            loss = self.gameframe.losses[self.gameframe.current_step]
            gof = self.gameframe.goodness_of_fit[self.gameframe.current_step]
            combo = self.gameframe.combo[self.gameframe.current_step]

            lang = SingletonConfig().config['language']
            is_zh = lang == 'zh'

            if is_zh:
                result_lines = [f"  {direction_map[key[0].lower()]}: {value}" for key, value in result.items()]
                result_text = "\n\n&nbsp;&nbsp;\n\n&nbsp;&nbsp;" + "\n\n&nbsp;&nbsp;".join(result_lines)
            else:
                result_lines = [f"  {key.capitalize()[0]}: {value}" for key, value in result.items()]
                result_text = "\n\n&nbsp;&nbsp;\n\n&nbsp;&nbsp;" + "\n\n&nbsp;&nbsp;".join(result_lines)

            if is_zh:
                move_dir = direction_map[move[0].lower()]
                best_dir = direction_map[best_move[0].lower()]
            else:
                move_dir = move.capitalize()
                best_dir = best_move.capitalize()

            separator = '\n\n--------------------------------------------------'
            is_match = move.lower() == best_move.lower()

            if is_zh:
                if is_match:
                    action_text = f"\n\n你走的是   **{move_dir}**\n\n最优解正是 **{best_dir}**"
                    stats_text = f"\n\nCombo: {combo}x, 吻合度: {gof:.4f}"
                else:
                    action_text = f"\n\n你走的是   **{move_dir}**\n\n但最优解是 **{best_dir}**"
                    stats_text = f'\n\n单步损失: {1 - loss:.4f}, 吻合度: {gof:.4f}'
            else:
                if is_match:
                    action_text = f"\n\nYou pressed          **{move_dir}**.\n\nAnd the best move is **{best_dir}**"
                    stats_text = (f"\n\nCombo: {combo}x, "
                                  f'goodness of fit: {gof:.4f}')
                else:
                    action_text = f"\n\nYou pressed          **{move_dir}**.\n\nBut the best move is **{best_dir}**"
                    stats_text = (f'\n\none-step loss: {1 - loss:.4f}, '
                                  f'goodness of fit: {gof:.4f}')

            results_text = f"{result_text}{separator}{action_text}{stats_text}"

            self.results_label.setMarkdown(results_text)
        else:
            self.results_label.setMarkdown('')

        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))

        if self.ai_state:
            steps_per_second = SingletonConfig().config['demo_speed'] / 10
            self.ai_timer.singleShot(int(1000 / steps_per_second), lambda: self.handle_one_step(1))

    def handleSliderMove(self, target_step):
        # # 如果AI正在运行则停止
        # if self.ai_state:
        #     self.ai_state = False
        max_steps = len(self.gameframe.record) - 1

        self.gameframe.current_step = min(max(target_step, 0), max_steps)
        self.gameframe.board_encoded = self.gameframe.record[self.gameframe.current_step][0]
        self.gameframe.board = bm.decode_board(self.gameframe.board_encoded)
        values = bm.decode_board(self.gameframe.board_encoded)
        self.gameframe.update_all_frame(values)
        self.update_results()
        self.board_state.setText(hex(self.gameframe.board_encoded)[2:].rjust(16, '0'))

    def syncSliderPosition(self):
        if self.gameframe.current_step >= len(self.gameframe.record):
            if self.ai_state:
                self.toggle_demo()
            return
        with QtCore.QSignalBlocker(self.slider):
            self.slider.setValue(self.gameframe.current_step)
            self.gameframe.board_encoded = self.gameframe.record[self.gameframe.current_step][0]
            self.gameframe.board = bm.decode_board(self.gameframe.board_encoded)

    def reset_record(self, record):
        self.gameframe.load_record(record)
        if 0 < len(self.gameframe.record):
            self.slider.deleteLater()
            self.slider = ColoredMarkSlider(self.gameframe.losses)
            self.slider.setMinimumSize(QtCore.QSize(400, 60))
            self.slider.valueChanged.connect(self.handleSliderMove)  # type: ignore
            self.gridLayout.addWidget(self.slider, 0, 0, 1, 1)
            self.syncSliderPosition()
            self.update_results()
        else:
            self.statusbar.showMessage(self.tr('Recording file corrupted'), 3000)

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu(self.tr('&File'))
        select_action = QtWidgets.QAction(self.tr('&Select Recording File...'), self)
        select_action.setShortcut('Ctrl+N')  # 设置快捷键
        select_action.setStatusTip(self.tr('Load new recording file'))  # 状态栏提示
        select_action.triggered.connect(self.select_rec_file)
        file_menu.addAction(select_action)

    def select_rec_file(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Recording File"),  # 对话框标题
            "",  # 初始目录
            "Recording Files (*.rpl);;All Files (*)",  # 文件过滤器
            options=options
        )

        # 如果用户选择了文件
        if file_path:
            # 调用reset_record方法处理新文件
            self.reset_record(file_path)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ReplayWindow(r"C:\Users\Administrator\Desktop\L3_512_0_20250702220058_0.2884_rec.txt")
    window.show()
    sys.exit(app.exec_())
