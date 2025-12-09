import sys
from typing import List

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from Config import SingletonConfig
from MiniGame import MinigameFrame, MinigameWindow


class BlitzkriegFrame(MinigameFrame):
    gameover = QtCore.pyqtSignal()  # 游戏结束停止计时

    def __init__(self, centralwidget=None, minigame_type='Blitzkrieg'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        self.is_over = False
        super().__init__(centralwidget, minigame_type)
        try:
            self.time_left = SingletonConfig().config['minigame_state'][self.difficulty][minigame_type][1][0]
        except KeyError:
            self.setup_new_game()

    def setup_new_game(self):
        super().setup_new_game()
        self.time_left = 3
        self.is_over = False

    def check_game_passed(self):
        self.current_max_num = max(self.current_max_num, self.board.max())
        if self.is_over:
            self.max_num = max(self.max_num, self.current_max_num)
            if self.max_num > 9:
                self.is_passed = {12: 3, 11: 2, 10: 1}.get(self.max_num, 4)

            if self.current_max_num <= 9:
                super().check_game_over()
            else:
                if self.score == self.max_score:
                    level = {12: 'gold', 11: 'silver', 10: 'bronze'}.get(self.current_max_num, 'gold')
                    message = f'You achieved {self.score} score!\n You get a {level} trophy!'
                else:
                    level = {12: 'gold', 11: 'silver', 10: 'bronze'}.get(self.current_max_num, 'gold')
                    message = f'You achieved {self.score} score!\n Nice game!'
                self.show_trophy(f'pic/{level}.png', message)

    def check_game_over(self):
        if not self.is_over and self.has_possible_move():
            pass
        else:
            self.is_over = True
            self.gameover.emit()
            self.check_game_passed()

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            ([self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [
             self.time_left])
        event.accept()


class CountdownTimer(QtWidgets.QLabel):
    timeout = QtCore.pyqtSignal()

    def __init__(self, initial_time_minutes=3):
        super().__init__()
        self.setStyleSheet("font: 30pt 'Consolas'; font-weight: bold; color: black; background-color: transparent;")
        self.default_initial_time = initial_time_minutes * 60 * 1000  # 默认初始时间
        self.current_initial_time = self.default_initial_time  # 当前计时基准
        self.remaining_time = self.current_initial_time
        self.additional_time = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.elapsed_timer = QtCore.QElapsedTimer()
        self.running = False
        self.reset()

    def start(self):
        if not self.running:
            self.elapsed_timer.start()
            self.timer.start(33)  # 每多少毫秒更新一次
            self.running = True

    def stop(self):
        self.timer.stop()
        self.running = False

    def reset(self):
        self.stop()
        # 同步重置当前计时基准
        self.current_initial_time = self.default_initial_time
        self.remaining_time = self.current_initial_time
        self.additional_time = 0
        self.update_display()

    def add_time(self, minutes):
        self.additional_time += minutes * 60 * 1000
        self.update_display()

    def set_time(self, minutes):
        self.current_initial_time = minutes * 60 * 1000
        self.remaining_time = self.current_initial_time
        self.additional_time = 0  # 重置额外时间
        self.update_display()

    def export_remaining_time(self):
        return self.remaining_time / (60 * 1000)

    def update_display(self):
        if self.running:
            elapsed = self.elapsed_timer.elapsed()
            self.remaining_time = self.current_initial_time - elapsed + self.additional_time
            if self.remaining_time <= 0:
                self.setText(f"00:00.00")
                self.remaining_time, self.additional_time = 0, 0
                self.timer.stop()
                self.timeout.emit()

        minutes = int(self.remaining_time // (60 * 1000))
        seconds = int((self.remaining_time % (60 * 1000)) // 1000)
        milliseconds = int(self.remaining_time % 1000 // 10)
        self.setText(f"{minutes:02}:{seconds:02}.{milliseconds:02}")


class BlitzkriegWindow(MinigameWindow):
    def __init__(self, minigame='Blitzkrieg1', frame_type=BlitzkriegFrame):
        self.timer_started = False
        self.timer: CountdownTimer | None = None
        super().__init__(minigame=minigame, frame_type=frame_type)
        self.count_1k = np.sum(self.gameframe.board == 10)
        self.gameframe.gameover.connect(self.timer.stop)
        self.timer_anims: List[QtCore.QAbstractAnimation] = []

    def setupUi(self):
        super().setupUi()
        self.operate_frame.setMaximumSize(QtCore.QSize(16777215, 270))
        self.operate_frame.setMinimumSize(QtCore.QSize(120, 200))
        self.init_timer()

    def init_timer(self):
        # 创建并添加倒计时器控件
        self.timer = CountdownTimer(initial_time_minutes=3)
        self.timer.timeout.connect(self.on_timeout)
        self.buttons.addWidget(self.timer, 1, 0, 1, 1)
        self.timer.set_time(self.gameframe.time_left)
        self.timer.update_display()

    def on_timeout(self):
        self.gameframe.is_over = True
        self.gameframe.check_game_over()

    def process_input(self, direction):
        if self.isProcessing:
            return
        if self.gameframe.is_over:
            return
        if not self.timer_started:
            self.timer_started = True
            self.timer.start()
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.time_bonus()
        self.update_score()
        self.isProcessing = False

    def handleNewGame(self):
        self.powerup_grid.set_power_ups_counts([10,10,10])
        self.timer_started = False
        self.timer.reset()
        self.timer_anims = []
        super().handleNewGame()

    def time_bonus(self):
        self.gameframe.time_left = self.timer.export_remaining_time()
        count_1k = np.sum(self.gameframe.board == 10)
        if self.count_1k < count_1k:
            bonus = 1 if self.gameframe.difficulty == 0 else 0.75
            self.timer.add_time(bonus)
            self.show_added_time_label(int(60 * bonus))
        self.count_1k = count_1k

    def show_added_time_label(self, seconds_added):
        # 获取 self.timer 相对于 self.operate_frame 的坐标
        timer_pos = self.timer.mapTo(self.operate_frame,
                                     QtCore.QPoint(self.timer.width(), 0))
        # 在 self.operate_frame 的坐标系上创建一个新的 QLabel
        label = QtWidgets.QLabel(self.operate_frame)
        label.setStyleSheet("color: green; background-color: transparent;")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setGeometry(timer_pos.x() - 100, timer_pos.y(), 100, self.timer.height())
        # 设置字体
        font = QtGui.QFont("Consolas")
        font.setPointSize(24)
        font.setWeight(QtGui.QFont.DemiBold)
        label.setFont(font)
        label.setText(f"+{seconds_added}s")
        label.show()

        # 透明度动画
        opacity_effect = QtWidgets.QGraphicsOpacityEffect(label)
        label.setGraphicsEffect(opacity_effect)
        opacity_anim = QtCore.QPropertyAnimation(opacity_effect, b"opacity")
        opacity_anim.setDuration(800)
        opacity_anim.setStartValue(1)
        opacity_anim.setEndValue(0)
        opacity_anim.setEasingCurve(QtCore.QEasingCurve.OutQuad)
        opacity_anim.finished.connect(label.deleteLater)

        self.timer_anims.append(opacity_anim)
        opacity_anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def show_message(self):
        text = self.tr('Act quickly to earn bonus time and rack up the highest score!')
        QtWidgets.QMessageBox.information(self, self.tr('Information'), text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = BlitzkriegWindow(minigame='Blitzkrieg', frame_type=BlitzkriegFrame)
    main.gameframe.board = np.array([[0, 1, 1, 3],
                                     [0, 0, 3, 1],
                                     [4, 4, 4, 2],
                                     [5, 9, 10, 11]])
    main.show()
    sys.exit(app.exec_())
