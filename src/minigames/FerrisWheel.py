import sys

from PyQt5 import QtCore, QtWidgets, QtGui

from MiniGame import MinigameFrame, MinigameWindow
from Config import SingletonConfig


class FerrisWheelFrame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Ferris Wheel'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        self.animation_group: QtCore.QAbstractAnimation | None = None
        try:
            self.count_down = SingletonConfig().config['minigame_state'][self.difficulty][minigame_type][1][0]
        except KeyError:
            self.count_down = 40 - self.difficulty * 10
        super().__init__(centralwidget, minigame_type)

    def setup_new_game(self):
        super().setup_new_game()
        self.count_down = 40 - self.difficulty * 10

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            ([self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [
                self.count_down])
        event.accept()

    def after_gen_num(self):
        QtWidgets.QApplication.processEvents()
        self.count_down -= 1
        if self.count_down == 0:
            self.count_down = 40 - self.difficulty * 10
            QtCore.QTimer.singleShot(120, self.spin)

    def spin(self):
        # 定义外围 12 个方块的当前和目标坐标
        positions = [
            (0, 0), (0, 1), (0, 2), (0, 3),  # 顶行
            (1, 3), (2, 3),  # 右列
            (3, 3), (3, 2), (3, 1), (3, 0),  # 底行
            (2, 0), (1, 0)  # 左列
        ]

        # 计算目标位置
        target_positions = positions[1:] + positions[:1]  # 顺时针旋转一格
        animations = []

        # 创建动画
        for (start_row, start_col), (end_row, end_col) in zip(positions, target_positions):
            frame_start = self.game_square.frames[start_row][start_col]
            frame_end = self.game_square.frames[end_row][end_col]

            anim = QtCore.QPropertyAnimation(frame_start, b"geometry")
            anim.setDuration(2000)
            anim.setStartValue(frame_start.geometry())
            anim.setEndValue(frame_end.geometry())
            animations.append(anim)

        self.animation_group = QtCore.QParallelAnimationGroup()
        for anim in animations:
            self.animation_group.addAnimation(anim)

        # 在动画结束时更新棋盘并交换标签的内容
        def swap_labels():
            values = [self.board[row, col] for row, col in positions]
            values = values[-1:] + values[:-1]
            for (row, col), value in zip(positions, values):
                self.board[row, col] = value
            self.update_all_frame()

        self.animation_group.finished.connect(swap_labels)
        self.animation_group.start()


# noinspection PyAttributeOutsideInit
class FerrisWheelWindow(MinigameWindow):
    def __init__(self, minigame='Ferris Wheel', frame_type=FerrisWheelFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def setupUi(self):
        super().setupUi()
        self.operate_frame.setMaximumSize(QtCore.QSize(16777215, 280))
        self.operate_frame.setMinimumSize(QtCore.QSize(120, 200))
        self.countdown_text = QtWidgets.QTextEdit(self.centralwidget)
        self.countdown_text.setObjectName("countdown_text")
        self.countdown_text.setReadOnly(True)
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        self.countdown_text.setFont(font)
        self.buttons.addWidget(self.countdown_text, 1, 0, 1, 1)
        font_metrics = QtGui.QFontMetrics(font)
        line_height = font_metrics.lineSpacing()
        min_height = int(line_height * 2.5)
        self.countdown_text.setFixedHeight(min_height)
        self.countdown_text.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # 禁用按钮的键盘焦点
        self.countdown_text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.countdown_text.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.update_countdown_text()

    def handleNewGame(self):
        super().handleNewGame()
        self.update_countdown_text()

    def update_countdown_text(self):
        text = f'The next rotation will occur in **{self.gameframe.count_down}** steps.'
        self.countdown_text.setMarkdown(text)

    def process_input(self, direction):
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.update_score()
        self.update_countdown_text()
        if self.gameframe.count_down == 40 - self.gameframe.difficulty * 10:
            QtCore.QTimer.singleShot(2150, self.allow_input)  # 配合gameframe.after_gen_num中动画的延迟
        else:
            self.allow_input()

    def allow_input(self):
        self.isProcessing = False

    def show_message(self):
        text = 'The Earth revolves around the Sun.'
        QtWidgets.QMessageBox.information(self, 'Information', text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = FerrisWheelWindow(minigame='Ferris Wheel', frame_type=FerrisWheelFrame)
    main.show()
    sys.exit(app.exec_())
