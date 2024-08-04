import sys
import random

from PyQt5 import QtCore, QtWidgets, QtGui

from MiniGame import MinigameFrame, MinigameWindow
from Config import SingletonConfig


class ColumnChaosFrame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Column Chaos'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        self.animation_group: QtCore.QAbstractAnimation | None = None
        try:
            self.count_down = SingletonConfig().config['minigame_state'][self.difficulty][minigame_type][1][0]
        except KeyError:
            self.count_down = 40 - 10 * self.difficulty
        super().__init__(centralwidget, minigame_type)

    def setup_new_game(self):
        super().setup_new_game()
        self.count_down = 40 - 10 * self.difficulty

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            ([self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [
             self.count_down])
        event.accept()

    def after_gen_num(self):
        QtWidgets.QApplication.processEvents()
        self.count_down -= 1
        if self.count_down == 0:
            self.count_down = 40 - 10 * self.difficulty
            QtCore.QTimer.singleShot(120, self.swap_lines)

    def swap_lines(self):
        col1, col2 = random.sample(range(4), 2)
        animations = []

        for row in range(4):
            frame1 = self.game_square.frames[row][col1]
            frame2 = self.game_square.frames[row][col2]

            # 创建 frame1 的动画
            anim1 = QtCore.QPropertyAnimation(frame1, b"geometry")
            anim1.setDuration(3000)
            anim1.setStartValue(frame1.geometry())
            anim1.setEndValue(frame2.geometry())
            animations.append(anim1)

            # 创建 frame2 的动画
            anim2 = QtCore.QPropertyAnimation(frame2, b"geometry")
            anim2.setDuration(3000)
            anim2.setStartValue(frame2.geometry())
            anim2.setEndValue(frame1.geometry())
            animations.append(anim2)

        # 创建一个动画组
        self.animation_group = QtCore.QParallelAnimationGroup()
        for anim in animations:
            self.animation_group.addAnimation(anim)

        # 在动画结束时交换标签的内容
        def swap_labels():
            self.board[:, [col1, col2]] = self.board[:, [col2, col1]]
            self.update_all_frame()
            self.setFocus()

        self.animation_group.finished.connect(swap_labels)
        self.animation_group.start()


# noinspection PyAttributeOutsideInit
class ColumnChaosWindow(MinigameWindow):
    def __init__(self, minigame='Column Chaos', frame_type=ColumnChaosFrame):
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
        text = f'The next chaos will occur in **{self.gameframe.count_down}** steps.'
        self.countdown_text.setMarkdown(text)

    def process_input(self, direction):
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.update_score()
        self.update_countdown_text()
        if self.gameframe.count_down == 40 - 10 * self.gameframe.difficulty:
            QtCore.QTimer.singleShot(3150, self.allow_input)  # 配合gameframe.after_gen_num中动画的延迟
        else:
            self.allow_input()

    def allow_input(self):
        self.isProcessing = False

    def show_message(self):
        text = 'Unpredictable shifts in columns are coming!'
        QtWidgets.QMessageBox.information(self, 'Information', text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = ColumnChaosWindow(minigame='Column Chaos', frame_type=ColumnChaosFrame)
    main.show()
    sys.exit(app.exec_())
    