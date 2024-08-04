import sys
import random

from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np

from MiniGame import MinigameFrame, MinigameWindow
from Config import SingletonConfig


class IsolatedIslandFrame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Isolated Island'):
        super().__init__(centralwidget, minigame_type)

    def gen_new_num(self, do_anim=True):
        if random.random() < 0.04 - (self.board == -3).sum() * 0.02 + self.difficulty * 0.01:
            zero_positions = np.where(self.board == 0)
            newtile_pos = random.choice(list(zip(zero_positions[0], zero_positions[1])))
            self.newtile_pos, self.newtile = newtile_pos[0] * 4 + newtile_pos[1], -3
            self.board[*newtile_pos] = -3
        else:
            self.board, _, new_tile_pos, val = self.mover.gen_new_num(
                self.board, SingletonConfig().config['4_spawn_rate'])
            self.newtile_pos, self.newtile = new_tile_pos, val
        self.update_all_frame(self.board)
        self.update_frame(self.newtile, self.newtile_pos // 4, self.newtile_pos % 4, anim=do_anim)

    def update_frame(self, value, row, col, anim=False):
        if self.board[row, col] != -3:
            label = self.game_square.labels[row][col]
            label.clear()
            super().update_frame(value, row, col, anim=False)
        else:
            label = self.game_square.labels[row][col]
            frame = self.game_square.frames[row][col]
            label.setText('')
            color = 'rgba(229, 229, 229, 1)'
            frame.setStyleSheet(f"background-color: {color};")

            label.setPixmap(QtGui.QPixmap('pic//portal.png').scaled(frame.width(), frame.height(
            ), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"""background-color: transparent;""")
            if anim:
                self.game_square.animate_appear(row, col)


# noinspection PyAttributeOutsideInit
class IsolatedIslandWindow(MinigameWindow):
    def __init__(self, minigame='Isolated Island', frame_type=IsolatedIslandFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def show_message(self):
        text = 'Discovered a tile that merges with none but itself.'
        QtWidgets.QMessageBox.information(self, 'Information', text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = IsolatedIslandWindow(minigame='Isolated Island', frame_type=IsolatedIslandFrame)
    main.show()
    sys.exit(app.exec_())
