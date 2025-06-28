import random
import sys

import numpy as np
from PyQt5 import QtWidgets

from Config import SingletonConfig
from MiniGame import MinigameFrame, MinigameWindow


class IsolatedIslandFrame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Isolated Island'):
        super().__init__(centralwidget, minigame_type)

    def gen_new_num(self, do_anim=True):
        if random.random() < 0.04 - np.sum(self.board == -3) * 0.02 + self.difficulty * 0.01:
            zero_positions = np.where(self.board == 0)
            newtile_pos = random.choice(list(zip(zero_positions[0], zero_positions[1])))
            self.newtile_pos, self.newtile = newtile_pos[0] * 4 + newtile_pos[1], -3
            self.board[*newtile_pos] = -3
        else:
            self.board, _, new_tile_pos, val = self.mover.gen_new_num(
                self.board, SingletonConfig().config['4_spawn_rate'])
            self.newtile_pos, self.newtile = new_tile_pos, val
        if do_anim:
            self.timer1.singleShot(125, lambda: self.game_square.animate_appear(
                self.newtile_pos // 4, self.newtile_pos % 4, self.newtile))

    def _set_special_frame(self, value, row, col):
        if value == -3:
            label = self.game_square.labels[row][col]
            frame = self.game_square.frames[row][col]
            label.setText('')
            frame.setStyleSheet(f"""
            QFrame#f{row * 4 + col} {{
                border-image: url(pic//portal.png) 0 0 0 0 stretch stretch;
            }}
            """)
            if (self.newtile_pos // self.cols != row or self.newtile_pos % self.cols != col
            ) or not SingletonConfig().config['do_animation']:
                frame.show()
        else:
            super()._set_special_frame(value, row, col)

# noinspection PyAttributeOutsideInit
class IsolatedIslandWindow(MinigameWindow):
    def __init__(self, minigame='Isolated Island', frame_type=IsolatedIslandFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def show_message(self):
        text = self.tr('Discovered a tile that merges with none but itself.')
        QtWidgets.QMessageBox.information(self, self.tr('Information'), text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = IsolatedIslandWindow(minigame='Isolated Island', frame_type=IsolatedIslandFrame)
    main.show()
    sys.exit(app.exec_())
