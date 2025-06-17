import sys
import random
from typing import Tuple, List

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from MiniGame import MinigameWindow
from Config import SingletonConfig
from Endless import EndlessFrame


class EndlessHybridFrame(EndlessFrame):
    def __init__(self, centralwidget=None, minigame_type='Endless Hybrid'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        try:
            self.bomb, self.current_level, self.bomb_type = \
                SingletonConfig().config['minigame_state'][self.difficulty][minigame_type][1]
        except KeyError:
            self.bomb: Tuple[int, int] | None = None
            self.current_level = 0
            self.bomb_type = 0

        self.levels = [
            (150000, 4, None),
            (100000, 3, 'gold'),
            (50000, 2, 'silver'),
            (20000, 1, 'bronze')
        ]

        self.pic_path_list = ['pic//bomb.png', 'pic//tilebg.png', 'pic//giftbox.png']
        self.pic_path = self.pic_path_list[self.bomb_type]
        self.bomb_gen_rate = 0.05
        self.explode_pic_list = ['pic//explode.png', 'pic//hole2.png', 'pic//hole.png']
        self.explode_pic = self.explode_pic_list[self.bomb_type]
        self.has_just_exploded = False
        self.animation_group: List[QtCore.QAbstractAnimation] = []
        self.explosion_frame: List[QtWidgets.QFrame] = []

        super(EndlessFrame, self).__init__(centralwidget=centralwidget, minigame_type=minigame_type)

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            [self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [
                self.bomb, self.current_level, self.bomb_type]
        event.accept()

    def setup_new_game(self):
        self.bomb_type = random.randint(0,2)
        self.explode_pic = self.explode_pic_list[self.bomb_type]
        self.pic_path = self.pic_path_list[self.bomb_type]
        super().setup_new_game()

    def gen_new_num(self, do_anim=True):
        if not self.bomb and random.random() < self.bomb_gen_rate:
            zero_positions = np.where(self.board == 0)
            self.bomb = random.choice(list(zip(zero_positions[0], zero_positions[1])))
            self.bomb_type = random.randint(0, 2)
            self.explode_pic = self.explode_pic_list[self.bomb_type]
            self.pic_path = self.pic_path_list[self.bomb_type]
            self.newtile_pos, self.newtile = self.bomb[0] * 4 + self.bomb[1], 0
        else:
            if self.bomb:
                self.board[*self.bomb] = 1
            if random.random() < 0.03 - (self.board == -3).sum() * 0.02 + self.difficulty * 0.015:
                zero_positions = np.where(self.board == 0)
                newtile_pos = random.choice(list(zip(zero_positions[0], zero_positions[1])))
                self.newtile_pos, self.newtile = newtile_pos[0] * 4 + newtile_pos[1], -3
                self.board[*newtile_pos] = -3
            else:
                self.board, _, new_tile_pos, val = self.mover.gen_new_num(
                    self.board, SingletonConfig().config['4_spawn_rate'])
                self.newtile_pos, self.newtile = new_tile_pos, val
            if self.bomb:
                self.board[*self.bomb] = 0

        self.update_all_frame(self.board)
        self.update_frame(self.newtile, self.newtile_pos // 4, self.newtile_pos % 4, anim=do_anim)

    def update_frame(self, value, row, col, anim=False):
        if self.board[row, col] == -3:
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

        elif self.bomb == (row, col):
            label = self.game_square.labels[row][col]
            frame = self.game_square.frames[row][col]
            label.clear()

            frame.setStyleSheet(f"""
            QFrame#f{row * 4 + col} {{
            border-image: url({self.pic_path}) 0 0 0 0 stretch stretch;;
            }}
            """)
            if self.bomb_type != 1:
                label.setText('')
                label.setStyleSheet(f"""background-color: transparent;""")
            else:
                label.setText('÷')
                label.setStyleSheet(f"""font: {self.game_square.base_font_size}pt 'Calibri'; font-weight: bold;
                color: white; background-color: transparent;""")
            if anim:
                self.game_square.animate_appear(row, col)

        else:
            label = self.game_square.labels[row][col]
            label.clear()
            super().update_frame(value, row, col, anim=False)

    def explode_effect(self, r, c):
        if self.bomb_type == 0:
            self.board[r, c] = 0
            self.trigger_explosion(r, c)
        elif self.bomb_type == 1:
            self.has_just_exploded = self.board[r, c]
            self.board[r, c] = -2
            if self.has_just_exploded > 1:
                self.board[*self.bomb] = -2
        elif self.bomb_type == 2:
            self.has_just_exploded = self.board[r, c]
            self.board[r, c] = -2

    def before_gen_num(self, direct):
        if self.has_just_exploded:
            if self.bomb_type == 1:
                positions = np.where(self.board == -2)
                positions_list = list(zip(positions[0], positions[1]))
                if self.has_just_exploded <= 1:
                    self.board[*positions_list[0]] = 0
                    self.trigger_explosion(*positions_list[0])
                else:
                    factor1 = random.randint(1, self.has_just_exploded - 1)
                    self.board[*positions_list[0]] = factor1
                    self.board[*positions_list[1]] = self.has_just_exploded - factor1
                    self.trigger_explosion(*positions_list[0])
                    self.trigger_explosion(*positions_list[1])
            elif self.bomb_type == 2:
                original_value = self.has_just_exploded
                positions = np.where(self.board == -2)
                positions_list = list(zip(positions[0], positions[1]))

                exponents = np.arange(2, 11)
                weights = 1 / (exponents ** 1.2)
                weights /= weights.sum()
                new_value = original_value
                while new_value == original_value:
                    new_value = np.random.choice(exponents, p=weights)
                self.board[*positions_list[0]] = new_value
                self.trigger_explosion(*positions_list[0])
            self.has_just_exploded = False

    def after_gen_num(self):
        pass
        #print(self.board)


class EndlessHybridWindow(MinigameWindow):
    def __init__(self, minigame='Endless Hybrid', frame_type=EndlessHybridFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def show_message(self):
        text = 'All-Stars.'
        QtWidgets.QMessageBox.information(self, 'Information', text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = EndlessHybridWindow()
    main.show()
    sys.exit(app.exec_())
