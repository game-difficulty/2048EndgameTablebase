import sys

import numpy as np
from numpy.typing import NDArray
from PyQt5 import QtCore, QtWidgets

from MiniGame import MinigameFrame, MinigameWindow
from Config import SingletonConfig


class GravityTwist1Frame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Gravity Twist1'):
        self.current_gravity = 4
        super().__init__(centralwidget, minigame_type)

    def after_gen_num(self):
        QtWidgets.QApplication.processEvents()
        func = self.gravity_move2 if self.minigame == 'Gravity Twist1' else self.gravity_move1
        delay = 360 if SingletonConfig().config['do_animation'] else 90
        QtCore.QTimer.singleShot(delay, func)

    def gravity_move1(self):
        do_anim = SingletonConfig().config['do_animation']
        board_new, new_score = self.mover.move_board(self.board, self.current_gravity)
        if np.any(board_new != self.board):
            if do_anim:
                direction = {1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}[self.current_gravity]
                self.slide_tiles(self.board, direction)
                self.timer3.singleShot(100, lambda: self.update_all_frame(board_new))
            else:
                self.update_all_frame(board_new)
            self._last_values = self.board.copy()
            self.board = board_new
            self.score += new_score
            self.max_score = max(self.max_score, self.score)
            self.check_game_passed()

    def gravity_move2(self):
        for col in range(4):
            current_col = self.board[:, col]
            non_zero_elements = current_col[current_col != 0]
            zero_elements = np.zeros(4 - len(non_zero_elements))
            new_col = np.concatenate((zero_elements, non_zero_elements))
            self.board[:, col] = new_col

        do_anim = SingletonConfig().config['do_animation']
        if do_anim:
            direction = {1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}[self.current_gravity]
            slide_distances = count_zeros_below(self._last_values)
            for row in range(self.game_square.rows):
                for col in range(self.game_square.cols):
                    self.game_square.animate_slide(row, col, direction, slide_distances[row][col])
            self.timer3.singleShot(100, lambda: self.update_all_frame(self.board))
        else:
            self.update_all_frame(self.board)
        self._last_values = self.board.copy()


def count_zeros_below(matrix: NDArray):
    zero_mask = (matrix == 0)
    result = np.zeros_like(matrix)
    for i in range(matrix.shape[0] - 2, -1, -1):
        result[i] = result[i + 1] + zero_mask[i + 1]
    result[zero_mask] = 0
    return result


GravityTwist2Frame = GravityTwist1Frame


# noinspection PyAttributeOutsideInit
class GravityTwistWindow(MinigameWindow):
    def __init__(self, minigame='Gravity Twist1', frame_type=GravityTwist1Frame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def process_input(self, direction):
        if self.isProcessing:
            return
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.update_score()
        delay = 400 if SingletonConfig().config['do_animation'] else 100
        QtCore.QTimer.singleShot(delay,self.allow_input)  # 配合gameframe.after_gen_num中重力效果的延迟

    def allow_input(self):
        self.isProcessing = False

    def show_message(self):
        text = self.tr('The tiles are affected by the unusual gravity.')
        QtWidgets.QMessageBox.information(self, self.tr('Information'), text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = GravityTwistWindow(minigame='Gravity Twist2', frame_type=GravityTwist2Frame)
    main.show()
    sys.exit(app.exec_())
