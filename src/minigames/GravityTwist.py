import sys

import numpy as np
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
        QtCore.QTimer.singleShot(100, func)

    def gravity_move1(self):
        do_anim = SingletonConfig().config['do_animation']
        board_new, new_score = self.mover.move_board(self.board, self.current_gravity)
        if np.any(board_new != self.board):
            if do_anim[1]:
                direction = {1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}[self.current_gravity]
                self.pop_merged(self.board, direction)
            self.board = board_new
            self.score += new_score
            self.max_score = max(self.max_score, self.score)
            self.update_all_frame(self.board)
            self.check_game_passed()

    def gravity_move2(self):
        for col in range(4):
            current_col = self.board[:, col]
            non_zero_elements = current_col[current_col != 0]
            zero_elements = np.zeros(4 - len(non_zero_elements))
            new_col = np.concatenate((zero_elements, non_zero_elements))
            self.board[:, col] = new_col
        self.update_all_frame(self.board)


GravityTwist2Frame = GravityTwist1Frame


# noinspection PyAttributeOutsideInit
class GravityTwistWindow(MinigameWindow):
    def __init__(self, minigame='Gravity Twist1', frame_type=GravityTwist1Frame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def process_input(self, direction):
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.update_score()
        QtCore.QTimer.singleShot(100,self.allow_input)  # 配合gameframe.after_gen_num中重力效果的延迟

    def allow_input(self):
        self.isProcessing = False

    def show_message(self):
        text = 'The tiles are affected by the unusual gravity.'
        QtWidgets.QMessageBox.information(self, 'Information', text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = GravityTwistWindow(minigame='Gravity Twist2', frame_type=GravityTwist2Frame)
    main.show()
    sys.exit(app.exec_())
