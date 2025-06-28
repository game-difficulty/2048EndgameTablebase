import sys

import numpy as np
from PyQt5 import QtWidgets

from MiniGame import MinigameFrame, MinigameWindow
from Config import SingletonConfig
from AIPlayer import EvilGen


class TrickyTilesFrame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Tricky Tiles'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        self.evil_gen_prob = 0.3 + np.random.rand() / 12 + self.difficulty * 0.1
        super().__init__(centralwidget, minigame_type)
        self.evil_gen = EvilGen(self.original_board())

    def before_gen_num(self, _):
        QtWidgets.QApplication.processEvents()

    def setup_new_game(self):
        super().setup_new_game()
        self.evil_gen_prob = 0.3 + np.random.rand() / 12 + self.difficulty * 0.1

    def original_board(self):
        board: np.ndarray = np.zeros_like(self.board, dtype=np.int32)
        for i in range(self.rows):
            for j in range(self.cols):
                board[i][j] = 2 ** self.board[i][j] if self.board[i][j] > 0 else 0
        return board

    def gen_new_num(self, do_anim=True):
        if np.random.rand() > self.evil_gen_prob:
            self.board, _, new_tile_pos, val = self.mover.gen_new_num(
                self.board, SingletonConfig().config['4_spawn_rate'])
        else:
            board = self.original_board()
            self.evil_gen.reset_board(board)
            depth = 5 if np.sum(board == 0) < 6 else 4
            board_encoded, new_tile_pos, val = self.evil_gen.gen_new_num(depth)
            self.board[new_tile_pos // self.cols][new_tile_pos % self.cols] = val
        self.newtile_pos, self.newtile = new_tile_pos, val
        if do_anim:
            self.timer1.singleShot(125, lambda: self.game_square.animate_appear(new_tile_pos // self.cols, new_tile_pos % self.cols, val))


# noinspection PyAttributeOutsideInit
class TrickyTilesWindow(MinigameWindow):
    def __init__(self, minigame='Tricky Tiles', frame_type=TrickyTilesFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def show_message(self):
        text = self.tr('Brace yourself! New numbers may appear in the most challenging spots!')
        QtWidgets.QMessageBox.information(self, self.tr('Information'), text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = TrickyTilesWindow(minigame='Tricky Tiles', frame_type=TrickyTilesFrame)
    main.show()
    sys.exit(app.exec_())
