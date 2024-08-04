import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets

from MiniGame import MinigameFrame, MinigameWindow
from Config import SingletonConfig
from Calculator import find_merge_positions


class MysteryMerge1Frame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Mystery Merge1'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        try:
            self.peek_count, self.masked_ = \
                SingletonConfig().config['minigame_state'][self.difficulty][minigame_type][1]
        except KeyError:
            self.peek_count = 0
            self.masked_ = np.zeros((4, 4), dtype=bool)
        super().__init__(centralwidget, minigame_type)

    def setup_new_game(self):
        super().setup_new_game()
        self.peek_count = 0

    def update_frame(self, value, row, col, anim=False):
        if self.board[row][col] == 0 or self.newtile_pos == col + row * 4 or self.masked_[row, col]:
            super().update_frame(value, row, col, anim=False)
        else:
            label = self.game_square.labels[row][col]
            frame = self.game_square.frames[row][col]
            label.setText('?')
            fontsize = self.game_square.base_font_size
            frame.setStyleSheet(f"""
            QFrame#f{row * 4 + col} {{
                border-image: url(pic//tilebg.png) 0 0 0 0 stretch stretch;
            }}
            """)
            label.setStyleSheet(f"""font: {fontsize}pt 'Calibri'; font-weight: bold; color: white;
            background-color: transparent;""")

            if anim:
                self.game_square.animate_appear(row, col)

    def show_all_frame(self):
        for i in range(self.game_square.rows):
            for j in range(self.game_square.cols):
                super().update_frame(self.board[i][j], i, j, anim=False)

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            ([self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [
                self.peek_count, self.masked_])
        event.accept()

    def check_game_over(self):
        if self.has_possible_move():
            print(self.peek_count)
            pass
        else:
            self.show_all_frame()
            QtCore.QTimer().singleShot(500, self.game_over)

    def pop_merged(self, board, direction):
        merged_pos = find_merge_positions(board, direction)
        self.masked_ = merged_pos
        for row in range(self.game_square.rows):
            for col in range(self.game_square.cols):
                if merged_pos[row][col] == 1:
                    self.game_square.animate_pop(row, col)


class MysteryMerge2Frame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Mystery Merge2'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        try:
            self.peek_count, self.masked_ = \
                SingletonConfig().config['minigame_state'][self.difficulty][minigame_type][1]
        except KeyError:
            self.peek_count = 0
            self.masked_ = np.zeros((4, 4), dtype=bool)
        super().__init__(centralwidget, minigame_type)

    def setup_new_game(self):
        self.masked_ = np.zeros((4, 4), dtype=bool)
        super().setup_new_game()
        self.peek_count = 0
        self.after_gen_num()

    def update_frame(self, value, row, col, anim=False):
        if not self.masked_[row][col] and self.newtile_pos != col + row * 4:
            super().update_frame(value, row, col, anim=False)
        else:
            label = self.game_square.labels[row][col]
            frame = self.game_square.frames[row][col]
            label.setText('?')
            fontsize = self.game_square.base_font_size
            frame.setStyleSheet(f"""
            QFrame#f{row * 4 + col} {{
                border-image: url(pic//tilebg.png) 0 0 0 0 stretch stretch;
            }}
            """)
            label.setStyleSheet(f"""font: {fontsize}pt 'Calibri'; font-weight: bold; color: white;
            background-color: transparent;""")

            if anim:
                self.game_square.animate_appear(row, col)

    def show_all_frame(self):
        for i in range(self.game_square.rows):
            for j in range(self.game_square.cols):
                super().update_frame(self.board[i][j], i, j, anim=False)

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            ([self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [
                self.peek_count, self.masked_])
        event.accept()

    @staticmethod
    def update_mask_line(line, mask, reverse=False):
        if reverse:
            line = line[::-1]
            mask = mask[::-1]

        non_zero = [i for i in line if i != 0]  # 去掉所有的0
        non_zero_mask = [m for i, m in zip(line, mask) if i != 0]  # 去掉所有数字对应的mask
        # merged = []
        merged_mask = []
        skip = False

        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # merged_value = 2 * non_zero[i]
                # merged.append(merged_value)
                merged_mask.append(0)  # 合并后不保留遮挡
                skip = True
            else:
                # merged.append(non_zero[i])
                merged_mask.append(non_zero_mask[i])

        # 补齐剩下的 0
        # merged += [0] * (len(line) - len(merged))
        merged_mask += [0] * (len(mask) - len(merged_mask))

        if reverse:
            # merged = merged[::-1]
            merged_mask = merged_mask[::-1]

        return np.array(merged_mask, dtype=bool)

    def before_move(self, direct):
        if direct == 1:  # Move left
            for i in range(4):
                self.masked_[i, :] = self.update_mask_line(self.board[i, :], self.masked_[i, :])
        elif direct == 2:  # Move right
            for i in range(4):
                self.masked_[i, :] = self.update_mask_line(self.board[i, :], self.masked_[i, :], reverse=True)
        elif direct == 3:  # Move up
            for j in range(4):
                self.masked_[:, j] = self.update_mask_line(self.board[:, j], self.masked_[:, j])
        elif direct == 4:  # Move down
            for j in range(4):
                self.masked_[:, j] = self.update_mask_line(self.board[:, j], self.masked_[:, j], reverse=True)

    def after_gen_num(self):
        self.masked_[self.newtile_pos // 4][self.newtile_pos % 4] = True

    def check_game_over(self):
        if self.has_possible_move():
            pass
        else:
            self.show_all_frame()
            QtCore.QTimer().singleShot(500, self.game_over)


# noinspection PyAttributeOutsideInit
class MysteryMergeWindow(MinigameWindow):
    def __init__(self, minigame='Mystery Merge1', frame_type=MysteryMerge1Frame):
        super().__init__(minigame=minigame, frame_type=frame_type)
        self.reset_peek_button_enabled()

    def setupUi(self):
        super().setupUi()
        self.peek = QtWidgets.QPushButton(self.operate_frame)
        self.peek.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # 禁用按钮的键盘焦点
        self.peek.setStyleSheet("font: 750 12pt \"Cambria\";")
        self.peek.setObjectName("peek")
        self.buttons.addWidget(self.peek, 0, 1, 1, 1)
        self.peek.pressed.connect(self.handlePeek)  # type: ignore
        self.peek.released.connect(self.handleUnpeek)  # type: ignore

    def reset_peek_button_enabled(self):
        if self.gameframe.difficulty == 0:
            self.peek.setEnabled(True)
        else:
            if self.gameframe.peek_count > self.gameframe.score // 10000:
                self.peek.setEnabled(False)
            else:
                self.peek.setEnabled(True)

    # noinspection PyTypeChecker
    def retranslateUi(self):
        super().retranslateUi()
        _translate = QtCore.QCoreApplication.translate
        self.peek.setText(_translate("Minigame", "peek"))

    def handlePeek(self):
        if self.gameframe.has_possible_move():
            self.gameframe.show_all_frame()
            self.gameframe.peek_count += 1

    def handleUnpeek(self):
        if self.gameframe.has_possible_move():
            self.gameframe.update_all_frame(self.gameframe.board)
            self.reset_peek_button_enabled()

    def handleNewGame(self):
        super().handleNewGame()
        self.reset_peek_button_enabled()

    def update_score(self):
        super().update_score()
        self.reset_peek_button_enabled()

    def show_message(self):
        if self.minigame == 'Mystery Merge1':
            text = 'Show only empty spaces and newly generated tiles.'
        elif self.minigame == 'Mystery Merge2':
            text = 'Not sure what newly generated tiles are unless a merge occurs.'
        else:
            text = ''
        QtWidgets.QMessageBox.information(self, 'Information', text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = MysteryMergeWindow(minigame='Mystery Merge1', frame_type=MysteryMerge1Frame)
    main.gameframe.board = np.array([[11, 10, 9, 8],
                                     [4, 5, 6, 7],
                                     [3, 2, 1, 1],
                                     [1, 1, 1, 1]])
    main.gameframe.masked_ = np.array([[1, 1, 1, 1],
                                       [1, 1, 1, 1],
                                       [1, 1, 1, 1],
                                       [1, 1, 1, 1]])
    main.show()
    main.gameframe.update_all_frame()
    sys.exit(app.exec_())
