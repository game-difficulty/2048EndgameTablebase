import sys

import numpy as np
from PyQt5 import QtWidgets, QtGui

from MiniGame import MinigameFrame, MinigameWindow


pattern = {
    'Design Master1': np.array([[0, 0, 0, 0],
                               [0, 2, 1, 0],
                               [0, 1, 2, 0],
                               [0, 0, 0, 0]]),
    'Design Master2': np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]),
    'Design Master3': np.array([[0, 0, 1, 0],
                               [0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1]]),
    'Design Master4': np.array([[0, 0, 0, 0],
                               [0, 0.0078125, 1, 0.5],
                               [0, 0.015625, 2, 0.25],
                               [0, 0.03125, 0.0625, 0.125]]),
}


class DesignMaster1Frame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Design Master1'):
        self.pattern = pattern[minigame_type]
        super().__init__(centralwidget, minigame_type)
        self.current_max_num = 7

    def setup_new_game(self):
        super().setup_new_game()
        self.current_max_num = 7

    def check_game_passed(self):
        pattern_level = self.check_pattern()
        if not pattern_level or pattern_level <= self.current_max_num:
            return
        if pattern_level > self.current_max_num:
            self.current_max_num = pattern_level
            if self.current_max_num > self.max_num:
                self.max_num = self.current_max_num
                self.is_passed = {10: 3, 9: 2, 8: 1}.get(self.max_num, 4)
                level = {10: 'golden', 9: 'silver', 8: 'bronze'}.get(self.max_num, 'golden')
                message = f'You achieved {2 ** self.max_num}!\n You get a {level} trophy!'
            else:
                level = {10: 'golden', 9: 'silver', 8: 'bronze'}.get(self.current_max_num, 'golden')
                message = f'You achieved {2 ** self.current_max_num}!\n Take it further!'
            self.show_trophy(f'pic/{level}.png', message)

    def check_pattern(self):
        mask = self.pattern != 0
        masked_original = 2 ** self.board * mask
        for num in (8,9,10,11,12,13,14):
            if np.all(masked_original == self.pattern * 2 ** num):
                return num
        return False

    def update_frame_small_label(self, row, col):
        if self.pattern[row][col] == 0:
            return
        small_label = self.game_square.small_labels[row][col]
        target = int(self.pattern[row][col] * 2 ** (max(self.current_max_num, self.max_num) + 1))
        target = str(target) if target < 1000 else str(target // 1000) + 'k'
        small_label.setText(target)


DesignMaster2Frame, DesignMaster3Frame, DesignMaster4Frame = DesignMaster1Frame, DesignMaster1Frame, DesignMaster1Frame


# noinspection PyAttributeOutsideInit
class DesignMasterWindow(MinigameWindow):
    def __init__(self, minigame='Design Master1', frame_type=DesignMaster1Frame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def show_message(self):
        text = 'Fit a particular pattern.\n'
        target = self.gameframe.pattern[0] * max(self.gameframe.current_max_num, self.gameframe.max_num) * 2
        text += self.array_to_formatted_string(target)

        # 创建自定义的QMessageBox
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle('Information')
        msg_box.setText(text)
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        font = QtGui.QFont("Consolas", 10)  # 设置等宽字体
        msg_box.setFont(font)
        msg_box.exec_()

        self.gameframe.setFocus()

    @staticmethod
    def format_number(num):
        if num == 0:
            return '_'
        return f"{num:3d}" if num < 1000 else f"{num // 1000}k"

    def array_to_formatted_string(self, array):
        formatted_rows = []
        for row in array:
            formatted_row = [self.format_number(int(num)).rjust(3, ' ') for num in row]
            formatted_rows.append(formatted_row)
        return "\n".join(" ".join(row) for row in formatted_rows)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = DesignMasterWindow(minigame='Design Master3', frame_type=DesignMaster1Frame)
    main.gameframe.board = np.array([[0, 1, 0, 1],
                                     [1, 0, 0, 2],
                                     [3, 4, 8, 3],
                                     [8, 7, 6, 8]])
    main.show()
    sys.exit(app.exec_())
