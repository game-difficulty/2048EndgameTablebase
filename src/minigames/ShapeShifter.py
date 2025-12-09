import sys
import random

import numpy as np
from PyQt5 import QtWidgets

from MiniGame import MinigameFrame, MinigameWindow, MinigameSquareFrame
from Config import SingletonConfig, ColorManager


class ShapeShifterFrame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Shape Shifter'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        self.n = 12
        self.game_square: MinigameSquareFrame | None = None
        try:
            self.rows, self.cols = \
                SingletonConfig().config['minigame_state'][self.difficulty][minigame_type][0][0].shape
        except KeyError:
            self.rows, self.cols = self.n, self.n
        super().__init__(centralwidget, minigame_type, shape=(self.rows, self.cols))

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            ([self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [
                ])
        SingletonConfig().save_config(SingletonConfig().config)
        event.accept()

    def setup_new_game(self):
        rectangle_area = 10000
        while rectangle_area > 13 - self.difficulty * 2 or rectangle_area < 8 - self.difficulty * 2:
            self.board = self.select_connected_cells(self.n, 18)
            rectangle_area = ShapeShifterFrame.max_rectangle_area(self.board)

        self.board = ShapeShifterFrame.min_bounding_rectangle(self.board)
        self.rows, self.cols = self.board.shape
        self.replace_game_square()
        self.mover.rows, self.mover.cols = self.rows, self.cols

        self.board, _, self.newtile_pos, self.newtile = self.mover.gen_new_num(
            self.mover.gen_new_num(self.board, SingletonConfig().config['4_spawn_rate'])[0],
            SingletonConfig().config['4_spawn_rate'])

        self.update_all_frame(self.board)
        self.score = 0
        self.current_max_num = self.board.max()

    def replace_game_square(self):
        # 删除当前的game_square
        if self.game_square is not None:
            self.game_square.deleteLater()
            self.game_square = None
        # 创建一个新的game_square
        self.game_square = MinigameSquareFrame(self, shape=(self.rows, self.cols))
        self.game_square.updateGeometry()
        self.game_square.show()

    def _set_special_frame(self, value, row, col):
        if value == -1:
            label = self.game_square.labels[row][col]
            frame = self.game_square.frames[row][col]
            label.setText('')
            color_mgr = ColorManager()
            frame.setStyleSheet(f"""
                        QFrame#f{row * self.cols + col} {{
                            background-color: {color_mgr.get_css_color(5)};
                        }}
                        """)
            fontsize = self.game_square.base_font_size if (value == -1 or len(str(2 ** value)) < 3) else int(
                self.game_square.base_font_size * 3 / (0.5 + len(str(2 ** value))))
            label.setStyleSheet(self.game_square.get_label_style(fontsize, value))
            self.game_square.grids[row][col].setVisible(False)

    @staticmethod
    def is_valid(x, y, visited, n):
        return 0 <= x < n and 0 <= y < n and (x, y) not in visited

    @staticmethod
    def get_neighbors(x, y):
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        random.shuffle(neighbors)  # 随机打乱邻居顺序
        return neighbors

    @staticmethod
    def select_connected_cells(n, m):
        # 初始化nxn的数组，所有元素初始化为-1
        array: np.ndarray = np.full((n, n), -1)
        visited = set()
        remaining = set()

        # 随机选择一个起始点
        start_x, start_y = random.randint(0, n - 1), random.randint(0, n - 1)
        stack = [(start_x, start_y)]

        while len(visited) < m:
            if not stack:
                # 如果栈为空且未达到m个格子，将remaining填入
                stack = [remaining.pop()]

            x, y = stack.pop()
            if (x, y) in visited:
                continue

            # 标记当前格子已访问
            visited.add((x, y))
            array[x, y] = 0

            # 获取并遍历邻居
            for nx, ny in ShapeShifterFrame.get_neighbors(x, y):
                if ShapeShifterFrame.is_valid(nx, ny, visited, n):
                    if random.random() > 0.5:
                        stack.append((nx, ny))
                    else:
                        remaining.add((nx, ny))

        return array

    @staticmethod
    def max_rectangle_area(matrix):
        if not np.any(matrix == 0):
            return 0

        n = len(matrix)
        m = len(matrix[0])
        heights = [0] * m
        max_area = 0

        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    heights[j] += 1
                else:
                    heights[j] = 0

            max_area = max(max_area, ShapeShifterFrame.largest_rectangle_area_in_histogram(heights))

        return max_area

    @staticmethod
    def largest_rectangle_area_in_histogram(heights):
        stack = []
        max_area = 0
        heights.append(0)

        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)

        heights.pop()
        return max_area

    @staticmethod
    def min_bounding_rectangle(matrix):
        n = len(matrix)
        if n == 0:
            return np.array([])

        min_x, max_x, min_y, max_y = n, 0, n, 0

        # 找到连通区域的边界
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 0:
                    min_x = min(min_x, i)
                    max_x = max(max_x, i)
                    min_y = min(min_y, j)
                    max_y = max(max_y, j)

        # 提取包含所有0格子的最小矩形
        bounding_rectangle = matrix[min_x:max_x + 1, min_y:max_y + 1]

        return bounding_rectangle


# noinspection PyAttributeOutsideInit
class ShapeShifterWindow(MinigameWindow):
    def __init__(self, minigame='Shape Shifter', frame_type=ShapeShifterFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def show_message(self):
        text = self.tr('Every game begins with a unique and unexpected board shape!')
        QtWidgets.QMessageBox.information(self, self.tr('Information'), text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = ShapeShifterWindow(minigame='Shape Shifter', frame_type=ShapeShifterFrame)
    main.show()
    sys.exit(app.exec_())
