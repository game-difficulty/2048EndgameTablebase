import sys
from typing import List, Optional, Tuple
import random

import numpy as np
from PyQt5 import QtCore, QtWidgets

from MiniGame import MinigameFrame, MinigameWindow
from Config import SingletonConfig


class IceAgeFrame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Ice Age'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        self.frozen_step = 80 + self.difficulty * 20
        self.animation_group: QtCore.QAbstractAnimation | None = None
        self.rows, self.cols = 4, 4
        try:
            self.count_down = SingletonConfig().config['minigame_state'][self.difficulty][minigame_type][1][0]
        except KeyError:
            self.count_down = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.movement_track = np.zeros((self.rows, self.cols), dtype=bool)
        self.frozen_frame: List[List[List[Optional[QtWidgets.QLabel]]]] = \
            [[[None] * 6 for _ in range(self.cols)] for __ in range(self.rows)]

        super().__init__(centralwidget, minigame_type)
        self.init_frozen_frame()

    def clear_all_frozen_frame(self):
        for row in range(self.rows):
            for col in range(self.cols):
                for i, frame in enumerate(self.frozen_frame[row][col]):
                    if frame is not None:
                        frame.close()

    def setup_new_game(self):
        self.clear_all_frozen_frame()
        self.count_down = np.zeros((self.rows, self.cols), dtype=np.int32)
        super().setup_new_game()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.set_frozen_frame_positions()

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            ([self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [
                self.count_down])
        event.accept()

    @staticmethod
    def hex_to_rgba(hex_color, alpha=255):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        rgba = (r, g, b, alpha)
        return f'rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})'

    def before_move(self, direct):
        self.movement_track = self.move_and_track(direct)

    def before_gen_num(self, _):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i, j] == - 1:
                    continue
                elif not self.movement_track[i, j]:  # 移动过
                    self.count_down[i, j] = 0
                else:
                    self.count_down[i, j] += 1
                    if self.count_down[i, j] >= self.frozen_step:
                        self.count_down[i, j] = self.board[i, j]
                        self.board[i, j] = -1

        self.update_frozen_frames()

    def update_frame(self, value, row, col, anim=False):
        label = self.game_square.labels[row][col]
        frame = self.game_square.frames[row][col]
        if value == -1:
            died_value = self.count_down[row, col]
            label.setText(str(2 ** died_value))
            color = self.game_square.colors[died_value - 1]
            color = self.hex_to_rgba(color, 160)
            frame.setStyleSheet(f"""
                        QFrame#f{row * self.cols + col} {{
                            background-color: {color};
                        }}
                        """)
            fontsize = self.game_square.base_font_size if len(str(2 ** died_value)) < 3 else int(
                self.game_square.base_font_size * 3 / (0.5 + len(str(2 ** died_value))))
        elif value != 0:
            label.setText(str(2 ** value))
            color = self.game_square.colors[value - 1]
            frame.setStyleSheet(f"background-color: {color};")
            fontsize = self.game_square.base_font_size if len(str(2 ** value)) < 3 else int(
                self.game_square.base_font_size * 3 / (0.5 + len(str(2 ** value))))
        else:
            label.setText('')
            color = 'rgba(229, 229, 229, 1)'
            frame.setStyleSheet(f"background-color: {color};")
            fontsize = self.game_square.base_font_size
        label.setStyleSheet(
            f"font: {fontsize}pt 'Calibri'; font-weight: bold; color: white; background-color: transparent;")

        self.update_frame_small_label(row, col)
        if anim:
            self.game_square.animate_appear(row, col)

    def init_frozen_frame(self):
        for row in range(self.rows):
            for col in range(self.cols):
                frame = self.game_square.frames[row][col]

                # Crystal1
                crystal1 = QtWidgets.QFrame(frame)
                crystal1.setObjectName(f'f{row * self.cols + col}_0')
                crystal1.setStyleSheet(f"""
                QFrame#f{row * self.cols + col}_0 {{
                border-image: url(pic//crystal1.png) 0 0 0 0 stretch stretch; background-color: transparent;
                }}
                """)
                self.frozen_frame[row][col][0] = crystal1

                # Crystal2
                crystal2 = QtWidgets.QFrame(frame)
                crystal2.setObjectName(f'f{row * self.cols + col}_1')
                crystal2.setStyleSheet(f"""
                QFrame#f{row * self.cols + col}_1 {{
                border-image: url(pic//crystal3.png) 0 0 0 0 stretch stretch; background-color: transparent;
                }}
                """)
                self.frozen_frame[row][col][1] = crystal2

                # Crystal3
                crystal3 = QtWidgets.QFrame(frame)
                crystal3.setObjectName(f'f{row * self.cols + col}_2')
                crystal3.setStyleSheet(f"""
                QFrame#f{row * self.cols + col}_2 {{
                border-image: url(pic//crystal2.png) 0 0 0 0 stretch stretch; background-color: transparent;
                }}
                """)
                self.frozen_frame[row][col][2] = crystal3

                # Crystal4
                crystal4 = QtWidgets.QFrame(frame)
                crystal4.setObjectName(f'f{row * self.cols + col}_3')
                crystal4.setStyleSheet(f"""
                QFrame#f{row * self.cols + col}_3 {{
                border-image: url(pic//ice_overlay.png) 0 0 0 0 stretch stretch; background-color: transparent;
                }}
                """)
                self.frozen_frame[row][col][3] = crystal4

                # Crystal5
                crystal5 = QtWidgets.QFrame(frame)
                crystal5.setObjectName(f'f{row * self.cols + col}_4')
                crystal5.setStyleSheet(f"""
                QFrame#f{row * self.cols + col}_4 {{
                border-image: url(pic//icetrap0.png) 0 0 0 0 stretch stretch; background-color: transparent;
                }}
                """)
                self.frozen_frame[row][col][4] = crystal5

                # Crystal6
                crystal6 = QtWidgets.QFrame(frame.parent())
                crystal6.setObjectName(f'f{row * self.cols + col}_5')
                crystal6.setStyleSheet(f"""
                QFrame#f{row * self.cols + col}_5 {{
                border-image: url(pic//icetrap.png) 0 0 0 0 stretch stretch; background-color: transparent;
                }}
                """)
                self.frozen_frame[row][col][5] = crystal6

    def set_frozen_frame_positions(self):
        for row in range(self.rows):
            for col in range(self.cols):
                frame = self.game_square.frames[row][col]

                # 设置 Crystal1 位置
                crystal1 = self.frozen_frame[row][col][0]
                self.set_frame_position(crystal1, frame, (1 / 9, 1 / 9),
                                        lambda pw, ph, fw, fh: (pw - 2 * fw, ph - 2 * fh))

                # 设置 Crystal2 位置
                crystal2 = self.frozen_frame[row][col][1]
                self.set_frame_position(crystal2, frame, (1 / 6, 1 / 6),
                                        lambda pw, ph, fw, fh: (pw - 3 * fw, ph - int(5.2 * fh)))

                # 设置 Crystal3 位置
                crystal3 = self.frozen_frame[row][col][2]
                self.set_frame_position(crystal3, frame, (1 / 4, 1 / 4),
                                        lambda pw, ph, fw, fh: (pw - int(3.75 * fw), ph - int(1.75 * fh)))

                # 设置 Crystal4 位置
                crystal4 = self.frozen_frame[row][col][3]
                self.set_frame_position(crystal4, frame, (1, 1),
                                        lambda pw, ph, fw, fh: (0, 0))

                # 设置 Crystal5 位置
                crystal5 = self.frozen_frame[row][col][4]
                self.set_frame_position(crystal5, frame, (1, 1 / 4),
                                        lambda pw, ph, fw, fh: (0, - int(0.33 * fh)))

                # 设置 Crystal6 位置
                crystal6 = self.frozen_frame[row][col][5]
                parent_width = frame.width()
                parent_height = frame.height()
                crystal6.resize(int(parent_width * 1.2), parent_height // 2)
                # 获取 frame 相对于其父控件的坐标
                frame_pos = frame.pos()
                new_x = frame_pos.x() + (parent_width - crystal6.width()) // 2  # 水平居中对齐
                new_y = frame_pos.y() + int(parent_height // 1.8)

                crystal6.move(new_x, new_y)

    @staticmethod
    def set_frame_position(frame, parent, size_ratio, position_func):
        parent_width = parent.width()
        parent_height = parent.height()
        frame.resize(int(parent_width * size_ratio[0]), int(parent_height * size_ratio[1]))
        new_x, new_y = position_func(parent_width, parent_height, frame.width(), frame.height())
        frame.move(new_x, new_y)

    def update_frozen_frames(self):
        for row in range(self.rows):
            for col in range(self.cols):
                self.update_frozen_frame(row, col)

    def update_frozen_frame(self, row, col):
        frozen_frames = self.frozen_frame[row][col]
        count_down = self.count_down[row][col]
        num = self.board[row][col]

        def _update_frame(i, th):
            if count_down >= th or num == -1:
                if frozen_frames[i].isHidden():
                    frozen_frames[i].show()
            elif frozen_frames[i].isVisible():
                frozen_frames[i].hide()

        if count_down == 0 and num != -1:
            for f in frozen_frames:
                if f is not None and f.isVisible():
                    f.close()

        thresholds = [
            (5, self.frozen_step),
            (4, self.frozen_step - 5),
            (3, 64 + self.difficulty * 16),
            (2, 50 + self.difficulty * 10),
            (1, 36 + self.difficulty * 4),
            (0, 20)
        ]

        for index, threshold in thresholds:
            _update_frame(index, threshold)

    @staticmethod
    def track_movement(line: np.ndarray, reverse: bool = False) -> np.ndarray:
        if reverse:
            line = line[::-1]

        result: np.ndarray = np.zeros_like(line, dtype=bool)  # 初始化为全False, 表示每个位置都发生了移动或合并

        segments = []
        current_segment = []

        # 分割行
        for value in line:
            if value == -1:
                if current_segment:
                    segments.append(current_segment)
                segments.append([-1])
                current_segment = []
            else:
                current_segment.append(value)

        if current_segment:
            segments.append(current_segment)

        result_idx = 0  # 用于记录结果数组的位置索引
        for segment in segments:
            if segment == [-1]:
                result[result_idx] = False
                result_idx += 1
            else:
                moved = False
                for i in range(len(segment)):
                    if moved:
                        result[result_idx + i] = False
                        continue
                    if segment[i] == 0:
                        moved = True
                    j = 1
                    while i + j < len(segment):
                        if segment[i] == segment[i + j]:
                            moved = True
                            break
                        elif segment[i + j] != 0:
                            break
                        j += 1
                    if moved:
                        result[result_idx + i] = False
                    else:
                        result[result_idx + i] = True
                result_idx += len(segment)

        if reverse:
            result = result[::-1]

        return result

    def move_and_track(self, direction: int) -> np.ndarray:
        movement_occurred = np.zeros_like(self.board, dtype=bool)

        if direction == 1:  # "left"
            for i in range(self.rows):
                movement_occurred[i, :] = self.track_movement(self.board[i, :])
        elif direction == 2:  # "right":
            for i in range(self.rows):
                movement_occurred[i, :] = self.track_movement(self.board[i, :], reverse=True)
        elif direction == 3:  # "up":
            for i in range(self.cols):
                movement_occurred[:, i] = self.track_movement(self.board[:, i])
        elif direction == 4:  # "down":
            for i in range(self.cols):
                movement_occurred[:, i] = self.track_movement(self.board[:, i], reverse=True)

        return movement_occurred


# noinspection PyAttributeOutsideInit
class IceAgeWindow(MinigameWindow):
    def __init__(self, minigame='Ice Age', frame_type=IceAgeFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

        # 设置定时器
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.random_show_icesparkle)
        self.timer.start(880)  # 每880毫秒触发一次

        self.sparkle_and_anim: Tuple[Optional[QtWidgets.QFrame], Optional[QtCore.QAbstractAnimation]] = (None, None)

    def random_show_icesparkle(self):
        if self.sparkle_and_anim[1] is None:
            if random.random() < 0.1:  # 0.1的概率
                self.show_icesparkle()

    def show_icesparkle(self):
        # 获取窗口的宽度和高度
        window_width = self.width()
        window_height = self.height()

        # 随机生成 QFrame 的位置
        x = random.randint(0, window_width - 50)
        y = random.randint(0, window_height - 50)

        w = int(window_width // random.randint(10, 20))
        h = int(w // 4)

        # 创建 QFrame
        sparkle_frame = QtWidgets.QFrame(self)
        sparkle_frame.setGeometry(x, y, w, h)
        sparkle_frame.setObjectName("icesparkle")
        sparkle_frame.setStyleSheet("""
                QFrame#icesparkle {
                    border-image: url(pic//icesparkle.png) 0 0 0 0 stretch stretch;
                    background-color: transparent;
                }
            """)

        # 创建透明度效果
        opacity_effect = QtWidgets.QGraphicsOpacityEffect()
        sparkle_frame.setGraphicsEffect(opacity_effect)

        # 创建动画
        animation = QtCore.QPropertyAnimation(opacity_effect, b"opacity")
        animation.setDuration(1000)  # 持续1秒
        animation.setStartValue(1)
        animation.setEndValue(0)
        animation.finished.connect(self.anim_ended)  # 动画结束后删除 QFrame
        animation.start()

        sparkle_frame.show()
        print(x,y)
        self.sparkle_and_anim = (sparkle_frame, animation)

    def anim_ended(self):
        self.sparkle_and_anim[0].deleteLater()
        self.sparkle_and_anim = (None, None)

    def show_message(self):
        text = 'Tiles left idle for too long will freeze in place!'
        QtWidgets.QMessageBox.information(self, 'Information', text)
        self.gameframe.setFocus()

    def show(self):
        super().show()
        print(self.gameframe.count_down)
        self.gameframe.update_frozen_frames()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = IceAgeWindow(minigame='Ice Age', frame_type=IceAgeFrame)
    # main.gameframe.board = np.array([[0, 0, 0, 1],
    #                                  [1, 2, 3, 4],
    #                                  [8, 7, 6, 5],
    #                                  [9, 10, 11, 17]])
    # main.gameframe.count_down = np.array([[0, 0, 0, 0],
    #                                       [0, 0, 0, 0],
    #                                       [0, 0, 0, 0],
    #                                       [0, 0, 0, 0]])
    main.show()
    sys.exit(app.exec_())
