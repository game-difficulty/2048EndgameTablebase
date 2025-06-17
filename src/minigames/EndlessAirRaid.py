import sys
from typing import Tuple, Union
import random

import numpy as np
from PyQt5 import QtWidgets, QtCore

from MiniGame import MinigameFrame, MinigameWindow
from Config import SingletonConfig


class EndlessAirRaidFrame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Endless AirRaid'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        self.rows, self.cols = 4, 4
        try:
            self.count_down, self.target_pos, self.current_level = \
                SingletonConfig().config['minigame_state'][self.difficulty][minigame_type][1]
        except KeyError:
            self.count_down: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.int32)
            self.target_pos: Union[Tuple[int, int], None] = None
            self.current_level = 0

        self.levels = [
            (300000, 4, None),
            (200000, 3, 'gold'),
            (100000, 2, 'silver'),
            (40000, 1, 'bronze')
        ]

        self.animation_group: QtCore.QAbstractAnimation | None = None
        self.target_layout: QtWidgets.QFrame | None = None
        super().__init__(centralwidget, minigame_type)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.target_pos:
            self.clear_target_layout()
            self.set_target_layout()

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            ([self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [
                self.count_down, self.target_pos, self.current_level])
        event.accept()

    def before_gen_num(self, _):
        mask = self.count_down > 0
        self.count_down[mask] -= 1
        mask_just_zero = (self.count_down == 0) & mask
        self.board[mask_just_zero] = 0
        if self.target_pos:
            self.animation_group.stop()
            self.target_layout.deleteLater()
            self.animation_group = None
            self.target_layout = None
            if self.board[*self.target_pos] != 0:
                self.update_all_frame()
                QtWidgets.QApplication.processEvents()
                self.set_explode_layout()
            else:
                self.target_pos = None

    def set_explode_layout(self):
        self.target_layout = QtWidgets.QFrame(self.parent())
        self.target_layout.setObjectName(f'f_fire_layout')
        self.target_layout.setStyleSheet(f"""
        QFrame#f_fire_layout {{
        border-image: url(pic//fire.png) 0 0 0 0 stretch stretch; background-color: transparent;
        }}
        """)
        self.target_layout.show()
        self.create_fire_animation()

    def create_fire_animation(self):
        # 获取目标 frame 的几何位置
        row, col = self.target_pos
        frame = self.game_square.frames[row][col]
        target_local_pos = self.mapToParent(self.game_square.mapToParent(frame.mapToParent(QtCore.QPoint(0, 0))))

        # 设置 fire_layout 初始位置
        self.target_layout.setGeometry(target_local_pos.x(), target_local_pos.y() - 360, frame.width(), frame.height())

        # 创建下落动画
        self.animation_group = QtCore.QPropertyAnimation(self.target_layout, b"geometry")
        self.animation_group.setDuration(250)
        self.animation_group.setStartValue(QtCore.QRect(target_local_pos.x(), target_local_pos.y() - 360, frame.width(),
                                                        frame.height()))
        self.animation_group.setEndValue(
            QtCore.QRect(target_local_pos.x(), target_local_pos.y(), frame.width(), frame.height()))
        self.animation_group.setEasingCurve(QtCore.QEasingCurve.InQuad)  # 加速下落

        # 下落动画结束后，显示爆炸效果
        self.animation_group.finished.connect(self.trigger_explosion)

        # 开始动画
        self.animation_group.start()

    def trigger_explosion(self):
        self.clear_target_layout()

        self.board[*self.target_pos] = -1
        self.count_down[*self.target_pos] = 60 + self.difficulty * 40
        self.update_all_frame()

        row, col = self.target_pos
        frame = self.game_square.frames[row][col]

        # 创建一个临时 QFrame 作为爆炸效果
        self.target_layout = QtWidgets.QFrame(frame)
        self.target_layout.setObjectName(f'f_explosion_layout')
        self.target_layout.setStyleSheet(f"""
        QFrame#f_explosion_layout {{
        border-image: url(pic//explode.png) 0 0 0 0 stretch stretch; background-color: transparent;
        }}
        """)
        self.target_layout.setGeometry(0, 0, frame.width(), frame.height())

        size_animation = QtCore.QPropertyAnimation(self.target_layout, b"geometry")
        size_animation.setDuration(200)
        size_animation.setStartValue(QtCore.QRect(frame.width() // 4, frame.height() // 4,
                                                  frame.width() // 2, frame.height() // 2))
        size_animation.setEndValue(QtCore.QRect(-frame.width() // 10, -frame.height() // 10,
                                                int(frame.width() * 1.2), int(frame.height() * 1.2)))
        size_animation.setEasingCurve(QtCore.QEasingCurve.OutQuart)

        # 创建透明度动画
        opacity_effect = QtWidgets.QGraphicsOpacityEffect(self.target_layout)
        opacity_effect.setOpacity(1)
        self.target_layout.setGraphicsEffect(opacity_effect)
        opacity_animation = QtCore.QPropertyAnimation(opacity_effect, b"opacity")
        opacity_animation.setDuration(200)
        opacity_animation.setStartValue(1)
        opacity_animation.setEndValue(1)  # 不设1有bug，待修
        opacity_animation.setEasingCurve(QtCore.QEasingCurve.InExpo)

        # 创建动画组
        self.animation_group = QtCore.QParallelAnimationGroup()
        self.animation_group.addAnimation(opacity_animation)
        self.animation_group.addAnimation(size_animation)

        # 在动画结束后移除临时 QFrame
        self.animation_group.finished.connect(self.explode_anim_finish)

        # 显示爆炸效果
        self.target_layout.show()
        self.animation_group.start()

    def explode_anim_finish(self):
        self.target_pos = None
        self.clear_target_layout()
        self.check_game_over()

    def clear_target_layout(self):
        if self.target_layout is not None:
            # self.target_layout.close()
            self.animation_group.stop()
            self.target_layout.deleteLater()
            self.animation_group = None
            self.target_layout = None
            QtWidgets.QApplication.processEvents()

    def setup_new_game(self):
        self.target_pos = None
        super().setup_new_game()
        self.clear_target_layout()

    def gen_new_num(self, do_anim=True):
        self.board, empty_count, new_tile_pos, val = self.mover.gen_new_num(
            self.board, SingletonConfig().config['4_spawn_rate'])
        self.newtile_pos, self.newtile = new_tile_pos, val
        p = max(0.1 - (self.count_down > 0).sum() / 40, 0.01)
        if empty_count > 1 and self.target_pos is None and random.random() < p:
            self.gen_target()
        self.update_all_frame(self.board)
        self.update_frame(val, new_tile_pos // self.cols, new_tile_pos % self.cols, anim=do_anim)

    def gen_target(self):
        empty_positions = [(i, j) for i in range(4) for j in range(4) if self.board[i, j] == 0]
        self.target_pos = random.choice(empty_positions)
        self.set_target_layout()

    def update_frame(self, value, row, col, anim=False):
        label = self.game_square.labels[row][col]
        frame = self.game_square.frames[row][col]
        if value == -1:
            if self.count_down[row, col] > (60 + self.difficulty * 40) / 2:
                pic_path = 'pic//crater1.png'
            else:
                pic_path = 'pic//crater2.png'
            label.setText('')
            frame.setStyleSheet(f"""
                        QFrame#f{row * self.cols + col} {{
                            border-image: url({pic_path}) 2 2 2 2 stretch stretch;
                        }}
                        """)
        elif value != 0:
            label.setText(str(2 ** value))
            color = self.game_square.colors[value - 1]
            frame.setStyleSheet(f"background-color: {color};")
        else:
            label.setText('')
            color = 'rgba(229, 229, 229, 1)'
            frame.setStyleSheet(f"background-color: {color};")
        fontsize = self.game_square.base_font_size if (value == -1 or len(str(2 ** value)) < 3) else int(
            self.game_square.base_font_size * 3 / (0.5 + len(str(2 ** value))))
        label.setStyleSheet(self.game_square.get_label_style(fontsize, value))
        self.update_frame_small_label(row, col)

        if anim:
            self.game_square.animate_appear(row, col)

    def set_target_layout(self):
        row, col = self.target_pos
        frame = self.game_square.frames[row][col]
        self.target_layout = QtWidgets.QFrame(frame)
        self.target_layout.setObjectName(f'f_target_layout')
        self.target_layout.setStyleSheet(f"""
        QFrame#f_target_layout {{
        border-image: url(pic//target.png) 0 0 0 0 stretch stretch; background-color: transparent;
        }}
        """)
        self.reset_target_layout_pos()
        self.create_breathing_animation()
        self.target_layout.show()

    def reset_target_layout_pos(self):
        row, col = self.target_pos
        frame = self.game_square.frames[row][col]
        self.target_layout.resize(frame.width(), frame.height())
        self.target_layout.move(0, 0)

    def create_breathing_animation(self):
        self.animation_group = QtCore.QSequentialAnimationGroup()

        start_rect = self.target_layout.geometry()
        end_rect = start_rect.adjusted(10, 10, -10, -10)

        expand_animation = QtCore.QPropertyAnimation(self.target_layout, b"geometry")
        expand_animation.setDuration(800)
        expand_animation.setStartValue(end_rect)
        expand_animation.setEndValue(start_rect)
        expand_animation.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        shrink_animation = QtCore.QPropertyAnimation(self.target_layout, b"geometry")
        shrink_animation.setDuration(800)
        shrink_animation.setStartValue(start_rect)
        shrink_animation.setEndValue(end_rect)
        shrink_animation.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        # 将动画添加到动画组中
        self.animation_group.addAnimation(shrink_animation)
        self.animation_group.addAnimation(expand_animation)
        self.animation_group.setLoopCount(-1)  # 无限循环

        self.animation_group.start()

    def check_game_passed(self):
        self.current_max_num = max(self.current_max_num, self.board.max())
        self.max_num = max(self.max_num, self.current_max_num)

        level = None
        for score_threshold, level_number, level_name in self.levels:
            if self.score >= score_threshold and self.current_level < level_number:
                self.is_passed = max(self.is_passed, level_number)
                self.current_level = level_number
                level = level_name
                break
        if not level:
            return
        score = str(self.max_score // 1000) + 'k'
        if self.max_score == self.score:
            message = f"You achieved {score} score!\n You get a {level} trophy!"
        else:
            message = f"You achieved {score} score!\n Let's go!"
        self.show_trophy(f'pic/{level}.png', message)

    def has_possible_move(self):
        for direct in (1, 2, 3, 4):
            board_new, _, is_valid_move = self.move_and_check_validity(direct)
            if is_valid_move:
                return True
        return False


# noinspection PyAttributeOutsideInit
class EndlessAirRaidWindow(MinigameWindow):
    def __init__(self, minigame='Endless AirRaid', frame_type=EndlessAirRaidFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def process_input(self, direction):
        self.isProcessing = True  # 设置标志防止进一步的输入
        self.gameframe.do_move(direction)
        self.update_score()
        if self.gameframe.animation_group and self.gameframe.target_layout.objectName() != 'f_target_layout':
            QtCore.QTimer.singleShot(480, self.allow_input)  # 配合gameframe中空袭动画的延迟
        else:
            self.allow_input()

    def allow_input(self):
        self.isProcessing = False

    def show_message(self):
        text = 'Airstrikes incoming! Avoid marked targets!'
        QtWidgets.QMessageBox.information(self, 'Information', text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = EndlessAirRaidWindow(minigame='Endless AirRaid', frame_type=EndlessAirRaidFrame)
    # main.gameframe.board = np.array([[2, 1, 0, -1],
    #                                  [3, 1, -1, 0],
    #                                  [6, 3, -1, -1],
    #                                  [1, 5, 3, 2]])
    main.show()
    sys.exit(app.exec_())
