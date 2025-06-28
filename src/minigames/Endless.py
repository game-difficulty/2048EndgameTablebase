import sys
import random
from typing import Tuple, List

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from MiniGame import MinigameFrame, MinigameWindow
from Config import SingletonConfig
from Calculator import count_zeros


class EndlessFrame(MinigameFrame):
    def __init__(self, centralwidget=None, minigame_type='Endless Explosions'):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        try:
            self.bomb, self.current_level = \
                SingletonConfig().config['minigame_state'][self.difficulty][minigame_type][1]
        except KeyError:
            self.bomb: Tuple[int, int] | None = None
            self.current_level = 0
        
        self.levels = [
            (300000, 4, None),
            (200000, 3, 'gold'),
            (100000, 2, 'silver'),
            (40000, 1, 'bronze')
        ]
            
        self.pic_path = 'pic//tilebg.png'
        self.bomb_gen_rate = 0.03
        self.explode_pic = "pic//explode.png"
        self.has_just_exploded = False
        self.animation_group: List[QtCore.QAbstractAnimation] = []
        self.explosion_frame: List[QtWidgets.QFrame] = []
        super().__init__(centralwidget, minigame_type)

    def closeEvent(self, event):
        SingletonConfig().config['minigame_state'][self.difficulty][self.minigame] = \
            [self.board, self.score, self.max_score, self.max_num, self.is_passed, self.newtile_pos], [
                self.bomb, self.current_level]
        event.accept()

    def set_pic_path(self, path):
        self.pic_path = path
        if self.bomb:
            self.update_frame(0, *self.bomb)

    def setup_new_game(self):
        self.bomb = None
        super().setup_new_game()
        zero_positions = np.where(self.board == 0)
        self.bomb = random.choice(list(zip(zero_positions[0], zero_positions[1])))
        self.update_frame(0, *self.bomb)

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

    def gen_new_num(self, do_anim=True):
        if not self.bomb and random.random() < self.bomb_gen_rate:
            zero_positions = np.where(self.board == 0)
            self.bomb = random.choice(list(zip(zero_positions[0], zero_positions[1])))
            self.newtile_pos, self.newtile = self.bomb[0] * 4 + self.bomb[1], 0
        else:
            if self.bomb:
                self.board[*self.bomb] = 1
            self.board, _, new_tile_pos, val = self.mover.gen_new_num(
                self.board, SingletonConfig().config['4_spawn_rate'])
            if self.bomb:
                self.board[*self.bomb] = 0
            self.newtile_pos, self.newtile = new_tile_pos, val
        if do_anim:
            self.timer1.singleShot(125, lambda: self.game_square.animate_appear(
                self.newtile_pos // self.cols, self.newtile_pos % self.cols, self.newtile))

    def _set_special_frame(self, value, row, col):
        if self.bomb and self.bomb[0] == row and self.bomb[1] == col:
            if not self.game_square.frames[row][col]:
                # 先创建控件
                self.game_square.update_tile_frame(None, row, col)

            label = self.game_square.labels[row][col]
            frame = self.game_square.frames[row][col]
            label.setText('')
            frame.setStyleSheet(f"""
            QFrame#f{row * 4 + col} {{
            border-image: url({self.pic_path}) 0 0 0 0 stretch stretch;;
            }}
            """)
            label.setStyleSheet(f"""background-color: transparent;""")
            if self.newtile > 0 or not SingletonConfig().config['do_animation']:
                # 炸弹不是刚刚生成
                frame.show()

    def move_and_check_validity(self, direct):
        is_valid_move = self.bomb_move(direct)
        board_new, new_score = self.mover.move_board(self.board, direct)
        is_valid_move |= np.any(board_new != self.board)
        return board_new, new_score, is_valid_move

    def _bomb_explode(self, _range_front, _range_back, r, c, _is_vertical):
        _range_front, _range_back = list(_range_front), list(_range_back)
        if not _is_vertical:
            for j in _range_front + _range_back:
                if self.board[r, j] != 0:
                    self.has_just_exploded = True
                    self.explode_effect(r, j)
                    self.bomb = None
                    return True
        else:
            for i in _range_front + _range_back:
                if self.board[i, c] != 0:
                    self.has_just_exploded = True
                    self.explode_effect(i, c)
                    self.bomb = None
                    return True
        return False

    def explode_effect(self, r, c):
        pass

    def bomb_move(self, direct):
        if not self.bomb:
            return False
        do_anim = SingletonConfig().config['do_animation']
        row, col = self.bomb

        if direct == 1:  # Left
            if col != 0 and do_anim:
                distance = count_zeros(self.board[row, :col][::-1])
                self.game_square.animate_slide(row, col, 'Left', distance)
            if self._bomb_explode(range(col - 1, -1, -1), range(col + 1, 4), row, col, 0):
                return True
            self.bomb = (row, 0)
            return col != 0
        elif direct == 2:  # Right
            if col != 3 and do_anim:
                distance = count_zeros(self.board[row, (col + 1):])
                self.game_square.animate_slide(row, col, 'Right', distance)
            if self._bomb_explode(range(col + 1, 4), range(col - 1, -1, -1), row, col, 0):
                return True
            self.bomb = (row, 3)
            return col != 3
        elif direct == 3:  # Up
            if row != 0 and do_anim:
                distance = count_zeros(self.board[:row, col][::-1])
                self.game_square.animate_slide(row, col, 'Up', distance)
            if self._bomb_explode(range(row - 1, -1, -1), range(row + 1, 4), row, col, 1):
                return True
            self.bomb = (0, col)
            return row != 0
        elif direct == 4:  # Down
            if row != 3 and do_anim:
                distance = count_zeros(self.board[(row + 1):, col])
                self.game_square.animate_slide(row, col, 'Down', distance)
            if self._bomb_explode(range(row + 1, 4), range(row - 1, -1, -1), row, col, 1):
                return True
            self.bomb = (3, col)
            return row != 3

    def has_possible_move(self):
        if np.sum(self.board == 0) > 0 or self.bomb:
            return True
        else:
            for direct in (1, 2, 3, 4):
                board_new, _, is_valid_move = self.move_and_check_validity(direct)
                if is_valid_move:
                    return True
            return False

    def before_gen_num(self, direct):
        if self.has_just_exploded:
            pass
            self.has_just_exploded = False

    def trigger_explosion(self, row, col):
        # 获取目标 QFrame
        target_frame = self.game_square.grids[row][col]

        # 获取目标 QFrame 的全局几何位置
        target_global_pos = target_frame.mapToGlobal(QtCore.QPoint(0, 0))
        target_size = target_frame.size()

        # 创建一个临时 QFrame 作为爆炸效果
        explosion_frame = QtWidgets.QFrame(self)
        explosion_frame.setStyleSheet("background: transparent;")
        explosion_frame.setGeometry(QtCore.QRect(self.mapFromGlobal(target_global_pos), target_size))

        # 创建一个 QLabel 用于显示爆炸图案
        explosion_label = QtWidgets.QLabel(explosion_frame)
        explosion_label.setPixmap(QtGui.QPixmap(self.explode_pic))
        explosion_label.setScaledContents(True)
        explosion_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        explosion_label.setGeometry(0, 0, target_frame.width(), target_frame.height())

        # 创建放大动画
        size_animation = QtCore.QPropertyAnimation(explosion_label, b"geometry")
        size_animation.setDuration(500)
        size_animation.setStartValue(QtCore.QRect(target_frame.width() // 4, target_frame.height() // 4,
                                                  target_frame.width() // 2, target_frame.height() // 2))
        size_animation.setEndValue(QtCore.QRect(-target_frame.width() // 10, -target_frame.height() // 10,
                                                int(target_frame.width() * 1.2), int(target_frame.height() * 1.2)))
        size_animation.setEasingCurve(QtCore.QEasingCurve.OutQuart)

        # 创建透明度动画
        opacity_effect = QtWidgets.QGraphicsOpacityEffect(explosion_label)
        explosion_label.setGraphicsEffect(opacity_effect)
        opacity_animation = QtCore.QPropertyAnimation(opacity_effect, b"opacity")
        opacity_animation.setDuration(500)
        opacity_animation.setStartValue(1)
        opacity_animation.setEndValue(0)
        opacity_animation.setEasingCurve(QtCore.QEasingCurve.InExpo)

        # 创建动画组
        animation_group = QtCore.QParallelAnimationGroup()
        animation_group.addAnimation(size_animation)
        animation_group.addAnimation(opacity_animation)

        def on_animation_finished():
            self.explosion_frame.remove(explosion_frame)
            explosion_label.deleteLater()
            explosion_frame.deleteLater()
            self.animation_group.remove(animation_group)

        # 在动画结束后移除临时 QFrame
        animation_group.finished.connect(on_animation_finished)

        def start_anim():
            # 显示爆炸效果
            explosion_frame.show()
            animation_group.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

        do_anim = SingletonConfig().config['do_animation']
        if do_anim:
            QtCore.QTimer.singleShot(100, start_anim)
        else:
            start_anim()

        self.animation_group.append(animation_group)
        self.explosion_frame.append(explosion_frame)


class EndlessExplosionsFrame(EndlessFrame):
    def __init__(self, centralwidget=None, minigame_type='Endless Explosions'):
        super().__init__(centralwidget, minigame_type)
        self.set_pic_path('pic//bomb.png')

    def explode_effect(self, r, c):
        self.board[r, c] = 0
        self.trigger_explosion(r, c)
        self.update_frame(0, r, c)


class EndlessGiftboxFrame(EndlessFrame):
    def __init__(self, centralwidget=None, minigame_type='Endless Giftbox'):
        super().__init__(centralwidget, minigame_type)
        self.set_pic_path('pic//giftbox.png')
        self.explode_pic = 'pic//hole.png'

    def explode_effect(self, r, c):
        self.has_just_exploded = self.board[r, c]
        self.board[r, c] = -2

    def before_gen_num(self, direct):
        if self.has_just_exploded:
            original_value = self.has_just_exploded
            positions = np.where(self.board == -2)
            positions_list = list(zip(positions[0], positions[1]))

            r0, c0 = positions_list[0]
            exponents = np.arange(2, 11)
            weights = 1 / (exponents ** 1.5)
            weights /= weights.sum()
            new_value = original_value
            while new_value == original_value:
                new_value = np.random.choice(exponents, p=weights)
            self.board[r0, c0] = new_value
            self.trigger_explosion(r0, c0)

            if SingletonConfig().config['do_animation']:
                def update_new_value(val, r, c):
                    self.game_square.update_tile_frame(val, r, c)
                    self.game_square.frames[r][c].raise_()

                QtCore.QTimer.singleShot(120, lambda: update_new_value(new_value, r0, c0))

            self.has_just_exploded = False


class EndlessFactorizationFrame(EndlessFrame):
    def __init__(self, centralwidget=None, minigame_type='Endless Factorization'):
        super().__init__(centralwidget, minigame_type)
        self.set_pic_path('pic//tilebg.png')
        self.explode_pic = 'pic//hole2.png'

    def explode_effect(self, r, c):
        self.has_just_exploded = self.board[r, c]
        if self.has_just_exploded != 1:
            self.board[*self.bomb] = -2
        else:
            self.board[*self.bomb] = 0  # 本来也是0
        self.board[r, c] = -2

    def before_gen_num(self, direct):
        if self.has_just_exploded:
            positions = np.where(self.board == -2)
            positions_list = list(zip(positions[0], positions[1]))
            if self.has_just_exploded == 1:
                r0, c0 = positions_list[0]
                self.board[r0, c0] = 0
                self.trigger_explosion(r0, c0)
                self.update_frame(0, r0, c0)
            else:
                r0, c0 = positions_list[0]
                r1, c1 = positions_list[1]
                factor1 = random.randint(1, max(self.has_just_exploded - 1, 1))
                self.board[r0, c0] = factor1
                self.board[r1, c1] = self.has_just_exploded - factor1
                self.trigger_explosion(r0, c0)
                self.trigger_explosion(r1, c1)

                if SingletonConfig().config['do_animation']:
                    def update_divided(val, r, c):
                        self.game_square.update_tile_frame(val, r, c)
                        self.game_square.frames[r][c].raise_()

                    factor0 = self.has_just_exploded - factor1
                    QtCore.QTimer.singleShot(120, lambda: update_divided(factor0, r1, c1))
                    QtCore.QTimer.singleShot(120, lambda: update_divided(factor1, r0, c0))

            self.has_just_exploded = False

    def _set_special_frame(self, value, row, col):
        if self.bomb and self.bomb[0] == row and self.bomb[1] == col:
            if not self.game_square.frames[row][col]:
                # 先创建控件
                self.game_square.update_tile_frame(None, row, col)

            label = self.game_square.labels[row][col]
            frame = self.game_square.frames[row][col]
            label.setText('÷')
            frame.setStyleSheet(f"""
            QFrame#f{row * 4 + col} {{
            border-image: url({self.pic_path}) 0 0 0 0 stretch stretch;;
            }}
            """)
            fontsize = self.game_square.base_font_size
            label.setStyleSheet(f"""font: {fontsize}pt 'Calibri'; font-weight: bold; color: white;
                                 background-color: transparent;""")
            if self.newtile > 0 or not SingletonConfig().config['do_animation']:
                # 炸弹不是刚刚生成
                frame.show()


# noinspection PyAttributeOutsideInit
class EndlessExplosionsWindow(MinigameWindow):
    def __init__(self, minigame='Endless Explosions', frame_type=EndlessExplosionsFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def show_message(self):
        text = self.tr('A small chance of generating a bomb\n that destroys the first tile it encounters!')
        QtWidgets.QMessageBox.information(self, self.tr('Information'), text)
        self.gameframe.setFocus()


class EndlessFactorizationWindow(MinigameWindow):
    def __init__(self, minigame='Endless Factorization', frame_type=EndlessFactorizationFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def show_message(self):
        text = self.tr('A small chance of generating a power-up\n that halves the first tile it encounters!')
        QtWidgets.QMessageBox.information(self, self.tr('Information'), text)
        self.gameframe.setFocus()


class EndlessGiftboxWindow(MinigameWindow):
    def __init__(self, minigame='Endless Giftbox', frame_type=EndlessGiftboxFrame):
        super().__init__(minigame=minigame, frame_type=frame_type)

    def show_message(self):
        text = self.tr('A small chance of generating a gift box\n that magically changes the first tile it encounters!')
        QtWidgets.QMessageBox.information(self, self.tr('Information'), text)
        self.gameframe.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = EndlessFactorizationWindow()
    main.gameframe.board = np.array([[0, 0, 0, 0],
                                     [0, 0, 2, 1],
                                     [3, 4, 4, 2],
                                     [5, 7, 7, 17]])
    main.gameframe.bomb = (0, 1)
    main.show()
    sys.exit(app.exec_())
