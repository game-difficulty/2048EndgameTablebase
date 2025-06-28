import sys
import importlib
import re
from typing import List

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui

from Config import SingletonConfig


class StrokedLabel(QtWidgets.QLabel):
    def __init__(self, text='', parent=None, shadow_offset=0, color=(100, 100, 100)):
        super().__init__(text, parent)
        self.shadow_offset = shadow_offset
        self.color = QtGui.QColor(*color)
        self.setWordWrap(True)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        lines = self.text().split('\n')
        line_height = self.fontMetrics().height()
        total_height = line_height * len(lines)
        y_offset = (self.height() - total_height) // 2

        for i, line in enumerate(lines):
            x = self.rect().center().x() - int(self.fontMetrics().horizontalAdvance(line) / 2)
            y = y_offset + i * line_height + self.fontMetrics().ascent()

            # Draw text shadow
            if self.shadow_offset > 0:
                painter.setPen(QtGui.QColor(0, 0, 0, 160))
                painter.drawText(QtCore.QRect(x, y - self.fontMetrics().ascent(), self.fontMetrics().horizontalAdvance(
                    line), line_height).translated(QtCore.QPoint(self.shadow_offset, self.shadow_offset)),
                                 QtCore.Qt.AlignmentFlag.AlignLeft, line)

            # Draw text outline
            path = QtGui.QPainterPath()
            path.addText(x, y, self.font(), line)

            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 4, QtCore.Qt.PenStyle.SolidLine,
                                      QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin))
            painter.drawPath(path)

            # Draw text fill
            painter.setPen(QtGui.QPen(self.color))
            painter.drawText(QtCore.QRect(x, y - self.fontMetrics().ascent(), self.fontMetrics().horizontalAdvance(
                line), line_height), QtCore.Qt.AlignmentFlag.AlignLeft, line)

        painter.end()


# noinspection PyAttributeOutsideInit
class GameWidget(QtWidgets.QFrame):
    def __init__(self, game_name, cover_path, font, parent=None):
        super().__init__(parent)
        self.game_name = game_name
        self.cover_path = cover_path
        self.font = font
        self.game_window: QtWidgets.QMainWindow | None = None
        self.is_passed = 0
        self.difficulty = SingletonConfig().config['minigame_difficulty']

        self.trophy_bg: QtWidgets.QFrame | None = None
        self.trophy: QtWidgets.QFrame | None = None
        self.max_score_label: StrokedLabel | None = None

        self.setupUi()
        self.set_trophy_state()

    def set_trophy_state(self):
        self.difficulty = SingletonConfig().config['minigame_difficulty']
        try:
            self.max_score, self.max_num, self.is_passed = \
                SingletonConfig().config['minigame_state'][self.difficulty][self.game_name][0][2:5]
        except KeyError:
            self.max_score, self.max_num, self.is_passed = 0, 0, 0
        self.delete_trophy_widgets()
        if self.is_passed == 0:
            return
        level = {1: 'bronze', 2: 'silver', 3: 'gold', 4: 'gold'}[self.is_passed]
        self.trophy_bg = QtWidgets.QFrame(self)
        self.trophy_bg.setObjectName(f"f_{self.game_name.replace(' ', '')}_trophy_bg")
        self.trophy_bg.setStyleSheet(f"""
        QFrame#f_{self.game_name.replace(' ', '')}_trophy_bg {{
            border-image: url(pic//trophybg.png) 0 0 0 0 stretch stretch;
            background-color: transparent;
        }}
        """)
        self.trophy = QtWidgets.QFrame(self)
        self.trophy.setObjectName(f"f_{self.game_name.replace(' ', '')}_trophy")
        self.trophy.setStyleSheet(f"""
        QFrame#f_{self.game_name.replace(' ', '')}_trophy {{
            border-image: url(pic//{level}.png) 0 0 0 0 stretch stretch;
            background-color: transparent;
        }}
        """)
        if self.is_passed == 4:
            self.max_score_label = StrokedLabel('', self.main_frame, shadow_offset=3, color=(70, 180, 80))
            self.max_score_label.setFont(self.font)
            self.max_score_label.setStyleSheet("background-color: transparent;")
            if 'Endless' in self.game_name:
                self.max_score_label.setText('score:\n' + str(self.max_score // 1000) + 'k')
            else:
                self.max_score_label.setText(str(2 ** self.max_num // 1000) + 'k')
            self.max_score_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.reset_trophy_pos()

    def delete_trophy_widgets(self):
        if self.trophy_bg is not None:
            self.trophy_bg.hide()
            self.trophy.hide()
            self.trophy_bg.deleteLater()
            self.trophy.deleteLater()
            self.trophy_bg = None
            self.trophy = None
        if self.max_score_label is not None:
            self.max_score_label.hide()
            self.max_score_label.deleteLater()
            self.max_score_label = None

    def reset_trophy_pos(self):
        if self.is_passed > 0:
            self.trophy_bg.setGeometry(0, 0, int(self.width() // 2.5), int(self.width() // 2.5))
            self.trophy.setGeometry(3, 3, int(self.trophy_bg.width() // 1.9), int(self.trophy_bg.width() // 1.9))
            self.trophy_bg.show()
            self.trophy.show()
            if self.max_score_label is not None:
                self.max_score_label.show()

    def setupUi(self):
        # 创建主框架
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName(f"f_{self.game_name.replace(' ', '')}")
        self.main_frame.setStyleSheet(f"""
        QFrame#f_{self.game_name.replace(' ', '')} {{
            border-image: url(pic//window.png) 0 0 0 0 stretch stretch;
            background-color: transparent;
        }}
        """)

        # 设置布局
        layout = QtWidgets.QVBoxLayout(self.main_frame)
        layout.setContentsMargins(10, 8, 10, 12)
        layout.setSpacing(0)

        # 设置封面图片
        self.cover = QtWidgets.QLabel(self.main_frame)
        self.cover.setPixmap(QtGui.QPixmap(self.cover_path).scaled(60, 60, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                                   QtCore.Qt.TransformationMode.SmoothTransformation))
        self.cover.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.cover)

        # 设置游戏名称
        self.name = QtWidgets.QLabel(self.game_name.replace(' ', '\n'), self.main_frame)
        self.name.setFont(self.font)
        self.name.setStyleSheet("color: rgb(50, 50, 95);")
        self.name.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.name.setFixedHeight(80)

        # 设置阴影效果
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(3)
        shadow.setOffset(1, 1)
        shadow.setColor(QtGui.QColor(0, 0, 0, 120))
        self.name.setGraphicsEffect(shadow)
        layout.addWidget(self.name)

    def enterEvent(self, event):
        self.main_frame.setStyleSheet(f"""
        QFrame#f_{self.game_name.replace(' ', '')} {{
            border-image: url(pic//window_highlight.png) 0 0 0 0 stretch stretch;
            background-color: transparent;
        }}
        """)
        self.name.setStyleSheet("color: rgb(220, 30, 5);")

    def leaveEvent(self, event):
        self.main_frame.setStyleSheet(f"""
        QFrame#f_{self.game_name.replace(' ', '')} {{
            border-image: url(pic//window.png) 0 0 0 0 stretch stretch;
            background-color: transparent;
        }}
        """)
        self.name.setStyleSheet("color: rgb(50, 50, 95);")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.main_frame.setGeometry(0, 0, self.width(), self.height())
        self.cover.setPixmap(
            QtGui.QPixmap(self.cover_path).scaled(self.main_frame.width(), self.main_frame.height() // 2,
                                                  QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                  QtCore.Qt.TransformationMode.SmoothTransformation))
        self.name.setFont(self.font)
        self.reset_trophy_pos()
        if self.is_passed == 4:
            self.max_score_label.setGeometry(self.main_frame.width() // 2 - self.max_score_label.width(
            ) // 2, self.main_frame.height() // 2 - self.name.height(), self.max_score_label.width(), self.name.height()
                                             )

    def mousePressEvent(self, event):
        self.move(self.x() + 2, self.y() + 2)
        QtWidgets.QApplication.processEvents()
        if self.game_window is not None and self.game_window.isVisible():
            self.game_window.setWindowState(
                self.game_window.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                | QtCore.Qt.WindowState.WindowActive)
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.game_name in ("Endless Factorization", "Endless Explosions", "Endless Giftbox"):
                module_name = "Endless"
                class_name = self.game_name.replace(' ', '') + 'Window'
            elif bool(re.search(r'\d', self.game_name)):
                module_name = re.sub(r'\d+', '', self.game_name.replace(' ', ''))
                class_name = module_name + 'Window'
            else:
                module_name = self.game_name.replace(' ', '')
                class_name = module_name + 'Window'
            frame_name = self.game_name.replace(' ', '') + 'Frame'
            try:
                module = importlib.import_module('minigames.' + module_name)
                window_class = getattr(module, class_name)
                frame_class = getattr(module, frame_name)
                self.game_window = window_class(minigame=self.game_name, frame_type=frame_class)
                self.game_window.closed.connect(self.set_trophy_state)
                self.game_window.show()
            except (ModuleNotFoundError, AttributeError) as e:
                print(f"Error loading {module_name}: {e}")

    def mouseReleaseEvent(self, event):
        self.move(self.x() - 2, self.y() - 2)  # 释放时返回原位


# noinspection PyAttributeOutsideInit
class CustomCheckBox(QtWidgets.QWidget):
    state_changed = QtCore.pyqtSignal()

    def __init__(self, text, font, parent=None, checkbox_size=24):
        super().__init__(parent)
        self.text = text
        self.checked = False
        self.checkbox_size = checkbox_size
        self.label_font = font
        self.setupUi()

    def setupUi(self):
        self.setMinimumSize(160, self.checkbox_size)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.checkbox_label = QtWidgets.QLabel(self)
        self.checkbox_label.setPixmap(QtGui.QPixmap("pic//checkbox0.png").scaled(
            self.checkbox_size, self.checkbox_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))
        self.checkbox_label.setFixedSize(self.checkbox_size, self.checkbox_size)
        self.checkbox_label.mousePressEvent = self.toggle  # 点击贴图时切换状态

        self.text_label = QtWidgets.QLabel(self.text, self)
        self.text_label.mousePressEvent = self.toggle  # 点击文本时切换状态
        self.text_label.setFont(self.label_font)
        self.text_label.setMinimumSize(80, self.checkbox_size)

        self.layout.addWidget(self.checkbox_label)
        self.layout.addWidget(self.text_label)

        self.setLayout(self.layout)

    def toggle(self, _):
        self.checked = not self.checked
        self.update_checkbox()

    def update_checkbox(self):
        if self.checked:
            self.checkbox_label.setPixmap(QtGui.QPixmap("pic//checkbox1.png").scaled(
                self.checkbox_size, self.checkbox_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
        else:
            self.checkbox_label.setPixmap(QtGui.QPixmap("pic//checkbox0.png").scaled(
                self.checkbox_size, self.checkbox_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
        SingletonConfig().config['minigame_difficulty'] = self.is_checked()
        self.state_changed.emit()

    def is_checked(self):
        return self.checked

    def set_checked(self, state):
        self.checked = state
        self.update_checkbox()

    def enterEvent(self, event):
        self.text_label.setStyleSheet("color: rgb(220, 30, 5);")

    def leaveEvent(self, event):
        self.text_label.setStyleSheet("color: rgb(0, 0, 0);")


# noinspection PyAttributeOutsideInit
class MenuWindow(QtWidgets.QWidget):
    def __init__(self, games, font):
        super().__init__()
        self.games = games
        self.font: QtGui.QFont = font

        self.setupUi()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        pixmap = QtGui.QPixmap("pic//background.jpg")
        scaled_pixmap = pixmap.scaled(self.size(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                      QtCore.Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap(self.rect(), scaled_pixmap)

    def setupUi(self):
        self.setWindowTitle('Game Menu')
        self.setFixedSize(1097, 707)
        self.GameWidgets: List[List[GameWidget]] = [[], []]

        # 设置布局
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setContentsMargins(60, 150, 60, 80)
        self.gridLayout.setSpacing(30)
        self.gridLayout.setObjectName("gridLayout")

        # 添加游戏框
        positions = [(i, j) for i in range(2) for j in range(5)]
        for position, game in zip(positions, self.games):
            game_widget = GameWidget(game, f"pic//covers//{game}.png", self.font)
            self.gridLayout.addWidget(game_widget, *position, 1, 1)
            self.GameWidgets[position[0]].append(game_widget)

        # 添加换页按钮
        self.next_button = QtWidgets.QPushButton(self)
        self.next_button.setIcon(QtGui.QIcon("pic//next.png"))
        self.next_button.setStyleSheet("background-color: transparent;")
        self.next_button.setFixedSize(50, 50)
        self.next_button.setIconSize(QtCore.QSize(50, 50))

        self.next_button.pressed.connect(self.button_pressed)  # type: ignore
        self.next_button.released.connect(self.button_released)  # type: ignore

        # 添加标题
        self.title = QtWidgets.QLabel(self)
        self.title.setFixedSize(160, 60)
        self.title.setFont(QtGui.QFont(self.font.family(), 16))
        self.title.setStyleSheet("""color: rgb(0, 0, 0); background-color: transparent;""")
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title.setText('Minigames')

        # 设置阴影效果
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(5)
        shadow.setOffset(2, 2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 160))
        self.title.setGraphicsEffect(shadow)

        # 添加困难模式复选框
        self.hardmode_checkbox = CustomCheckBox(self.tr('HardMode'), self.font, self, 26)
        self.hardmode_checkbox.set_checked(SingletonConfig().config['minigame_difficulty'])
        self.hardmode_checkbox.state_changed.connect(self.update_difficulty)

    def update_difficulty(self):
        for i in range(len(self.GameWidgets)):
            for widget in self.GameWidgets[i]:
                widget.set_trophy_state()

    def button_pressed(self):
        self.next_button.move(self.width() - 78, self.height() - 58)  # 下沉效果

    def button_released(self):
        self.next_button.move(self.width() - 80, self.height() - 60)  # 弹起效果

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.next_button.move(self.width() - 80, self.height() - 60)  # 放置在右下角
        self.title.move(self.width() // 2 - self.title.width() // 2, 32)
        self.hardmode_checkbox.move(40, self.height() - 50)  # 放置在左下角

    def reset_checkbox_dificulty(self):
        self.hardmode_checkbox.set_checked(SingletonConfig().config['minigame_difficulty'])


class MinigamesMainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        # 启动预加载线程
        self.preload_thread = PreloadThread()
        self.preload_thread.start()

        super().__init__(parent)
        font_id = QtGui.QFontDatabase.addApplicationFont("pic//fzjz.ttf")
        font_families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if font_families:
            font_family = font_families[0]
        else:
            font_family = "Sans Serif"
        self.font = QtGui.QFont(font_family, 12)

        self.stacked_widget = QtWidgets.QStackedWidget()

        self.page1 = MenuWindow(
            ["Design Master1", "Mystery Merge1", "Column Chaos", "Gravity Twist1", "Blitzkrieg", "Tricky Tiles",
             "Design Master2", "Shape Shifter", "Ferris Wheel", "Gravity Twist2"], self.font)
        self.page2 = MenuWindow(
            ["Design Master3", "Mystery Merge2", "Ice Age", "Isolated Island", "Design Master4",
             "Endless Factorization", "Endless Explosions", "Endless Giftbox", "Endless Hybrid", "Endless AirRaid"],
            self.font)

        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)

        self.page1.next_button.clicked.connect(self.show_page2)  # type: ignore
        self.page2.next_button.clicked.connect(self.show_page1)  # type: ignore

        self.setCentralWidget(self.stacked_widget)
        self.setWindowTitle('2048 Minigames')
        self.setFixedSize(1097, 707)
        self.show()

    def show_page1(self):
        self.stacked_widget.setCurrentWidget(self.page1)
        self.page1.reset_checkbox_dificulty()

    def show_page2(self):
        self.stacked_widget.setCurrentWidget(self.page2)
        self.page2.reset_checkbox_dificulty()

    def closeEvent(self, event):
        # 保存配置
        SingletonConfig().save_config(SingletonConfig().config)
        super().closeEvent(event)


class PreloadThread(QtCore.QThread):
    def run(self):
        from AIPlayer import EvilGen  # , BoardMover
        board = np.array([[0, 0, 2, 2],
                          [0, 0, 2, 4],
                          [32, 2, 8, 16],
                          [8, 128, 64, 8192]], dtype=np.int32)
        eg = EvilGen(board)
        _ = eg.gen_new_num(3)
        print("EvilGen preloading complete.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MinigamesMainWindow()
    main_window.show()
    sys.exit(app.exec_())
