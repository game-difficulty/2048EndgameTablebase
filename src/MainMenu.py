import sys
import os

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui

from Config import SingletonConfig


# noinspection PyAttributeOutsideInit
class MainMenuWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # 启动预加载线程
        self.preload_thread = PreloadThread()
        self.preload_thread.start()

        super().__init__()
        self.setupUi()
        self.game_window = None
        self.minigames_window = None
        from Trainer import TrainWindow
        self.train_window = TrainWindow()  # 方便回放直接跳转
        self.test_window = None
        self.settings_window = None
        self.view_window = None

        if hasattr(sys, '_MEIPASS'):
            # noinspection PyProtectedMember
            current_dir = sys._MEIPASS
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        variants_dir = os.path.join(current_dir, 'Variants')
        if variants_dir not in sys.path:
            sys.path.insert(0, variants_dir)

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QtGui.QIcon(r"pic\2048_2.ico"))
        self.resize(560, 480)
        self.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(10, 50, 10, 10)  # 左，上，右，下边距
        self.innerContainer = QtWidgets.QWidget(self.centralwidget)
        self.innerContainerLayout = QtWidgets.QVBoxLayout(self.innerContainer)
        self.verticalLayout.addWidget(self.innerContainer,
                                      alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.Game = QtWidgets.QPushButton()
        self.Game.setObjectName("Game")
        self.Game.clicked.connect(self.openGameWindow)  # type: ignore
        self.Game.setMaximumSize(QtCore.QSize(600, 80))
        self.Game.setStyleSheet("font: 640 14pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Game)
        self.Minigames = QtWidgets.QPushButton(self.centralwidget)
        self.Minigames.setObjectName("Minigames")
        self.Minigames.clicked.connect(self.openMinigamesWindow)  # type: ignore
        self.Minigames.setMaximumSize(QtCore.QSize(600, 80))
        self.Minigames.setStyleSheet("font: 640 14pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Minigames)
        self.Practise = QtWidgets.QPushButton(self.centralwidget)
        self.Practise.setObjectName("Practise")
        self.Practise.clicked.connect(self.openTrainWindow)  # type: ignore
        self.Practise.setMaximumSize(QtCore.QSize(600, 80))
        self.Practise.setStyleSheet("font: 640 14pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Practise)
        self.Test = QtWidgets.QPushButton(self.centralwidget)
        self.Test.setObjectName("Test")
        self.Test.clicked.connect(self.openTestWindow)  # type: ignore
        self.Test.setMaximumSize(QtCore.QSize(600, 80))
        self.Test.setStyleSheet("font: 640 14pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Test)
        self.Settings = QtWidgets.QPushButton(self.centralwidget)
        self.Settings.setObjectName("Settings")
        self.Settings.clicked.connect(self.openSettingsWindow)  # type: ignore
        self.Settings.setMaximumSize(QtCore.QSize(600, 80))
        self.Settings.setStyleSheet("font: 640 14pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Settings)
        self.Help = QtWidgets.QPushButton(self.centralwidget)
        self.Help.setObjectName("Settings")
        self.Help.clicked.connect(self.openHelpWindow)  # type: ignore
        self.Help.setMaximumSize(QtCore.QSize(600, 80))
        self.Help.setStyleSheet("font: 640 14pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Help)
        # 创建一个QLabel，用于显示引导文字
        self.info_label = QtWidgets.QLabel(self.centralwidget)
        self.info_label.setObjectName("InfoLabel")
        self.info_label.setOpenExternalLinks(True)
        self.info_label.setStyleSheet("font: 420 12pt \"Cambria\";")
        self.info_label.setMaximumWidth(600)
        self.info_label.setMinimumHeight(100)
        self.info_label.setWordWrap(True)  # 启用自动换行
        self.info_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.innerContainerLayout.addWidget(self.info_label)
        self.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.width() > 600:
            self.innerContainer.setMaximumWidth(600)
            self.verticalLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        else:
            self.innerContainer.setFixedWidth(self.width())
            self.verticalLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

    @staticmethod
    def show_and_center_window(window):
        # 检测窗口位置
        screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
        screen_geometry = QtWidgets.QApplication.desktop().screenGeometry(screen)
        window_geometry = window.frameGeometry()

        # 检测窗口是否在屏幕之外
        if (window_geometry.left() < screen_geometry.left() or
                window_geometry.right() > screen_geometry.right() or
                window_geometry.top() < screen_geometry.top() or
                window_geometry.bottom() > screen_geometry.bottom()):
            # 将窗口移动到屏幕中心
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            window.move(window_geometry.topLeft())

    def openGameWindow(self):
        if self.game_window is None:
            from Gamer import GameWindow
            self.game_window = GameWindow()
        if self.game_window.windowState() & QtCore.Qt.WindowState.WindowMinimized:
            self.game_window.setWindowState(
                self.game_window.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                | QtCore.Qt.WindowState.WindowActive)
        self.game_window.show()
        self.game_window.activateWindow()
        self.game_window.raise_()
        self.show_and_center_window(self.game_window)
        self.game_window.gameframe.update_all_frame(self.game_window.gameframe.board)

    def openMinigamesWindow(self):
        if self.minigames_window is None:
            if hasattr(sys, '_MEIPASS'):
                # noinspection PyProtectedMember
                current_dir = sys._MEIPASS
            else:
                current_dir = os.path.dirname(os.path.abspath(__file__))

            minigames_dir = os.path.join(current_dir, 'minigames')
            if minigames_dir not in sys.path:
                sys.path.insert(0, minigames_dir)

            from minigames.MinigameMenu import MinigamesMainWindow
            self.minigames_window = MinigamesMainWindow()

        if self.minigames_window.windowState() & QtCore.Qt.WindowState.WindowMinimized:
            self.minigames_window.setWindowState(
                self.minigames_window.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                | QtCore.Qt.WindowState.WindowActive)
        self.minigames_window.show()
        self.minigames_window.activateWindow()
        self.minigames_window.raise_()
        self.show_and_center_window(self.minigames_window)

    def openTrainWindow(self):
        if self.train_window.windowState() & QtCore.Qt.WindowState.WindowMinimized:
            self.train_window.setWindowState(
                self.train_window.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                | QtCore.Qt.WindowState.WindowActive)
        self.train_window.show()
        self.train_window.activateWindow()
        self.train_window.raise_()
        self.show_and_center_window(self.train_window)
        self.train_window.gameframe.update_all_frame(self.train_window.gameframe.board)

    def openTestWindow(self):
        if self.test_window is None:
            from Tester import TestWindow
            self.test_window = TestWindow()
        if self.test_window.windowState() & QtCore.Qt.WindowState.WindowMinimized:
            self.test_window.setWindowState(
                self.test_window.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                | QtCore.Qt.WindowState.WindowActive)
        self.test_window.show()
        self.test_window.activateWindow()
        self.test_window.raise_()
        self.show_and_center_window(self.test_window)
        self.test_window.gameframe.update_all_frame(self.test_window.gameframe.board)

    def openSettingsWindow(self):
        if self.settings_window is None:
            from Settings import SettingsWindow
            self.settings_window = SettingsWindow()
        if self.settings_window.windowState() & QtCore.Qt.WindowState.WindowMinimized:
            self.settings_window.setWindowState(
                self.settings_window.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                | QtCore.Qt.WindowState.WindowActive)
        self.settings_window.show()
        self.settings_window.activateWindow()
        self.show_and_center_window(self.settings_window)
        self.settings_window.raise_()

    def openHelpWindow(self):
        if self.view_window is None:
            from MDViewer import MDViewer
            if SingletonConfig().config['language'] == 'zh':
                self.view_window = MDViewer('helpZH.md')
            else:
                self.view_window = MDViewer('help.md')
        if self.view_window.windowState() & QtCore.Qt.WindowState.WindowMinimized:
            self.view_window.setWindowState(
                self.view_window.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                | QtCore.Qt.WindowState.WindowActive)
        self.view_window.show()
        self.view_window.activateWindow()
        self.show_and_center_window(self.view_window)
        self.view_window.raise_()

    # noinspection PyTypeChecker
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainMenu", "2048"))
        self.Game.setText(_translate("MainMenu", "Game"))
        self.Minigames.setText(_translate("MainMenu", "Minigames"))
        self.Practise.setText(_translate("MainMenu", "Practise"))
        self.Test.setText(_translate("MainMenu", "Test"))
        self.Settings.setText(_translate("MainMenu", "Settings"))
        self.Help.setText(_translate("MainMenu", "Help"))
        self.info_label.setText(_translate("MainMenu",
            'Support this project by giving it a star on '
            '<a href="https://github.com/game-difficulty/2048EndgameTablebase">GitHub</a>.'
        ))

    def closeEvent(self, event):
        SingletonConfig().save_config(SingletonConfig().config)
        if self.preload_thread.isRunning():
            self.preload_thread.quit()
            self.preload_thread.wait()
        super().closeEvent(event)


class PreloadThread(QtCore.QThread):
    def run(self):
        from Config import SingletonConfig
        SingletonConfig()
        import BoardMover as bm
        bm.move_all_dir(np.uint64(0x10120342216902ac))
        for d in (1, 2, 3, 4):
            bm.decode_board(bm.s_move_board(np.uint64(0x10120342216902ac), d)[0])
        import Variants.vBoardMover as vbm
        vbm.move_all_dir(np.uint64(0x10120342216ff2ac))
        for d in (1, 2, 3, 4):
            vbm.decode_board(vbm.s_move_board(np.uint64(0x10120342216ff2ac), d)[0])
        print("Preloading complete.")
