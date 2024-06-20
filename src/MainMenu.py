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
        self.train_window = None
        self.test_window = None
        self.settings_window = None
        self.view_window = None

    def setupUi(self):
        self.setObjectName("self")
        self.setWindowIcon(QtGui.QIcon(r"pic\2048_2.ico"))
        self.resize(360, 160)
        self.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.innerContainer = QtWidgets.QWidget(self.centralwidget)
        self.innerContainerLayout = QtWidgets.QVBoxLayout(self.innerContainer)
        self.verticalLayout.addWidget(self.innerContainer, alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)

        self.Game = QtWidgets.QPushButton(self.centralwidget)
        self.Game.setObjectName("Game")
        self.Game.clicked.connect(self.openGameWindow)
        self.Game.setMaximumSize(QtCore.QSize(480, 60))
        self.Game.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Game)
        self.Practise = QtWidgets.QPushButton(self.centralwidget)
        self.Practise.setObjectName("Practise")
        self.Practise.clicked.connect(self.openTrainWindow)
        self.Practise.setMaximumSize(QtCore.QSize(480, 60))
        self.Practise.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Practise)
        self.Test = QtWidgets.QPushButton(self.centralwidget)
        self.Test.setObjectName("Test")
        self.Test.clicked.connect(self.openTestWindow)
        self.Test.setMaximumSize(QtCore.QSize(480, 60))
        self.Test.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Test)
        self.Settings = QtWidgets.QPushButton(self.centralwidget)
        self.Settings.setObjectName("Settings")
        self.Settings.clicked.connect(self.openSettingsWindow)
        self.Settings.setMaximumSize(QtCore.QSize(480, 60))
        self.Settings.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Settings)
        self.Help = QtWidgets.QPushButton(self.centralwidget)
        self.Help.setObjectName("Settings")
        self.Help.clicked.connect(self.openHelpWindow)
        self.Help.setMaximumSize(QtCore.QSize(480, 60))
        self.Help.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Help)
        self.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.width() > 360:
            self.innerContainer.setMaximumWidth(360)
            self.verticalLayout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        else:
            self.innerContainer.setFixedWidth(self.width())
            self.verticalLayout.setAlignment(QtCore.Qt.AlignTop)

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
        if self.game_window.windowState() & QtCore.Qt.WindowMinimized:
            self.game_window.setWindowState(
                self.game_window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.game_window.show()
        self.game_window.activateWindow()
        self.game_window.raise_()
        self.show_and_center_window(self.game_window)
        self.game_window.gameframe.update_all_frame(self.game_window.gameframe.board)

    def openTrainWindow(self):
        if self.train_window is None:
            from Trainer import TrainWindow
            self.train_window = TrainWindow()
        if self.train_window.windowState() & QtCore.Qt.WindowMinimized:
            self.train_window.setWindowState(
                self.train_window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.train_window.show()
        self.train_window.activateWindow()
        self.train_window.raise_()
        self.show_and_center_window(self.train_window)
        self.train_window.gameframe.update_all_frame(self.train_window.gameframe.board)

    def openTestWindow(self):
        if self.test_window is None:
            from Tester import TestWindow
            self.test_window = TestWindow()
        if self.test_window.windowState() & QtCore.Qt.WindowMinimized:
            self.test_window.setWindowState(
                self.test_window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.test_window.show()
        self.test_window.activateWindow()
        self.test_window.raise_()
        self.show_and_center_window(self.test_window)
        self.test_window.gameframe.update_all_frame(self.test_window.gameframe.board)

    def openSettingsWindow(self):
        if self.settings_window is None:
            from Settings import SettingsWindow
            self.settings_window = SettingsWindow()
        if self.settings_window.windowState() & QtCore.Qt.WindowMinimized:
            self.settings_window.setWindowState(
                self.settings_window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.settings_window.show()
        self.settings_window.activateWindow()
        self.show_and_center_window(self.settings_window)
        self.settings_window.raise_()

    def openHelpWindow(self):
        if self.view_window is None:
            from MDViewer import MDViewer
            self.view_window = MDViewer('help.md')
        if self.view_window.windowState() & QtCore.Qt.WindowMinimized:
            self.view_window.setWindowState(
                self.view_window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.view_window.show()
        self.view_window.activateWindow()
        self.show_and_center_window(self.view_window)
        self.view_window.raise_()

    # noinspection PyTypeChecker
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainMenu", "2048"))
        self.Game.setText(_translate("MainMenu", "Game"))
        self.Practise.setText(_translate("MainMenu", "Practise"))
        self.Test.setText(_translate("MainMenu", "Test"))
        self.Settings.setText(_translate("MainMenu", "Settings"))
        self.Help.setText(_translate("MainMenu", "Help"))

    def closeEvent(self, event):
        SingletonConfig().save_config(SingletonConfig().config)
        if self.preload_thread.isRunning():
            self.preload_thread.quit()
            self.preload_thread.wait()
        super().closeEvent(event)


class PreloadThread(QtCore.QThread):
    def run(self):
        from BoardMover import BoardMoverWithScore  # , BoardMover
        bm = BoardMoverWithScore()
        bm.move_all_dir(np.uint64(0x010120342216902ac))
        bm.move_board(np.uint64(0x010120342216902ac), 2)
        # bm = BoardMover()
        # bm.move_all_dir(np.uint64(0x010120342216902ac))
        print("Preloading complete.")
