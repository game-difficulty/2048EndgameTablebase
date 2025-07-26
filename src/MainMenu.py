from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui

from Config import SingletonConfig


# noinspection PyAttributeOutsideInit
class MainMenuWindow(QtWidgets.QMainWindow):
    def __init__(self):

        super().__init__()
        self.setupUi()
        from Trainer import TrainWindow
        self.train_window = TrainWindow()  # 方便回放直接跳转
        self.settings_window = None

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

        self.Practise = QtWidgets.QPushButton(self.centralwidget)
        self.Practise.setObjectName("Practise")
        self.Practise.clicked.connect(self.openTrainWindow)  # type: ignore
        self.Practise.setMaximumSize(QtCore.QSize(600, 80))
        self.Practise.setStyleSheet("font: 640 14pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Practise)

        self.Settings = QtWidgets.QPushButton(self.centralwidget)
        self.Settings.setObjectName("Settings")
        self.Settings.clicked.connect(self.openSettingsWindow)  # type: ignore
        self.Settings.setMaximumSize(QtCore.QSize(600, 80))
        self.Settings.setStyleSheet("font: 640 14pt \"Cambria\";")
        self.innerContainerLayout.addWidget(self.Settings)

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

    # noinspection PyTypeChecker
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainMenu", "2048"))
        self.Practise.setText(_translate("MainMenu", "Practise"))
        self.Settings.setText(_translate("MainMenu", "Settings"))


    def closeEvent(self, event):
        SingletonConfig().save_config(SingletonConfig().config)
        super().closeEvent(event)
