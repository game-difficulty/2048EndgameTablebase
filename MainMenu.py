from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui


from Gamer import GameWindow
from Settings import SettingsWindow
from Trainer import TrainWindow
from Config import SingletonConfig
from HTMLViewer import HTMLViewer


# noinspection PyAttributeOutsideInit
class MainMenuWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.game_window = GameWindow()
        self.train_window = TrainWindow()
        self.settings_window = SettingsWindow()
        self.view_window = HTMLViewer('help.htm')

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

    def openGameWindow(self):
        self.game_window.show()
        self.game_window.gameframe.update_all_frame(self.game_window.gameframe.board)

    def openTrainWindow(self):
        self.train_window.show()

    def openSettingsWindow(self):
        self.settings_window.show()

    def openHelpWindow(self):
        self.view_window.show()

    # noinspection PyTypeChecker
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainMenu", "2048"))
        self.Game.setText(_translate("MainMenu", "Game"))
        self.Practise.setText(_translate("MainMenu", "Practise"))
        self.Settings.setText(_translate("MainMenu", "Settings"))
        self.Help.setText(_translate("MainMenu", "Help"))

    def closeEvent(self, event):
        SingletonConfig().save_config(SingletonConfig().config)
        event.accept()
