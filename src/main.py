import sys
import time
from multiprocessing import freeze_support

from PyQt5 import QtCore, QtWidgets, QtGui

from MainMenu import MainMenuWindow


def main():
    app = QtWidgets.QApplication(sys.argv)

    main_win = MainMenuWindow()
    splash_pix = QtGui.QPixmap("pic/cover.jpg")
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowType.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()

    time.sleep(4)
    main_win.show()
    splash.close()
    main_win.raise_()
    main_win.activateWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    freeze_support()
    main()
