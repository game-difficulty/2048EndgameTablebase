import sys
import time
import multiprocessing

from PyQt5 import QtCore, QtWidgets, QtGui


def main():
    app = QtWidgets.QApplication(sys.argv)
    from MainMenu import MainMenuWindow
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
    multiprocessing.freeze_support()
    main()
