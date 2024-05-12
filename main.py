import sys

from PyQt5 import QtCore, QtWidgets, QtGui


def main():
    app = QtWidgets.QApplication(sys.argv)

    splash_pix = QtGui.QPixmap("pic/cover.jpg")
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()

    from MainMenu import MainMenuWindow

    main_win = MainMenuWindow()
    main_win.show()
    splash.close()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
