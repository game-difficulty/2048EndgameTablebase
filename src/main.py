import os
import sys
import time
import multiprocessing

from PyQt5 import QtCore, QtWidgets, QtGui


def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    from MainMenu import MainMenuWindow

    main_win = MainMenuWindow()
    splash_pix = QtGui.QPixmap("pic/cover.jpg")
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowType.WindowStaysOnTopHint |
                                     QtCore.Qt.WindowType.FramelessWindowHint)
    splash.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    splash.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()

    time.sleep(4)
    main_win.show()
    splash.close()
    main_win.raise_()
    main_win.activateWindow()

    sys.exit(app.exec_())


# os.environ["NUMBA_SLP_VECTORIZE"] = "1"
# os.environ["NUMBA_DEVELOPER_MODE"] = "1"
# os.environ['NUMBA_THREADING_LAYER']='tbb'

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
