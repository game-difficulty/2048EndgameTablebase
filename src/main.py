import os
import sys
import multiprocessing

from PyQt5 import QtCore, QtWidgets, QtGui


def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

    # 加载默认字体
    QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform == 'win32':
        default_font = QtGui.QFont("Segoe UI")
    else:
        default_font = QtGui.QFont("Noto Sans")
    app.setFont(default_font)

    # 加载默认语言
    from Config import SingletonConfig, logger
    if SingletonConfig().config.get('language', 'en') == 'zh':
        SingletonConfig.apply_language('zh')

    # 加载ClearSans字体
    font_files = [
        r"font/ClearSans/ClearSans-Bold.ttf",
        r"font/ClearSans/ClearSans-Regular.ttf"
    ]
    for font_file in font_files:
        # 返回值为字体ID，负数表示失败
        if QtGui.QFontDatabase.addApplicationFont(font_file) < 0:
            logger.warning(f"Font loading failure: {font_file}")

    splash_pix = QtGui.QPixmap("pic/cover.jpg")
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowType.WindowStaysOnTopHint |
                                     QtCore.Qt.WindowType.FramelessWindowHint)
    splash.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    splash.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
    splash.setMask(splash_pix.mask())
    splash.show()
    QtWidgets.QApplication.processEvents()

    from MainMenu import MainMenuWindow

    main_win = MainMenuWindow()

    splash.close()  # 关闭启动画面
    main_win.show()  # 显示主窗口
    main_win.raise_()  # 置顶窗口
    main_win.activateWindow()  # 激活窗口获取焦点

    sys.exit(app.exec_())


# os.environ["NUMBA_SLP_VECTORIZE"] = "1"
# os.environ["NUMBA_DEVELOPER_MODE"] = "1"
# os.environ['NUMBA_THREADING_LAYER']='tbb'

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
