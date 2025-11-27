"""用于跨页面信号"""
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


class ProgressSignal(QObject):
    # 更新计算进度
    progress_updated = pyqtSignal(int, int)


progress_signal = ProgressSignal()


class PracticeSignal(QObject):
    # 回放界面跳转练习界面
    board_update = pyqtSignal(np.uint64, str)


practice_signal = PracticeSignal()

