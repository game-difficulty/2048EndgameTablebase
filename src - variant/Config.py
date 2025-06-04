import os
import sys
import pickle
import logging
from typing import Callable

import numpy as np
import cpuinfo
from PyQt5 import QtWidgets
import Calculator

PatternCheckFunc = Callable[[np.uint64], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]

formation_info = {
    'LL': [-14, Calculator.is_LL_pattern, Calculator.minUL, np.uint64(0x210012),
             np.array([np.uint64(0x1000000012002100), np.uint64(0x1000000021001200)], dtype=np.uint64)],
    '4411': [-8, Calculator.is_4411_pattern, Calculator.re_self, np.uint64(0x1210212),
           np.array([np.uint64(0x1000000010002000), np.uint64(0x1000000020001000)], dtype=np.uint64)],
}

# noinspection PyAttributeOutsideInit
class SingletonConfig:
    _instance = None  # 用于存储单例实例的类属性
    config_file_path = os.path.join(os.path.dirname(__file__), 'config')  # 配置文件路径
    use_avx = 1

    def __new__(cls):
        # 检查是否已经有实例存在
        if cls._instance is None:
            cls._instance = super(SingletonConfig, cls).__new__(cls)
            # 初始化配置数据
            cls._instance.config = cls.load_config()
            cls.check_cpuinfo()
        return cls._instance

    @classmethod
    def load_config(cls, filename=None):
        if filename is None:
            filename = cls.config_file_path

        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as file:
                    return pickle.load(file)
            except EOFError:
                pass
        return {'filepath_map': dict(),
                'colors': ['#043c24', '#06643d', '#1b955b', '#20c175', '#fc56a0', '#e4317f', '#e900ad', '#bf009c',
                           '#94008a', '#6a0079', '#3f0067', '#00406b', '#006b9a', '#0095c8', '#00c0f7', '#00c0f7'] + [
                              '#000000'] * 20,
                'demo_speed': 40,
                '4_spawn_rate': 0.1,
                'do_animation': [False, False],
                'dis_32k': False,
                'compress': False,
                'font_size_factor': 100,
                }

    @classmethod
    def save_config(cls, config, filename=None):
        if filename is None:
            filename = cls.config_file_path

        with open(filename, 'wb') as file:
            pickle.dump(config, file)

    @classmethod
    def check_cpuinfo(cls):
        # 获取当前cpu指令集信息
        info = cpuinfo.get_cpu_info()
        if 'avx512f' in info['flags'] and 'avx512vl' in info['flags'] and \
                ('avx512dq' in info['flags'] or ('avx512bw' in info['flags'] and 'avx512vbmi2' in info['flags'])):
            cls.use_avx = 2
            logger.info("CPU info: AVX512 is supported")
        elif 'avx2' in info['flags']:
            cls.use_avx = 1
            logger.info("CPU info: AVX2 is supported; AVX512 is not supported")
        else:
            cls.use_avx = 0
            logger.info("CPU info: AVX512/AVX2 is not supported")
        return


# 创建日志记录
logger = logging.getLogger('debug_logger')

logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 创建文件处理器，设置级别为WARNING，用于输出到文件
file_handler = logging.FileHandler('logger.txt')
file_handler.setLevel(logging.WARNING)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 将处理器添加到日志记录器中
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # 防止应用崩溃
    if not issubclass(exc_type, KeyboardInterrupt):
        QtWidgets.QMessageBox.critical(None, "Error", f"An unexpected error occurred: {exc_value}\n {exc_traceback}")


sys.excepthook = handle_exception

