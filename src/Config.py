import os
import sys
import pickle
import logging
from typing import Callable, Dict, Tuple, Union

import numpy as np
import cpuinfo
from PyQt5 import QtWidgets
import Calculator
import Variants.vCalculator as vCalculator

PatternCheckFunc = Callable[[np.uint64], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]

formation_info: Dict[str, Tuple[int, PatternCheckFunc, ToFindFunc,
                                SuccessCheckFunc, np.typing.NDArray[np.uint64] | None]] = {
    'LL': [-131072 - 14, Calculator.is_LL_pattern, Calculator.minUL, Calculator.is_LL_success,
           np.array([np.uint64(0x1000000021ff12ff), np.uint64(0x0000000112ff21ff)], dtype=np.uint64)],
    '4431': [-131072 - 14, Calculator.is_4431_pattern, Calculator.re_self, Calculator.is_4431_success,
             np.array([np.uint64(0x10000000121f2fff), np.uint64(0x00000001121f2fff)], dtype=np.uint64)],
    '444': [-131072 - 2, Calculator.is_444_pattern, Calculator.re_self, Calculator.is_444_success,
            np.array([np.uint64(0x100000000000ffff), np.uint64(0x000000010000ffff)], dtype=np.uint64)],
    'free8': [-229376 - 16, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free9': [-196608 - 18, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free10': [-163840 - 20, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free11': [-131072 - 22, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free12': [-98304 - 24, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free13': [-65536 - 26, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free14': [-32768 - 28, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'L3': [-196608 - 8, Calculator.is_L3_pattern, Calculator.re_self, Calculator.is_L3_success,
           np.array([np.uint64(0x100000001fff2fff), np.uint64(0x000000011fff2fff)], dtype=np.uint64)],
    'L3t': [-196608 - 8, Calculator.is_L3t_pattern, Calculator.re_self, Calculator.is_L3t_success,
            np.array([np.uint64(0x100000001fff2fff), np.uint64(0x000000011fff2fff)], dtype=np.uint64)],
    '442': [-196608 - 8, Calculator.is_442_pattern, Calculator.re_self, Calculator.is_442_success,
            np.array([np.uint64(0x1000000012ffffff), np.uint64(0x0000000112ffffff)], dtype=np.uint64)],
    't': [-196608 - 8, Calculator.is_t_pattern, Calculator.re_self, Calculator.is_t_success,
          np.array([np.uint64(0x10000000f1fff2ff), np.uint64(0x00000001f1fff2ff)], dtype=np.uint64)],
    '4441': [-98304 - 4, Calculator.is_4441_pattern, Calculator.re_self, Calculator.is_4441_success,
             np.array([np.uint64(0x0000100000001fff), np.uint64(0x0001000000001fff)], dtype=np.uint64)],
    '4432': [-98304 - 16, Calculator.is_4432_pattern, Calculator.minUL, Calculator.is_4432_success,
             np.array([np.uint64(0x00001000121f21ff), np.uint64(0x00010000121f21ff)], dtype=np.uint64)],
    '4442': [-98304 - 8, Calculator.is_4442_pattern, Calculator.re_self, Calculator.is_4442_success,
             np.array([np.uint64(0x00001000000012ff), np.uint64(0x00010000000012ff)], dtype=np.uint64)],
    '4442f': [-98304 - 16, Calculator.is_4442f_pattern, Calculator.re_self, Calculator.is_4442f_success,
              np.array([np.uint64(0x00001000121f21ff), np.uint64(0x0001000012112fff)], dtype=np.uint64)],
    '4442ff': [-131072 - 14, Calculator.is_4442ff_pattern, Calculator.re_self, Calculator.is_4442ff_success,
              np.array([np.uint64(0x0000100012ff21ff), np.uint64(0x000100001212ffff),
                        np.uint64(0x00ff0012102100ff)], dtype=np.uint64)],
    'free8w': [-262144 - 14, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free9w': [-229376 - 16, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free10w': [-196608 - 18, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free11w': [-163840 - 20, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free12w': [-131072 - 22, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free13w': [-98304 - 24, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free14w': [-65536 - 26, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    'free15w': [-32768 - 28, Calculator.is_free_pattern, Calculator.min_all_symm, Calculator.is_free_success, None],
    '2x4': [-262144, vCalculator.is_variant_pattern, vCalculator.min24, vCalculator.is_2x4_success,
            np.array([np.uint64(0xffff00000000ffff)], dtype=np.uint64)],
    '3x3': [-229376, vCalculator.is_variant_pattern, vCalculator.min33, vCalculator.is_3x3_success,
            np.array([np.uint64(0x000f000f000fffff)], dtype=np.uint64)],
    '3x4': [-131072, vCalculator.is_variant_pattern, vCalculator.min34, vCalculator.is_3x4_success,
            np.array([np.uint64(0x000000000000ffff)], dtype=np.uint64)],
    '4432f': [-131072 - 14, Calculator.is_4432f_pattern, Calculator.minUL, Calculator.is_4432f_success,
              np.array([np.uint64(0x00001000121f2fff), np.uint64(0x00010000121ff2ff)], dtype=np.uint64)],
    "3433": [-98304 - 6, Calculator.is_3433_pattern, Calculator.re_self, Calculator.is_3433_success,
             np.array([np.uint64(0x100000000000f2ff), np.uint64(0x000000000001f2ff)], dtype=np.uint64)],
    "3442": [-98304 - 8, Calculator.is_3442_pattern, Calculator.re_self, Calculator.is_3442_success,
             np.array([np.uint64(0x10000000000ff21f), np.uint64(0x00000000100ff21f)], dtype=np.uint64)],
    "3432": [-131072 - 6, Calculator.is_3432_pattern, Calculator.minUL, Calculator.is_3432_success,
             np.array([np.uint64(0x10000000000ff2ff), np.uint64(0x00000000100ff2ff)], dtype=np.uint64)],
    "2433": [-131072 - 6, Calculator.is_2433_pattern, Calculator.re_self, Calculator.is_2433_success,
             np.array([np.uint64(0x10000000f000f2ff), np.uint64(0x00000000f001f2ff)], dtype=np.uint64)],
    "movingLL": [-131072 - 14, Calculator.is_movingLL_pattern, Calculator.min_all_symm, Calculator.is_movingLL_success,
                 np.array([np.uint64(0x100000001ff12ff2), np.uint64(0x1000000012ff21ff)], dtype=np.uint64)],
}

pattern_32k_tiles_map: Dict[str, list] = {
    # 32k, free32k, fix32k_pos,
    '': [0, 0, np.array([], dtype=np.uint8)],
    'LL': [4, 0, np.array([0, 4, 16, 20], dtype=np.uint8)],
    '4431': [4, 0, np.array([0, 4, 8, 16], dtype=np.uint8)],
    '444': [4, 0, np.array([0, 4, 8, 12], dtype=np.uint8)],
    'free8': [7, 7, np.array([], dtype=np.uint8)],
    'free9': [6, 6, np.array([], dtype=np.uint8)],
    'free10': [5, 5, np.array([], dtype=np.uint8)],
    'free11': [4, 4, np.array([], dtype=np.uint8)],
    'free12': [3, 3, np.array([], dtype=np.uint8)],
    'free13': [2, 2, np.array([], dtype=np.uint8)],
    'free14': [1, 1, np.array([], dtype=np.uint8)],
    'L3': [6, 0, np.array([0, 4, 8, 16, 20, 24], dtype=np.uint8)],
    'L3t': [6, 2, np.array([4, 8, 20, 24], dtype=np.uint8)],
    '442': [6, 3, np.array([4, 16, 20], dtype=np.uint8)],
    't': [6, 2, np.array([0, 4, 16, 20], dtype=np.uint8)],
    '4441': [3, 0, np.array([0, 4, 8], dtype=np.uint8)],
    '4432': [3, 0, np.array([0, 4, 16], dtype=np.uint8)],
    '4432f': [4, 1, np.array([0, 4, 16], dtype=np.uint8)],
    '4442': [2, 0, np.array([0, 4], dtype=np.uint8)],
    '4442f': [3, 1, np.array([0, 4], dtype=np.uint8)],
    '4442ff': [4, 2, np.array([0, 4], dtype=np.uint8)],
    'free8w': [8, 8, np.array([], dtype=np.uint8)],
    'free9w': [7, 7, np.array([], dtype=np.uint8)],
    'free10w': [6, 6, np.array([], dtype=np.uint8)],
    'free11w': [5, 5, np.array([], dtype=np.uint8)],
    'free12w': [4, 4, np.array([], dtype=np.uint8)],
    'free13w': [3, 3, np.array([], dtype=np.uint8)],
    'free14w': [2, 2, np.array([], dtype=np.uint8)],
    'free15w': [1, 1, np.array([], dtype=np.uint8)],
    '2x4': [0, 0, np.array([], dtype=np.uint8)],
    '3x3': [0, 0, np.array([], dtype=np.uint8)],
    '3x4': [0, 0, np.array([], dtype=np.uint8)],
    "3433": [3, 1, np.array([0, 4], dtype=np.uint8)],
    "3442": [3, 0, np.array([0, 12, 16], dtype=np.uint8)],
    "3432": [4, 1, np.array([0, 4, 16], dtype=np.uint8)],
    "2433": [4, 2, np.array([0, 4], dtype=np.uint8)],
    "movingLL": [4, 4, np.array([], dtype=np.uint8)],
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
                'do_animation': [True, True],
                'game_state': [np.uint64(0), 0, 0],
                'dis_32k': False,
                'compress': False,
                'optimal_branch_only': False,
                'compress_temp_files': False,
                'SmallTileSumLimit': 56,
                'advanced_algo': False,
                'deletion_threshold': 0,
                'font_size_factor': 100,
                'minigame_state': [dict(), dict()],  # [盘面，得分，最高分，最大数，是否曾过关, 新数位置], []
                'power_ups_state': [dict(), dict()],
                'minigame_difficulty': 1,
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
