import os
import pickle
import logging

import numpy as np
import cpuinfo


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
                'dis_32k': True,
                'compress': True,
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
logger.addHandler(console_handler)
