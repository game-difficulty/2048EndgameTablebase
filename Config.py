import os
import pickle

import numpy as np


# noinspection PyAttributeOutsideInit
class SingletonConfig:
    _instance = None  # 用于存储单例实例的类属性

    def __new__(cls):
        # 检查是否已经有实例存在
        if cls._instance is None:
            cls._instance = super(SingletonConfig, cls).__new__(cls)
            # 初始化配置数据
            cls._instance.config = cls.load_config()
        return cls._instance

    @staticmethod
    def load_config(filename='config'):
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                return pickle.load(file)
        else:
            return {
                'filepath_map': dict(),
                'colors': ['#043c24', '#06643d', '#1b955b', '#20c175', '#fc56a0', '#e4317f', '#e900ad', '#bf009c',
                           '#94008a', '#6a0079', '#3f0067', '#00406b', '#006b9a', '#0095c8', '#00c0f7', '#00c0f7'] + [
                              '#ffffff'] * 20,
                'demo_speed': 40,
                '4_spawn_rate': 0.1,
                'do_animation': [True, False],
                'game_state':[np.uint64(0), 0, 0],
                'dis_32k': True,
                'compress': True,
                'font_size_factor': 100,
            }

    @staticmethod
    def save_config(config, filename='config'):
        with open(filename, 'wb') as file:
            pickle.dump(config, file)

    def reload_config(self, filename='config'):
        self.config = self.load_config(filename)
