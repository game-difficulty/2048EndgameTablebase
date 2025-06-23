import logging
import os
import pickle
import sys
from typing import Callable, Dict, Tuple

import cpuinfo
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import QFontDatabase
from numpy.typing import NDArray

import Calculator
import Variants.vCalculator as vCalculator

PatternCheckFunc = Callable[[np.uint64], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]


category_info = {'10 space': ['L3', 'L3t', '442', 't'],
                 '12 space': ['444', '4431', 'LL', '4432f', '4442ff'],
                 'free': [f'free{i}w' for i in range(8, 15)],
                 'free halfway': [f'free{i}' for i in range(8, 12)],
                 'variant': ['2x4', '3x3', '3x4'],
                 'others': ['4442', '4442f', '4441']}


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
    '4442': [-65536 - 8, Calculator.is_4442_pattern, Calculator.re_self, Calculator.is_4442_success,
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
    '4432ff': [-163840 - 10, Calculator.is_4432ff_pattern, Calculator.minUL, Calculator.is_4432ff_success,
              np.array([np.uint64(0x00001000112fffff), np.uint64(0x00001000f12ff1ff)], dtype=np.uint64)],
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
    '4432ff': [5, 2, np.array([0, 4, 16], dtype=np.uint8)],
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


theme_map = {
          'Default': ['#043c24', '#06643d', '#1b955b', '#20c175', '#fc56a0', '#e4317f', '#e900ad', '#bf009c',
                      '#94008a', '#6a0079', '#3f0067', '#00406b', '#006b9a', '#0095c8', '#00c0f7', '#00c0f7'],
          'Chrome': ['#eee4da', '#ede0c7', '#f2b17a', '#f59563', '#ff7455', '#f55f44', '#f1cf5f', '#f3cb49',
                     '#f3c82c', '#f3c300', '#ebb800', '#33b4a9', '#27a59a', '#12998d', '#24a7f5', '#24a7f5'],
          'Classic': ['#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', '#f65e3b', '#edcf72', '#edcc61',
                      '#edc850', '#edc53f', '#edc22e', '#3c3a32', '#3c3a32', '#3c3a32', '#3c3a32', '#3c3a32'],
          'Coral cave': ['#eef5e8', '#dcedc8', '#aed581', '#6fab3c', '#4b942f', '#247538', '#195a3f', '#166b60',
                         '#1b6e6e', '#1d8287', '#01949e', '#008bb3', '#007ec2', '#506ac1', '#8a4daa', '#8a4daa'],
          'Dice': ['#40007f', '#4c0099', '#5900b2', '#6600cc', '#7300e5', '#8000ff', '#8c19ff', '#00418a',
                   '#004b9e', '#0054b2', '#005ec6', '#196ecb', '#337ed1', '#4c8ed7', '#e4317f', '#e4317f'],
          'Eclipse': ['#f4da98', '#cabb90', '#a09b89', '#767c81', '#6a7278', '#5e6770', '#525d67', '#47525e',
                      '#3b4855', '#2f3d4d', '#233334', '#fffff0', '#ffffcb', '#ffffa5', '#000000', '#000000'],
          'Eight': ['#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', '#f65e3b', '#edcf72', '#edcc61',
                    '#edc850', '#edc53f', '#edc22e', '#77a136', '#2db388', '#2d83b3', '#2d2db3', '#2d2db3'],
          'Estoty': ['#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', '#f65e3b', '#edcf72', '#edcc61',
                     '#edc850', '#edc53f', '#edc22e', '#de6d6e', '#dc595f', '#d54e42', '#80b2d1', '#80b2d1'],
          'Fruitsational': ['#eee6d6', '#eedab3', '#ffbd5b', '#f5982d', '#ed7520', '#e34e13', '#cf3e34', '#cc3a5a',
                            '#bf3976', '#bd3b9f', '#7c18ff', '#2a7cd9', '#09a5cc', '#0cbfcc', '#15d4a1', '#15d4a1'],
          'Galaxy': ['#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', '#f65e3b', '#edcf72', '#edcc61',
                     '#edc850', '#edc53f', '#edc22e', '#e67272', '#d05480', '#bd398e', '#991399', '#991399'],
          'Green': ['#eef5e8', '#dcedc8', '#aed581', '#8fc94c', '#6fab3c', '#4b942f', '#018c79', '#02a8b4',
                    '#0992c4', '#0271c7', '#029acc', '#01949e', '#1d8287', '#1b6e6e', '#16666b', '#16666b'],
          'Hobbel': ['#a4a3cb', '#8a8fb5', '#81799e', '#c599aa', '#e3998c', '#f3b177', '#6f4762', '#8e536f',
                     '#a25670', '#ba5672', '#cd5b74', '#fe595d', '#f57246', '#e88d3a', '#f9c65b', '#f9c65b'],
          'Holiday': ['#eee4da', '#fff18c', '#ffe840', '#ffe212', '#f5c300', '#f6af02', '#005e00', '#007400',
                      '#008900', '#009f00', '#00b400', '#b11e31', '#c2182e', '#d4112a', '#e50b27', '#e50b27'],
          'Icy': ['#8258ff', '#7649fc', '#6d38ff', '#632aff', '#5618ff', '#4400ff', '#003c7d', '#00418a', '#004b9e',
                  '#0054b2', '#005ec6', '#5f5be4', '#9400ff', '#c236f8', '#fe52c1', '#fe52c1'],
          'Kaiser': ['#186818', '#25a725', '#4cd64c', '#02e3a7', '#00e0e0', '#02bae3', '#4f70bd', '#3e5ca4',
                     '#334b86', '#263864', '#1d2c4e', '#420e61', '#631492', '#8a1dcb', '#b359e8', '#b359e8'],
          'Ketchapp': ['#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', '#f65e3b', '#edcf72', '#edcc61',
                       '#edc850', '#edc53f', '#edc22e', '#00eb6d', '#00ba4e', '#00873b', '#00692d', '#00692d'],
          'New Year': ['#4f3b36', '#633c3c', '#672422', '#822422', '#9c2220', '#b71d1d', '#672422', '#822422',
                       '#9c2220', '#b71d1d', '#b81414', '#bd9a3a', '#cf9d1f', '#e6ae16', '#ffbb00', '#ffbb00'],
          'Mint': ['#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', '#f65e3b', '#edcf72', '#edcc61',
                   '#edc850', '#edc53f', '#edc22e', '#1a9b52', '#2ecc71', '#1dceaa', '#3498db', '#3498db'],
          'Mochiica': ['#0a6669', '#07555f', '#0b3d4d', '#243861', '#332e6b', '#432b82', '#4c3699', '#5c45a6',
                       '#7e5fb9', '#a777d3', '#cf99e3', '#e8aedb', '#febfbf', '#febe88', '#fdb070', '#fdb070'],
          'Nebula': ['#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', '#f65e3b', '#edcf72', '#edcc61',
                     '#edc850', '#edc53f', '#edc22e', '#9100cf', '#611ecc', '#303ec9', '#005ec6', '#005ec6'],
          'Old Ketchapp': ['#eee4da', '#ece0c7', '#f2b079', '#ec8c54', '#f67b5e', '#ea5837', '#f2d86b', '#f0d04b',
                           '#e4c02a', '#e1ba14', '#ebc302', '#5eda93', '#24bb64', '#228c50', '#206d3c', '#206d3c'],
          'Pink': ['#c41197', '#d713a6', '#eb1bb7', '#f14593', '#f0464f', '#df0022', '#9122ff', '#8000ff',
                   '#6600cc', '#5500aa', '#45008a', '#005ec6', '#0079ff', '#00acff', '#47c5ff', '#47c5ff'],
          'Playful': ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#e84e9b', '#27a28b', '#f67c5f', '#2b5bad',
                      '#edc850', '#b79473', '#976f43', '#3c3a32', '#ea4335', '#ea4335', '#ea4335', '#ea4335'],
          'Rainbow': ['#dd2620', '#ed431a', '#fd6013', '#fb7f19', '#fa9e1f', '#f8bd25', '#bfb933', '#85b440',
                      '#4cb04e', '#3e8c55', '#31685c', '#234463', '#4a3475', '#722387', '#991399', '#991399'],
          'Royal': ['#fffdf8', '#fcf1d4', '#f9e5b0', '#f6da8d', '#f3ce69', '#f0c245', '#e67372', '#d35b7c',
                    '#c04386', '#ac2b8f', '#991399', '#6469ee', '#5a83ee', '#4f9eee', '#45b8ee', '#45b8ee'],
          'Spooky': ['#565656', '#5e3232', '#702626', '#781111', '#6e0909', '#560000', '#ff9041', '#fd7e24',
                     '#db6400', '#aa4e00', '#8a4000', '#1e0076', '#14004f', '#0a0027', '#000000', '#000000'],
          'Sunrise': ['#271732', '#50264a', '#6e3e70', '#905fae', '#b15beb', '#c163ff', '#b43f6c', '#ce577d',
                      '#dc6581', '#e1758f', '#e38d94', '#dd4f39', '#e70127', '#b0022a', '#a90a40', '#a90a40'],
          'Sunset': ['#eee6d6', '#eedab3', '#eec269', '#e09b43', '#db881f', '#c76c1e', '#ba4e39', '#b0402d',
                     '#ad3136', '#8c1d55', '#732172', '#5b297d', '#562185', '#5f1d75', '#78194a', '#78194a'],
          'Supernova': ['#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', '#f65e3b', '#edcf72', '#edcc61',
                        '#edc850', '#edc53f', '#edc22e', '#e900ad', '#bf009c', '#94008a', '#6a0079', '#6a0079'],
          'Tropical': ['#eef5e8', '#dcedc8', '#aed581', '#6fab3c', '#4b942f', '#247538', '#195a3f', '#166b60',
                       '#1b6e6e', '#1d8287', '#01949e', '#ccb400', '#c79522', '#c66421', '#a11c20', '#a11c20'],
          'Verse': ['#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', '#f65e3b', '#edcf72', '#edcc61',
                    '#edc850', '#edc53f', '#edc22e', '#9100cf', '#590080', '#36004d', '#000000', '#000000'],
          '10 years': ['#f57322', '#cf621f', '#bd5418', '#b0aad1', '#a29dbf', '#908cab', '#ffc832', '#ffc421',
                       '#ffbb00', '#66e8ff', '#45e3ff', '#3698fa', '#187fed', '#b92fdb', '#d041e3', '#d041e3'],
}


def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB元组（0-255）"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def is_darker_than(color_hex, reference="#ede0c7"):
    r, g, b = hex_to_rgb(color_hex)
    r_ref, g_ref, b_ref = hex_to_rgb(reference)

    # 计算亮度
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    luminance_ref = 0.299 * r_ref + 0.587 * g_ref + 0.114 * b_ref

    return luminance_ref > luminance


def fill_mid_falses(lst):
    if len(lst) < 3:
        return lst
    result = lst.copy()
    for i in range(1, len(lst) - 1):
        if not lst[i] and lst[i - 1] and lst[i + 1]:
            result[i] = True

    return result


# noinspection PyAttributeOutsideInit
class SingletonConfig:
    _instance = None  # 用于存储单例实例的类属性
    config_file_path = os.path.join(os.path.dirname(__file__), 'config')  # 配置文件路径
    use_avx = 1
    font_colors = [True for _ in range(36)]

    def __new__(cls):
        # 检查是否已经有实例存在
        if cls._instance is None:
            cls._instance = super(SingletonConfig, cls).__new__(cls)
            # 初始化配置数据
            cls._instance.config = cls.load_config()
            cls.tile_font_colors()
            cls.check_cpuinfo()
        return cls._instance

    @classmethod
    def tile_font_colors(cls):
        if not cls._instance:
            return
        font_colors = []
        bg_colors = cls._instance.config['colors']
        for color in bg_colors:
            font_colors.append(is_darker_than(color))
        cls.font_colors = fill_mid_falses(font_colors)

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
                'game_state': [np.uint64(0), 0, 0],
                'dis_32k': False,
                'compress': False,
                'optimal_branch_only': False,
                'compress_temp_files': False,
                'SmallTileSumLimit': 56,
                'advanced_algo': False,
                'chunked_solve': False,
                'deletion_threshold': 0.0,
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


def _load_fonts():
    """动态加载Clear Sans字体"""
    font_files = [
        r"font/ClearSans/ClearSans-Bold.ttf",
        r"font/ClearSans/ClearSans-Regular.ttf"
    ]

    for font_file in font_files:
        # 返回值为字体ID，负数表示失败
        if QFontDatabase.addApplicationFont(font_file) < 0:
            logger.warning(f"Font loading failure: {font_file}")

_load_fonts()


""" CPU-time returning clock() function which works from within njit-ted code """
import ctypes
import platform

if platform.system() == "Windows":
    from ctypes.util import find_msvcrt

    __LIB = find_msvcrt()
    if __LIB is None:
        __LIB = "msvcrt.dll"
else:
    from ctypes.util import find_library

    __LIB = find_library("c")

clock = ctypes.CDLL(__LIB).clock
clock.argtypes = []

