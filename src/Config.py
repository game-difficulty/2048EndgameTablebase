import glob
import logging
import os
import pickle
import sys
import json
from typing import Callable

import cpuinfo
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QLocale, QTranslator

import Calculator
from BoardMover import decode_board

PatternCheckFunc = Callable[[np.uint64], bool]
CanonicalFunc = Callable[[np.uint64], np.uint64]
SuccessCheckFunc = Callable[[np.uint64, int], bool]


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


DTYPE_CONFIG = {
    'uint32': (np.zeros(2,dtype='uint64,uint32'), np.uint32, np.uint32(4e9), np.uint32(0)),
    'uint64': (np.zeros(2,dtype='uint64,uint64'), np.uint64, np.uint64(1.6e18), np.uint64(0)),
    'float32': (np.zeros(2,dtype='uint64,float32'), np.float32, np.float32(1.0), np.float32(0.0)),
    'float64': (np.zeros(2,dtype='uint64,float64'), np.float64, np.float64(1.0), np.float64(0.0)),
    '1-float32': (np.zeros(2,dtype='uint64,float32'), np.float32, np.float32(0.0), np.float32(-1.0)),
    '1-float64': (np.zeros(2,dtype='uint64,float64'), np.float64, np.float64(0.0), np.float64(-1.0)),
}


# 预定义的默认配置（当文件读取失败时使用）
DEFAULT_PATTERNS = {
  "L3": {
    "category": "10 space",
      "valid pattern": ["0xfff0fff"],
      "target pos": "0xf00f0000000",
      "canonical mode": "identity",
      "seed boards": ["0x100000001fff2fff", "0x000000012fff1fff"],
      "extra steps": 48
    },
  "L3t": {
    "category": "10 space",
      "valid pattern": ["0xfff0fff", "0xf0fff0ff0", "0xf000f0ff00ff0"],
      "target pos": "0xffff00000000",
      "canonical mode": "identity",
      "seed boards": ["0x100000001fff2fff", "0x000000012fff1fff"],
      "extra steps": 48
    },
  "442": {
    "category": "10 space",
      "valid pattern": ["0xffffff"],
      "target pos": "0xf000000",
      "canonical mode": "identity",
      "seed boards": ["0x1000000021ffffff", "0x0000000112ffffff"],
      "extra steps": 48
    },
  "442t": {
    "category": "10 space",
      "valid pattern": [
        "0xffffff", "0xf0ff0fff", "0xffff0ff", "0xffff00ff", "0xf00fffff0",
        "0xff0ff0ff0", "0xf00000ff0fff", "0xf0000fff00ff", "0xf00f00ff0ff0", "0xf00f0fff00f0"
      ],
      "target pos": "0xff0ff000000",
      "canonical mode": "identity",
      "seed boards": ["0x1000000021ffffff", "0x0000000112ffffff"],
      "extra steps": 48
    },
  "t": {
    "category": "10 space",
      "valid pattern": ["0xf0fff0ff", "0xf000f0ff00ff", "0xf000f00000ff00ff"],
      "target pos": "0xff0f000f00",
      "canonical mode": "identity",
      "seed boards": ["0x10000000f1fff2ff", "0x00000001f2fff1ff"],
      "extra steps": 48
    },
  "LL": {
    "category": "12 space",
      "valid pattern": ["0xff00ff"],
      "target pos": "0xff0f000f00",
      "canonical mode": "diagonal",
      "seed boards": ["0x1000000021ff12ff", "0x0000000112ff21ff"],
      "extra steps": 48
    },
  "4431": {
    "category": "12 space",
      "valid pattern": ["0xf0fff"],
      "target pos": "0xf00000",
      "canonical mode": "identity",
      "seed boards": ["0x10000000121f2fff", "0x00000001121f2fff"],
      "extra steps": 48
    },
  "444": {
    "category": "12 space",
      "valid pattern": ["0xffff"],
      "target pos": "0xff00000",
      "canonical mode": "horizontal",
      "seed boards": ["0x100000000000ffff", "0x000000010000ffff"],
      "extra steps": 60
    },
  "4432f": {
    "category": "12 space",
      "valid pattern": ["0xf00ff"],
      "target pos": "0xf00000",
      "canonical mode": "diagonal",
      "seed boards": ["0x00001000121f2fff", "0x00010000121ff2ff"],
      "extra steps": 48
    },
  "4442ff": {
    "category": "12 space",
      "valid pattern": ["0xff"],
      "target pos": "0xff0fff0f00",
      "canonical mode": "identity",
      "seed boards": ["0x0000100012ff21ff", "0x000100001212ffff", "0x00ff0012102100ff"],
      "extra steps": 48
    },
  "4421": {
    "category": "others",
      "valid pattern": ["0xff0fff"],
      "target pos": "0xf00000",
      "canonical mode": "identity",
      "seed boards": ["0x1000000012ff2fff", "0x1000000121ff1fff"],
      "extra steps": 48
    },
  "4441": {
    "category": "others",
      "valid pattern": ["0xfff"],
      "target pos": "0xfff000",
      "canonical mode": "identity",
      "seed boards": ["0x0000100000001fff", "0x0001000000001fff"],
      "extra steps": 48
    },
  "4432": {
    "category": "others",
      "valid pattern": ["0xf00ff"],
      "target pos": "0xf00000",
      "canonical mode": "diagonal",
      "seed boards": ["0x00001000121f21ff", "0x00010000121f21ff"],
      "extra steps": 48
    },
  "4442": {
    "category": "others",
      "valid pattern": ["0xff"],
      "target pos": "0xff0f00",
      "canonical mode": "identity",
      "seed boards": ["0x00001000000012ff", "0x00010000000021ff"],
      "extra steps": 48
    },
  "4442f": {
    "category": "others",
      "valid pattern": ["0xff"],
      "target pos": "0xffff00",
      "canonical mode": "identity",
      "seed boards": ["0x00001000121f21ff", "0x0001000012112fff"],
      "extra steps": 48
    },
  "free8": {
    "category": "free",
      "valid pattern": [],
      "target pos": "0xffffffffffffffff",
      "canonical mode": "full",
      "seed boards": ["0x01111111ffffffff"],
      "extra steps": 36
    },
  "free9": {
    "category": "free",
      "valid pattern": [],
      "target pos": "0xffffffffffffffff",
      "canonical mode": "full",
      "seed boards": ["0x011111111fffffff"],
      "extra steps": 36
    },
  "free10": {
    "category": "free",
      "valid pattern": [],
      "target pos": "0xffffffffffffffff",
      "canonical mode": "full",
      "seed boards": ["0x0111111111ffffff"],
      "extra steps": 36
    },
  "free11": {
    "category": "free",
      "valid pattern": [],
      "target pos": "0xffffffffffffffff",
      "canonical mode": "full",
      "seed boards": ["0x01111111111fffff"],
      "extra steps": 36
    },
  "free12": {
    "category": "free",
      "valid pattern": [],
      "target pos": "0xffffffffffffffff",
      "canonical mode": "full",
      "seed boards": ["0x011111111111ffff"],
      "extra steps": 36
    },
  "2x4": {
    "category": "variant",
      "valid pattern": [],
      "target pos": "0xffffffff0000",
      "canonical mode": "min24",
      "seed boards": ["0xffff00000000ffff"],
      "extra steps": 48
    },
  "3x3": {
    "category": "variant",
      "valid pattern": [],
      "target pos": "0xfff0fff0fff00000",
      "canonical mode": "min33",
      "seed boards": ["0x000f000f000fffff"],
      "extra steps": 60
    },
  "3x4": {
    "category": "variant",
      "valid pattern": [],
      "target pos": "0xffffffffffff0000",
      "canonical mode": "min34",
      "seed boards": ["0x000000000000ffff"],
      "extra steps": 60
    }
}


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
        QtWidgets.QMessageBox.critical(None, "Error",
                                       f"An unexpected error occurred: {exc_value}\n {exc_traceback}")


sys.excepthook = handle_exception


def find_f_nibble_positions(board: np.uint64) -> list[int]:
    """输入 uint64，输出所有值为 0xf 的 nibble 的起始位索引。"""
    positions = []
    for i in range(16):
        if (board >> np.uint64(i * 4)) & np.uint64(0xf) == 0xf:
            positions.append(i * 4)
    return positions


def get_nibble_intersection(encoded_boards):
    if len(encoded_boards) == 0:
        return np.uint64(0)
    return np.bitwise_and.reduce(encoded_boards)


def load_patterns_from_file(file_path=None):
    """
    读取配置文件，并初始化所有定式定义相关的全局字典。
    """
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), 'patterns_config.json')

    raw_data = None
    needs_restore = False

    # 尝试读取文件
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Config file not found: {file_path}. Restoring defaults...")
            needs_restore = True
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                if not raw_data:
                    raise ValueError("File is empty")
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning(f"Error loading {file_path}: {e}. Restoring defaults...")
        needs_restore = True

    # 如果读取失败，加载默认值并写回文件
    if needs_restore:
        raw_data = DEFAULT_PATTERNS
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Successfully restored DEFAULT_PATTERNS to {file_path}")
        except OSError as e:
            logger.critical(f"Critical Warning: Could not write default config to {file_path}: {e}")

    # 1. 初始化容器
    new_category_info = {}
    new_pattern_data = {}
    new_pattern_32k_tiles_map = {}

    # 2. 第一轮遍历：构建基础数据和 PATTERN_DATA
    for name, data in raw_data.items():
        # --- 构建 category_info ---
        cat = data.get('category', 'others')
        if cat not in new_category_info:
            new_category_info[cat] = []
        new_category_info[cat].append(name)

        # --- 构建 PATTERN_DATA ---
        # 解析十六进制字符串为 uint64
        def_masks = tuple(np.uint64(int(m, 16)) for m in data.get('valid pattern', []))
        def_shifts = tuple(find_f_nibble_positions(np.uint64(int(data.get('target pos', '0xffffffffffffffff'), 16))))
        new_pattern_data[name] = (def_masks, def_shifts)

    # 3. 将 PATTERN_DATA 注入 Calculator 并触发 Numba 编译
    # 这一步必须在构建 formation_info 之前完成，因为 formation_info 依赖生成的函数
    Calculator.PATTERN_DATA = new_pattern_data
    Calculator.update_logic_functions()

    # 4. 第二轮遍历：构建 formation_info
    # 现在 Calculator.is_L3_pattern 等函数已经存在了
    new_formation_info = {}

    for name, data in raw_data.items():
        # 动态获取刚刚生成的 Numba 函数
        # update_logic_functions 会将函数注入 Calculator 的全局命名空间
        # 使用 getattr 从模块中获取
        try:
            is_pattern_func = getattr(Calculator, f'is_{name}_pattern')
            is_success_func = getattr(Calculator, f'is_{name}_success')
        except AttributeError:
            print(f"Warning: Functions for {name} not found in Calculator after injection.")
            continue

        # 获取对称函数
        symm_name = 'canonical_' + data.get('canonical mode', 'identity')
        symm_func = getattr(Calculator, symm_name)

        # 解析初始局面
        fmt_seeds_raw = data.get('seed boards')
        fmt_seeds = [np.uint64(int(m, 16)) for m in fmt_seeds_raw]

        ini_decoded = decode_board(fmt_seeds[0])
        board_sum = ini_decoded.sum()
        fmt_seeds = np.array([board for board in fmt_seeds if ini_decoded.sum() == board_sum],
                              dtype=np.uint64)

        # 组装元组
        # [score, pattern_func, find_func, success_func, masks_array]
        new_formation_info[name] = [
            -board_sum,
            is_pattern_func,
            symm_func,
            is_success_func,
            fmt_seeds,
            data.get('extra steps', 36)
        ]

        # --- 构建 pattern_32k_tiles_map ---
        count = np.sum(ini_decoded == 32768)
        fixed_pos = find_f_nibble_positions(get_nibble_intersection(new_pattern_data[name][0]))
        fixed_pos = np.array(fixed_pos, dtype=np.uint8)
        free_count = count - len(fixed_pos)
        new_pattern_32k_tiles_map[name] = [count, free_count, fixed_pos]

    if 'others' in new_category_info:
        # pop 会取出该键值对，重新赋值会将其插入到字典末尾
        new_category_info['others'] = new_category_info.pop('others')

    return new_category_info, new_formation_info, new_pattern_32k_tiles_map, new_pattern_data


category_info, formation_info, pattern_32k_tiles_map, _ = load_patterns_from_file()


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
    font_colors = [True for _ in range(36)]
    _current_translator: None | QTranslator = None

    def __new__(cls):
        # 检查是否已经有实例存在
        if cls._instance is None:
            cls._instance = super(SingletonConfig, cls).__new__(cls)
            # 初始化配置数据
            cls._instance.config = cls.load_config()
            cls.tile_font_colors()
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
                'do_animation': True,
                'game_state': [np.uint64(0), 0, 0],
                'dis_32k': False,
                'dis_text': True,
                'compress': False,
                'optimal_branch_only': False,
                'compress_temp_files': False,
                'SmallTileSumLimit': 96,
                'advanced_algo': False,
                'chunked_solve': False,
                'deletion_threshold': 0.0,
                'notebook_threshold': 0.999,
                'font_size_factor': 100,
                'minigame_state': [dict(), dict()],  # [盘面，得分，最高分，最大数，是否曾过关, 新数位置], []
                'power_ups_state': [dict(), dict()],
                'minigame_difficulty': 1,
                'language': SingletonConfig.get_system_language(),
                'theme': "Default",
                'success_rate_dtype': "uint32",
                }

    @classmethod
    def save_config(cls, config, filename=None):
        if filename is None:
            filename = cls.config_file_path

        with open(filename, 'wb') as file:
            pickle.dump(config, file)

    @staticmethod
    def check_cpuinfo():
        # 获取当前cpu指令集信息
        info = cpuinfo.get_cpu_info()
        if 'avx512f' in info['flags'] and 'avx512vl' in info['flags'] and \
                ('avx512dq' in info['flags'] or ('avx512bw' in info['flags'] and 'avx512vbmi2' in info['flags'])):
            return 'avx512'
        elif 'avx2' in info['flags']:
            return 'avx2'
        else:
            return 'None'

    @staticmethod
    def get_system_language():
        language_name = QLocale.languageToString(QLocale().language())
        bcp47_name = QLocale().bcp47Name()

        logger.info(f"Language name: {language_name}, BCP47 code: {bcp47_name}")

        if bcp47_name.startswith("zh") or language_name.startswith("Chinese"):
            return 'zh'
        else:
            return 'en'  # 默认英语

    @staticmethod
    def apply_language(lang):
        app = QtWidgets.QApplication.instance()
        SingletonConfig().config['language'] = lang

        # 1. 清除旧翻译器
        if SingletonConfig._current_translator:
            SingletonConfig._current_translator = None

        # 2. 加载新翻译（中文）
        if lang == 'zh':
            zh_translator = QTranslator()
            if zh_translator.load(os.path.join("translations", "app_zh_CN.qm")):
                SingletonConfig._current_translator = zh_translator
                app.installTranslator(zh_translator)

        # 3. 重新翻译所有窗口
        # noinspection PyUnresolvedReferences
        for widget in app.topLevelWidgets():
            if hasattr(widget, 'retranslateUi'):
                widget.retranslateUi()

    @classmethod
    def check_pattern_file(cls, pattern):  # todo done
        if not cls._instance:
            return False
        spawn_rate4 = SingletonConfig().config['4_spawn_rate']
        filepath_map = cls._instance.config['filepath_map']
        file_path_list: list | None = filepath_map.get((pattern, spawn_rate4), None)
        prefix = f"{pattern}_"
        if not file_path_list:
            return False

        for file_path, success_rate_dtype in file_path_list:
            # 检查文件夹是否存在
            if not file_path or not os.path.exists(file_path) or not os.path.isdir(file_path):
                file_path_list.remove((file_path, success_rate_dtype))
                continue

            # 遍历文件夹中的所有项
            for item in os.listdir(file_path):
                if not item.startswith(prefix):
                    continue
                if item.endswith('.book') or item.endswith('.z') or item.endswith('b'):
                    return True

            file_path_list.remove((file_path, success_rate_dtype))
        SingletonConfig().save_config(SingletonConfig().config)

        return False

    @staticmethod
    def read_success_rate_dtype(folder_path, pattern):
        """
        从配置文件中读取success_rate_dtype的值。
        如果找不到该字段，则在配置文件末尾追加默认值'uint32'并返回该值。
        """
        if not os.path.isdir(folder_path):
            return 'uint32'

        # 查找 config 文件
        config_files = glob.glob(os.path.join(folder_path, f"{pattern}_config.txt"))
        if not config_files:
            return 'uint32'

        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('success_rate_dtype:'):
                            return line.split(':', 1)[1].strip()

                # 如果代码运行到这里，说明文件存在但没找到字段，写入默认值
                with open(config_file, 'a', encoding='utf-8') as f:
                    f.write('\nsuccess_rate_dtype: uint32')

                return 'uint32'

            except Exception as e:
                logger.error(f"Unexpected error processing {config_file}: {e}", exc_info=True)

        return 'uint32'


    @staticmethod
    def read_4sr(folder_path, pattern):
        """ 从配置文件中读取4_spawn_rate的值。 """
        if not os.path.isdir(folder_path):
            return None

        config_files = glob.glob(os.path.join(folder_path, f"{pattern}_config.txt"))
        if not config_files:
            return None

        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('4_spawn_rate:'):
                            spawn_rate = line.split(':', 1)[1].strip()
                            return float(spawn_rate)
                return None

            except Exception as e:
                logger.error(f"Unexpected error processing {config_file}: {e}", exc_info=True)

        return None


# 用于管理除数字块之外的配色
class ColorManager:
    _instance = None

    def __new__(cls, config_file="color_schemes.txt"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file="color_schemes.txt"):
        if not self._initialized:
            self.config_file = config_file
            self.schemes = {}  # 配色字典
            self.current_theme = SingletonConfig().config.get('theme', "Default")
            self.current_colors = []  # 当前主题的颜色列表

            self._load_schemes()
            self._initialized = True

    @staticmethod
    def _normalize_line(line):
        """将中文标点替换为英文标点"""
        chinese_to_english = {
            '，': ',',
            '（': '(',
            '）': ')',
            '；': ';',
            '：': ':',
            '　': ' ',  # 全角空格
        }

        for cn_char, en_char in chinese_to_english.items():
            line = line.replace(cn_char, en_char)
        return line

    def _load_schemes(self):
        """从配置文件加载配色方案"""
        if not os.path.exists(self.config_file):
            self._create_default_config()
            return

        with open(self.config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                line = self._normalize_line(line)

                if ':' in line:
                    theme, colors_str = line.split(':', 1)
                    theme = theme.strip()

                    # 分割颜色字符串
                    colors = []
                    for color in colors_str.split('),('):
                        color = color.strip()
                        if color.startswith('(') and not color.endswith(')'):
                            color += ')'
                        elif not color.startswith('(') and color.endswith(')'):
                            color = '(' + color
                        elif not color.startswith('(') and not color.endswith(')'):
                            color = '(' + color + ')'
                        colors.append(color)

                    self.schemes[theme] = colors

        # 设置默认当前主题
        if self.schemes:
            self.current_colors = self.schemes.get(self.current_theme, "Default")

    def _create_default_config(self):
        """创建默认配置文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            f.write(
                "Default:(255,255,255),(245,245,247),(244,241,232),(236,236,236),(222,222,222),(209,209,209),(205,193,180),(187,173,160),(167,167,167),(160,160,160),(0,0,0)\n")
            f.write(
                "Dark:(32,32,32),(30,36,42),(135,130,125),(33,33,33),(36,36,36),(38,38,38),(114,111,107),(48,48,48),(53,53,53),(62,60,58),(245,235,219)\n")

    def add_scheme(self, theme_name, colors_list):
        self.schemes[theme_name] = colors_list
        self._save_to_file()

    def switch_theme(self, theme_name):
        if theme_name in self.schemes:
            self.current_theme = theme_name
            SingletonConfig().config['theme'] = theme_name
            SingletonConfig().save_config(SingletonConfig().config)
            self.current_colors = self.schemes[theme_name]
            return True
        return False

    def _save_to_file(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            for theme, colors in self.schemes.items():
                f.write(f"{theme}:{','.join(colors)}\n")

    def get_css_color(self, index, a=None):
        """获取CSS格式的颜色"""
        if 0 <= index < len(self.current_colors):
            if a is None:
                return f"rgb{self.current_colors[index]}"
            else:
                return f"rgba{self.current_colors[index][:-1]},{a})"
        return "rgb(255,255,255)"

    def get_rgb(self, index):
        if 0 <= index < len(self.current_colors):
            r, g, b = self.current_colors[index].strip('()').split(',')
            return int(r), int(g), int(b)
        return 255, 255, 255


def apply_global_theme(app):
    """应用全局主题到整个应用程序"""
    color_mgr = ColorManager()
    if color_mgr.current_theme == "Default":
        return

    palette = QtGui.QPalette()

    font_rgb = color_mgr.get_rgb(10)
    bg1_rgb = color_mgr.get_rgb(0)
    bg2_rgb = color_mgr.get_rgb(9)

    font_qcolor = QtGui.QColor(*font_rgb)
    bg1_qcolor = QtGui.QColor(*bg1_rgb)
    bg2_qcolor = QtGui.QColor(*bg2_rgb)

    # 1. 窗口相关
    palette.setColor(QtGui.QPalette.Window, bg1_qcolor)  # 窗口背景
    palette.setColor(QtGui.QPalette.WindowText, font_qcolor)  # 窗口文字

    # 2. 按钮相关
    palette.setColor(QtGui.QPalette.Button, bg2_qcolor)  # 按钮背景
    palette.setColor(QtGui.QPalette.ButtonText, font_qcolor)  # 按钮文字

    # 3. 基础控件相关
    palette.setColor(QtGui.QPalette.Base, bg2_qcolor)  # 输入框等背景
    palette.setColor(QtGui.QPalette.Text, font_qcolor)  # 输入框等文字

    # 4. 工具提示
    palette.setColor(QtGui.QPalette.ToolTipBase, bg2_qcolor)  # 工具提示背景
    palette.setColor(QtGui.QPalette.ToolTipText, font_qcolor)  # 工具提示文字

    app.setStyle("Fusion")
    app.setPalette(palette)


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
