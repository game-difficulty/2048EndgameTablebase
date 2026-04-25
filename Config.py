import glob
import logging
import locale
import os
import pickle
import sys
import json
import threading
from typing import Callable, Optional, Any
import ctypes
import platform

import cpuinfo
import numpy as np

import egtb_core.Calculator as Calculator
from egtb_core.BoardMover import decode_board
from error_bridge import publish_frontend_exception

PatternCheckFunc = Callable[[np.uint64], bool]
CanonicalFunc = Callable[[np.uint64], np.uint64]
SuccessCheckFunc = Callable[[np.uint64, int], bool]


def load_config_json(filename):
    """Utility to load JSON configuration."""
    path = os.path.join(os.path.dirname(__file__), "docs_and_configs", filename)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


theme_map = load_config_json("themes.json")


DTYPE_CONFIG = {
    "uint32": (
        np.zeros(2, dtype="uint64,uint32"),
        np.uint32,
        np.uint32(4e9),
        np.uint32(0),
    ),
    "uint64": (
        np.zeros(2, dtype="uint64,uint64"),
        np.uint64,
        np.uint64(1.6e18),
        np.uint64(0),
    ),
    "float32": (
        np.zeros(2, dtype="uint64,float32"),
        np.float32,
        np.float32(1.0),
        np.float32(0.0),
    ),
    "float64": (
        np.zeros(2, dtype="uint64,float64"),
        np.float64,
        np.float64(1.0),
        np.float64(0.0),
    ),
    "1-float32": (
        np.zeros(2, dtype="uint64,float32"),
        np.float32,
        np.float32(0.0),
        np.float32(-1.0),
    ),
    "1-float64": (
        np.zeros(2, dtype="uint64,float64"),
        np.float64,
        np.float64(0.0),
        np.float64(-1.0),
    ),
}


DEFAULT_PATTERNS = load_config_json("default_patterns.json")


def restore_patterns_config_file(file_path, raw_data):
    """Write default patterns back to patterns_config.json."""
    try:
        default_path = os.path.join(os.path.dirname(__file__), "docs_and_configs", "default_patterns.json")
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as src:
                default_text = src.read()
            with open(file_path, "w", encoding="utf-8") as dst:
                dst.write(default_text)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)
                f.write("\n")
        logger.info(f"Restored patterns config from defaults: {file_path}")
    except OSError as e:
        logger.error(f"Failed to restore default patterns config {file_path}: {e}")


logger = logging.getLogger("debug_logger")

logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler("logger.txt")
file_handler.setLevel(logging.WARNING)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


MAX_DELETION_THRESHOLD = 0.999999


def normalize_deletion_threshold(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return min(MAX_DELETION_THRESHOLD, max(0.0, parsed))


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    try:
        publish_frontend_exception(
            "Unhandled Exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
    except Exception:
        logger.error("Failed to publish frontend error", exc_info=True)


sys.excepthook = handle_exception


def _handle_thread_exception(args):
    handle_exception(args.exc_type, args.exc_value, args.exc_traceback)


threading.excepthook = _handle_thread_exception


def find_f_nibble_positions(board: np.uint64) -> list[int]:
    """Return nibble start positions whose value is 0xF."""
    positions = []
    for i in range(16):
        if (board >> np.uint64(i * 4)) & np.uint64(0xF) == 0xF:
            positions.append(i * 4)
    return positions


def get_nibble_intersection(encoded_boards):
    if len(encoded_boards) == 0:
        return np.uint64(0)
    return np.bitwise_and.reduce(encoded_boards)


def load_patterns_from_file(file_path=None):
    """Load pattern metadata from patterns_config.json."""
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "docs_and_configs", "patterns_config.json")

    raw_data: dict[str, dict] = {}
    config_missing = not os.path.exists(file_path)
    needs_restore = False

    try:
        if config_missing:
            logger.warning(f"Config file not found: {file_path}. Using defaults...")
            needs_restore = True
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                if not raw_data:
                    raise ValueError("File is empty")
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning(f"Error loading {file_path}: {e}. Using defaults...")
        needs_restore = True

    # 如果读取失败，加载内存中的默认值
    if needs_restore:
        raw_data = DEFAULT_PATTERNS
        if config_missing:
            restore_patterns_config_file(file_path, raw_data)
        # 如果文件存在但格式错误，则更名以提醒用户
        if not config_missing and os.path.exists(file_path):
            error_path = file_path + ".error"
            try:
                if os.path.exists(error_path):
                    os.remove(error_path)
                os.rename(file_path, error_path)
                logger.warning(f"Problematic config file renamed to {error_path}")
            except OSError as e:
                logger.error(f"Failed to rename problematic config {file_path}: {e}")

    # 1. 初始化容器
    new_category_info = {}
    new_pattern_data = {}
    new_pattern_32k_tiles_map = {}

    # 2. 第一轮遍历：构建基础数据和 PATTERN_DATA
    for name, data in raw_data.items():
        # --- 构建 category_info ---
        cat = data.get("category", "others")
        if cat not in new_category_info:
            new_category_info[cat] = []
        new_category_info[cat].append(name)

        # --- 构建 PATTERN_DATA ---
        # 解析十六进制字符串为 uint64
        def_masks = tuple(np.uint64(int(m, 16)) for m in data.get("valid pattern", []))
        def_shifts = tuple(
            find_f_nibble_positions(
                np.uint64(int(data.get("target pos", "0xffffffffffffffff"), 16))
            )
        )
        new_pattern_data[name] = (def_masks, def_shifts)

    # 3. 将 PATTERN_DATA 注入 Calculator 并触发 Numba 编译
    # 这一步必须在构建 formation_info 之前完成，因为 formation_info 依赖生成的函数
    Calculator.PATTERN_DATA = new_pattern_data  # type: ignore
    Calculator.update_logic_functions()

    # 4. 第二轮遍历：构建 formation_info
    # 现在 Calculator.is_L3_pattern 等函数已经存在了
    new_formation_info = {}

    for name, data in raw_data.items():
        # 动态获取刚刚生成的 Numba 函数
        # update_logic_functions 会将函数注入 Calculator 的全局命名空间
        # 使用 getattr 从模块中获取
        try:
            is_pattern_func = getattr(Calculator, f"is_{name}_pattern")
            is_success_func = getattr(Calculator, f"is_{name}_success")
        except AttributeError:
            print(
                f"Warning: Functions for {name} not found in Calculator after injection."
            )
            continue

        # 获取对称函数
        symm_name = "canonical_" + data.get("canonical mode", "identity")
        symm_func = getattr(Calculator, symm_name)

        # 解析初始局面
        fmt_seeds_raw = data.get("seed boards", ["0xffffffff"])
        fmt_seeds = [np.uint64(int(m, 16)) for m in fmt_seeds_raw]

        ini_decoded = decode_board(fmt_seeds[0])
        board_sum = ini_decoded.sum()
        fmt_seeds = np.array(
            [board for board in fmt_seeds if ini_decoded.sum() == board_sum],
            dtype=np.uint64,
        )

        new_formation_info[name] = [
            -board_sum,
            is_pattern_func,
            symm_func,
            is_success_func,
            fmt_seeds,
            data.get("extra steps", 36),
        ]

        # --- 构建 pattern_32k_tiles_map ---
        count = np.sum(ini_decoded == 32768)
        fixed_pos = find_f_nibble_positions(
            get_nibble_intersection(new_pattern_data[name][0])
        )
        fixed_pos = np.array(fixed_pos, dtype=np.uint8)
        free_count = count - len(fixed_pos)
        new_pattern_32k_tiles_map[name] = [count, free_count, fixed_pos]

    if "others" in new_category_info:
        # pop 会取出该键值对，重新赋值会将其插入到字典末尾
        new_category_info["others"] = new_category_info.pop("others")

    return (
        new_category_info,
        new_formation_info,
        new_pattern_32k_tiles_map,
        new_pattern_data,
    )


category_info, formation_info, pattern_32k_tiles_map, _ = load_patterns_from_file()


def hex_to_rgb(hex_color):
    """Convert a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def is_darker_than(color_hex, reference="#ede0c7"):
    r, g, b = hex_to_rgb(color_hex)
    r_ref, g_ref, b_ref = hex_to_rgb(reference)

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
    _instance: Optional["SingletonConfig"] = None
    config: dict[str, Any]

    config_file_path = os.path.join(os.path.dirname(__file__), "docs_and_configs", "config")
    font_colors = [True for _ in range(36)]
    _current_translator: Any = None

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

        # Safe access to avoid KeyErrors with legacy config files
        config = cls._instance.config
        use_custom = config.get("use_custom_theme", False)

        default_colors = ["#000000"] * 20
        if use_custom:
            bg_colors = config.get(
                "custom_colors", config.get("colors", default_colors)
            )
        else:
            bg_colors = config.get("colors", default_colors)

        for color in bg_colors:
            font_colors.append(is_darker_than(color))
        cls.font_colors = fill_mid_falses(font_colors)

    @classmethod
    def load_config(cls, filename=None):
        if filename is None:
            filename = cls.config_file_path

        # Mandatory Defaults
        defaults = {
            "filepath_map": dict(),
            "colors": [
                "#fffdf8",
                "#fcf1d4",
                "#f9e5b0",
                "#f6da8d",
                "#f3ce69",
                "#f0c245",
                "#e67372",
                "#d35b7c",
                "#c04386",
                "#ac2b8f",
                "#991399",
                "#6469ee",
                "#5a83ee",
                "#4f9eee",
                "#45b8ee",
                "#45b8ee",
            ]
            + ["#000000"] * 20,
            "custom_colors": [
                "#fffdf8",
                "#fcf1d4",
                "#f9e5b0",
                "#f6da8d",
                "#f3ce69",
                "#f0c245",
                "#e67372",
                "#d35b7c",
                "#c04386",
                "#ac2b8f",
                "#991399",
                "#6469ee",
                "#5a83ee",
                "#4f9eee",
                "#45b8ee",
                "#45b8ee",
            ]
            + ["#000000"] * 20,
            "demo_speed": 40,
            "4_spawn_rate": 0.1,
            "do_animation": True,
            "game_state": [np.uint64(0), 0, 0],
            "dis_32k": False,
            "dis_text": True,
            "compress": False,
            "optimal_branch_only": False,
            "compress_temp_files": False,
            "SmallTileSumLimit": 96,
            "advanced_algo": False,
            "chunked_solve": False,
            "deletion_threshold": 0.0,
            "notebook_threshold": 0.999,
            "font_size_factor": 100,
            "ui_scale": 100,
            "minigame_state": [dict(), dict()],
            "power_ups_state": [dict(), dict()],
            "minigame_difficulty": 1,
            "language": cls.get_system_language(),
            "theme": "Default",
            "use_custom_theme": False,
            "success_rate_dtype": "uint32",
            "record_player_slider_threshold": 0.99999,
            "dark_mode": False,
        }

        if os.path.exists(filename):
            try:
                with open(filename, "rb") as file:
                    data = pickle.load(file)
                    # Merge data with defaults to ensure missing keys are added
                    updated = False
                    for k, v in defaults.items():
                        if k not in data:
                            data[k] = v
                            updated = True
                    normalized_deletion_threshold = normalize_deletion_threshold(
                        data.get("deletion_threshold", defaults["deletion_threshold"])
                    )
                    if data.get("deletion_threshold") != normalized_deletion_threshold:
                        data["deletion_threshold"] = normalized_deletion_threshold
                        updated = True
                    if updated:
                        cls.save_config(data, filename)
                    return data
            except (EOFError, pickle.UnpicklingError, Exception):
                pass

        return defaults

    @classmethod
    def save_config(cls, config, filename=None):
        if filename is None:
            filename = cls.config_file_path

        with open(filename, "wb") as file:
            pickle.dump(config, file)

    @staticmethod
    def check_cpuinfo():
        # 获取当前cpu指令集信息
        info = cpuinfo.get_cpu_info()
        if (
            "avx512f" in info["flags"]
            and "avx512vl" in info["flags"]
            and (
                "avx512dq" in info["flags"]
                or ("avx512bw" in info["flags"] and "avx512vbmi2" in info["flags"])
            )
        ):
            return "avx512"
        elif "avx2" in info["flags"]:
            return "avx2"
        else:
            return "None"

    @staticmethod
    def get_system_language():
        lang_code, _ = locale.getlocale()
        if not lang_code:
            lang_code = locale.getdefaultlocale()[0]
        normalized = str(lang_code or "").lower()

        logger.info(f"System locale: {normalized}")

        if normalized.startswith("zh"):
            return "zh"
        else:
            return "en"  # 默认英语

    @staticmethod
    def apply_language(lang):
        SingletonConfig().config["language"] = lang
        SingletonConfig().save_config(SingletonConfig().config)
        return

    @classmethod
    def get_pattern_key(cls, pattern, spawn_rate4):
        if cls._instance is None:
            return (pattern, float(spawn_rate4))
        filepath_map = cls._instance.config.get("filepath_map", {})
        target_sr4 = float(spawn_rate4)
        for k in filepath_map.keys():
            if k[0] == pattern and abs(float(k[1]) - target_sr4) <= 1e-4:
                return k
        return (pattern, target_sr4)

    @classmethod
    def check_pattern_file(cls, pattern):
        if not cls._instance:
            return False
        spawn_rate4 = SingletonConfig().config["4_spawn_rate"]
        filepath_map = cls._instance.config["filepath_map"]
        pattern_key = cls.get_pattern_key(pattern, float(spawn_rate4))
        file_path_list: list | None = filepath_map.get(pattern_key, None)
        prefix = f"{pattern}_"
        if not file_path_list:
            return False

        for file_path, success_rate_dtype in file_path_list:
            # 检查文件夹是否存在
            if (
                not file_path
                or not os.path.exists(file_path)
                or not os.path.isdir(file_path)
            ):
                file_path_list.remove((file_path, success_rate_dtype))
                continue

            # 遍历文件夹中的所有项
            for item in os.listdir(file_path):
                if not item.startswith(prefix):
                    continue
                if item.endswith(".book") or item.endswith(".z") or item.endswith("b"):
                    return True

            file_path_list.remove((file_path, success_rate_dtype))
        SingletonConfig().save_config(SingletonConfig().config)

        return False

    @staticmethod
    def read_success_rate_dtype(folder_path, pattern):
        """Read success_rate_dtype from the pattern config file."""
        if not os.path.isdir(folder_path):
            return "uint32"

        # 查找 config 文件
        config_files = glob.glob(os.path.join(folder_path, f"{pattern}_config.txt"))
        if not config_files:
            return "uint32"

        for config_file in config_files:
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("success_rate_dtype:"):
                            return line.split(":", 1)[1].strip()

                # 如果代码运行到这里，说明文件存在但没找到字段，写入默认值
                with open(config_file, "a", encoding="utf-8") as f:
                    f.write("\nsuccess_rate_dtype: uint32")

                return "uint32"

            except Exception as e:
                logger.error(
                    f"Unexpected error processing {config_file}: {e}", exc_info=True
                )

        return "uint32"

    @staticmethod
    def read_4sr(folder_path, pattern):
        """Read 4_spawn_rate from the pattern config file."""
        if not os.path.isdir(folder_path):
            return None

        config_files = glob.glob(os.path.join(folder_path, f"{pattern}_config.txt"))
        if not config_files:
            return None

        for config_file in config_files:
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("4_spawn_rate:"):
                            spawn_rate = line.split(":", 1)[1].strip()
                            return float(spawn_rate)
                return None

            except Exception as e:
                logger.error(
                    f"Unexpected error processing {config_file}: {e}", exc_info=True
                )

        return None


# 用于管理除数字块之外的配色
class ColorManager:
    _instance = None

    def __new__(cls, config_file=None):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), "docs_and_configs", "color_schemes.txt")
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), "docs_and_configs", "color_schemes.txt")
        if not self._initialized:
            self.config_file = config_file
            self.schemes = {}  # 配色字典
            self.current_theme = SingletonConfig().config.get("theme", "Default")
            self.current_colors = []  # 当前主题的颜色列表

            self._load_schemes()
            self._initialized = True

    @staticmethod
    def _normalize_line(line):
        """Normalize localized punctuation to ASCII punctuation."""
        chinese_to_english = {
            "，": ",",
            "（": "(",
            "）": ")",
            "；": ";",
            "：": ":",
            "　": " ",
        }

        for cn_char, en_char in chinese_to_english.items():
            line = line.replace(cn_char, en_char)
        return line

    def _load_schemes(self):
        """Load color schemes from the config file."""
        if not os.path.exists(self.config_file):
            self._create_default_config()
            return

        with open(self.config_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                line = self._normalize_line(line)

                if ":" in line:
                    theme, colors_str = line.split(":", 1)
                    theme = theme.strip()

                    # 分割颜色字符串
                    colors = []
                    for color in colors_str.split("),("):
                        color = color.strip()
                        if color.startswith("(") and not color.endswith(")"):
                            color += ")"
                        elif not color.startswith("(") and color.endswith(")"):
                            color = "(" + color
                        elif not color.startswith("(") and not color.endswith(")"):
                            color = "(" + color + ")"
                        colors.append(color)

                    self.schemes[theme] = colors

        # 设置默认当前主题
        if self.schemes:
            self.current_colors = self.schemes.get(self.current_theme, "Default")

    def _create_default_config(self):
        """Create a default color scheme config file."""
        with open(self.config_file, "w", encoding="utf-8") as f:
            f.write(
                "Default:(255,255,255),(245,245,247),(244,241,232),(236,236,236),(222,222,222),(209,209,209),(205,193,180),(187,173,160),(167,167,167),(160,160,160),(0,0,0)\n"
            )
            f.write(
                "Dark:(32,32,32),(30,36,42),(135,130,125),(33,33,33),(36,36,36),(38,38,38),(114,111,107),(48,48,48),(53,53,53),(62,60,58),(245,235,219)\n"
            )

    def add_scheme(self, theme_name, colors_list):
        self.schemes[theme_name] = colors_list
        self._save_to_file()

    def switch_theme(self, theme_name):
        if theme_name in self.schemes:
            self.current_theme = theme_name
            SingletonConfig().config["theme"] = theme_name
            SingletonConfig().save_config(SingletonConfig().config)
            self.current_colors = self.schemes[theme_name]
            return True
        return False

    def _save_to_file(self):
        with open(self.config_file, "w", encoding="utf-8") as f:
            for theme, colors in self.schemes.items():
                f.write(f"{theme}:{','.join(colors)}\n")

    def get_css_color(self, index, a=None):
        """Return a CSS color string for the selected theme color."""
        if 0 <= index < len(self.current_colors):
            if a is None:
                return f"rgb{self.current_colors[index]}"
            else:
                return f"rgba{self.current_colors[index][:-1]},{a})"
        return "rgb(255,255,255)"

    def get_rgb(self, index):
        if 0 <= index < len(self.current_colors):
            r, g, b = self.current_colors[index].strip("()").split(",")
            return int(r), int(g), int(b)
        return 255, 255, 255


def apply_global_theme(app):
    return None


""" CPU-time returning clock() function which works from within njit-ted code """
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
