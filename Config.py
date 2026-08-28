from __future__ import annotations

import glob
import logging
import locale
import os
import pickle
import sys
import json
import threading
from collections.abc import Iterator, Mapping
from typing import Callable, Optional, Any
import ctypes
import platform

import cpuinfo
import numpy as np

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

LOGGER_FILE_PATH = os.path.abspath("logger.txt")
os.environ.setdefault("TABLEBASE_NATIVE_LOG_FILE", LOGGER_FILE_PATH)

file_handler = logging.FileHandler(LOGGER_FILE_PATH, encoding="utf-8")
file_handler.setLevel(logging.WARNING)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


MAX_DELETION_THRESHOLD = 0.999999
MIN_BC_FAMILY_MODULUS = 13
MAX_BC_FAMILY_MODULUS = 256
DEFAULT_BC_FAMILY_MODULUS = 29
RUNTIME_DELETION_THRESHOLD_SIGNAL_PATH = os.path.join(
    os.path.dirname(__file__),
    "docs_and_configs",
    "runtime_deletion_threshold.txt",
)


def normalize_deletion_threshold(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return min(MAX_DELETION_THRESHOLD, max(0.0, parsed))


def normalize_deletion_threshold_mode(value):
    if value == "relative":
        return "relative"
    if value == "off":
        return "off"
    return "absolute"


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False

    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def normalize_bc_family_modulus(value):
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_BC_FAMILY_MODULUS

    clamped = min(
        MAX_BC_FAMILY_MODULUS,
        max(MIN_BC_FAMILY_MODULUS, parsed),
    )
    if _is_prime(clamped):
        return clamped

    max_distance = max(
        clamped - MIN_BC_FAMILY_MODULUS,
        MAX_BC_FAMILY_MODULUS - clamped,
    )
    for distance in range(1, max_distance + 1):
        lower = clamped - distance
        if lower >= MIN_BC_FAMILY_MODULUS and _is_prime(lower):
            return lower

        upper = clamped + distance
        if upper <= MAX_BC_FAMILY_MODULUS and _is_prime(upper):
            return upper

    return DEFAULT_BC_FAMILY_MODULUS


def deletion_threshold_components(value, mode="absolute"):
    threshold = normalize_deletion_threshold(value)
    normalized_mode = normalize_deletion_threshold_mode(mode)
    if normalized_mode == "off":
        return 0.0, 0.0
    if normalized_mode == "relative":
        return 0.0, threshold
    return threshold, 0.0


def write_runtime_deletion_threshold_signal(value, relative_value=0.0, mode=None):
    threshold = normalize_deletion_threshold(value)
    if mode is None:
        absolute_threshold = threshold
        relative_threshold = normalize_deletion_threshold(relative_value)
    else:
        absolute_threshold, relative_threshold = deletion_threshold_components(
            threshold,
            mode,
        )
    path = RUNTIME_DELETION_THRESHOLD_SIGNAL_PATH
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, "w", encoding="ascii") as file:
        file.write(f"{absolute_threshold:.12g} {relative_threshold:.12g}\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)
    return threshold


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


def decode_board_pure(encoded_board: np.uint64) -> np.ndarray:
    encoded_board = np.uint64(encoded_board)
    board = np.zeros((4, 4), dtype=np.int32)
    for i in range(3, -1, -1):
        for j in range(3, -1, -1):
            encoded_num = (encoded_board >> np.uint64(4 * ((3 - i) * 4 + (3 - j)))) & np.uint64(0xF)
            board[i, j] = 2 ** encoded_num if encoded_num > 0 else 0
    return board


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
    new_pattern_catalog = {}

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

        fmt_seeds_raw = data.get("seed boards", ["0xffffffff"])
        fmt_seeds = [np.uint64(int(m, 16)) for m in fmt_seeds_raw]
        ini_decoded = decode_board_pure(fmt_seeds[0])
        board_sum = int(ini_decoded.sum())
        fmt_seeds = np.array(
            [board for board in fmt_seeds if int(decode_board_pure(board).sum()) == board_sum],
            dtype=np.uint64,
        )

        count = int(np.sum(ini_decoded == 32768))
        fixed_pos = np.array(
            find_f_nibble_positions(get_nibble_intersection(def_masks)),
            dtype=np.uint8,
        )
        free_count = count - len(fixed_pos)
        new_pattern_32k_tiles_map[name] = [count, free_count, fixed_pos]
        new_pattern_catalog[name] = {
            "name": name,
            "category": cat,
            "pattern_masks": def_masks,
            "success_shifts": def_shifts,
            "canonical_mode": data.get("canonical mode", "identity"),
            "seed_boards": fmt_seeds,
            "nums_adjust": -board_sum,
            "extra_steps": data.get("extra steps", 36),
            "count_32k": count,
            "free_count_32k": free_count,
            "fixed_pos_32k": fixed_pos,
        }

    # 3. 将 PATTERN_DATA 注入 Calculator

    if "others" in new_category_info:
        # pop 会取出该键值对，重新赋值会将其插入到字典末尾
        new_category_info["others"] = new_category_info.pop("others")

    return (
        new_category_info,
        new_pattern_catalog,
        new_pattern_32k_tiles_map,
        new_pattern_data,
    )


def _build_formation_info(pattern_catalog, pattern_data):
    import engine_core.Calculator as Calculator

    Calculator.PATTERN_DATA = pattern_data  # type: ignore[attr-defined]
    Calculator.update_logic_functions()

    new_formation_info = {}
    for name, meta in pattern_catalog.items():
        try:
            is_pattern_func = getattr(Calculator, f"is_{name}_pattern")
            is_success_func = getattr(Calculator, f"is_{name}_success")
        except AttributeError:
            logger.warning(f"Functions for {name} not found in Calculator after injection.")
            continue

        symm_func = getattr(Calculator, "canonical_" + meta["canonical_mode"])
        new_formation_info[name] = [
            meta["nums_adjust"],
            is_pattern_func,
            symm_func,
            is_success_func,
            meta["seed_boards"],
            meta["extra_steps"],
        ]
    return new_formation_info


class _LazyFormationInfo(Mapping):
    def __init__(self, pattern_catalog, pattern_data):
        self._pattern_catalog = pattern_catalog
        self._pattern_data = pattern_data
        self._lock = threading.Lock()
        self._data: Optional[dict[str, list]] = None

    def _ensure_loaded(self):
        if self._data is None:
            with self._lock:
                if self._data is None:
                    self._data = _build_formation_info(self._pattern_catalog, self._pattern_data)
        return self._data

    def get(self, key, default=None):
        return self._ensure_loaded().get(key, default)

    def __getitem__(self, key):
        return self._ensure_loaded()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._ensure_loaded())

    def __len__(self) -> int:
        return len(self._ensure_loaded())

    def __contains__(self, key):
        return key in self._ensure_loaded()

    def keys(self):
        return self._ensure_loaded().keys()

    def items(self):
        return self._ensure_loaded().items()

    def values(self):
        return self._ensure_loaded().values()


category_info, pattern_catalog, pattern_32k_tiles_map, _pattern_data = load_patterns_from_file()
formation_info = _LazyFormationInfo(pattern_catalog, _pattern_data)


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
            cls.clean_pattern_paths()
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
            "algorithm_mode": "ex",
            "SmallTileSumLimit": 96,
            "advanced_algo": False,
            "zmask_algo": True,
            "chunked_solve": False,
            "bc_family_modulus": DEFAULT_BC_FAMILY_MODULUS,
            "direct_io": True,
            "direct_io_queue_depth": 16,
            "direct_io_chunk_mib": 8,
            "deletion_threshold": 0.0,
            "deletion_threshold_mode": "absolute",
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
                    if data.get("algorithm_mode") not in ("classic", "ad", "ex", "exad", "bc"):
                        advanced = bool(data.get("advanced_algo", defaults["advanced_algo"]))
                        zmask = bool(data.get("zmask_algo", defaults["zmask_algo"]))
                        if advanced and zmask:
                            data["algorithm_mode"] = "exad"
                        elif advanced:
                            data["algorithm_mode"] = "ad"
                        elif zmask:
                            data["algorithm_mode"] = "ex"
                        else:
                            data["algorithm_mode"] = "classic"
                        updated = True
                    for k, v in defaults.items():
                        if k not in data:
                            data[k] = v
                            updated = True
                    bc_family_modulus = normalize_bc_family_modulus(
                        data.get("bc_family_modulus", defaults["bc_family_modulus"])
                    )
                    if data.get("bc_family_modulus") != bc_family_modulus:
                        data["bc_family_modulus"] = bc_family_modulus
                        updated = True
                    for direct_io_key in (
                        "direct_io",
                        "direct_io_queue_depth",
                        "direct_io_chunk_mib",
                    ):
                        if data.get(direct_io_key) != defaults[direct_io_key]:
                            data[direct_io_key] = defaults[direct_io_key]
                            updated = True
                    normalized_deletion_threshold = normalize_deletion_threshold(
                        data.get("deletion_threshold", defaults["deletion_threshold"])
                    )
                    if data.get("deletion_threshold") != normalized_deletion_threshold:
                        data["deletion_threshold"] = normalized_deletion_threshold
                        updated = True
                    normalized_deletion_threshold_mode = normalize_deletion_threshold_mode(
                        data.get("deletion_threshold_mode", defaults["deletion_threshold_mode"])
                    )
                    if data.get("deletion_threshold_mode") != normalized_deletion_threshold_mode:
                        data["deletion_threshold_mode"] = normalized_deletion_threshold_mode
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

        tmp_filename = os.fspath(filename) + ".tmp"
        with open(tmp_filename, "wb") as file:
            pickle.dump(config, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_filename, filename)
        if "deletion_threshold" in config or "deletion_threshold_mode" in config:
            try:
                write_runtime_deletion_threshold_signal(
                    config.get("deletion_threshold", 0.0),
                    mode=config.get("deletion_threshold_mode", "absolute"),
                )
            except OSError as exc:
                logger.error(f"Failed to write runtime deletion threshold signal: {exc}")

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
    def clean_pattern_paths(cls, pattern=None, spawn_rate4=None, persist=True):
        if not cls._instance:
            return False
        config = cls._instance.config
        filepath_map = config.get("filepath_map", {})
        if not isinstance(filepath_map, dict):
            config["filepath_map"] = {}
            if persist:
                cls.save_config(config)
            return False

        if pattern is None:
            pattern_keys = list(filepath_map.keys())
        else:
            if spawn_rate4 is None:
                spawn_rate4 = config.get("4_spawn_rate", 0.1)
            pattern_keys = [cls.get_pattern_key(pattern, float(spawn_rate4))]

        valid_paths_found = False
        changed = False
        table_suffixes = (
            ".book",
            ".z",
            "b",
            ".zbook",
            ".exzbook",
            ".exadbook",
            ".exadzbook",
            ".bccmp",
        )

        for pattern_key in pattern_keys:
            try:
                pattern_name = str(pattern_key[0])
            except (TypeError, IndexError):
                continue

            original_paths = filepath_map.get(pattern_key, [])
            path_entries = (
                original_paths if isinstance(original_paths, (list, tuple)) else []
            )
            valid_paths = []
            prefix = f"{pattern_name}_"

            for entry in path_entries:
                try:
                    file_path, success_rate_dtype = entry
                    file_path = os.fspath(file_path)
                except (TypeError, ValueError):
                    continue

                if not isinstance(file_path, str) or not file_path:
                    continue

                try:
                    if not os.path.isdir(file_path):
                        continue
                    with os.scandir(file_path) as items:
                        table_files = [
                            item.name for item in items if item.name.startswith(prefix)
                        ]
                    bc_positions = {
                        name[:-len(".bcpos")]
                        for name in table_files
                        if name.endswith(".bcpos")
                    }
                    bc_successes = {
                        name[:-len(".bcsuc")]
                        for name in table_files
                        if name.endswith(".bcsuc")
                    }
                    has_table_file = any(
                        name.endswith(table_suffixes) for name in table_files
                    ) or bool(bc_positions.intersection(bc_successes))
                except OSError as exc:
                    logger.warning(f"Unable to inspect table path {file_path}: {exc}")
                    continue

                if has_table_file:
                    valid_paths.append((file_path, success_rate_dtype))

            if valid_paths:
                valid_paths_found = True
            if valid_paths != original_paths:
                filepath_map[pattern_key] = valid_paths
                changed = True

        if changed and persist:
            cls.save_config(config)

        return valid_paths_found

    @classmethod
    def check_pattern_file(cls, pattern):
        if not cls._instance:
            return False
        spawn_rate4 = cls._instance.config.get("4_spawn_rate", 0.1)
        return cls.clean_pattern_paths(pattern, spawn_rate4)

    @classmethod
    def get_available_pattern_targets(cls):
        if not cls._instance:
            return {}

        config = cls._instance.config
        current_spawn_rate4 = float(config.get("4_spawn_rate", 0.1))
        available_targets = {}
        for pattern_key, path_list in config.get("filepath_map", {}).items():
            try:
                full_pattern, table_spawn_rate4 = pattern_key
                table_spawn_rate4 = float(table_spawn_rate4)
            except (TypeError, ValueError):
                continue

            if not path_list or abs(table_spawn_rate4 - current_spawn_rate4) > 1e-4:
                continue

            pattern_name, separator, target = str(full_pattern).rpartition("_")
            if not separator or not pattern_name or not target:
                continue
            available_targets.setdefault(pattern_name, set()).add(target)

        def target_sort_key(value):
            try:
                return (0, int(value))
            except ValueError:
                return (1, value)

        return {
            pattern_name: sorted(targets, key=target_sort_key)
            for pattern_name, targets in available_targets.items()
        }

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


"""CPU-time returning clock() function."""
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
