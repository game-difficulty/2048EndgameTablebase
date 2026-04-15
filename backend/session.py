import numpy as np

from egtb_core.BookReader import BookReaderDispatcher
from backend.minigames.session import MinigameSessionState
from Config import SingletonConfig
from egtb_core.VBoardMover import decode_board
from egtb_core.replay_utils import empty_replay

_SHARED_EVIL_GEN = None
_SHARED_AI_DISPATCHER = None
_SHARED_BOOK_READER = None


def u64(val):
    if isinstance(val, str):
        try:
            return int(val, 16) & 0xFFFFFFFFFFFFFFFF
        except ValueError:
            return int(float(val)) & 0xFFFFFFFFFFFFFFFF
    return int(val) & 0xFFFFFFFFFFFFFFFF


def safe_hex(val):
    return format(u64(val), "016x")


def np_u64(val):
    return np.uint64(u64(val))


def normalize_gamer_special_tiles(raw):
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = raw
    else:
        return {}

    normalized = {}
    for item in items:
        try:
            if isinstance(item, tuple) and len(item) == 2:
                index, value = item
            else:
                index, value = item[0], item[1]
            index = int(index)
            value = int(value)
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= index < 16 and value > 32768:
            normalized[index] = value
    return normalized


class GameSession:
    def __init__(self, client_id=""):
        self.client_id = client_id
        if client_id.startswith("gamer_"):
            game_state = SingletonConfig().config.get("game_state", [0, 0, 0])
        else:
            game_state = [0, 0, 0]

        self.board_encoded = np_u64(game_state[0])
        self.score = int(game_state[1])
        self.best_score = int(game_state[2])
        self.gamer_special_tiles = (
            normalize_gamer_special_tiles(game_state[3]) if len(game_state) > 3 else {}
        )
        self.state = "active"
        self.history = [(self.board_encoded, self.score)]
        self.move_history = [None]
        self.gamer_special_history = [dict(self.gamer_special_tiles)]

        self.evil_gen = None
        self.difficulty = 0.0

        self.ai_dispatcher = None

        self.book_reader = None
        self.current_pattern = ""
        self.pattern_settings = ["", ""]
        self.spawn_mode = 0
        self.moved = 0

        self.speed = 100.0
        self.played_length = 0
        self.trainer_results = {}
        self.use_variant = False

        self.recording_state = False
        self.records = np.empty(
            0, dtype=[("changes", np.uint8), ("rates", np.uint32, (4,))]
        )
        self.record_length = 0
        self.success_rate_dtype = "uint32"
        self.record_result_history = []
        self.record_result_dtype = None
        self.record_animation_history = []

        self.tester_pattern = ["?", "?"]
        self.tester_full_pattern = ""
        self.tester_results = {}
        self.tester_result_dtype = "?"
        self.tester_best_move = None
        self.tester_text_visible = bool(
            SingletonConfig().config.get("dis_text", True)
        )
        self.tester_logs = []
        self.tester_combo = 0
        self.tester_goodness_of_fit = 1.0
        self.tester_max_combo = 0
        self.tester_performance_stats = {
            "Perfect!": 0,
            "Excellent!": 0,
            "Nice try!": 0,
            "Not bad!": 0,
            "Mistake!": 0,
            "Blunder!": 0,
            "Terrible!": 0,
        }
        self.tester_ready = False
        self.tester_table_found = False
        self.tester_status = ""
        self.tester_record = np.zeros(
            4000, dtype="uint64,uint8,uint32,uint32,uint32,uint32"
        )
        self.tester_step_count = 0
        self.tester_last_step = {
            "board_lines": [],
            "result_lines": [],
            "message_lines": [],
            "evaluation": "",
            "direction": None,
            "best_move": None,
            "loss": None,
            "goodness_of_fit": None,
        }

        self.replay_record = empty_replay()
        self.replay_pattern = ""
        self.replay_source = ""
        self.replay_status = ""
        self.replay_loaded = False
        self.replay_use_variant = False
        self.replay_current_step = 0
        self.replay_board_encoded = np_u64(0)
        self.replay_results = {}
        self.replay_current_move = None
        self.replay_best_move = None
        self.replay_loss = None
        self.replay_gof = None
        self.replay_combo = 0
        self.replay_points_rank = []
        self.replay_losses = []
        self.replay_summary = {
            "total_moves": 0,
            "final_gof": 0.0,
            "max_combo": 0,
            "counts": {},
        }

        self.notebook_pattern = ""
        self.notebook_board_encoded = np_u64(0)
        self.notebook_best_move = None
        self.notebook_current_count = 0
        self.notebook_combo = 0
        self.notebook_correct = 0
        self.notebook_incorrect = 0
        self.notebook_weight_mode = 0
        self.notebook_unseen_boards = []
        self.notebook_answered = False
        self.notebook_last_direction = None
        self.notebook_answer_correct = None
        self.notebook_status = ""

        self.minigame_session = MinigameSessionState()

    def ensure_evil_gen(self):
        global _SHARED_EVIL_GEN
        if _SHARED_EVIL_GEN is None:
            try:
                from ai_and_sort.ai_core import EvilGen
            except Exception as exc:
                raise RuntimeError(
                    "AI core module is unavailable. Gamer evil spawn requires ai_and_sort.ai_core."
                ) from exc
            _SHARED_EVIL_GEN = EvilGen(self.board_encoded)  # type: ignore
        self.evil_gen = _SHARED_EVIL_GEN
        return self.evil_gen

    def ensure_ai_dispatcher(self):
        global _SHARED_AI_DISPATCHER
        if _SHARED_AI_DISPATCHER is None:
            try:
                from egtb_core.AIPlayer import Dispatcher
            except Exception as exc:
                raise RuntimeError(
                    "AI dispatcher is unavailable. Gamer AI requires ai_and_sort.ai_core."
                ) from exc
            board_encoded = np_u64(self.board_encoded)
            _SHARED_AI_DISPATCHER = Dispatcher(
                decode_board(board_encoded),
                board_encoded,
            )
        self.ai_dispatcher = _SHARED_AI_DISPATCHER
        exponent = (100.0 - float(self.speed)) / 100.0
        self.ai_dispatcher.time_limit_ratio = 10.0**exponent
        return self.ai_dispatcher

    def ensure_book_reader(self):
        global _SHARED_BOOK_READER
        if _SHARED_BOOK_READER is None:
            _SHARED_BOOK_READER = BookReaderDispatcher()
        self.book_reader = _SHARED_BOOK_READER
        return self.book_reader
