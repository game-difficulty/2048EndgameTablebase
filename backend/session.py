import numpy as np

from ai_and_sort.ai_core import EvilGen
from egtb_core.AIPlayer import Dispatcher
from egtb_core.BookReader import BookReaderDispatcher
from backend.minigames.session import MinigameSessionState
from Config import SingletonConfig
from egtb_core.VBoardMover import decode_board
from egtb_core.replay_utils import empty_replay


def u64(val):
    if isinstance(val, str):
        try:
            return int(val, 16) & 0xFFFFFFFFFFFFFFFF
        except ValueError:
            return int(float(val)) & 0xFFFFFFFFFFFFFFFF
    return int(val) & 0xFFFFFFFFFFFFFFFF


def safe_hex(val):
    return format(u64(val), "016x")


class GameSession:
    def __init__(self, client_id=""):
        self.client_id = client_id
        if client_id.startswith("gamer_"):
            game_state = SingletonConfig().config.get("game_state", [0, 0, 0])
        else:
            game_state = [0, 0, 0]

        self.board_encoded = np.uint64(u64(game_state[0]))
        self.score = int(game_state[1])
        self.best_score = int(game_state[2])
        self.state = "active"
        self.history = [(self.board_encoded, self.score)]
        self.move_history = [None]

        self.evil_gen = EvilGen(self.board_encoded)  # type: ignore
        self.difficulty = 0.0

        self.ai_dispatcher = Dispatcher(
            decode_board(np.uint64(u64(self.board_encoded))),
            np.uint64(u64(self.board_encoded)),
        )

        self.book_reader = BookReaderDispatcher()
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
        self.replay_board_encoded = np.uint64(0)
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
        self.notebook_board_encoded = np.uint64(0)
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
