from __future__ import annotations

from dataclasses import dataclass

from backend.replay_2048next import (
    EXT_AI_USED,
    EXT_DIFFICULTY_CHANGE,
    EXT_RANKED_METADATA,
    EXT_RULESET,
    CheckpointRecord,
    EndRecord,
    ExtensionRecord,
    MoveRecord,
    RANKED_RULES_VERSION,
    UndoRecord,
    decode_2048next_replay,
)

from .rules import DIRECTIONS, board_codes, evil_spawn, initial_board, legal_moves, random_spawn, simulate_move


MAX_RANKED_MOVES = 100_000
GAMER_HIGH_SCORE_BOARD = "gamer_high_score"
GAMER_ADVERSARIAL_BOARD = "gamer_adversarial"


class RankedValidationError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ValidatedGame:
    board_key: str
    score: int
    max_tile: int
    move_count: int
    used_ai: bool
    final_board_codes: tuple[int, ...]


def classify_ranked_candidate(
    *,
    seed_hex: str,
    rules_version: int,
    record_encoding: str,
) -> str:
    """Classify a structurally valid ranked record without replaying the game."""
    try:
        replay = decode_2048next_replay(record_encoding)
    except ValueError as exc:
        raise RankedValidationError("invalid_record") from exc
    if replay.width != 4 or replay.height != 4:
        raise RankedValidationError("invalid_board_size")
    try:
        metadata = replay.ranked_metadata()
    except ValueError as exc:
        raise RankedValidationError("invalid_record") from exc
    if metadata != (rules_version, seed_hex.lower()):
        raise RankedValidationError("seed_mismatch")
    if rules_version != RANKED_RULES_VERSION:
        raise RankedValidationError("unsupported_rules")

    current_difficulty: int | None = None
    metadata_count = 0
    ruleset_count = 0
    ai_count = 0
    move_count = 0
    all_adversarial = True
    ended = False
    for position, record in enumerate(replay.records):
        if ended:
            raise RankedValidationError("data_after_end")
        if isinstance(record, (UndoRecord, CheckpointRecord)):
            raise RankedValidationError("disallowed_record")
        if isinstance(record, EndRecord):
            if position != len(replay.records) - 1:
                raise RankedValidationError("data_after_end")
            ended = True
            continue
        if isinstance(record, ExtensionRecord):
            if record.extension_type == EXT_RANKED_METADATA:
                metadata_count += 1
                if metadata_count != 1 or move_count:
                    raise RankedValidationError("invalid_metadata_order")
            elif record.extension_type == EXT_RULESET:
                ruleset_count += 1
                if ruleset_count != 1 or record.payload != b"pow2" or move_count:
                    raise RankedValidationError("invalid_ruleset")
            elif record.extension_type == EXT_DIFFICULTY_CHANGE:
                if len(record.payload) != 1 or record.payload[0] > 100:
                    raise RankedValidationError("invalid_difficulty")
                current_difficulty = int(record.payload[0])
            elif record.extension_type == EXT_AI_USED:
                ai_count += 1
                if ai_count != 1 or record.payload:
                    raise RankedValidationError("invalid_ai_marker")
            else:
                raise RankedValidationError("unknown_extension")
            continue
        if not isinstance(record, MoveRecord):
            raise RankedValidationError("unknown_record")
        if current_difficulty is None:
            raise RankedValidationError("missing_difficulty")
        move_count += 1
        if move_count > MAX_RANKED_MOVES:
            raise RankedValidationError("too_many_moves")
        all_adversarial = all_adversarial and current_difficulty == 100

    if metadata_count != 1 or ruleset_count != 1 or not ended or move_count < 1:
        raise RankedValidationError("incomplete_record")
    return GAMER_ADVERSARIAL_BOARD if all_adversarial else GAMER_HIGH_SCORE_BOARD


def validate_ranked_game(
    *,
    seed_hex: str,
    rules_version: int,
    record_encoding: str,
    claimed_score: int,
    claimed_final_board: list[int],
) -> ValidatedGame:
    try:
        replay = decode_2048next_replay(record_encoding)
    except ValueError as exc:
        raise RankedValidationError("invalid_record") from exc
    if replay.width != 4 or replay.height != 4:
        raise RankedValidationError("invalid_board_size")
    try:
        metadata = replay.ranked_metadata()
    except ValueError as exc:
        raise RankedValidationError("invalid_record") from exc
    if metadata != (rules_version, seed_hex.lower()):
        raise RankedValidationError("seed_mismatch")
    if rules_version != RANKED_RULES_VERSION:
        raise RankedValidationError("unsupported_rules")

    board, expected_initial, rng = initial_board(seed_hex)
    if replay.initial_tiles != expected_initial:
        raise RankedValidationError("initial_tiles_mismatch")

    current_difficulty: int | None = None
    metadata_count = 0
    ruleset_count = 0
    ai_count = 0
    used_ai = False
    ended = False
    move_count = 0
    score = 0
    all_adversarial = True

    for position, record in enumerate(replay.records):
        if ended:
            raise RankedValidationError("data_after_end")
        if isinstance(record, (UndoRecord, CheckpointRecord)):
            raise RankedValidationError("disallowed_record")
        if isinstance(record, EndRecord):
            if position != len(replay.records) - 1:
                raise RankedValidationError("data_after_end")
            ended = True
            continue
        if isinstance(record, ExtensionRecord):
            if record.extension_type == EXT_RANKED_METADATA:
                metadata_count += 1
                if metadata_count != 1 or move_count:
                    raise RankedValidationError("invalid_metadata_order")
            elif record.extension_type == EXT_RULESET:
                ruleset_count += 1
                if ruleset_count != 1 or record.payload != b"pow2" or move_count:
                    raise RankedValidationError("invalid_ruleset")
            elif record.extension_type == EXT_DIFFICULTY_CHANGE:
                if len(record.payload) != 1 or record.payload[0] > 100:
                    raise RankedValidationError("invalid_difficulty")
                current_difficulty = int(record.payload[0])
            elif record.extension_type == EXT_AI_USED:
                ai_count += 1
                if ai_count != 1 or record.payload:
                    raise RankedValidationError("invalid_ai_marker")
                used_ai = True
            else:
                raise RankedValidationError("unknown_extension")
            continue
        if not isinstance(record, MoveRecord):
            raise RankedValidationError("unknown_record")
        if current_difficulty is None:
            raise RankedValidationError("missing_difficulty")
        move_count += 1
        if move_count > MAX_RANKED_MOVES:
            raise RankedValidationError("too_many_moves")
        direction = DIRECTIONS.get(record.direction)
        if direction is None:
            raise RankedValidationError("invalid_direction")
        moved_board, score_delta = simulate_move(board, direction)
        if moved_board == board:
            raise RankedValidationError("invalid_move")

        branch = rng.next_float()
        use_evil = current_difficulty >= 100 or (
            current_difficulty > 0 and branch < current_difficulty / 100.0
        )
        try:
            expected_index, expected_exponent = (
                evil_spawn(moved_board, depth=5)
                if use_evil
                else random_spawn(moved_board, rng)
            )
        except Exception as exc:
            raise RankedValidationError("spawn_validation_failed") from exc
        if (
            record.spawn_index != expected_index
            or record.spawn_value_bit + 1 != expected_exponent
        ):
            raise RankedValidationError("spawn_mismatch")
        moved_board[expected_index] = 2**expected_exponent
        board = moved_board
        score += score_delta
        all_adversarial = all_adversarial and current_difficulty == 100

    if metadata_count != 1 or ruleset_count != 1 or not ended or move_count < 1:
        raise RankedValidationError("incomplete_record")
    if legal_moves(board):
        raise RankedValidationError("game_not_over")
    try:
        actual_codes = tuple(board_codes(board))
    except ValueError as exc:
        raise RankedValidationError("unsupported_final_board") from exc
    if claimed_score != score:
        raise RankedValidationError("score_mismatch")
    if tuple(claimed_final_board) != actual_codes:
        raise RankedValidationError("final_board_mismatch")
    return ValidatedGame(
        board_key=GAMER_ADVERSARIAL_BOARD if all_adversarial else GAMER_HIGH_SCORE_BOARD,
        score=score,
        max_tile=max(board),
        move_count=move_count,
        used_ai=used_ai,
        final_board_codes=actual_codes,
    )
