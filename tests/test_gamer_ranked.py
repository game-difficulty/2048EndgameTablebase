import unittest
import base64
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import zlib

from backend.gamer_ranked.prng import Xoshiro128StarStar
from backend.gamer_ranked.rules import board_codes, initial_board, legal_moves, random_spawn, simulate_move
from backend.gamer_ranked.validator import (
    GAMER_ADVERSARIAL_BOARD,
    GAMER_HIGH_SCORE_BOARD,
    RankedValidationError,
    validate_ranked_game,
)
from backend.auth.db import auth_db, init_auth_db
from backend.gamer_ranked.service import (
    create_ranked_run,
    get_ranked_run,
    process_one_pending_run,
    prune_ranked_replays,
    public_replay,
    submit_ranked_run,
)
from backend.leaderboards.service import leaderboard_payload
from backend.replay_2048next import (
    EXT_AI_USED,
    EXT_DIFFICULTY_CHANGE,
    EXT_RANKED_METADATA,
    EXT_RULESET,
    EndRecord,
    ExtensionRecord,
    MoveRecord,
    decode_2048next_replay,
)


SEED = "00000001000000020000000300000004"
JS_FIXTURE = (
    "REPLAY_v1RPL_B64_"
    "UlBMMUQAAhARg2QRAQAAAAEAAAACAAAAAwAAAASDAgRwb3cyg2UBAAt7g2YAhMwwCkc="
)

DIRECTION_CODES = {"up": 0, "right": 1, "down": 2, "left": 3}


def _uleb(value: int) -> bytes:
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        result.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(result)


def _extension(extension_type: int, payload: bytes) -> bytes:
    return bytes((131,)) + _uleb(extension_type) + _uleb(len(payload)) + payload


def _completed_random_game(seed: str = SEED) -> tuple[str, int, list[int]]:
    board, initial_tiles, rng = initial_board(seed)
    payload = bytearray(b"RPL1")
    payload.extend((0x44, 0, 2))
    for index, value_bit in initial_tiles:
        payload.append(index | (value_bit << 4))
    payload.extend(_extension(EXT_RANKED_METADATA, bytes((1,)) + bytes.fromhex(seed)))
    payload.extend(_extension(EXT_RULESET, b"pow2"))
    payload.extend(_extension(EXT_DIFFICULTY_CHANGE, b"\x00"))
    score = 0
    while legal_moves(board):
        direction = legal_moves(board)[0]
        moved, delta = simulate_move(board, direction)
        rng.next_float()  # Every ranked move consumes the branch draw.
        spawn_index, spawn_exponent = random_spawn(moved, rng)
        moved[spawn_index] = 2**spawn_exponent
        payload.append(
            DIRECTION_CODES[direction]
            | (spawn_index << 2)
            | ((spawn_exponent - 1) << 6)
        )
        payload.extend(_uleb(10))
        board = moved
        score += delta
    payload.append(132)
    payload.extend((zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "little"))
    return "REPLAY_v1RPL_B64_" + base64.b64encode(payload).decode("ascii"), score, board_codes(board)


def _completed_adversarial_game(seed: str = SEED) -> tuple[str, int, list[int]]:
    board, initial_tiles, rng = initial_board(seed)
    payload = bytearray(b"RPL1")
    payload.extend((0x44, 0, 2))
    for index, value_bit in initial_tiles:
        payload.append(index | (value_bit << 4))
    payload.extend(_extension(EXT_RANKED_METADATA, bytes((1,)) + bytes.fromhex(seed)))
    payload.extend(_extension(EXT_RULESET, b"pow2"))
    payload.extend(_extension(EXT_DIFFICULTY_CHANGE, b"\x64"))
    payload.extend(_extension(EXT_AI_USED, b""))
    score = 0
    while legal_moves(board):
        direction = legal_moves(board)[0]
        moved, delta = simulate_move(board, direction)
        rng.next_float()
        spawn_index = next(index for index, value in enumerate(moved) if value == 0)
        moved[spawn_index] = 2
        payload.append(DIRECTION_CODES[direction] | (spawn_index << 2))
        payload.extend(_uleb(10))
        board = moved
        score += delta
    payload.append(132)
    payload.extend((zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "little"))
    return "REPLAY_v1RPL_B64_" + base64.b64encode(payload).decode("ascii"), score, board_codes(board)


class GamerRankedContractTests(unittest.TestCase):
    def test_xoshiro_matches_javascript_vector(self):
        rng = Xoshiro128StarStar.from_seed_hex(SEED)
        self.assertEqual(
            [rng.next_u32() for _ in range(8)],
            [11520, 0, 5927040, 70819200, 2031721883, 1637235492, 1287239034, 3734860849],
        )

    def test_python_decodes_javascript_ranked_fixture(self):
        replay = decode_2048next_replay(JS_FIXTURE)
        self.assertEqual(replay.initial_tiles, ((0, 1), (1, 1)))
        self.assertEqual(replay.ranked_metadata(), (1, SEED))
        self.assertEqual(replay.text_extension(EXT_RULESET), "pow2")
        self.assertEqual(
            [type(record) for record in replay.records],
            [ExtensionRecord, ExtensionRecord, ExtensionRecord, MoveRecord, ExtensionRecord, EndRecord],
        )
        self.assertEqual(replay.records[0].extension_type, EXT_RANKED_METADATA)
        self.assertEqual(replay.records[2].extension_type, EXT_DIFFICULTY_CHANGE)
        self.assertEqual(replay.records[4].extension_type, EXT_AI_USED)

    def test_validator_replays_random_game_and_rejects_claimed_score_tampering(self):
        record, score, final_board = _completed_random_game()
        validated = validate_ranked_game(
            seed_hex=SEED,
            rules_version=1,
            record_encoding=record,
            claimed_score=score,
            claimed_final_board=final_board,
        )
        self.assertEqual(validated.board_key, GAMER_HIGH_SCORE_BOARD)
        self.assertEqual(validated.score, score)
        with self.assertRaisesRegex(RankedValidationError, "score_mismatch"):
            validate_ranked_game(
                seed_hex=SEED,
                rules_version=1,
                record_encoding=record,
                claimed_score=score + 4,
                claimed_final_board=final_board,
            )

    def test_validator_accepts_ai_and_classifies_all_difficulty_100_separately(self):
        record, score, final_board = _completed_adversarial_game()

        def first_empty_spawn(board, *, depth=5):
            self.assertEqual(depth, 5)
            return next(index for index, value in enumerate(board) if value == 0), 1

        with patch("backend.gamer_ranked.validator.evil_spawn", side_effect=first_empty_spawn):
            validated = validate_ranked_game(
                seed_hex=SEED,
                rules_version=1,
                record_encoding=record,
                claimed_score=score,
                claimed_final_board=final_board,
            )
        self.assertEqual(validated.board_key, GAMER_ADVERSARIAL_BOARD)
        self.assertTrue(validated.used_ai)


class GamerRankedServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.previous_db = os.environ.get("CLOUD_AUTH_DB")
        os.environ["CLOUD_AUTH_DB"] = str(Path(self.temporary.name) / "auth.sqlite3")
        init_auth_db()
        now = datetime.now(timezone.utc).isoformat()
        with auth_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name, display_name_key,
                 status, created_at, updated_at)
                VALUES ('ranked@example.com', 'ranked@example.com', 'x',
                        'Ranked Test', 'ranked test', 'active', ?, ?)
                """,
                (now, now),
            )
            self.user_id = int(cursor.lastrowid)

    def tearDown(self):
        if self.previous_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self.previous_db
        self.temporary.cleanup()

    def test_pending_run_is_verified_and_published(self):
        run = create_ranked_run(
            user_id=self.user_id,
            request_id="request-1",
            ip_address="127.0.0.1",
        )
        record, score, final_board = _completed_random_game(run["seed_hex"])
        submitted = submit_ranked_run(
            run_id=run["run_id"],
            user_id=self.user_id,
            score=score,
            final_board_codes=final_board,
            record_encoding=record,
            ip_address="127.0.0.1",
        )
        self.assertEqual(submitted["status"], "pending")
        self.assertTrue(process_one_pending_run())
        verified = get_ranked_run(run_id=run["run_id"], user_id=self.user_id)
        self.assertEqual(verified["status"], "verified")
        with auth_db() as db:
            score_row = db.execute(
                "SELECT replay_id FROM gamer_high_scores WHERE user_id = ?",
                (self.user_id,),
            ).fetchone()
        replay = public_replay(score_row["replay_id"])
        self.assertEqual(replay["record_blob"], record)
        board = leaderboard_payload("gamer_high_score")
        self.assertEqual(board["unit"], "points")
        self.assertEqual(board["entries"][0]["score"], score)
        self.assertEqual(board["entries"][0]["replay_id"], score_row["replay_id"])

    def test_replay_retention_keeps_only_the_highest_scores_per_board(self):
        now = datetime.now(timezone.utc).isoformat()
        with auth_db() as db:
            user_ids = [self.user_id]
            for index in range(2):
                cursor = db.execute(
                    """
                    INSERT INTO users
                    (email, email_identity, password_hash, display_name, display_name_key,
                     status, created_at, updated_at)
                    VALUES (?, ?, 'x', ?, ?, 'active', ?, ?)
                    """,
                    (
                        f"ranked-{index}@example.com",
                        f"ranked-{index}@example.com",
                        f"Ranked {index}",
                        f"ranked {index}",
                        now,
                        now,
                    ),
                )
                user_ids.append(int(cursor.lastrowid))
            for index, (user_id, points) in enumerate(zip(user_ids, (100, 300, 200))):
                run_id = f"retention-run-{index}"
                db.execute(
                    """
                    INSERT INTO gamer_ranked_runs
                    (run_id, user_id, request_id, seed_hex, rules_version, status,
                     started_at, expires_at, completed_at)
                    VALUES (?, ?, ?, ?, 1, 'verified', ?, ?, ?)
                    """,
                    (run_id, user_id, f"retention-{index}", SEED, now, now, now),
                )
                db.execute(
                    """
                    INSERT INTO gamer_high_scores
                    (user_id, board_key, score, max_tile, move_count, used_ai,
                     final_board, record_blob, replay_id, run_id, achieved_at, updated_at)
                    VALUES (?, 'gamer_high_score', ?, 128, 10, 0,
                            '[]', ?, ?, ?, ?, ?)
                    """,
                    (user_id, points, f"record-{index}", f"replay-{index}", run_id, now, now),
                )

        self.assertEqual(prune_ranked_replays(per_board_limit=2), 1)
        with auth_db() as db:
            remaining = [
                int(row["score"])
                for row in db.execute(
                    "SELECT score FROM gamer_high_scores ORDER BY score DESC"
                ).fetchall()
            ]
        self.assertEqual(remaining, [300, 200])


if __name__ == "__main__":
    unittest.main()
