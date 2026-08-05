import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import engine_core.BoardMover as bm
import engine_core.VBoardMover as vbm
from backend.analysis_core import ReplayDecoder
from backend.replay import _replay_sync_step
from engine_core.replay_utils import (
    REPLAY_DTYPE,
    build_step_transition,
    load_replay_file_with_terminal_board,
    replay_sentinel,
    replay_transition_matches_next_snapshot,
)


def _change(move_bits, spawn_pos, spawn_exp):
    return np.uint8(
        ((move_bits & 0b11) << 5)
        | ((spawn_pos & 0b1111) << 1)
        | ((spawn_exp - 1) & 0b1)
    )


class ReplayDiscontinuityTests(unittest.TestCase):
    def test_transition_uses_row_major_spawn_position(self):
        record = np.zeros(2, dtype=REPLAY_DTYPE)
        record[0]["f0"] = np.uint64(0x0210030129AB4CDE)
        record[0]["f1"] = _change(1, 0, 1)
        record[1]["f0"] = np.uint64(0x1021003129AB4CDE)

        transition = build_step_transition(record, 0)

        self.assertEqual(
            int(transition["next_board_encoded"]),
            int(record[1]["f0"]),
        )
        self.assertEqual(transition["appear_tile"], {"index": 0, "value": 2})
        self.assertTrue(replay_transition_matches_next_snapshot(record, 0))

    def test_discontinuity_does_not_match_next_snapshot(self):
        record = np.zeros(2, dtype=REPLAY_DTYPE)
        record[0]["f0"] = np.uint64(0x0244157829AB1CDE)
        record[0]["f1"] = _change(2, 12, 2)
        record[1]["f0"] = np.uint64(0x4211167829AB1CDE)

        transition = build_step_transition(record, 0)

        self.assertEqual(
            int(transition["next_board_encoded"]),
            0x1244257819AB2CDE,
        )
        self.assertFalse(replay_transition_matches_next_snapshot(record, 0))

    def test_legacy_final_record_transition_remains_playable(self):
        record = np.zeros(1, dtype=REPLAY_DTYPE)
        record[0]["f0"] = np.uint64(0x0210030129AB4CDE)
        record[0]["f1"] = _change(1, 0, 1)

        self.assertTrue(replay_transition_matches_next_snapshot(record, 0))

    def test_final_32k_merge_uses_capped_board_representation(self):
        board = np.array(
            [
                [32768, 0, 32768, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        record = np.zeros(1, dtype=REPLAY_DTYPE)
        record[0]["f0"] = bm.encode_board(board)
        record[0]["f1"] = _change(0, 15, 1)

        transition = build_step_transition(record, 0)

        np.testing.assert_array_equal(
            np.array(
                [
                    [32768, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 2],
                ],
                dtype=np.int32,
            ),
            bm.decode_board(transition["next_board_encoded"]),
        )
        self.assertEqual(transition["score_delta"], 65536)

    def test_terminal_snapshot_is_loaded_from_sentinel(self):
        terminal_board = np.uint64(0x123456789ABCDEF0)
        replay = np.zeros(2, dtype=REPLAY_DTYPE)
        replay[0]["f0"] = np.uint64(0x0210030129AB4CDE)
        replay[0]["f1"] = _change(1, 0, 1)
        replay[1] = replay_sentinel(terminal_board)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal.rpl"
            replay.tofile(path)
            record, loaded_terminal = load_replay_file_with_terminal_board(path)

        self.assertEqual(len(record), 1)
        self.assertEqual(int(loaded_terminal), int(terminal_board))

    def test_mismatched_terminal_snapshot_disables_final_animation(self):
        record = np.zeros(1, dtype=REPLAY_DTYPE)
        record[0]["f0"] = np.uint64(0x0210030129AB4CDE)
        record[0]["f1"] = _change(1, 0, 1)
        terminal_board = np.uint64(0x123456789ABCDEF0)
        session = SimpleNamespace(
            replay_record=record,
            replay_current_step=0,
            replay_use_variant=False,
            replay_terminal_board_encoded=terminal_board,
            replay_losses=[1.0],
        )

        metadata = _replay_sync_step(
            session,
            1,
            animate=True,
            previous_step=0,
        )

        self.assertEqual(metadata, {})
        self.assertEqual(int(session.replay_board_encoded), int(terminal_board))

    def test_tester_snapshot_replay_does_not_create_a_fake_final_move(self):
        snapshot_dtype = np.dtype(
            [("f0", "uint64"), ("f1", "uint32"), ("f2", "uint8")]
        )
        snapshots = np.zeros(3, dtype=snapshot_dtype)
        initial = bm.encode_board(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 2, 0, 2],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
        )
        after_right, score = bm.s_move_board(initial, 2)
        after_right |= np.uint64(1) << np.uint64((15 - 3) * 4)
        after_down, next_score = bm.s_move_board(after_right, 4)
        after_down |= np.uint64(2) << np.uint64((15 - 0) * 4)
        snapshots[0] = (initial, 0, 0)
        snapshots[1] = (after_right, score, 2)
        snapshots[2] = (after_down, int(score) + int(next_score), 4)

        decoder = ReplayDecoder("", bm, vbm)
        decoder._decode_test_replay(snapshots)

        self.assertEqual(len(decoder.record_list), 2)
        self.assertEqual(decoder.record_list["f2"].tolist(), [2, 4])
        self.assertEqual(decoder.record_list["f3"].tolist(), [1, 2])
        self.assertEqual(decoder.record_list["f4"].tolist(), [3, 0])

    def test_step_across_discontinuity_uses_target_snapshot_without_animation(self):
        record = np.zeros(2, dtype=REPLAY_DTYPE)
        record[0]["f0"] = np.uint64(0x0244157829AB1CDE)
        record[0]["f1"] = _change(2, 12, 2)
        record[1]["f0"] = np.uint64(0x4211167829AB1CDE)
        session = SimpleNamespace(
            replay_record=record,
            replay_current_step=0,
            replay_use_variant=False,
            replay_losses=[1.0, 1.0],
        )

        metadata = _replay_sync_step(
            session,
            1,
            animate=True,
            previous_step=0,
        )

        self.assertEqual(metadata, {})
        self.assertEqual(
            int(session.replay_board_encoded),
            int(record[1]["f0"]),
        )


if __name__ == "__main__":
    unittest.main()
