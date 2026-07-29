import unittest
from types import SimpleNamespace

import numpy as np

from backend.replay import _replay_sync_step
from engine_core.replay_utils import (
    REPLAY_DTYPE,
    build_step_transition,
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

    def test_final_record_transition_remains_playable(self):
        record = np.zeros(1, dtype=REPLAY_DTYPE)
        record[0]["f0"] = np.uint64(0x0210030129AB4CDE)
        record[0]["f1"] = _change(1, 0, 1)

        self.assertTrue(replay_transition_matches_next_snapshot(record, 0))

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
