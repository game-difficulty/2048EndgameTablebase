from __future__ import annotations

import struct
import unittest

from backend.battle.replay_records import (
    REPLAY_RATE_SCALE,
    REPLAY_RECORD_BYTES,
    append_replay_step,
    battle_replay_filename,
    encode_replay_step,
    finalize_replay,
)


class BattleReplayRecordTests(unittest.TestCase):
    def test_encodes_standard_rpl_record_and_sentinel(self) -> None:
        step = encode_replay_step(
            board=0x1234,
            selected_direction="right",
            spawn_index=7,
            spawn_value=4,
            rates={"left": 0.25, "right": 0.5, "up": 0.75, "down": 1.0},
        )
        self.assertEqual(len(step), REPLAY_RECORD_BYTES)
        board, change, left, right, up, down = struct.unpack("<QB4I", step)
        self.assertEqual(board, 0x1234)
        self.assertEqual((change >> 5) & 0b11, 1)
        self.assertEqual((change >> 1) & 0b1111, 7)
        self.assertEqual(change & 1, 1)
        self.assertEqual(
            (left, right, up, down),
            (
                REPLAY_RATE_SCALE // 4,
                REPLAY_RATE_SCALE // 2,
                REPLAY_RATE_SCALE * 3 // 4,
                REPLAY_RATE_SCALE,
            ),
        )

        replay_blob, recorded = append_replay_step(b"", step)
        replay = finalize_replay(replay_blob, terminal_board=0x5678)
        self.assertTrue(recorded)
        self.assertEqual(len(replay), REPLAY_RECORD_BYTES * 2)
        sentinel = struct.unpack("<QB4I", replay[REPLAY_RECORD_BYTES:])
        self.assertEqual(sentinel[0], 0x5678)
        self.assertEqual(sentinel[1:], (88, 666666666, 233333333, 314159265, 987654321))

    def test_scaled_rates_and_filename_remain_compatible(self) -> None:
        step = encode_replay_step(
            board=1,
            selected_direction="left",
            spawn_index=0,
            spawn_value=2,
            rates=(1, 2, 3, 4),
            rates_already_scaled=True,
        )
        self.assertEqual(struct.unpack("<QB4I", step)[2:], (1, 2, 3, 4))
        self.assertEqual(
            battle_replay_filename(
                mode_key="free goodness",
                full_pattern="L3/256",
                goodness_of_fit=0.98765,
            ),
            "battle_free_goodness_L3_256_0.9877.rpl",
        )


if __name__ == "__main__":
    unittest.main()
