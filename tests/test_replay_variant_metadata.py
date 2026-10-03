import asyncio
from types import SimpleNamespace
import unittest

import numpy as np
from backend.replay import _replay_load_record, _replay_pattern_from_path, _replay_sync_step, send_replay_state
from engine_core.replay_utils import REPLAY_DTYPE


class ReplayVariantMetadataTests(unittest.TestCase):
    def test_filename_patterns(self):
        for name, pattern in [
            ("3x3_sum-1800.rpl", "3x3_sum-1800"),
            ("C:/replays/2x4_sum-900_42.rpl", "2x4_sum-900"),
            ("3x3free8_512_42.rpl", "3x3free8_512"),
            ("custom_pattern_512_42.rpl", "custom_pattern_512"),
            ("renamed.rpl", ""),
        ]:
            self.assertEqual(_replay_pattern_from_path(name), pattern)

    def test_variant_flag_and_terminal_movement_are_shared_with_frontend(self):
        for pattern, first, last in [
            ("3x3_sum-1800", "011f000f000fffff", "210f000f000fffff"),
            ("2x4_sum-900", "ffff01100000ffff", "ffff21000000ffff"),
            ("3x4_1024", "011000000000ffff", "210000000000ffff"),
        ]:
            record = np.zeros(1, dtype=REPLAY_DTYPE)
            record[0]["f0"] = int(first, 16)
            record[0]["f1"] = 10 if pattern.startswith("2x4") else 2
            session = SimpleNamespace(user_id=None)
            _replay_load_record(session, record, pattern, use_variant=False)
            self.assertTrue(session.replay_use_variant)
            self.assertTrue(session.replay_variant_conflict)
            _replay_sync_step(session, 1, animate=True, previous_step=0)
            self.assertEqual(int(session.replay_board_encoded), int(last, 16))
            messages = []
            async def send_json(payload):
                messages.append(payload)
            asyncio.run(send_replay_state(SimpleNamespace(send_json=send_json), session))
            self.assertTrue(messages[0]["data"]["use_variant"])
            self.assertTrue(messages[0]["data"]["variant_conflict"])

    def test_classic_replay_keeps_full_board(self):
        record = np.zeros(1, dtype=REPLAY_DTYPE)
        record[0]["f0"] = int("10010332fff1fff3", 16)
        session = SimpleNamespace(user_id=None)
        _replay_load_record(session, record, "free10_512", use_variant=True)
        self.assertFalse(session.replay_use_variant)


if __name__ == "__main__":
    unittest.main()
