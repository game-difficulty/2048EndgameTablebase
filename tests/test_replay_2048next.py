import base64
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

import engine_core.BoardMover as bm
import engine_core.VBoardMover as vbm
from backend.analysis_core import ReplayDecoder
from backend.replay_2048next import (
    REPLAY_PREFIX,
    Replay2048NextError,
    decode_2048next_replay,
)


def _uleb128(value: int) -> bytes:
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        encoded.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(encoded)


def _fixture_replay() -> str:
    payload = bytearray(b"RPL1")
    payload.extend((0x44, 0, 2))  # 4x4, no flags, two initial tiles
    payload.extend((5, 7))  # Two 2 tiles at row-major cells 5 and 7
    payload.extend((131, 2, 4))
    payload.extend(b"pow2")
    payload.extend((13, 0))  # Right, spawn 2 at cell 3
    payload.extend((66, 0))  # Down, spawn 4 at cell 0 (later undone)
    payload.extend((128, 0))  # Undo one move
    payload.extend((63, 0))  # Left, spawn 2 at cell 15
    payload.append(132)  # End
    payload.extend((zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "little"))
    return REPLAY_PREFIX + base64.b64encode(payload).decode("ascii")


class Replay2048NextTests(unittest.TestCase):
    def test_codec_parses_envelope_and_records(self):
        replay = decode_2048next_replay(_fixture_replay())

        self.assertEqual((replay.width, replay.height), (4, 4))
        self.assertEqual(replay.initial_tiles, ((5, 0), (7, 0)))
        self.assertEqual(replay.text_extension(2), "pow2")
        self.assertEqual(len(replay.records), 6)

    def test_decoder_folds_undo_into_effective_move_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2048next.txt"
            path.write_text(_fixture_replay(), encoding="ascii")
            decoder = ReplayDecoder(str(path), bm, vbm)
            decoder.decode()

        record = decoder.record_list
        self.assertEqual(decoder.variant, "4x4")
        self.assertEqual(len(record), 2)
        np.testing.assert_array_equal(
            bm.decode_board(record[0]["f0"]),
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 2, 0, 2],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ),
        )
        self.assertEqual(record["f1"].tolist(), [0, 4])
        self.assertEqual(record["f2"].tolist(), [2, 1])
        self.assertEqual(record["f3"].tolist(), [1, 1])
        self.assertEqual(record["f4"].tolist(), [3, 15])

    def test_codec_rejects_crc_mismatch(self):
        encoded = _fixture_replay()[len(REPLAY_PREFIX) :]
        payload = bytearray(base64.b64decode(encoded))
        payload[10] ^= 1
        corrupted = REPLAY_PREFIX + base64.b64encode(payload).decode("ascii")

        with self.assertRaisesRegex(Replay2048NextError, "CRC32 mismatch"):
            decode_2048next_replay(corrupted)


if __name__ == "__main__":
    unittest.main()
