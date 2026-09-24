import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from backend.gamer_ranked.rules import packed_native_board
from tools.live_runner import NativeAI


class NativeLargeTileTests(unittest.TestCase):
    def ai(self):
        ai = NativeAI.__new__(NativeAI)
        ai.np = np
        ai.player = SimpleNamespace(board=0)
        ai.dispatcher = SimpleNamespace(reset=Mock(), dispatcher=lambda: 'AI', current_table='AI', counts=[])
        ai.core = SimpleNamespace(resolve_32768_doubles=Mock(return_value=123))
        ai.logic = SimpleNamespace(calculate_step=Mock(return_value=1))
        return ai

    def test_resolve_only_exactly_two_equal_large_tiles(self):
        for tiles,expected in [([65536,32768],False),([32768,32768],True),
                               ([65536,65536],True),([131072,65536],False),
                               ([32768],False),([32768,32768,65536],False)]:
            ai = self.ai()
            board = [0,*tiles]+[0]*(15-len(tiles))
            self.assertEqual(ai.choose(board),('left','AI'))
            self.assertEqual(ai.core.resolve_32768_doubles.called,expected)
            self.assertEqual(ai.player.board,123 if expected else packed_native_board(board))

    def test_invalid_move_logs_real_values(self):
        ai = self.ai()
        ai.logic.calculate_step.return_value=3
        board=[2,2,65536,8,0,4,32768,16,0,0,0,2,0,0,0,0]
        with self.assertRaisesRegex(RuntimeError,'65536'):
            ai.choose(board)
