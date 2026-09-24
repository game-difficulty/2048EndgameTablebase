import unittest
from types import SimpleNamespace

import numpy as np

from engine_core.AIPlayer import DispatcherCommon
from engine_core.reader_results import ReaderResults, missing_immediate_merge_result


def board(code):
    return np.array([2 ** int(c, 16) if c != '0' else 0 for c in code]).reshape(4, 4)


class TerminalMergeTests(unittest.TestCase):
    def make_dispatcher(self, code='e931a88f102d0013', results=None, dtype='uint32'):
        d = DispatcherCommon.__new__(DispatcherCommon)
        d._table_cooldowns = {}
        d._restore_reader_state = lambda state: None
        d.book_reader = SimpleNamespace(move_on_dic=lambda *args: (results, dtype))
        d.reset(board(code), int(code, 16))
        d.last_operator = 1
        d.current_table = 'free11_512'
        candidate = (14, 5, 5, 'free11', 9, '512', 'free11_512', 1, None)
        return d, candidate

    def test_real_case_exits_to_unrestricted_search_without_cooldown(self):
        d, candidate = self.make_dispatcher(results=ReaderResults(
            {'down': .978171212, 'left': 0, 'right': 0, 'up': None}, 8))
        d.ai_search_moves = (4,)
        self.assertEqual(d.check_table(candidate, 1), 'AI')
        self.assertEqual(d.current_table, 'AI')
        self.assertEqual(d.last_operator, 0)
        self.assertIsNone(d.ai_search_moves)
        self.assertEqual(d._table_cooldowns, {})
        d.get_endgame_lvls = lambda: ([candidate], [candidate], [])
        calls = []
        original = d.check_table
        def tracked(*args):
            calls.append(args)
            return original(*args)
        d.check_table = tracked
        self.assertEqual(d.dispatcher(), 'AI')
        self.assertEqual(len(calls), 1)

    def test_missing_zero_nonfinite_and_failure_rate_offset(self):
        b = board('e931a88f102d0013')
        for value in (None, '', 0, float('nan'), float('inf'), False):
            self.assertTrue(missing_immediate_merge_result(b, 512, {'left': value, 'right': .9}))
        self.assertTrue(missing_immediate_merge_result(b, 512, {'right': .9}))
        self.assertFalse(missing_immediate_merge_result(b, 512, {'left': -.1, 'right': -.2}, -1))
        self.assertTrue(missing_immediate_merge_result(b, 512, {'left': -1, 'right': -.2}, -1))

    def test_zero_without_immediate_merge_keeps_table(self):
        d, candidate = self.make_dispatcher('e931a78f102d0013',
            ReaderResults({'down': .9, 'left': 0, 'right': 0}, 11))
        self.assertEqual(d.check_table(candidate, 1), 'down')
        self.assertEqual(d._table_cooldowns, {})
        # Existing 512 is not a newly merged 512; an intervening tile blocks halves.
        self.assertFalse(missing_immediate_merge_result(board('8189000000000000'), 512, {}))

    def test_positive_terminal_results_keep_existing_policy(self):
        d, candidate = self.make_dispatcher(results=ReaderResults(
            {'down': .98, 'left': .9, 'right': .8, 'up': 0}, 15))
        self.assertEqual(d.check_table(candidate, 1), 'down')

    def test_gaps_vertical_and_no_chain_merge(self):
        self.assertTrue(missing_immediate_merge_result(board('8089000000000000'), 512, {}))
        self.assertTrue(missing_immediate_merge_result(board('8000000080009000'), 512, {'left': .9, 'right': .9}))
        self.assertFalse(missing_immediate_merge_result(board('7789000000000000'), 512, {}))
        # Existing targets can merge away simultaneously; count delta alone is insufficient.
        self.assertTrue(missing_immediate_merge_result(board('9988000000000000'), 512, {}))


if __name__ == '__main__':
    unittest.main()

