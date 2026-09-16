import unittest
from types import SimpleNamespace

import numpy as np

from engine_core.AIPlayer import DispatcherCommon


BOARD = '39107a1266d813b1'


def reader(name, target, large, free):
    exponent = target.bit_length() - 1
    return (large + exponent, large, free, name.split('_')[0],
            exponent, str(target), name, 1, None)


class DispatcherCooldownTests(unittest.TestCase):
    def make_dispatcher(self):
        dispatcher = DispatcherCommon.__new__(DispatcherCommon)
        dispatcher._table_cooldowns = {}
        dispatcher.ad_readers = {}
        dispatcher._restore_reader_state = lambda state: None
        dispatcher.book_reader = SimpleNamespace(
            move_on_dic=lambda *args: ({'down': 1.0}, 'uint32'))
        for item in (
            reader('free10_128', 128, 6, 6),
            reader('ordinary_512', 512, 4, 0),
            reader('plusone_1024', 1024, 4, 0),
            reader('free12_2048', 2048, 4, 4),
            reader('missing_4096', 4096, 1, 0),
            reader('free10_512', 512, 6, 6),
        ):
            dispatcher.ad_readers.setdefault(item[:2], []).append(item)
        self.reset(dispatcher)
        return dispatcher

    def reset(self, dispatcher, code=BOARD):
        board = np.array([2 ** int(c, 16) if c != '0' else 0 for c in code]).reshape(4, 4)
        dispatcher.reset(board, int(code, 16))

    def names(self, dispatcher):
        return {item[6] for group in dispatcher.get_endgame_lvls() for item in group}

    def trigger(self, dispatcher):
        item = reader('free10_128', 128, 6, 6)
        self.assertEqual(dispatcher.check_table(item, 1), 'AI')

    def test_handoff_blocks_both_higher_target_branches_but_keeps_normal_candidates(self):
        d = self.make_dispatcher()
        self.assertTrue({'plusone_1024', 'free12_2048'} <= self.names(d))
        self.trigger(d)
        names = self.names(d)
        self.assertTrue({'ordinary_512', 'missing_4096'} <= names)
        self.assertTrue({'plusone_1024', 'free12_2048'}.isdisjoint(names))
        self.assertIsNone(d.check_table(reader('free10_128', 128, 6, 6), 1))

    def test_candidates_return_at_existing_twentieth_reset_boundary(self):
        d = self.make_dispatcher()
        self.trigger(d)
        for _ in range(19):
            self.reset(d)
            self.assertNotIn('free12_2048', self.names(d))
        self.reset(d)
        self.assertIn('free12_2048', self.names(d))
        self.assertIn('plusone_1024', self.names(d))

    def test_other_cooldowns_keep_higher_targets_blocked_and_normal_free_tables_available(self):
        d = self.make_dispatcher()
        self.trigger(d)
        d._table_cooldowns['other'] = 25
        for _ in range(20):
            self.reset(d)
        self.assertNotIn('free12_2048', self.names(d))
        self.reset(d, '1011178729ab1cde')
        self.assertIn('free10_512', self.names(d))
        # free12 is still permitted via the normal branch on this different board.
        self.assertIn('free12_2048', self.names(d))


if __name__ == '__main__':
    unittest.main()
