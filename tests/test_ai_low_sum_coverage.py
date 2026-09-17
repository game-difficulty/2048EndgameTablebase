import unittest
from types import SimpleNamespace

import numpy as np

from engine_core.AIPlayer import DispatcherCommon
from engine_core.reader_results import ReaderResults, complete_positive_moves


class LowSumCoverageTests(unittest.TestCase):
    def dispatcher(self, total=22, pattern="ordinary", results=None, dtype="uint32"):
        d = DispatcherCommon.__new__(DispatcherCommon)
        d._table_cooldowns = {}
        d._restore_reader_state = lambda state: None
        d.book_reader = SimpleNamespace(move_on_dic=lambda *args: (results, dtype))
        board = np.array([2**int(c,16) if c != "0" else 0 for c in "101022109830edba"]).reshape(4,4)
        board[0,1] = total-22
        d.reset(board, 0)
        candidate = (14,6,6,pattern,8,"256",pattern+"_256",1,None)
        return d, candidate

    def test_complete_positive_and_missing_legal_direction(self):
        self.assertTrue(complete_positive_moves(ReaderResults({"left":.9,"right":None},1)))
        for values in ({"left":.9}, {"left":.9,"right":None},
                       {"left":.9,"right":0}, {"left":.9,"right":float("nan")}):
            self.assertFalse(complete_positive_moves(ReaderResults(values,3)))
        self.assertFalse(complete_positive_moves({"left":1}))
        self.assertFalse(complete_positive_moves(ReaderResults({"left":1},0)))
        self.assertTrue(complete_positive_moves(ReaderResults({"left":-.1},1),-1))
        self.assertFalse(complete_positive_moves(ReaderResults({"left":-1},1),-1))

    def test_low_sum_certain_complete_result_bypasses_only_small_remainder_handoff(self):
        for kind in (1,3):
            d,c = self.dispatcher(results=ReaderResults({"left":1.0,"right":None},1))
            self.assertEqual(d.check_table(c,kind),"left")
            self.assertEqual(d._table_cooldowns,{})
        d,c = self.dispatcher(results=ReaderResults({"left":1.0},1))
        self.assertEqual(d.check_table(c,2),"AI")

    def test_threshold_28_and_free10_32(self):
        for total,pattern,expected in ((26,"ordinary",None),(28,"ordinary","left"),
                                       (30,"free10",None),(32,"free10","left")):
            d,c = self.dispatcher(total,pattern,ReaderResults({"left":.9,"right":0},3))
            self.assertEqual(d.check_table(c,1),expected)

    def test_incomplete_candidate_does_not_start_cooldown(self):
        d,c = self.dispatcher(results=ReaderResults({"left":.9,"right":0},3))
        self.assertIsNone(d.check_table(c,1))
        self.assertEqual(d._table_cooldowns,{})
        self.assertEqual(int(d.mask(6).sum())-6*32768,22)


if __name__ == "__main__":
    unittest.main()
