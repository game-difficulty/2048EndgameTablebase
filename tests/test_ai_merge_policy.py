import unittest

from engine_core.ai_merge_policy import allows_five_tiler_relaxation, merge_urgency_for_readers


class MergePolicyTests(unittest.TestCase):
    def test_five_tiler_table_allowlist(self):
        for pattern in ('free12', 'free11', 'free13', '4442', '4442f', '4442ff'):
            self.assertTrue(allows_five_tiler_relaxation({0: [(0, 0, 0, pattern)]}))
        for pattern in ('LL', '444', 'free10', 'free12custom'):
            self.assertFalse(allows_five_tiler_relaxation({0: [(0, 0, 0, pattern)]}))
        self.assertFalse(allows_five_tiler_relaxation({}))

    def test_reported_board_relaxes_only_with_allowed_table(self):
        import numpy as np
        from types import SimpleNamespace
        from engine_core.AIPlayer import CoreAILogic
        code = '46ec3578239a11fd'
        board = np.array([2 ** int(c, 16) for c in code]).reshape(4, 4)
        counts = np.bincount([int(c, 16) for c in code], minlength=16)
        logic = CoreAILogic.__new__(CoreAILogic)
        logic.manager = SimpleNamespace(probe=lambda *args: (None, False, None, None, None))
        logic.perform_iterative_search = lambda *args: (1, 10, [0] * 4)
        for enabled, expected in ((False, 1), (True, 0), (False, 1)):
            logic.allow_five_tiler_relaxation = enabled
            player = SimpleNamespace(board=int(code, 16))
            logic.calculate_step(player, board, counts)
            self.assertEqual(player.prune, expected)
        logic.allow_five_tiler_relaxation = True
        logic.danbianhuichuan_patch = lambda *args: True
        logic.calculate_step(player, board, counts)
        self.assertEqual(player.prune, 1)

    def test_available_patterns(self):
        for patterns, expected in [
            ([], 0), (['444', 'free10'], 0), (['free11'], 1),
            (['LL'], 1), (['4442'], 1), (['4442f'], 1),
            (['4442ff'], 1), (['free12'], 1.5),
            (['free11', 'LL', 'free12'], 1.5),
        ]:
            with self.subTest(patterns=patterns):
                readers = {0: [(0, 0, 0, pattern) for pattern in patterns]}
                self.assertEqual(merge_urgency_for_readers(readers), expected)

    def test_native_parameter_and_search(self):
        from native_core import ai_core
        player = ai_core.AIPlayer(0x1122334455667788)
        self.assertEqual(player.merge_urgency, 0)
        player.max_threads = 1
        for urgency in (0, 1, 1.5, 0):
            player.merge_urgency = urgency
            self.assertEqual(player.merge_urgency, urgency)
            player.start_search(2)
            self.assertIn(player.best_operation, (1, 2, 3, 4))


if __name__ == '__main__':
    unittest.main()
