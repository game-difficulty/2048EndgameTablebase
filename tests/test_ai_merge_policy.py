import unittest

from engine_core.ai_merge_policy import merge_urgency_for_readers


class MergePolicyTests(unittest.TestCase):
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
