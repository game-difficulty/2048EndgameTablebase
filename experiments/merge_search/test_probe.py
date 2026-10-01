import json
import subprocess
import unittest
from functools import lru_cache

from build import EXE, build, environment


def move(board, direction):
    cells = list(board)
    lines = []
    if direction in (0, 1):
        lines = [[4 * r + c for c in range(4)] for r in range(4)]
    else:
        lines = [[4 * r + c for r in range(4)] for c in range(4)]
    for indices in lines:
        if direction in (1, 3):
            indices.reverse()
        values = [board[i] for i in indices if board[i]]
        merged = []
        j = 0
        while j < len(values):
            if j + 1 < len(values) and values[j] == values[j + 1] and values[j] < 15:
                merged.append(values[j] + 1)
                j += 2
            else:
                merged.append(values[j])
                j += 1
        merged += [0] * (4 - len(merged))
        for index, value in zip(indices, merged):
            cells[index] = value
    return tuple(cells)


def reference(code, target, depth, p4=.1):
    target_exp = target.bit_length() - 1

    def after(board, remaining):
        if target_exp in board:
            return (1., 0., 1., 1.)
        empty = [i for i, value in enumerate(board) if not value]
        total = [0.] * 4
        for index in empty:
            for tile, p in ((1, 1 - p4), (2, p4)):
                child = list(board)
                child[index] = tile
                success, death, steps, upper = player(tuple(child), remaining - 1)
                for i, v in enumerate((success, death, steps + success, upper)):
                    total[i] += p * v / len(empty)
        return tuple(total)

    @lru_cache(None)
    def player(board, remaining):
        children = [move(board, d) for d in range(4)]
        children = [child for child in children if child != board]
        if not children:
            return (0., 1., 0., 0.)
        if remaining == 0:
            return (0., 0., 0., 1.)
        candidates = [after(child, remaining) for child in children]
        best = max(candidates, key=lambda r: (r[0], -r[1], -r[2]))
        return (*best[:3], max(r[3] for r in candidates))

    board = tuple(int(c, 16) for c in code)
    return {name: after(move(board, d), depth) for d, name in enumerate(('left', 'right', 'up', 'down'))
            if move(board, d) != board}


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build()
        cls.env = environment()[1]

    def run_probe(self, board, target, depth, *extra):
        command = [str(EXE), '--board', board, '--target', str(target), '--depth', str(depth),
                   '--seconds', '30', '--nodes', '10000000', *extra]
        result = subprocess.run(command, env=self.env, text=True, capture_output=True, timeout=130, check=True)
        return json.loads(result.stdout)

    def test_against_independent_small_tree(self):
        for code, target in [('2200000000000000', 8), ('1212323434545656', 128),
                             ('1121324354657677', 256)]:
            for depth in (1, 2, 3):
                with self.subTest(board=code, depth=depth):
                    actual = self.run_probe(code, target, depth)
                    expected = reference(code, target, depth)
                    self.assertEqual({r['direction'] for r in actual['directions']}, set(expected))
                    for r in actual['directions']:
                        success, death, steps, upper = expected[r['direction']]
                        self.assertTrue(r['complete'])
                        self.assertAlmostEqual(r['success_lower'], success, places=11)
                        self.assertAlmostEqual(r['policy_death'], death, places=11)
                        self.assertAlmostEqual(r['success_upper'], upper, places=11)
                        if success:
                            self.assertAlmostEqual(r['success_steps'], steps / success, places=5)

    def test_goal_dead_and_budget(self):
        goal = self.run_probe('2200000000000000', 8, 1, '--nodes', '0')
        left = next(r for r in goal['directions'] if r['direction'] == 'left')
        self.assertEqual(left['success_lower'], 1)
        self.assertEqual(left['success_steps'], 1)
        dead = self.run_probe('1212212112122121', 8, 2)
        self.assertEqual(dead['directions'], [])
        self.assertIsNone(dead['best_observed'])
        truncated = self.run_probe('330043598671da10', 2048, 8, '--nodes', '0')
        for r in truncated['directions']:
            self.assertFalse(r['complete'])
            self.assertEqual((r['success_lower'], r['success_upper'], r['policy_unresolved']), (0, 1, 1))

    def test_cache_and_root_order(self):
        args = ('1121324354657677', 256, 3)
        variants = [self.run_probe(*args, *extra) for extra in ((), ('--reverse',), ('--cache-mib', '0'))]
        def values(data):
            return {r['direction']: tuple(r[k] for k in ('success_lower', 'success_upper', 'policy_death', 'success_steps'))
                    for r in data['directions']}
        self.assertEqual(values(variants[0]), values(variants[1]))
        self.assertEqual(values(variants[0]), values(variants[2]))

    def test_invalid_target(self):
        result = subprocess.run([str(EXE), '--target', '65536'], env=self.env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_budget_bounds_and_depth_monotonicity(self):
        previous = {}
        for depth in (1, 2, 3):
            full = self.run_probe('1121324354657677', 256, depth)
            limited = self.run_probe('1121324354657677', 256, depth, '--nodes', '3')
            bounded = {r['direction']: r for r in limited['directions']}
            for r in full['directions']:
                self.assertTrue(r['complete'])
                self.assertLessEqual(r['success_lower'], r['success_upper'] + 1e-12)
                self.assertAlmostEqual(r['success_lower'] + r['policy_death'] + r['policy_unresolved'], 1)
                cut = bounded[r['direction']]
                self.assertLessEqual(cut['success_lower'], r['success_lower'] + 1e-12)
                self.assertGreaterEqual(cut['success_upper'] + 1e-12, r['success_upper'])
                if r['direction'] in previous:
                    before = previous[r['direction']]
                    self.assertGreaterEqual(r['success_lower'] + 1e-12, before['success_lower'])
                    self.assertLessEqual(r['success_upper'], before['success_upper'] + 1e-12)
                previous[r['direction']] = r


if __name__ == '__main__':
    unittest.main()
