import unittest
from unittest.mock import patch

from tools import live_runner


class LiveRunnerTimingTests(unittest.IsolatedAsyncioTestCase):
    async def wait_step(self, source, elapsed, *, wake_early=False, board=None):
        now = elapsed
        waits = []

        async def sleep(delay):
            nonlocal now
            waits.append(delay)
            now += delay / 2 if wake_early and len(waits) == 1 else delay

        with patch.object(live_runner.time, 'monotonic', side_effect=lambda: now), \
                patch.object(live_runner.asyncio, 'sleep', side_effect=sleep):
            await live_runner.wait_for_step(0, source, .08, .05, board)
        return now, waits

    async def test_search_only_waits_to_fifty_milliseconds(self):
        now, waits = await self.wait_step('AI', .02)
        self.assertAlmostEqual(now, .05)
        self.assertAlmostEqual(sum(waits), .03)

    async def test_table_keeps_eighty_milliseconds(self):
        now, waits = await self.wait_step('free11_2048', .02)
        self.assertAlmostEqual(now, .08)
        self.assertAlmostEqual(sum(waits), .06)

    async def test_slow_computation_adds_no_wait(self):
        for source, elapsed in [('AI', .06), ('free12_2048', .12)]:
            now, waits = await self.wait_step(source, elapsed)
            self.assertEqual(now, elapsed)
            self.assertEqual(waits, [])

    async def test_early_wake_still_observes_minimum(self):
        for source, minimum in [('AI', .05), ('LL_1024', .08)]:
            now, waits = await self.wait_step(source, .01, wake_early=True)
            self.assertAlmostEqual(now, minimum)
            self.assertEqual(len(waits), 2)

    async def test_switching_source_uses_current_step(self):
        for source, minimum in [('free11_2048', .08), ('AI', .05), ('LL_1024', .08)]:
            now, _ = await self.wait_step(source, .01)
            self.assertAlmostEqual(now, minimum)

    async def test_stage_thresholds_and_priority(self):
        for source in ['AI', 'free12_2048']:
            for tiles, minimum in [
                ([2, 4], 0),
                ([1024] * 2 + [512] * 6, 0),
                ([1024] * 3 + [512] * 2, .05 if source == 'AI' else .08),
                ([1024] * 3 + [512] * 3, .12),
                ([1024] * 3 + [512] * 4, .18),
                ([2048] * 8, .18),
            ]:
                board = tiles + [0] * (16 - len(tiles))
                now, waits = await self.wait_step(source, .01, board=board)
                self.assertAlmostEqual(now, max(.01, minimum))
                if minimum == 0:
                    self.assertEqual(waits, [])

    async def test_stage_wait_includes_computation_and_handles_early_wake(self):
        board = [1024] * 7 + [0] * 9
        now, waits = await self.wait_step('AI', .05, board=board, wake_early=True)
        self.assertAlmostEqual(now, .18)
        self.assertAlmostEqual(sum(waits[1:]) + waits[0] / 2, .13)
        now, waits = await self.wait_step('AI', .25, board=board)
        self.assertEqual(now, .25)
        self.assertEqual(waits, [])


if __name__ == '__main__':
    unittest.main()
