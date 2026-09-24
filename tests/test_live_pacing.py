import random
import unittest
from tools.live_pacing import batch_means, step_drift


class BatchPacingTests(unittest.TestCase):
    def test_distinct_means_differ_by_two_to_five_ms(self):
        fastest = set()
        for batch in range(100):
            means = batch_means(str(batch))
            fastest.add(means.index(min(means)))
            self.assertGreaterEqual(min(means), .002)
            self.assertLessEqual(max(means), .008)
            for i in range(3):
                for j in range(i):
                    self.assertGreaterEqual(abs(means[i]-means[j]), .002)
                    self.assertLessEqual(abs(means[i]-means[j]), .005)
        self.assertEqual(fastest, {0, 1, 2})

    def test_bounded_jitter_and_mean(self):
        for lane, mean in enumerate(batch_means('sample-batch')):
            samples = [step_drift('sample-batch', lane, seq) for seq in range(10000)]
            self.assertGreaterEqual(min(samples), mean-.001)
            self.assertLessEqual(max(samples), mean+.001)
            self.assertAlmostEqual(sum(samples)/len(samples), mean, delta=.00003)

    def test_resume_reproduces_profile_without_game_rng_or_added_checkpoint_state(self):
        state = random.getstate()
        means = batch_means('batch-one')
        steps = [step_drift('batch-one', lane, 100) for lane in range(3)]
        batch_means.cache_clear()
        self.assertEqual(batch_means('batch-one'), means)
        self.assertEqual([step_drift('batch-one', lane, 100) for lane in range(3)], steps)
        self.assertNotEqual(batch_means('batch-two'), means)
        self.assertNotEqual(step_drift('batch-one', 0, 101), steps[0])
        self.assertEqual(random.getstate(), state)
        self.assertEqual(step_drift(None, 0, 0), 0)
