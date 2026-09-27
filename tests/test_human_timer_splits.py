import unittest

from backend.human_play import engine, service


class HumanTimerSplitSettingsTests(unittest.TestCase):
    def test_accepts_four_variant_ordered_combined_splits(self):
        payload = {variant: ["32", "512", "512+256", "1024"] for variant in engine.VARIANTS}
        self.assertEqual(service._timer_splits(payload), payload)

    def test_rejects_invalid_tiles_duplicates_and_partial_variants(self):
        valid = {variant: ["32"] for variant in engine.VARIANTS}
        for payload in (
            {"4x4": ["32"]},
            {**valid, "4x4": ["3"]},
            {**valid, "4x4": ["256+512"]},
            {**valid, "4x4": ["512+256", "512+256"]},
            {**valid, "4x4": ["32"] * 33},
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(service.RunError, "invalid_timer_splits"):
                service._timer_splits(payload)


if __name__ == "__main__":
    unittest.main()
