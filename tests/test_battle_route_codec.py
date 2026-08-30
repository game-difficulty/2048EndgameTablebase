from __future__ import annotations

import hashlib
import struct
import unittest

from backend.battle.route_codec import (
    MAX_ROUTE_BYTES,
    MAX_ROUTE_STEPS,
    RATE_SCALE,
    RECORD_SIZE,
    RouteCodecError,
    RouteStep,
    decode_changes,
    decode_direction,
    decode_route,
    decode_spawn,
    encode_changes,
    encode_route,
    route_sha256,
)
from backend.battle.scoring import (
    ABSOLUTE_DIFFERENCE_TOLERANCE,
    accumulate_goodness,
    calculate_step_ratio,
    score_choice,
    score_recorded_choice,
)


class BattleRouteCodecTests(unittest.TestCase):
    def test_round_trip_uses_exact_trainer_layout(self) -> None:
        board = 0xFEDCBA9876543210
        steps = (
            RouteStep(
                encode_changes("up", 0, 2),
                (RATE_SCALE, 3_000_000_000, 2_000_000_000, 0),
            ),
            RouteStep(
                encode_changes("right", 15, 4),
                (1, 2, 3, 4),
            ),
        )

        encoded = encode_route(board, steps)

        self.assertEqual(len(encoded), RECORD_SIZE * 3)
        self.assertEqual(
            struct.unpack("<B4I", encoded[:RECORD_SIZE]),
            (0, 0x3210, 0x7654, 0xBA98, 0xFEDC),
        )
        self.assertEqual(
            encoded[RECORD_SIZE],
            encode_changes("up", 0, 2),
        )
        self.assertEqual(decode_route(encoded), decode_route(bytearray(encoded)))
        self.assertEqual(decode_route(encoded).initial_board, board)
        self.assertEqual(decode_route(encoded).steps, steps)

    def test_direction_and_spawn_bits_match_trainer(self) -> None:
        for code, direction in enumerate(("up", "down", "left", "right")):
            for spawn_index in (0, 7, 15):
                for spawn_value in (2, 4):
                    changes = encode_changes(direction, spawn_index, spawn_value)
                    decoded = decode_changes(changes)
                    self.assertEqual(decoded.direction_code, code)
                    self.assertEqual(decoded.direction, direction)
                    self.assertEqual(decoded.spawn_index, spawn_index)
                    self.assertEqual(decoded.spawn_value, spawn_value)
                    self.assertEqual(decoded.spawn_exponent, 1 if spawn_value == 2 else 2)
                    self.assertEqual(decode_direction(changes), direction)
                    self.assertEqual(decode_spawn(changes), (spawn_index, spawn_value))

    def test_sha256_is_over_exact_encoded_bytes(self) -> None:
        encoded = encode_route(
            0x0123456789ABCDEF,
            [RouteStep(encode_changes("left", 4, 2), (0, 1, 2, RATE_SCALE))],
        )
        self.assertEqual(route_sha256(encoded), hashlib.sha256(encoded).hexdigest())

    def test_header_only_route_is_valid(self) -> None:
        route = decode_route(encode_route(0, []))
        self.assertEqual(route.initial_board, 0)
        self.assertEqual(route.steps, ())

    def test_rejects_invalid_lengths_and_headers(self) -> None:
        with self.assertRaisesRegex(RouteCodecError, "missing"):
            decode_route(b"")
        with self.assertRaisesRegex(RouteCodecError, "multiple of 17"):
            decode_route(b"\0" * (RECORD_SIZE + 1))

        nonzero_changes = struct.pack("<B4I", 1, 0, 0, 0, 0)
        with self.assertRaisesRegex(RouteCodecError, "must be zero"):
            decode_route(nonzero_changes)

        noncanonical_chunk = struct.pack("<B4I", 0, 0x10000, 0, 0, 0)
        with self.assertRaisesRegex(RouteCodecError, "non-canonical"):
            decode_route(noncanonical_chunk)

    def test_rejects_reserved_bits_and_out_of_range_values(self) -> None:
        header = struct.pack("<B4I", 0, 0, 0, 0, 0)
        reserved_step = struct.pack("<B4I", 0x80, 0, 0, 0, 0)
        with self.assertRaisesRegex(RouteCodecError, "reserved"):
            decode_route(header + reserved_step)

        with self.assertRaises(RouteCodecError):
            RouteStep(0, (0, 0, 0, RATE_SCALE + 1))
        with self.assertRaises(RouteCodecError):
            encode_changes("up", 16, 2)
        with self.assertRaises(RouteCodecError):
            encode_changes("diagonal", 0, 2)
        with self.assertRaises(RouteCodecError):
            encode_changes("up", 0, 8)
        with self.assertRaises(RouteCodecError):
            encode_route(1 << 64, [])

    def test_enforces_step_and_byte_limits(self) -> None:
        step = RouteStep(encode_changes("up", 0, 2), (0, 0, 0, 0))
        with self.assertRaisesRegex(RouteCodecError, "1-step limit"):
            encode_route(0, [step, step], max_steps=1)

        two_step_payload = encode_route(0, [step, step])
        with self.assertRaisesRegex(RouteCodecError, "1-step limit"):
            decode_route(two_step_payload, max_steps=1)

        oversized = b"\0" * (MAX_ROUTE_BYTES + RECORD_SIZE)
        self.assertEqual(len(oversized) % RECORD_SIZE, 0)
        with self.assertRaisesRegex(RouteCodecError, "byte codec limit"):
            decode_route(oversized)
        self.assertEqual(MAX_ROUTE_STEPS, 9_999)


class BattleScoringTests(unittest.TestCase):
    def test_absolute_tolerance_matches_tester(self) -> None:
        best = 0.8
        selected = best - (ABSOLUTE_DIFFERENCE_TOLERANCE / 2)
        self.assertEqual(calculate_step_ratio(selected, best), 1.0)

        outside = best - (ABSOLUTE_DIFFERENCE_TOLERANCE * 2)
        self.assertAlmostEqual(calculate_step_ratio(outside, best), outside / best)

    def test_cumulative_goodness_multiplies_without_logarithms(self) -> None:
        current = accumulate_goodness(1.0, 0.5, 1.0)
        current = accumulate_goodness(current, 0.25, 0.5)
        self.assertEqual(current, 0.25)

    def test_zero_best_rate_preserves_goodness(self) -> None:
        self.assertEqual(calculate_step_ratio(0.0, 0.0), 1.0)
        self.assertEqual(accumulate_goodness(0.75, 0.0, 0.0), 0.75)

    def test_scores_direction_and_uses_fixed_order_for_ties(self) -> None:
        result = score_choice(
            {"up": 0.9, "down": 0.9, "left": 0.45, "right": 0.1},
            "left",
            current_goodness=0.8,
        )
        self.assertEqual(result.best_direction, "up")
        self.assertEqual(result.step_ratio, 0.5)
        self.assertEqual(result.goodness_of_fit, 0.4)
        self.assertEqual(result.goodness_drop, 0.5)
        self.assertFalse(result.is_best)

    def test_recorded_adjacent_uint32_rates_are_equal_within_tolerance(self) -> None:
        result = score_recorded_choice(
            (RATE_SCALE - 1, RATE_SCALE, 0, 0),
            "up",
            current_goodness=0.9,
        )
        self.assertTrue(result.is_best)
        self.assertEqual(result.step_ratio, 1.0)
        self.assertEqual(result.goodness_of_fit, 0.9)

    def test_invalid_scoring_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_step_ratio(0.9, 0.8)
        with self.assertRaises(ValueError):
            score_choice((1.0, 0.5, 0.25), "up")
        with self.assertRaises(ValueError):
            score_choice((1.0, 0.5, 0.25, 0.0), "diagonal")
        with self.assertRaises(RouteCodecError):
            score_recorded_choice((0, 0, 0, RATE_SCALE + 1), "right")


if __name__ == "__main__":
    unittest.main()
