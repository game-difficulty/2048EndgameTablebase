import unittest

import numpy as np

from backend.animation import build_move_animation_metadata
from engine_core.BoardMover import s_move_board
from engine_core.Calculator import find_merge_positions, slide_distance


def _repeated_row_board(row):
    return np.tile(np.asarray(row, dtype=np.int64), (4, 1))


def _repeated_row_encoded(row_hex):
    return np.uint64(int(row_hex * 4, 16))


class AnimationMetadataTests(unittest.TestCase):
    def test_movable_32k_does_not_shift_right_merge_position(self):
        cases = (
            ("11f0", "002f"),
            ("22f0", "003f"),
            ("33f0", "004f"),
        )

        for source_row, expected_row in cases:
            with self.subTest(source_row=source_row):
                board = _repeated_row_encoded(source_row)
                moved_board, _ = s_move_board(board, 2)
                metadata = build_move_animation_metadata(
                    "right",
                    board_encoded=board,
                    use_variant=False,
                )

                self.assertEqual(int(moved_board), int(_repeated_row_encoded(expected_row)))
                self.assertEqual(
                    np.asarray(metadata["slide_distances"]).reshape(4, 4).tolist(),
                    [[2, 1, 1, 0]] * 4,
                )
                self.assertEqual(
                    np.asarray(metadata["pop_positions"]).reshape(4, 4).tolist(),
                    [[0, 0, 1, 0]] * 4,
                )

    def test_movable_32k_merge_positions_are_symmetric(self):
        horizontal_right = _repeated_row_board([2, 2, 32768, 0])
        horizontal_left = _repeated_row_board([32768, 0, 2, 2])
        vertical_down = horizontal_right.T
        vertical_up = horizontal_left.T

        cases = (
            (horizontal_right, "right", [[0, 0, 1, 0]] * 4),
            (horizontal_left, "left", [[0, 1, 0, 0]] * 4),
            (vertical_down, "down", np.asarray([[0, 0, 1, 0]] * 4).T.tolist()),
            (vertical_up, "up", np.asarray([[0, 1, 0, 0]] * 4).T.tolist()),
        )

        for board, direction, expected in cases:
            with self.subTest(direction=direction):
                self.assertEqual(
                    find_merge_positions(board, direction).tolist(),
                    expected,
                )

    def test_32k_tiles_slide_without_merging(self):
        board = _repeated_row_board([32768, 32768, 0, 0])

        self.assertEqual(
            slide_distance(board, "right").tolist(),
            [[2, 2, 0, 0]] * 4,
        )
        self.assertFalse(np.any(find_merge_positions(board, "right")))

    def test_variant_wall_remains_an_immovable_barrier(self):
        normalized_variant_board = _repeated_row_board([2, 2, -1, 0])

        self.assertEqual(
            slide_distance(
                normalized_variant_board,
                "right",
                (32768, 16384),
            ).tolist(),
            [[1, 0, 0, 0]] * 4,
        )
        self.assertEqual(
            find_merge_positions(
                normalized_variant_board,
                "right",
                (32768, 16384),
            ).tolist(),
            [[0, 1, 0, 0]] * 4,
        )

    def test_variant_16k_tiles_move_without_merging(self):
        board = _repeated_row_board([16384, 16384, 0, 0])
        non_merging_values = (32768, 16384)

        self.assertEqual(
            slide_distance(board, "right", non_merging_values).tolist(),
            [[2, 2, 0, 0]] * 4,
        )
        self.assertFalse(
            np.any(find_merge_positions(board, "right", non_merging_values))
        )

    def test_regular_merge_positions_without_32k_are_unchanged(self):
        for row in (
            [2, 2, 4, 0],
            [2, 2, 8, 0],
            [4, 4, 256, 0],
        ):
            with self.subTest(row=row):
                board = _repeated_row_board(row)
                self.assertEqual(
                    find_merge_positions(board, "right").tolist(),
                    [[0, 0, 1, 0]] * 4,
                )


if __name__ == "__main__":
    unittest.main()
