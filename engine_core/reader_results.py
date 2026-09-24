"""Direction metadata that preserves the existing two-value reader API."""
import math
import numbers

DIRECTIONS = ("left", "right", "up", "down")


class ReaderResults(dict):
    def __init__(self, values=(), legal_moves_mask=None):
        super().__init__(values)
        self.legal_moves_mask = legal_moves_mask


def positive_result(value, zero_value=0):
    return (not isinstance(value, bool) and isinstance(value, numbers.Real)
            and math.isfinite(value) and value - zero_value > 0)


def missing_immediate_merge_result(board, target, results, zero_value=0):
    """Detect omitted goal states using real moves, not the table's legal mask."""
    half = target // 2
    for directions, lines in ((DIRECTIONS[:2], board), (DIRECTIONS[2:], zip(*board))):
        if all(positive_result(results.get(d), zero_value) for d in directions):
            continue
        for line in lines:
            occupied = [value for value in line if value != 0]
            # Adjacent equal halves merge in either direction along this axis.
            if any(a == half and b == half for a, b in zip(occupied, occupied[1:])):
                return True
    return False


def complete_positive_moves(results, zero_value=0):
    mask = getattr(results, "legal_moves_mask", None)
    if not isinstance(mask, int) or isinstance(mask, bool) or not 0 < mask < 16:
        return False
    for index, direction in enumerate(DIRECTIONS):
        if mask & (1 << index):
            value = results.get(direction)
            if not positive_result(value, zero_value):
                return False
    return True


def certain_legal_moves(results, zero_value=0, threshold=0.9999999):
    """Return legal move codes whose table result is effectively certain."""
    mask = getattr(results, "legal_moves_mask", None)
    if not isinstance(mask, int) or isinstance(mask, bool) or not 0 < mask < 16:
        return ()
    moves = []
    for index, direction in enumerate(DIRECTIONS):
        if not (mask & (1 << index)):
            continue
        value = results.get(direction)
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or value - zero_value <= threshold):
            continue
        moves.append(index + 1)
    return tuple(moves)
