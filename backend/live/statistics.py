"""Count completed 32k endgame stages from retained live replays."""
from collections import Counter

from backend.gamer_ranked.rules import DIRECTIONS, legal_moves, simulate_move
from backend.replay_2048next import (
    decode_2048next_replay, MoveRecord, EndRecord, ExtensionRecord,
)

# A 32k stage ends when the remaining tail reaches final 2k.  The same
# five-merge sequence is reused after entering the 65k phase.
VERSION = 2
TARGETS = (32768, 16384, 8192, 4096, 2048)


class StageCounter:
    def __init__(self):
        self.stage = 0
        self.passed = 0

    def observe(self, before, after):
        previous, current = Counter(before), Counter(after)
        # Reconstruct merge counts top-down; this also handles simultaneous merges.
        merges = {}
        value = max(max(before), max(after))
        while value >= 1024:
            merges[value] = current[value] - previous[value] + 2 * merges.get(value * 2, 0)
            value //= 2
        if any(value >= 65536 and count > 0 for value, count in merges.items()):
            self.stage = 0  # Cancel, rather than fail, any unfinished old stage.
        while self.stage < len(TARGETS) and merges.get(TARGETS[self.stage], 0) > 0:
            self.passed += 1
            self.stage += 1

    def result(self, dead):
        return self.passed, int(dead and self.stage < len(TARGETS))


def replay_stages(text):
    if not isinstance(text, str):
        raise ValueError('missing_live_replay')
    replay = decode_2048next_replay(text)
    if (replay.width, replay.height) != (4, 4):
        raise ValueError('unsupported_live_board')
    board = [0] * 16
    for index, bit in replay.initial_tiles:
        board[index] = 2 ** (bit + 1)
    counter = StageCounter()
    ended = False
    for record in replay.records:
        if ended:
            raise ValueError('data_after_end')
        if isinstance(record, EndRecord):
            ended = True
        elif isinstance(record, ExtensionRecord):
            continue
        elif isinstance(record, MoveRecord):
            moved, score_delta = simulate_move(board, DIRECTIONS[record.direction])
            if moved == board or moved[record.spawn_index]:
                raise ValueError('invalid_live_move')
            if score_delta >= 1024:
                counter.observe(board, moved)
            moved[record.spawn_index] = 2 ** (record.spawn_value_bit + 1)
            board = moved
        else:
            raise ValueError('unsupported_live_record')
    if not ended:
        raise ValueError('unfinished_live_replay')
    return counter.result(not legal_moves(board))
