import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from backend.live.protocol import LiveRun, STEP, DIRECTIONS
from backend.live.store import LiveStore
from backend.live.routes import LiveHub
from backend.gamer_ranked.rules import legal_moves, simulate_move
from backend.replay_2048next import decode_2048next_replay, MoveRecord

SEED = '12345678123456791234568012345681'


def completed_run():
    run = LiveRun(SEED)
    for _ in range(10000):
        legal = legal_moves(run.board)
        if not legal:
            run.end()
            return run
        run.apply(run.make_step(legal[0], 50))
    raise AssertionError('fixture did not finish')


class LiveProtocolTests(unittest.TestCase):
    def test_compact_steps_and_checkpoint_rebuild_exactly(self):
        run = completed_run()
        restored = LiveRun.restore(run.checkpoint())
        self.assertEqual(STEP.size, 9)
        self.assertEqual(restored.snapshot(), run.snapshot())
        self.assertEqual(restored.rng.state, run.rng.state)
        self.assertEqual(restored.replay(), run.replay())

    def test_replay_round_trip_preserves_board_score_and_time(self):
        run = completed_run()
        parsed = decode_2048next_replay(run.replay())
        board, score, elapsed = [0] * 16, 0, 0
        for index, bit in parsed.initial_tiles:
            board[index] = 2 ** (bit + 1)
        for record in parsed.records:
            if isinstance(record, MoveRecord):
                board, delta = simulate_move(board, DIRECTIONS[record.direction])
                board[record.spawn_index] = 2 ** (record.spawn_value_bit + 1)
                score += delta
                elapsed += record.delta_ms
        self.assertEqual((board, score, elapsed), (run.board, run.score, run.elapsed))

    def test_large_tiles_keep_real_values(self):
        run = LiveRun(SEED)
        run.board = [32768, 32768] + [0] * 14
        run.apply(run.make_step('left', 50))
        self.assertEqual(run.board[0], 65536)
        self.assertEqual(run.score, 65536)
        self.assertEqual(run.nodes['65536'], 50)

    def test_bad_spawn_does_not_advance_rng_or_state(self):
        run = LiveRun(SEED)
        packet = bytearray(run.make_step(legal_moves(run.board)[0], 50))
        packet[8] ^= 64
        before = run.checkpoint(), run.rng.state.copy()
        with self.assertRaises(ValueError):
            run.apply(packet)
        self.assertEqual((run.checkpoint(), run.rng.state), before)

    def test_sequence_gaps_duplicates_and_false_deaths_rejected(self):
        run = LiveRun(SEED)
        packet = run.make_step(legal_moves(run.board)[0], 50)
        with self.assertRaises(ValueError):
            run.apply(STEP.pack(2, 50, packet[-1]))
        run.apply(packet)
        with self.assertRaises(ValueError):
            run.apply(packet)
        with self.assertRaises(ValueError):
            run.end()

    def test_idempotent_completion_and_storage_recovery(self):
        run = completed_run()
        with tempfile.TemporaryDirectory() as directory:
            store = LiveStore(Path(directory) / 'live.db')
            store.save(run)
            store.finish(run)
            store.finish(run)
            self.assertEqual(store.summary()['today']['games'], 1)
            self.assertEqual(len(store.summary()['history']), 1)
            self.assertEqual(store.replay(run.id), run.replay())
            self.assertEqual(LiveRun.restore(store.load()).snapshot(), run.snapshot())
            store.add_likes(3)
            self.assertEqual(store.summary()['likes'], 3)

    def test_slow_viewer_queue_is_bounded_and_resyncs(self):
        hub = LiveHub()
        hub.run = LiveRun(SEED)
        queue = asyncio.Queue(maxsize=2)
        hub.viewers[object()] = queue
        for _ in range(3):
            hub.broadcast(b'packet')
        self.assertEqual(queue.qsize(), 1)
        self.assertEqual(queue.get_nowait()['type'], 'snapshot')

    def test_chat_rate_limit_expires_and_is_per_actor(self):
        hub = LiveHub()
        with patch('backend.live.routes.time.monotonic', return_value=10):
            for _ in range(5):
                hub.limit(('chat', 'g:a'), 5)
            with self.assertRaises(HTTPException):
                hub.limit(('chat', 'g:a'), 5)
            hub.limit(('chat', 'g:b'), 5)
        with patch('backend.live.routes.time.monotonic', return_value=70):
            hub.limit(('chat', 'g:a'), 5)

    def test_cross_language_fixture(self):
        # Also consumed by the JS test without needing Python at frontend test time.
        run = LiveRun(SEED, '11111111-1111-4111-8111-111111111111', 1700000000)
        fixture = json.loads((Path(__file__).parents[1] / 'frontend/tests/fixtures/live.json').read_text())
        for packet in fixture['packets']:
            run.apply(bytes(packet))
        self.assertEqual(run.snapshot()['board'], fixture['final']['board'])
        self.assertEqual(run.score, fixture['final']['score'])
