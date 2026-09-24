import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from backend.gamer_ranked.rules import legal_moves
from backend.live.protocol import LiveRun, STEP
from backend.live.rooms import RoomDefinition
from backend.live.multi_content import MultiAiContent
from backend.live.multi_protocol import UPLINK, PUBLISH_PROTOCOL, encode_step, pack_batch, unpack_uplink
from backend.live.multi_store import MultiLiveStore
from tools.live_multi_runner import MultiCheckpoint


class Room:
    def __init__(self, path): self.path = path
    def store_path(self): return self.path


class Hub:
    producer = None
    def __init__(self, content):
        self.content = content
        self.control = dict(enabled=True, revision=0)
        self.events = []
        self.like_total = 0
    async def publisher_joined(self): pass
    def broadcast(self, event): self.events.append(event)
    def snapshot(self): return dict(type='snapshot', **self.content.snapshot())


class Socket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.closed = None
    async def accept(self): pass
    async def close(self, code): self.closed = code
    async def send_json(self, data): self.sent.append(data)
    async def receive(self):
        if not self.messages: return dict(type='websocket.disconnect')
        value = self.messages.pop(0)
        return dict(type='websocket.receive', **({'bytes': value} if isinstance(value, bytes) else {'text': json.dumps(value)}))


def handshake(runs=None):
    return [dict(type='hello', protocol=PUBLISH_PROTOCOL, control_version=1)] + [
        dict(type='sync_lane', lane=i, generation=1 if runs else 0, run=runs[i].checkpoint() if runs else None)
        for i in range(3)]


class MultiContentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.content = MultiAiContent(Room(Path(self.temp.name) / 'live.db'))
        await self.content.start()
        self.hub = Hub(self.content)

    async def test_three_lanes_sync_steps_source_batch_and_restore(self):
        runs = [LiveRun() for _ in range(3)]
        messages = handshake(runs)
        for lane, run in enumerate(runs):
            step = run.make_step(legal_moves(run.board)[0], 15 + lane * 65)
            run.apply(step)
            messages.append(UPLINK.pack(lane, 1, *STEP.unpack(step)))
        messages.append(dict(type='source', lane=1, generation=1, source='free10_1024'))
        ws = Socket(messages)
        await self.content.publish(ws, self.hub)
        self.assertIsNone(ws.closed)
        self.assertEqual([r.seq for r in self.content.runs], [1, 1, 1])
        self.assertEqual([r.board for r in self.content.runs], [r.board for r in runs])
        self.assertEqual(self.content.seq, 5)  # 3 steps, dictionary, source.
        self.assertTrue(any(isinstance(e, bytes) and e[0] == 0x21 for e in self.hub.events))
        restored = MultiAiContent(self.content.room)
        await restored.start()
        self.assertEqual([r.id for r in restored.runs], [r.id for r in runs])
        self.assertEqual(restored.snapshot()['lanes'][1]['run']['source_id'], 1)

    async def test_paused_ack_fences_all_lanes(self):
        self.hub.control = dict(enabled=False, revision=7)
        runs = [LiveRun() for _ in range(3)]
        step = runs[2].make_step(legal_moves(runs[2].board)[0], 50)
        ws = Socket(handshake(runs) + [dict(type='control_ack', revision=7), UPLINK.pack(2, 1, *STEP.unpack(step))])
        await self.content.publish(ws, self.hub)
        self.assertEqual(ws.closed, 1008)
        self.assertEqual([r.seq for r in self.content.runs], [0, 0, 0])

    async def test_stale_generation_and_duplicate_step_cannot_advance_other_lane(self):
        for generation in [0, 2]:
            runs = [LiveRun() for _ in range(3)]
            self.content.runs = [None] * 3
            step = runs[0].make_step(legal_moves(runs[0].board)[0], 50)
            ws = Socket(handshake(runs) + [UPLINK.pack(0, generation, *STEP.unpack(step))])
            await self.content.publish(ws, self.hub)
            self.assertEqual(ws.closed, 1008)
            self.assertEqual([r.seq for r in self.content.runs], [0, 0, 0])

    async def test_reconnect_adopts_valid_longer_prefix_but_rejects_fork(self):
        run = LiveRun()
        self.content.runs[0], self.content.generations[0] = run, 1
        longer = LiveRun.restore(run.checkpoint())
        longer.apply(longer.make_step(legal_moves(longer.board)[0], 80))
        await self.content.sync_lane(dict(lane=0, generation=1, run=longer.checkpoint()))
        self.assertEqual(self.content.runs[0].seq, 1)
        fork = LiveRun(run.seed, run.id, run.started)
        fork.apply(fork.make_step(legal_moves(fork.board)[0], 90))
        with self.assertRaisesRegex(ValueError, 'record_conflict'):
            await self.content.sync_lane(dict(lane=0, generation=1, run=fork.checkpoint()))

    async def test_lane_restart_is_a_barrier_after_old_steps(self):
        run = LiveRun()
        # A known terminal board can be used for lifecycle validation independent of AI search.
        run.board = [2,4,2,4,4,2,4,2,2,4,2,4,4,2,4,2]
        run.end()
        self.content.runs[0], self.content.generations[0] = run, 4
        self.content.hub = self.hub
        self.content.append(encode_step(1, STEP.pack(1, 15, 0)))
        self.content.event('lane_end', slot=self.content.slot_snapshot(0))
        self.assertIsInstance(self.hub.events[0], bytes)
        self.assertEqual(self.hub.events[1]['content_seq'], 2)
        self.assertEqual(self.hub.events[1]['slot']['generation'], 4)

    async def test_epoch_rotation_happens_before_applying_the_next_step(self):
        self.content.hub = self.hub
        before = self.content.epoch
        self.content.seq = 0xffffffff - 100
        self.content.ensure_capacity()
        self.assertNotEqual(self.content.epoch, before)
        self.assertEqual(self.hub.events[-1]['content_seq'], 0)


class MultiStorageTests(unittest.TestCase):
    def test_slow_viewer_queue_bounds_bytes_and_requests_full_reconnect(self):
        from backend.live.routes import LiveHub, ViewerQueue
        hub = LiveHub()
        queue = ViewerQueue()
        hub.viewers['slow'] = queue
        queue.put_nowait(b'x' * (256 * 1024 - 10))
        hub.broadcast({'type': 'gift', 'id': 'committed-gift'})
        self.assertTrue(queue.closing)
        self.assertIsNone(queue.get_nowait())
        self.assertEqual(queue.bytes, 0)
        hub.broadcast(b'ignored until reconnect')
        self.assertTrue(queue.empty())

    def test_migrate_original_game_and_atomic_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            store = MultiLiveStore(path / 'live.db')
            run = LiveRun()
            store.save(run)
            slots = store.load_slots()
            self.assertEqual(slots[0]['run']['run_id'], run.id)
            self.assertNotEqual(slots[1]['run']['run_id'], run.id)
            self.assertEqual(slots[1]['generation'], 1)
            legacy = path / 'old.json'
            legacy.write_text(json.dumps(run.checkpoint()), encoding='utf-8')
            checkpoint = MultiCheckpoint(path / 'multi', legacy)
            runs, generations = checkpoint.load()
            runs[1], generations[1] = LiveRun(), 1
            checkpoint.save(runs, generations)
            restored, actual = checkpoint.load()
            self.assertEqual([r.id if r else None for r in runs], [r.id if r else None for r in restored])
            self.assertEqual(actual, generations)

    def test_codec_size_and_bad_uplink(self):
        self.assertEqual(len(encode_step(2, STEP.pack(1, 15, 127))), 2)
        self.assertEqual(len(encode_step(2, STEP.pack(1, 80, 127))), 3)
        self.assertEqual(len(encode_step(2, STEP.pack(1, 3600000, 127))), 5)
        for raw in [b'', b'0', UPLINK.pack(3, 1, 1, 15, 0)]:
            with self.assertRaises(ValueError): unpack_uplink(raw)


if __name__ == '__main__': unittest.main()
