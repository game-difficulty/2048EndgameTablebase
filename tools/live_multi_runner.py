"""Three persistent native AI processes, owned by one publishing coordinator."""
import argparse
import asyncio
import contextlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.live.protocol import LiveRun, STEP
from backend.live.multi_protocol import PUBLISH_PROTOCOL, UPLINK
from backend.gamer_ranked.rules import legal_moves
from tools.live_runner import NativeAI, RunnerControl, wait_for_step
from tools.live_pacing import batch_means, step_drift

LOG = logging.getLogger('live_multi_runner')


def ai_process(connection, engine_root, tables, threads, time_ratio):
    # spawn gives each lane its own configuration, native caches, and crash boundary.
    parent = mp.parent_process()
    def watch_parent():
        while parent and parent.is_alive():
            time.sleep(1)
        if parent:
            os._exit(1)
    threading.Thread(target=watch_parent, daemon=True, name='publisher-watchdog').start()
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    try:
        ai = NativeAI(Path(engine_root), tables, threads, time_ratio)
        run_id = None
        connection.send(('ready', None))
        while True:
            message = connection.recv()
            if message is None:
                break
            identity, board = message
            if run_id != identity[1]:
                # Never transfer depth prediction or strategy cooldowns into a new game.
                from engine_core.ai_merge_policy import allows_five_tiler_relaxation
                ai.logic = type(ai.logic)()
                ai.logic.time_limit_ratio = time_ratio
                ai.dispatcher = ai.Dispatcher(ai.np.zeros((4, 4), dtype=ai.np.int32), 0)
                ai.logic.allow_five_tiler_relaxation = allows_five_tiler_relaxation(ai.dispatcher.ad_readers)
                ai.player.clear_cache()
                run_id = identity[1]
            started = time.monotonic()
            direction, source = ai.choose(board)
            connection.send(('result', identity, direction, source, time.monotonic() - started))
    except BaseException as error:
        with contextlib.suppress(Exception):
            connection.send(('error', type(error).__name__, str(error)[:300]))
    finally:
        connection.close()


class AiChild:
    def __init__(self, args, tables, lane):
        self.args, self.tables, self.lane = args, tables, lane
        self.process = self.connection = None
        self.request = 0

    async def start(self):
        self.close()
        ctx = mp.get_context('spawn')
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=ai_process, name=f'Live-AI-{self.lane}',
            args=(child, self.args.engine_root, self.tables, self.args.threads, self.args.time_ratio), daemon=True)
        self.process.start()
        child.close()
        message = await self.receive(180)
        if message[0] != 'ready':
            raise RuntimeError(str(message))
        LOG.info('Lane %s ready, PID %s', self.lane, self.process.pid)

    async def receive(self, timeout):
        deadline = time.monotonic() + timeout
        while self.process and self.process.is_alive():
            if self.connection.poll():
                return self.connection.recv()
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Lane {self.lane} search watchdog')
            await asyncio.sleep(.003)
        raise RuntimeError(f'Lane {self.lane} process exited')

    async def choose(self, run, generation, revision):
        self.request += 1
        identity = (generation, run.id, run.seq, revision, self.request)
        self.connection.send((identity, run.board.copy()))
        reply = await self.receive(self.args.search_timeout)
        if reply[0] != 'result' or reply[1] != identity:
            raise RuntimeError(f'Lane {self.lane}: invalid AI response {reply[:2]}')
        return reply[2], reply[3]

    def close(self):
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=3)
            self.process.close()
        if self.connection:
            self.connection.close()
        self.process = self.connection = None


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.next')
    temporary.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)


class MultiCheckpoint:
    def __init__(self, directory, legacy):
        self.directory, self.legacy = Path(directory), Path(legacy)
        self.batch = None

    def load(self):
        manifest = self.directory / 'manifest.json'
        if not manifest.exists():
            legacy = json.loads(self.legacy.read_text(encoding='utf-8')) if self.legacy.exists() else None
            return [LiveRun.restore(legacy) if legacy else None, None, None], [1 if legacy else 0, 0, 0]
        data = json.loads(manifest.read_text(encoding='utf-8'))
        self.batch = data.get('batch')
        if data.get('version') != 1 or len(data.get('lanes', [])) != 3:
            raise ValueError('invalid_multi_checkpoint')
        runs, generations = [], []
        for lane, slot in enumerate(data['lanes']):
            # Slots are embedded in the atomically replaced manifest: no partial three-file commit.
            if slot.get('lane') != lane or type(slot.get('generation')) is not int or not 0 <= slot['generation'] <= 0xffffffff:
                raise ValueError('invalid_checkpoint_lane')
            runs.append(LiveRun.restore(slot['run']) if slot['run'] else None)
            generations.append(slot['generation'])
        return runs, generations

    def save(self, runs, generations, batch=None):
        atomic_json(self.directory / 'manifest.json', dict(version=1, batch=batch,
            lanes=[dict(lane=i, generation=generations[i], run=r.checkpoint() if r else None)
                   for i, r in enumerate(runs)]))


class MultiRunner:
    def __init__(self, args):
        self.args = args
        self.checkpoint = MultiCheckpoint(args.checkpoint_dir, args.legacy_checkpoint)
        self.runs, self.generations = self.checkpoint.load()
        self.batch = self.checkpoint.batch
        self.batch_ack = asyncio.Event()
        self.batch_active = asyncio.Event()
        tables = json.loads(Path(args.tables).read_text(encoding='utf-8')) if args.tables else None
        self.children = [AiChild(args, tables, i) for i in range(3)]

    def save(self):
        self.checkpoint.save(self.runs, self.generations, self.batch)

    async def receive_batch(self, data):
        if data.get('type') not in ('batch_ready','batch_state'):
            return
        if data['type'] == 'batch_state' and self.batch and self.batch['phase'] == 'preparing':
            return
        if data['type'] == 'batch_ready' and (not self.batch or data.get('batch',{}).get('id') != self.batch['id']):
            raise ValueError('unexpected_batch_ack')
        self.batch = data.get('batch')
        if data['type'] == 'batch_ready':
            for run in self.runs:
                run.started = self.batch['started_at']
            self.batch_active.set()
        if self.batch and self.batch['phase'] == 'void':
            self.batch_active.clear()
        self.save()
        if data['type'] == 'batch_ready':
            self.batch_ack.set()

    async def coordinate(self, socket, control):
        while True:
            revision = await control.wait()
            if self.batch and self.batch['phase'] == 'running' and any(r and not r.ended for r in self.runs):
                self.batch_active.set()
                await asyncio.sleep(.2)
                continue
            self.batch_active.clear()
            if self.batch and self.batch['phase'] not in ('cooldown','void'):
                async with control.lock:
                    await socket.send('{"type":"batch_status"}')
                await asyncio.sleep(.5)
                continue
            if self.batch and time.time() < self.batch.get('next_at',0):
                await asyncio.sleep(min(.5,self.batch['next_at']-time.time()))
                continue
            # Prepare all engines before starting the shared three-minute window.
            await asyncio.gather(*(child.start() for child in self.children if not child.process))
            async with control.lock:
                if not control.enabled or revision != control.revision:
                    continue
                runs = [LiveRun() for _ in range(3)]
                self.runs = runs
                self.generations = [g+1 for g in self.generations]
                self.batch = dict(id=str(uuid.uuid4()),phase='preparing')
                LOG.info('Batch %s mean step drift (ms): %s', self.batch['id'],
                         [round(value * 1000, 3) for value in batch_means(self.batch['id'])])
                self.save()
                self.batch_ack.clear()
                await socket.send(json.dumps(dict(type='batch_start',batch_id=self.batch['id'],
                    lanes=[dict(lane=i,generation=self.generations[i],run_id=r.id,seed=r.seed) for i,r in enumerate(runs)])))
            await asyncio.wait_for(self.batch_ack.wait(),30)

    async def lane_loop(self, lane, socket, control):
        child = self.children[lane]
        while True:
            revision = await control.wait()
            await self.batch_active.wait()
            run = self.runs[lane]
            if run is None or run.ended:
                await asyncio.sleep(.1)
                continue
            generation = self.generations[lane]
            if not legal_moves(run.board):
                async with control.lock:
                    if not control.enabled or revision != control.revision:
                        continue
                    run.end()
                    await socket.send(json.dumps(dict(type='end', lane=lane, generation=generation)))
                    self.save()
                continue
            if not child.process:
                async with control.lock:
                    await socket.send(json.dumps(dict(type='lane_status', lane=lane, generation=generation, status='recovering')))
                try:
                    await child.start()
                except Exception as error:
                    child.close()
                    LOG.warning('Lane %s restart: %s', lane, error)
                    await asyncio.sleep(5)
                    continue
                async with control.lock:
                    await socket.send(json.dumps(dict(type='lane_status', lane=lane, generation=generation, status='running')))
                # A control revision may have changed while the engine was initializing.
                continue
            started = time.monotonic()
            try:
                direction, source = await child.choose(run, generation, revision)
            except Exception as error:
                child.close()
                LOG.warning('Lane %s AI failed: %s', lane, error)
                continue
            await wait_for_step(started, source, self.args.interval, self.args.search_interval, run.board,
                                drift=step_drift((self.batch or {}).get('id'), lane, run.seq))
            async with control.lock:
                if not control.enabled or revision != control.revision or self.runs[lane] is not run or not self.batch_active.is_set():
                    continue
                if source != run.source:
                    run.source = source
                    await socket.send(json.dumps(dict(type='source', lane=lane, generation=generation, source=source)))
                step = run.make_step(direction, round((time.monotonic() - started) * 1000))
                run.apply(step)
                seq, delta, move = STEP.unpack(step)
                await socket.send(UPLINK.pack(lane, generation, seq, delta, move))

    async def connected(self, socket):
        await socket.send(json.dumps(dict(type='hello', protocol=PUBLISH_PROTOCOL, control_version=1)))
        reply = json.loads(await asyncio.wait_for(socket.recv(), 30))
        if reply.get('type') != 'sync':
            raise ValueError('expected_sync')
        for lane, run in enumerate(self.runs):
            await socket.send(json.dumps(dict(type='sync_lane', lane=lane, generation=self.generations[lane],
                                               run=run.checkpoint() if run else None)))
            reply = json.loads(await asyncio.wait_for(socket.recv(), 180))
            if reply.get('type') != 'resume_lane' or reply.get('lane') != lane:
                raise ValueError('invalid_resume')
            self.runs[lane] = LiveRun.restore(reply['run']) if reply['run'] else None
            self.generations[lane] = reply['generation']
        reply = json.loads(await asyncio.wait_for(socket.recv(), 30))
        if reply.get('type') != 'ready':
            raise ValueError('expected_ready')
        self.batch = reply.get('batch')
        self.batch_active.clear()
        if self.batch and self.batch['phase'] == 'running':
            self.batch_active.set()
            LOG.info('Resuming batch %s, mean step drift (ms): %s', self.batch['id'],
                     [round(value * 1000, 3) for value in batch_means(self.batch['id'])])
        self.save()
        control = RunnerControl(socket, reply['control'], self.save, self.receive_batch)
        await control.acknowledge()
        control.reader = asyncio.create_task(control.receive())

        async def heartbeat():
            while True:
                await asyncio.sleep(5)
                async with control.lock:
                    await socket.send('{"type":"ping"}' if self.batch and self.batch['phase']=='preparing' else '{"type":"batch_status"}')
                    self.save()

        tasks = [control.reader, asyncio.create_task(heartbeat())]
        tasks.append(asyncio.create_task(self.coordinate(socket, control)))
        tasks.extend(asyncio.create_task(self.lane_loop(i, socket, control)) for i in range(3))
        LOG.info('Three-lane publisher connected; enabled=%s', control.enabled)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            raise ConnectionError('Publisher task stopped')
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Discard pending IPC results before reconnecting; no stale search can advance a restored game.
            for child in self.children:
                child.close()
            self.save()

    async def run(self):
        from websockets.asyncio.client import connect
        token = os.environ.get('LIVE_PUBLISH_TOKEN', '')
        if len(token) < 32:
            raise ValueError('Missing LIVE_PUBLISH_TOKEN')
        retry = 1
        try:
            while True:
                try:
                    async with connect(self.args.url, additional_headers={'Authorization': 'Bearer ' + token},
                                       max_size=800_000, compression=None, ping_interval=10, ping_timeout=15) as socket:
                        retry = 1
                        await self.connected(socket)
                except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as error:
                    LOG.warning('Reconnecting in %ss: %s', retry, error)
                except Exception as error:
                    LOG.warning('Connection lost, reconnecting in %ss: %s', retry, error)
                await asyncio.sleep(retry)
                retry = min(60, retry * 2)
        finally:
            for child in self.children:
                child.close()
            self.save()


def main():
    from logging.handlers import RotatingFileHandler
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='wss://live.2048tables.online/api/live/rooms/ai-classic/publish')
    parser.add_argument('--engine-root', default=str(ROOT.parent / 'src'))
    parser.add_argument('--tables')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--time-ratio', type=float, default=1.6)
    parser.add_argument('--interval', type=float, default=.08)
    parser.add_argument('--search-interval', type=float, default=.05)
    parser.add_argument('--search-timeout', type=float, default=600)
    parser.add_argument('--checkpoint-dir', default=str(ROOT / 'data/live-runs'))
    parser.add_argument('--legacy-checkpoint', default=str(ROOT / 'data/live-runner.json'))
    parser.add_argument('--log-file')
    args = parser.parse_args()
    args.engine_root = str(Path(args.engine_root).resolve())
    args.interval, args.search_interval = max(.08, args.interval), max(.05, args.search_interval)
    handlers = None
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers = [RotatingFileHandler(args.log_file, maxBytes=1_000_000, backupCount=2, encoding='utf-8')]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)
    asyncio.run(MultiRunner(args).run())


if __name__ == '__main__':
    mp.freeze_support()
    main()
