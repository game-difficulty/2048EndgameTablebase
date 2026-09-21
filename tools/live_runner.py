"""Local native AI broadcaster. Never runs a search on the cloud server."""
import argparse
import asyncio
import contextlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.live.protocol import LiveRun
from backend.gamer_ranked.rules import packed_native_board, legal_moves

LOG = logging.getLogger('live_runner')


class NativeAI:
    def __init__(self, engine_root, tables=None, threads=1, time_ratio=1.0):
        sys.path.insert(0, str(engine_root))
        os.chdir(engine_root)
        import numpy as np
        from Config import SingletonConfig
        # Desktop path validation persists by default. A broadcaster must never
        # overwrite the desktop's configuration, including during initialization.
        SingletonConfig.save_config = classmethod(lambda cls, *args, **kwargs: None)
        from engine_core.AIPlayer import Dispatcher, CoreAILogic
        from engine_core.ai_merge_policy import allows_five_tiler_relaxation, merge_urgency_for_readers
        from native_core import ai_core
        config = SingletonConfig().config
        config['4_spawn_rate'] = 0.1
        if tables is not None:
            config['filepath_map'] = {
                (name, 0.1): [(str(path), str(dtype)) for path, dtype in entries]
                for name, entries in tables.items()
            }
        self.np, self.Dispatcher, self.core = np, Dispatcher, ai_core
        self.logic = CoreAILogic()
        self.logic.time_limit_ratio = time_ratio
        self.player = ai_core.AIPlayer(0)
        self.player.max_threads = threads
        self.dispatcher = Dispatcher(np.zeros((4, 4), dtype=np.int32), 0)
        self.player.merge_urgency = merge_urgency_for_readers(self.dispatcher.ad_readers)
        self.logic.allow_five_tiler_relaxation = allows_five_tiler_relaxation(self.dispatcher.ad_readers)
        LOG.info('Five-tiler pruning relaxation: %s', self.logic.allow_five_tiler_relaxation)
        LOG.info('Search merge urgency: %.1f', self.player.merge_urgency)

    def choose(self, values):
        board = self.np.array(values, dtype=self.np.int64).reshape((4, 4))
        encoded = packed_native_board(values)
        self.dispatcher.reset(board, encoded)
        move = self.dispatcher.dispatcher()
        source = str(self.dispatcher.current_table)
        if move == 'AI':
            large_tiles = [value for value in values if value >= 32768]
            # Packed 0xf cannot distinguish 32k from 65k and higher tiles.
            equal_pair = len(large_tiles) == 2 and large_tiles[0] == large_tiles[1]
            self.player.board = self.core.resolve_32768_doubles(encoded) if equal_pair else encoded
            code = self.logic.calculate_step(
                self.player,
                board,
                self.dispatcher.counts,
                preferred_moves=getattr(self.dispatcher, 'ai_search_moves', None),
            )
            move = {1: 'left', 2: 'right', 3: 'up', 4: 'down'}.get(code)
            source = 'AI'
        direction = str(move).lower()
        if direction not in legal_moves(values):
            raise RuntimeError(f'Invalid AI move {direction!r}; source={source}; '
                               f'board={encoded:016x}; values={values}')
        return direction, source


def save_checkpoint(path, run):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.next')
    temporary.write_text(json.dumps(run.checkpoint()), encoding='utf-8')
    temporary.replace(path)


def step_interval(board, source, table_interval, search_interval):
    if board is not None:
        if sum(value >= 1024 for value in board) <= 2:
            return .015
        large_tiles = sum(value >= 512 for value in board)
        if large_tiles >= 7:
            return .18
        if large_tiles >= 6:
            return .12
    return search_interval if source == 'AI' else table_interval


async def wait_for_step(started, source, table_interval, search_interval, board=None):
    interval = step_interval(board, source, table_interval, search_interval)
    # Windows timers can wake early; enforce the minimum on the clock.
    remaining = interval - (time.monotonic() - started)
    while remaining > 0:
        await asyncio.sleep(remaining)
        remaining = interval - (time.monotonic() - started)


class RunnerControl:
    """Fence steps at the send boundary; a native search already running may finish."""
    def __init__(self, socket, initial, checkpoint):
        self.socket, self.checkpoint = socket, checkpoint
        self.enabled = initial.get('enabled', True)
        self.revision = initial.get('revision', 0)
        self.lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.reader = None
        if self.enabled:
            self.wake.set()

    async def acknowledge(self):
        await self.socket.send(json.dumps(dict(type='control_ack', revision=self.revision)))

    async def receive(self):
        try:
            while True:
                data = json.loads(await self.socket.recv())
                if data.get('type') != 'control' or data['revision'] < self.revision:
                    continue
                async with self.lock:
                    self.enabled, self.revision = data['enabled'], data['revision']
                    if self.enabled:
                        self.wake.set()
                    else:
                        self.wake.clear()
                        self.checkpoint()
                    await self.acknowledge()
                    LOG.info('Broadcast %s', 'resumed' if self.enabled else 'paused')
        finally:
            self.wake.set()

    async def wait(self):
        await self.wake.wait()
        if self.reader and self.reader.done():
            self.reader.result()
            raise ConnectionError('Control channel closed')
        return self.revision


async def run_forever(args):
    from websockets.asyncio.client import connect
    token = os.environ.get('LIVE_PUBLISH_TOKEN', '')
    if len(token) < 32:
        raise ValueError('Set LIVE_PUBLISH_TOKEN to a random secret of at least 32 characters')
    checkpoint = Path(args.checkpoint).resolve()
    tables = json.loads(Path(args.tables).read_text(encoding='utf-8')) if args.tables else None
    ai = NativeAI(Path(args.engine_root).resolve(), tables, args.threads, args.time_ratio)
    run = LiveRun.restore(json.loads(checkpoint.read_text(encoding='utf-8'))) if checkpoint.exists() else None
    retry = 1
    while True:
        try:
            async with connect(args.url, additional_headers={'Authorization': 'Bearer ' + token},
                               max_size=800_000, ping_interval=10, ping_timeout=15,
                               compression=None) as socket:
                await socket.send(json.dumps({'type': 'hello', 'control_version': 1,
                                              'run': run.checkpoint() if run else None}))
                reply = json.loads(await asyncio.wait_for(socket.recv(), 20))
                run = LiveRun.restore(reply['run']) if reply.get('run') else None
                LOG.info('Connected')
                retry = 1
                control = RunnerControl(socket, reply.get('control', {}),
                                        lambda: save_checkpoint(checkpoint, run) if run else None)
                if 'control' in reply:
                    await control.acknowledge()
                control.reader = asyncio.create_task(control.receive())

                async def heartbeat():
                    while True:
                        await asyncio.sleep(5)
                        await socket.send('{"type":"ping"}')

                ping = asyncio.create_task(heartbeat())
                saved_at = time.monotonic()
                try:
                    while True:
                        revision = await control.wait()
                        if run is None or run.ended:
                            if run:
                                await asyncio.sleep(max(0, (run.restart_at or 0) - time.time()))
                            async with control.lock:
                                if not control.enabled or revision != control.revision:
                                    continue
                                run = LiveRun()
                                await socket.send(json.dumps({'type': 'start', 'run_id': run.id, 'seed': run.seed}))
                                save_checkpoint(checkpoint, run)
                        if not legal_moves(run.board):
                            async with control.lock:
                                if not control.enabled or revision != control.revision:
                                    continue
                                delay = random.randint(10, 20)
                                run.end(time.time() + delay)
                                await socket.send(json.dumps({'type': 'end', 'delay': delay}))
                                save_checkpoint(checkpoint, run)
                            continue
                        started = time.monotonic()
                        direction, source = await asyncio.to_thread(ai.choose, run.board.copy())
                        await wait_for_step(started, source, args.interval, args.search_interval, run.board)
                        async with control.lock:
                            if not control.enabled or revision != control.revision:
                                continue
                            if source != run.source:
                                run.source = source
                                await socket.send(json.dumps({'type': 'source', 'source': source}))
                            packet = run.make_step(direction, round((time.monotonic()-started)*1000))
                            run.apply(packet)
                            await socket.send(packet)
                        if time.monotonic()-saved_at >= 5:
                            save_checkpoint(checkpoint, run)
                            saved_at = time.monotonic()
                finally:
                    ping.cancel()
                    control.reader.cancel()
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await ping
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await control.reader
        except asyncio.CancelledError:
            if run:
                save_checkpoint(checkpoint, run)
            raise
        except Exception as error:
            if run:
                save_checkpoint(checkpoint, run)
            LOG.warning('Paused: %s: %s; reconnect in %ss', type(error).__name__, error, retry)
            await asyncio.sleep(retry)
            retry = min(60, retry * 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='wss://live.2048tables.online/api/live/publish')
    parser.add_argument('--engine-root', default=str(ROOT.parent / 'src'))
    parser.add_argument('--checkpoint', default=str(ROOT / 'data/live-runner.json'))
    parser.add_argument('--tables', help='Optional JSON object: full pattern -> [[local path, dtype], ...]')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--time-ratio', type=float, default=1.0)
    parser.add_argument('--interval', type=float, default=0.08, help='Minimum table step interval in seconds')
    parser.add_argument('--search-interval', type=float, default=0.05, help='Minimum AI search step interval in seconds')
    parser.add_argument('--log-file', help='Optional bounded rotating log file')
    args = parser.parse_args()
    args.interval = max(.08, args.interval)
    args.search_interval = max(.05, args.search_interval)
    handlers = None
    if args.log_file:
        path = Path(args.log_file).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers = [RotatingFileHandler(path, maxBytes=512_000, backupCount=1, encoding='utf-8')]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)
    LOG.info('Step minimums: AI search %.0fms, table %.0fms', args.search_interval * 1000, args.interval * 1000)
    LOG.info('Stage overrides: <=2 tiles >=1024: 15ms; >=7 tiles >=512: 180ms; >=6: 120ms')
    asyncio.run(run_forever(args))


if __name__ == '__main__':
    main()
