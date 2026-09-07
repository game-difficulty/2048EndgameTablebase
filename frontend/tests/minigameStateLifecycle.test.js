import assert from 'node:assert/strict';
import test from 'node:test';
import { MinigameController } from '../src/features/minigames/engine/controller.js';
import { createMinigameRuntime, createVirtualMinigameClock } from '../src/features/minigames/engine/runtime.js';
import { animationInputLockMs } from '../src/features/minigames/model/animationInputLock.js';
import { createBufferedMinigameStore } from '../src/features/minigames/services/bufferedMinigameStore.js';

async function game(gameId) {
  const clock = createVirtualMinigameClock(1000);
  const controller = new MinigameController({ runtime: createMinigameRuntime({
    seedHex: '0123456789abcdeffedcba9876543210', clock,
  }) });
  await controller.startGame(gameId);
  controller.engine.board = [[0, 10, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  return { controller, clock };
}

test('Ice Age stage reveals keep animation without locking the next input', async () => {
  const { controller } = await game('ice-age');
  controller.engine.board = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  controller.engine.countDown[0][0] = 19;
  const moved = await controller.move('left');
  assert.equal(moved.animation.effects[0].type, 'ice_stage_reveal');
  assert.equal(animationInputLockMs(moved.animation), 0);
  const next = await controller.move('down');
  assert.equal(next.animation.direction, 'down');
  assert.notDeepEqual(next.board, moved.board);
  assert.equal(animationInputLockMs({ effects: [{ durationMs: 600 }] }), 600);
  assert.equal(animationInputLockMs({ followUp: { delayMs: 300, durationMs: 200 } }), 500);
});

test('Blitzkrieg expiry and rejected inputs never repeat the previous movement', async () => {
  const { controller, clock } = await game('blitzkrieg');
  const moved = await controller.move('left');
  assert.equal(moved.animation.direction, 'left');
  clock.advance(180001);
  const expired = controller.tick();
  assert.equal(expired.snapshot.engine.isOver, true);
  assert.equal(expired.status, 'trophy');
  assert.equal(expired.hud.customPanels[0].running, false);
  assert.deepEqual(expired.board, moved.board);
  assert.equal(expired.animation.direction, undefined);
  for (const direction of ['down', 'right', 'up']) {
    const rejected = await controller.move(direction);
    assert.deepEqual(rejected.board, expired.board);
    assert.equal(rejected.animation.direction, undefined);
    assert.equal(rejected.snapshot.engine.isOver, true);
  }
});

test('serializing Blitzkrieg HUD and snapshots does not advance or finish the engine', async () => {
  const { controller, clock } = await game('blitzkrieg');
  await controller.move('left');
  const engine = controller.engine;
  const remaining = engine.remainingMs;
  clock.advance(180001);
  const state = controller.statePayload();
  assert.equal(state.status, 'running');
  assert.equal(state.snapshot.engine.isOver, false);
  assert.equal(engine.remainingMs, remaining);
  assert.equal(state.hud.customPanels[0].syncedAt, 1000);
  assert.equal(controller.tick().snapshot.engine.isOver, true);
});

test('info/state refresh consumes no previous animation and preserves the logical board', async () => {
  for (const id of ['blitzkrieg', 'ice-age', 'gravity-twist-1', 'gravity-twist-2']) {
    const { controller } = await game(id);
    const moved = await controller.move('left');
    const info = controller.requestInfo();
    assert.deepEqual(info.board, moved.board, id);
    assert.equal(info.animation.direction, undefined, id);
    assert.equal(info.animation.followUp, undefined, id);
  }
});

function storageFixture() {
  let disk = { difficulty: 1, summaries: {}, activeGameSnapshots: {} };
  let writes = 0;
  let fail = false;
  const tasks = new Set();
  const buffer = createBufferedMinigameStore({
    read: () => structuredClone(disk),
    update: (fn) => {
      if (fail) throw new Error('quota');
      writes += 1;
      disk = structuredClone(fn(disk));
      return disk;
    },
  }, {
    schedule: (fn) => { tasks.add(fn); return fn; },
    cancel: (fn) => tasks.delete(fn),
  });
  return { buffer, tasks, disk: () => disk, writes: () => writes,
    fail: (value) => { fail = value; },
    external: (key, value) => { disk.activeGameSnapshots[key] = value; } };
}

function save(buffer, key, score) {
  buffer.update((s) => ({ ...s,
    activeGameSnapshots: { ...s.activeGameSnapshots, [key]: { score } },
  }));
}

test('rapid gameplay saves stay in memory and coalesce to the latest snapshot', () => {
  const f = storageFixture();
  for (let score = 1; score <= 100; score++) save(f.buffer, 'ice-age:1', score);
  assert.equal(f.writes(), 0);
  assert.equal(f.tasks.size, 1);
  assert.equal(f.buffer.read().activeGameSnapshots['ice-age:1'].score, 100);
  f.buffer.flush();
  assert.equal(f.writes(), 1);
  assert.equal(f.disk().activeGameSnapshots['ice-age:1'].score, 100);
  assert.equal(f.tasks.size, 0);
});

test('flush preserves another tab game, supports deletion and does not resave after lock loss', () => {
  const f = storageFixture();
  save(f.buffer, 'ice-age:1', 100);
  f.external('blitzkrieg:1', { score: 200 });
  f.buffer.flush();
  assert.equal(f.disk().activeGameSnapshots['blitzkrieg:1'].score, 200);
  save(f.buffer, 'ice-age:1', 300);
  f.buffer.discardSnapshot('ice-age:1');
  f.external('ice-age:1', { score: 400 });
  f.buffer.flush();
  assert.equal(f.disk().activeGameSnapshots['ice-age:1'].score, 400);
  f.buffer.update((s) => ({ ...s, activeGameSnapshots: {} }));
  f.buffer.flush();
  assert.equal(f.disk().activeGameSnapshots['ice-age:1'], undefined);
  assert.equal(f.disk().activeGameSnapshots['blitzkrieg:1'].score, 200);
});

test('failed storage write never rolls memory back and a later flush saves the newest state', () => {
  const f = storageFixture();
  save(f.buffer, 'blitzkrieg:1', 100);
  f.fail(true);
  assert.equal(f.buffer.flush(), false);
  save(f.buffer, 'blitzkrieg:1', 200);
  assert.equal(f.buffer.read().activeGameSnapshots['blitzkrieg:1'].score, 200);
  f.fail(false);
  assert.equal(f.buffer.flush(), true);
  assert.equal(f.disk().activeGameSnapshots['blitzkrieg:1'].score, 200);
});

test('a buffered summary cannot downgrade another tab personal best or trophy', () => {
  const f = storageFixture();
  f.buffer.update((s) => ({ ...s, summaries: { 'ice-age:1': { bestScore: 100, trophy: 1 } } }));
  f.disk().summaries['ice-age:1'] = { bestScore: 1000, trophy: 3, highestExp: 12, highestTile: 4096 };
  f.buffer.flush();
  assert.deepEqual(f.disk().summaries['ice-age:1'], {
    bestScore: 1000, trophy: 3, highestExp: 12, highestTile: 4096,
  });
});
