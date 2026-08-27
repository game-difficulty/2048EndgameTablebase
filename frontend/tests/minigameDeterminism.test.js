import assert from 'node:assert/strict';
import test from 'node:test';

import { MinigameController } from '../src/features/minigames/engine/controller.js';
import { MINIGAME_REGISTRY } from '../src/features/minigames/engine/registry.js';
import { createMinigameRuntime } from '../src/features/minigames/engine/runtime.js';
import { MinigameRankedRecorder } from '../src/features/minigames/engine/rankedRecorder.js';
import { replayMgo1 } from '../src/features/minigames/engine/rankedReplay.js';
import { encodeMgo1, MGO1_END_REASON } from '../src/features/minigames/protocol/index.js';
import { flattenBoard } from '../src/features/minigames/engine/utils.js';

const SEED = '0123456789abcdeffedcba9876543210';
const CLOCK = { now: () => 1_000_000 };

const deterministicEvilSpawn = async (board) => {
  const index = flattenBoard(board).findIndex((value) => Number(value) === 0);
  return index >= 0 ? { index, value: 1 } : null;
};

const makeRuntime = () => createMinigameRuntime({
  seedHex: SEED,
  clock: CLOCK,
  evilSpawn: deterministicEvilSpawn,
});

const comparableState = (state) => ({
  board: state.board,
  score: state.score,
  best: state.best,
  status: state.status,
  snapshot: state.snapshot,
  powerups: state.powerups,
  interaction: state.interaction,
});

test('all registered minigames start deterministically from the same seed', async () => {
  for (const definition of MINIGAME_REGISTRY) {
    const left = new MinigameController({ difficulty: 1, runtime: makeRuntime() });
    const right = new MinigameController({ difficulty: 1, runtime: makeRuntime() });
    const leftState = await left.startGame(definition.id);
    const rightState = await right.startGame(definition.id);
    assert.deepEqual(comparableState(leftState), comparableState(rightState), definition.id);
  }
});

test('runtime state survives a snapshot roundtrip', async () => {
  const first = new MinigameController({ difficulty: 1, runtime: makeRuntime() });
  await first.startGame('column-chaos');
  await first.move('left');
  const saved = first.statePayload().snapshot;

  const restored = new MinigameController({ difficulty: 1, runtime: makeRuntime() });
  await restored.startGame('column-chaos', saved);

  const nextFirst = await first.move('down');
  const nextRestored = await restored.move('down');
  assert.deepEqual(comparableState(nextFirst), comparableState(nextRestored));
});

test('all registered minigames continue deterministically after refresh', async () => {
  for (const definition of MINIGAME_REGISTRY) {
    const first = new MinigameController({ difficulty: 1, runtime: makeRuntime() });
    await first.startGame(definition.id);
    await first.move('left');
    const saved = first.statePayload().snapshot;

    const restored = new MinigameController({ difficulty: 1, runtime: makeRuntime() });
    await restored.startGame(definition.id, saved);
    const nextFirst = await first.move('down');
    const nextRestored = await restored.move('down');
    assert.deepEqual(comparableState(nextFirst), comparableState(nextRestored), definition.id);
  }
});

test('ranked recorder snapshot keeps the owning user across refresh', () => {
  const recorder = new MinigameRankedRecorder({
    runId: '123e4567-e89b-42d3-a456-426614174000',
    userId: 42,
    gameId: 'column-chaos',
    difficulty: 1,
    seedHex: SEED,
    startedAtMs: CLOCK.now(),
  });
  const restored = MinigameRankedRecorder.restore(recorder.exportSnapshot());
  assert.equal(restored?.userId, 42);
  assert.equal(restored?.runId, recorder.runId);
});

test('ranked elapsed time is derived from compact action deltas', () => {
  const recorder = new MinigameRankedRecorder({
    runId: '123e4567-e89b-42d3-a456-426614174000',
    userId: 42,
    gameId: 'column-chaos',
    difficulty: 1,
    seedHex: SEED,
    actions: [
      { type: 'move', direction: 'left', deltaMs: 120 },
      { type: 'move', direction: 'down', deltaMs: 340 },
    ],
    mutableActionCount: 2,
    startedAtMs: 1,
    lastActionAtMs: 999_999,
  });
  assert.equal(recorder.elapsedMs, 460);
});

test('ranked recorder reserves space for a terminal action', () => {
  const actions = Array.from({ length: 49_999 }, () => ({
    type: 'move',
    direction: 'left',
    deltaMs: 0,
  }));
  const recorder = new MinigameRankedRecorder({
    runId: '123e4567-e89b-42d3-a456-426614174000',
    userId: 42,
    gameId: 'column-chaos',
    difficulty: 1,
    seedHex: SEED,
    actions,
    mutableActionCount: actions.length,
    startedAtMs: CLOCK.now(),
  });
  assert.equal(recorder.finish(CLOCK.now(), null), true);
  assert.equal(recorder.actions.length, 50_000);
  assert.equal(recorder.actions.at(-1).type, 'end');
});

test('a deterministic operation stream replays to the claimed terminal state', async () => {
  const runtime = makeRuntime();
  let recorder = null;
  const controller = new MinigameController({
    difficulty: 1,
    runtime,
    onOperation({ operation, atMs, state }) {
      recorder.record(operation, atMs, state);
      if (state.engine?.isOver) recorder.finish(atMs, state);
    },
  });
  recorder = new MinigameRankedRecorder({
    runId: '123e4567-e89b-42d3-a456-426614174000',
    gameId: 'design-master-1',
    difficulty: 1,
    seedHex: SEED,
    startedAtMs: CLOCK.now(),
  });
  await controller.startGame('design-master-1', null, runtime);
  const directions = ['left', 'down', 'right', 'up'];
  for (let index = 0; !controller.engine.isOver && index < 20_000; index += 1) {
    await controller.move(directions[index % directions.length]);
  }
  assert.equal(controller.engine.isOver, true);
  assert.equal(recorder.ended, true);

  const replayed = await replayMgo1(recorder.encode(), { evilSpawn: deterministicEvilSpawn });
  assert.equal(replayed.score, controller.engine.score);
  assert.equal(replayed.trophyTier, controller.engine.isPassed);
  assert.deepEqual(replayed.finalBoard, flattenBoard(controller.engine.board));
});

test('a retired ranked stream verifies before natural game over', async () => {
  const runtime = makeRuntime();
  let recorder;
  const controller = new MinigameController({
    difficulty: 1,
    runtime,
    onOperation({ operation, atMs, state }) {
      recorder.record(operation, atMs, state);
    },
  });
  recorder = new MinigameRankedRecorder({
    runId: '123e4567-e89b-42d3-a456-426614174000',
    gameId: 'column-chaos',
    difficulty: 1,
    seedHex: SEED,
    startedAtMs: CLOCK.now(),
  });
  await controller.startGame('column-chaos', null, runtime);
  await controller.move('left');
  assert.equal(controller.engine.isOver, false);
  assert.equal(recorder.finish(CLOCK.now(), null, MGO1_END_REASON.RETIRED), true);

  const replayed = await replayMgo1(recorder.encode(), { evilSpawn: deterministicEvilSpawn });
  assert.equal(replayed.endReason, MGO1_END_REASON.RETIRED);
  assert.equal(replayed.score, controller.engine.score);
  assert.deepEqual(replayed.finalBoard, flattenBoard(controller.engine.board));
});

test('a premature game-over end remains invalid', async () => {
  const runtime = makeRuntime();
  const controller = new MinigameController({ difficulty: 1, runtime });
  const recorder = new MinigameRankedRecorder({
    runId: '123e4567-e89b-42d3-a456-426614174000',
    gameId: 'column-chaos',
    difficulty: 1,
    seedHex: SEED,
    startedAtMs: CLOCK.now(),
  });
  await controller.startGame('column-chaos', null, runtime);
  recorder.finish(CLOCK.now(), null, MGO1_END_REASON.GAME_OVER);
  await assert.rejects(
    replayMgo1(recorder.encode(), { evilSpawn: deterministicEvilSpawn }),
    /premature_end/u,
  );
});

test('all 20 minigames replay the same deterministic move prefix', async () => {
  for (const definition of MINIGAME_REGISTRY) {
    const runtime = makeRuntime();
    let recorder;
    const controller = new MinigameController({
      difficulty: 1,
      runtime,
      onOperation({ operation, atMs, state }) {
        recorder.record(operation, atMs, state);
      },
    });
    recorder = new MinigameRankedRecorder({
      runId: '123e4567-e89b-42d3-a456-426614174000',
      gameId: definition.id,
      difficulty: 1,
      seedHex: SEED,
      startedAtMs: CLOCK.now(),
    });
    await controller.startGame(definition.id, null, runtime);
    const directions = ['left', 'down', 'right', 'up'];
    for (let index = 0; !controller.engine.isOver && index < 16; index += 1) {
      await controller.move(directions[index % directions.length]);
    }
    const encoded = encodeMgo1({
      rulesVersion: 1,
      gameId: definition.id,
      difficulty: 1,
      runId: recorder.runId,
      seedHex: SEED,
      actions: recorder.actions,
    });
    const replayed = await replayMgo1(encoded, {
      evilSpawn: deterministicEvilSpawn,
      requireEnd: false,
    });
    assert.equal(replayed.score, controller.engine.score, definition.id);
    assert.deepEqual(replayed.finalBoard, flattenBoard(controller.engine.board), definition.id);
  }
});

test('powerups and custom actions use compact semantic records', async () => {
  const exercise = async (gameId, action) => {
    const runtime = makeRuntime();
    let recorder;
    const controller = new MinigameController({
      difficulty: 1,
      runtime,
      onOperation({ operation, atMs, state }) {
        recorder.record(operation, atMs, state);
      },
    });
    recorder = new MinigameRankedRecorder({
      runId: '123e4567-e89b-42d3-a456-426614174000',
      gameId,
      difficulty: 1,
      seedHex: SEED,
      startedAtMs: CLOCK.now(),
    });
    await controller.startGame(gameId, null, runtime);
    await action(controller);
    const encoded = encodeMgo1({
      rulesVersion: 1,
      gameId,
      difficulty: 1,
      runId: recorder.runId,
      seedHex: SEED,
      actions: recorder.actions,
    });
    const replayed = await replayMgo1(encoded, { evilSpawn: deterministicEvilSpawn, requireEnd: false });
    assert.equal(replayed.score, controller.engine.score, gameId);
    assert.deepEqual(replayed.finalBoard, flattenBoard(controller.engine.board), gameId);
    return recorder.actions;
  };

  const bombActions = await exercise('design-master-1', async (controller) => {
    const target = flattenBoard(controller.engine.board).findIndex((value) => value > 0);
    controller.usePowerup('bomb');
    controller.targetAction(target);
  });
  assert.equal(bombActions.at(-1).type, 'bomb');

  const gloveActions = await exercise('design-master-1', async (controller) => {
    const board = flattenBoard(controller.engine.board);
    const source = board.findIndex((value) => value > 0);
    const target = board.findIndex((value) => value === 0);
    controller.usePowerup('glove');
    controller.targetAction(source);
    controller.targetAction(target);
  });
  assert.equal(gloveActions.at(-1).type, 'glove');

  const twistActions = await exercise('design-master-1', async (controller) => {
    controller.usePowerup('twist');
    const target = controller.statePayload().interaction.validTargets[0];
    controller.targetAction(target);
  });
  assert.equal(twistActions.at(-1).type, 'twist');

  const customActions = await exercise('mystery-merge-1', async (controller) => {
    controller.triggerCustomAction({ key: 'peek', phase: 'start' });
    controller.triggerCustomAction({ key: 'peek', phase: 'end' });
  });
  assert.deepEqual(customActions.map((action) => action.type), ['custom', 'custom']);
});
