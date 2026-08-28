import assert from 'node:assert/strict';
import test from 'node:test';

import {
  canApplyPracticeSeed,
  createPracticeSession,
  reducePracticeSession,
} from '../src/features/practice/engine/practiceSession.js';

const rolls = (...values) => {
  let index = 0;
  return () => values[index++] ?? 0;
};

test('move and undo are local monotonic state transitions', () => {
  const initial = createPracticeSession({
    board: [2, 2, ...new Array(14).fill(0)],
    score: 7,
    context: { rng: [1, 2, 3, 4] },
  });
  const moved = reducePracticeSession(initial, {
    type: 'MOVE_RANDOM',
    direction: 'left',
    spawnRate4: 0.1,
    randomSource: rolls(0, 0.9),
    nextContext: { rng: [5, 6, 7, 8] },
  });
  assert.equal(moved.accepted, true);
  assert.equal(moved.state.revision, 1);
  assert.equal(moved.state.score, 11);
  assert.equal(moved.state.transition.kind, 'move');
  assert.deepEqual(moved.state.transition.fromBoard, initial.board);
  assert.deepEqual(moved.state.transition.toBoard, moved.state.board);

  const undone = reducePracticeSession(moved.state, { type: 'UNDO' });
  assert.equal(undone.accepted, true);
  assert.equal(undone.state.revision, 2);
  assert.deepEqual(undone.state.board, initial.board);
  assert.equal(undone.state.score, 7);
  assert.deepEqual(undone.state.context, { rng: [1, 2, 3, 4] });
  assert.equal(undone.state.transition.kind, 'snapshot');
});

test('undo can preserve a live random cursor while restoring the board', () => {
  const initial = createPracticeSession({
    board: [2, 2, ...new Array(14).fill(0)],
    context: { state: [1, 2, 3, 4], turn: 0 },
  });
  const moved = reducePracticeSession(initial, {
    type: 'MOVE_RANDOM',
    direction: 'left',
    randomSource: rolls(0, 0),
    nextContext: { state: [5, 6, 7, 8], turn: 1 },
  }).state;
  const liveCursor = { state: [9, 10, 11, 12], turn: 2 };
  const undone = reducePracticeSession(moved, {
    type: 'UNDO',
    nextContext: liveCursor,
  }).state;

  assert.deepEqual(undone.board, initial.board);
  assert.deepEqual(undone.context, liveCursor);
  assert.deepEqual(undone.history.at(-1).context, liveCursor);
});

test('late spawn responses cannot mutate a newer revision', () => {
  const initial = createPracticeSession({
    board: [2, 2, ...new Array(14).fill(0)],
  });
  const moved = reducePracticeSession(initial, {
    type: 'MOVE_ONLY',
    direction: 'left',
  }).state;
  const undone = reducePracticeSession(moved, { type: 'UNDO' }).state;
  const lateSpawn = reducePracticeSession(undone, {
    type: 'SPAWN',
    expectedRevision: moved.revision,
    index: 1,
    value: 2,
  });
  assert.equal(lateSpawn.accepted, false);
  assert.equal(lateSpawn.reason, 'stale_revision');
  assert.equal(lateSpawn.state, undone);
});

test('move-only and spawn form one undoable history entry', () => {
  const initial = createPracticeSession({
    board: [2, 2, ...new Array(14).fill(0)],
  });
  const moved = reducePracticeSession(initial, {
    type: 'MOVE_ONLY',
    direction: 'left',
  }).state;
  assert.equal(moved.phase, 'awaiting_spawn');
  assert.equal(moved.history.length, 1);

  const spawned = reducePracticeSession(moved, {
    type: 'SPAWN',
    expectedRevision: moved.revision,
    index: 1,
    value: 4,
  }).state;
  assert.equal(spawned.phase, 'ready');
  assert.equal(spawned.history.length, 2);
  assert.equal(spawned.transition.kind, 'spawn');

  const undone = reducePracticeSession(spawned, { type: 'UNDO' }).state;
  assert.deepEqual(undone.board, initial.board);
});

test('set board invalidates an older async command', () => {
  const initial = createPracticeSession({ boardHex: '0000000000000011' });
  const baseRevision = initial.revision;
  const replaced = reducePracticeSession(initial, {
    type: 'SET_BOARD',
    boardHex: '0000000000000123',
  }).state;
  const staleMove = reducePracticeSession(replaced, {
    type: 'MOVE_RANDOM',
    expectedRevision: baseRevision,
    direction: 'left',
    randomSource: rolls(0, 0),
  });
  assert.equal(staleMove.accepted, false);
  assert.equal(staleMove.reason, 'stale_revision');
  assert.equal(staleMove.state.boardHex, '0000000000000123');
});

test('a delayed board seed cannot replace a newer local revision', () => {
  const initial = createPracticeSession({ boardHex: '0000000000000011' });
  assert.equal(canApplyPracticeSeed(initial, 0, 0), true);
  assert.equal(canApplyPracticeSeed(initial, 0, null), false);
  assert.equal(canApplyPracticeSeed(initial, 0, '0'), false);

  const moved = reducePracticeSession(initial, {
    type: 'MOVE_RANDOM',
    direction: 'left',
    randomSource: rolls(0, 0),
  }).state;
  assert.equal(canApplyPracticeSeed(moved, 0, 0), false);
  assert.equal(canApplyPracticeSeed(moved, 1, 0), false);
  assert.equal(canApplyPracticeSeed(moved, 1, 1), true);
});
