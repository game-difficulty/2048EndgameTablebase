import assert from 'node:assert/strict';
import test from 'node:test';

import { MinigameRankedRecorder } from '../src/features/minigames/engine/rankedRecorder.js';
import { MinigameController } from '../src/features/minigames/engine/controller.js';
import { decodeMgo1 } from '../src/features/minigames/protocol/index.js';

const RUN_ID = '123e4567-e89b-42d3-a456-426614174000';
const SEED = '0123456789abcdeffedcba9876543210';

test('death checkpoints do not seal the live operation stream', () => {
  const recorder = new MinigameRankedRecorder({
    runId: RUN_ID,
    userId: 7,
    gameId: 'column-chaos',
    difficulty: 1,
    seedHex: SEED,
    startedAtMs: 1000,
  });
  assert.equal(recorder.record({ type: 'move', direction: 'left' }, 1010, null), true);

  const first = decodeMgo1(recorder.encodeCheckpoint());
  assert.deepEqual(first.actions.map(({ type }) => type), ['move', 'end']);
  assert.equal(recorder.ended, false);
  assert.deepEqual(recorder.actions.map(({ type }) => type), ['move']);

  assert.equal(recorder.record({ type: 'bomb', index: 3 }, 1020, null), true);
  const second = decodeMgo1(recorder.encodeCheckpoint());
  assert.deepEqual(second.actions.map(({ type }) => type), ['move', 'bomb', 'end']);
  assert.equal(recorder.mutableActionCount, 2);
  assert.equal(recorder.ended, false);
});

test('play state displays the persisted best without changing the engine score', async () => {
  const controller = new MinigameController({
    difficulty: 1,
    summaries: {
      'gravity-twist-1:1': { bestScore: 33392 },
    },
    snapshotKey: (gameId, difficulty) => `${gameId}:${difficulty}`,
  });

  const state = await controller.startGame('gravity-twist-1');

  assert.equal(state.score, 0);
  assert.equal(state.best, 33392);
  assert.equal(state.hud.best, 33392);
  assert.equal(state.snapshot.engine.maxScore, 0);
});
