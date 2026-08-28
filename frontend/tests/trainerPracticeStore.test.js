import assert from 'node:assert/strict';
import test from 'node:test';

import {
  clearTrainerPracticeState,
  restoreTrainerPracticeState,
  saveTrainerPracticeState,
} from '../src/features/trainer/services/trainerPracticeStore.js';

const values = new Map();
globalThis.window = {
  sessionStorage: {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  },
};

test('trainer practice snapshot roundtrips local board ownership state', () => {
  clearTrainerPracticeState();
  assert.equal(saveTrainerPracticeState({
    userId: 9,
    pattern: '442t',
    target: '512',
    spawnMode: 2,
    practice: {
      board: new Array(16).fill(0),
      boardHex: '0000000000000000',
      history: [{ boardHex: '0000000000000000' }],
    },
  }), true);
  const restored = restoreTrainerPracticeState();
  assert.equal(restored.userId, 9);
  assert.equal(restored.pattern, '442t');
  assert.equal(restored.spawnMode, 2);
});

test('trainer practice snapshot rejects malformed state', () => {
  values.set('2048tables:trainer-practice:v1', JSON.stringify({ version: 1, userId: 9 }));
  assert.equal(restoreTrainerPracticeState(), null);
});
