import assert from 'node:assert/strict';
import test from 'node:test';

import {
  clearTesterPracticeState,
  restoreTesterPracticeState,
  saveTesterPracticeState,
} from '../src/features/tester/services/testerPracticeStore.js';

const values = new Map();
globalThis.window = {
  sessionStorage: {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  },
};

test('tester practice session roundtrips bigint replay records', () => {
  clearTesterPracticeState();
  const saved = saveTesterPracticeState({
    userId: 7,
    pattern: 'L3',
    target: '256',
    session: {
      practice: {
        board: new Array(16).fill(0),
        boardHex: '0000000000000011',
      },
      records: [{ board: 0x123456789abcdef0n, change: 1, rates: [1, 2, 3, 4] }],
    },
  });

  assert.equal(saved, true);
  const restored = restoreTesterPracticeState();
  assert.equal(restored.userId, 7);
  assert.equal(restored.session.records[0].board, 0x123456789abcdef0n);
});

test('tester practice session rejects malformed snapshots', () => {
  values.set('2048tables:tester-practice:v1', JSON.stringify({ version: 1, userId: 7 }));
  assert.equal(restoreTesterPracticeState(), null);
});
