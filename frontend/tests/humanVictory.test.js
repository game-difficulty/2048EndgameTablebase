import test from 'node:test';
import assert from 'node:assert/strict';
import { reachedVictory } from '../src/human/victory.js';

test('victory only triggers on a configured target crossing', () => {
  const before = { variant: '4x4', board: [1024, 1024] };
  assert.equal(reachedVictory(before, { board: [2048] }), 2048);
  assert.equal(reachedVictory(before, { board: [1024] }), 0);
  assert.equal(reachedVictory({ ...before, board: [2048] }, { board: [2048] }), 0);
  assert.equal(reachedVictory({ ...before, victoryShown: true }, { board: [2048] }), 0);
  for (const variant of ['3x4', '3x3', '2x4']) {
    assert.equal(reachedVictory({ ...before, variant }, { board: [2048] }), 0);
  }
  assert.equal(reachedVictory({ variant: '2x4', board: [256, 256] }, { board: [512] }, { '2x4': 512 }), 512);
});
