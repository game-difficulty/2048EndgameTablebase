import assert from 'node:assert/strict';
import test from 'node:test';

import { calculateFourSpawnRate } from '../src/human/bestTenPoster.js';

test('Best 10 poster derives the four-spawn rate from final board and score', () => {
  assert.equal(calculateFourSpawnRate([2, 2], 0), 0);
  assert.equal(calculateFourSpawnRate([4, 2], 0), 0.5);
  assert.equal(calculateFourSpawnRate([4], 4), 0);
  const rate = calculateFourSpawnRate(
    [2, 4, 2, 4, 8192, 512, 16, 8, 16384, 1024, 64, 32, 32768, 2048, 256, 4],
    795032,
  );
  assert.ok(rate > 0.09 && rate < 0.11);
});

test('Best 10 poster rejects impossible board and score combinations', () => {
  assert.equal(calculateFourSpawnRate([], 0), null);
  assert.equal(calculateFourSpawnRate([3, 2], 0), null);
  assert.equal(calculateFourSpawnRate([2, 2], 100), null);
});
