import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildTesterPrefetchPayload,
  createTesterSpawnRandomSource,
} from '../src/services/tablebases/testerPrefetchRng.js';

const INITIAL = { state: [1, 2, 3, 4], turn: 7 };

test('tester spawn RNG consumes exactly two values per turn', () => {
  const spawn = createTesterSpawnRandomSource(INITIAL);
  assert.equal(spawn.randomSource(), 11520 / 0x100000000);
  assert.equal(spawn.randomSource(), 0);
  assert.deepEqual(spawn.nextState(), {
    state: [12295, 1029, 1029, 25165824],
    turn: 8,
  });
});

test('tester prefetch payload is compact and clamps the spawn rate', () => {
  assert.deepEqual(buildTesterPrefetchPayload(INITIAL, 2), {
    version: 1,
    state: [1, 2, 3, 4],
    turn: 7,
    spawn_rate_4: 1,
  });
});
