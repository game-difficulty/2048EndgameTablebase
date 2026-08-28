import assert from 'node:assert/strict';
import test from 'node:test';

import {
  GAMER_SPAWN_POLICY,
  resolveGamerSpawnPolicy,
} from '../src/features/gamer/engine/spawnPolicy.js';

test('an undo reroll takes priority over ranked and difficulty spawning', () => {
  assert.equal(resolveGamerSpawnPolicy({
    randomAfterUndo: true,
    rankedEligible: true,
    hasRankedRng: true,
  }), GAMER_SPAWN_POLICY.RANDOM);
});

test('eligible ranked games keep their deterministic spawn policy', () => {
  assert.equal(resolveGamerSpawnPolicy({
    rankedEligible: true,
    hasRankedRng: true,
  }), GAMER_SPAWN_POLICY.RANKED);
});

test('ordinary games continue to use their configured difficulty policy', () => {
  assert.equal(resolveGamerSpawnPolicy(), GAMER_SPAWN_POLICY.DIFFICULTY);
});
