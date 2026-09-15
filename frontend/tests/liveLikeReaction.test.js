import test from 'node:test';
import assert from 'node:assert/strict';
import { likeReactionWeights, pickLikeReaction } from '../src/live/likeReaction.js';

test('like artwork follows the exact requested weights', () => {
  const counts = {};
  for (let ticket = 0; ticket < 100; ticket++) {
    const id = pickLikeReaction(() => (ticket + .5) / 100);
    counts[id] = (counts[id] || 0) + 1;
  }
  assert.deepEqual(counts, {
    heart:25, flowers:25, '666':20, two:15, tea:5, whale:5, button:3, moai:2,
  });
  assert.equal(likeReactionWeights.reduce((total, [, weight]) => total + weight, 0), 100);
});

test('weighted selection handles the first and last ticket', () => {
  assert.equal(pickLikeReaction(() => 0), 'heart');
  assert.equal(pickLikeReaction(() => 1 - Number.EPSILON), 'moai');
});
