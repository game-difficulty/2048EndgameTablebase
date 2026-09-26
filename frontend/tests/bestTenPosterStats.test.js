import assert from 'node:assert/strict';
import test from 'node:test';

import { calculateFourSpawnRate, LANDSCAPE_POSTER_HEIGHT, LANDSCAPE_POSTER_WIDTH,
  posterCardBounds } from '../src/human/bestTenPoster.js';

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

test('landscape Best 10 poster gives the podium more space than the supporting results', () => {
  const featured = posterCardBounds(0, 'landscape');
  const cards = Array.from({ length: 9 }, (_, index) => posterCardBounds(index + 1, 'landscape'));
  assert.ok(featured.x + featured.width < cards[0].x);
  assert.equal(cards[0].width, cards[1].width);
  assert.ok(cards[0].width > cards[2].width);
  assert.equal(new Set(cards.slice(2, 6).map(card => card.x)).size, 4);
  assert.equal(new Set(cards.slice(6).map(card => card.x)).size, 3);
  assert.equal(new Set(cards.map(card => card.y)).size, 3);
  for (const card of [featured, ...cards]) {
    assert.ok(card.x >= 0 && card.y >= 0);
    assert.ok(card.x + card.width <= LANDSCAPE_POSTER_WIDTH);
    assert.ok(card.y + card.height <= LANDSCAPE_POSTER_HEIGHT);
  }
});
