import test from 'node:test';
import assert from 'node:assert/strict';
import { animationLayoutTarget } from '../src/components/useBoardAnimation.js';

test('animation layout flush stays inside the board surface', () => {
  const tile = { offsetHeight: 42 };
  const surface = { querySelector: selector => selector === '.moving-tile, .tile' ? tile : null };
  const page = { offsetHeight: 900 };
  assert.equal(animationLayoutTarget(surface, page), tile);
  assert.equal(animationLayoutTarget(null, page), page);
});
