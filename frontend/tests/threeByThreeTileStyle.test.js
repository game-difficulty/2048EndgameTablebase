import test from 'node:test';
import assert from 'node:assert/strict';
import { threeByThreeTileStyle } from '../src/components/threeByThreeTileStyle.js';

test('3x3 labels and corners scale with tile width without changing other boards', () => {
  for (const side of [60, 100, 180]) {
    const style = threeByThreeTileStyle(3, 3, side);
    const boardWidth = side * 3 / (1 - 4 * 0.036);
    assert.ok(Math.abs(parseFloat(style['--board-gap']) / boardWidth - 0.036) < 1e-12);
    assert.ok(Math.abs(parseFloat(style['--tile-label-small']) * 1.2 / side - 0.504) < 1e-12);
    assert.ok(Math.abs(parseFloat(style['--tile-label-medium']) / side - 0.33) < 1e-12);
    assert.ok(Math.abs(parseFloat(style['--tile-label-large']) / side - 0.28) < 1e-12);
    assert.ok(Math.abs(parseFloat(style['--tile-corner-radius']) / side - 0.03) < 1e-12);
  }
  for (const dimensions of [[4, 4], [3, 4], [4, 3]]) {
    assert.deepEqual(threeByThreeTileStyle(...dimensions, 100), {});
  }
  for (const side of [0, -1, NaN, Infinity]) assert.deepEqual(threeByThreeTileStyle(3, 3, side), {});
});
