import test from 'node:test';
import assert from 'node:assert/strict';
import { humanLiveBoardGeometry, humanLiveVisibleNodeCount } from '../src/live/content/humanBoardGeometry.js';

test('all live human variants render square tiles after gaps and padding', () => {
  for (const [rows, cols] of [[4, 4], [3, 4], [2, 4], [3, 3]]) {
    const geometry = humanLiveBoardGeometry(rows, cols);
    const tileWidth = (geometry.width - (cols + 1) * geometry.gap) / cols;
    const tileHeight = (geometry.height - (rows + 1) * geometry.gap) / rows;
    assert.ok(Math.abs(tileWidth - tileHeight) < 1e-9, `${rows}x${cols}`);
  }
});

test('live sidebars fit the same number of fixed rows as the visible board', () => {
  assert.equal(humanLiveVisibleNodeCount(humanLiveBoardGeometry(4, 4).height), 11);
  assert.equal(humanLiveVisibleNodeCount(humanLiveBoardGeometry(3, 4).height), 8);
});
