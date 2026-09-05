import assert from 'node:assert/strict';
import test from 'node:test';

import {
  boardViewportSignature,
  createBoardViewport,
  createBoardViewportLayout,
  logicalIndexToVisual,
} from '../src/utils/boardViewport.js';

const boardFromHex = (hex) => [...hex].map((digit) => {
  const exponent = Number.parseInt(digit, 16);
  return exponent === 15 ? 32768 : (exponent > 0 ? 2 ** exponent : 0);
});

test('crops complete variant wall borders while retaining logical indices', () => {
  const twoByFour = createBoardViewport(boardFromHex('ffff00101021ffff'), true);
  assert.deepEqual(twoByFour, {
    rows: 2,
    cols: 4,
    rowStart: 1,
    colStart: 0,
    visibleIndices: [4, 5, 6, 7, 8, 9, 10, 11],
  });
  assert.deepEqual(logicalIndexToVisual(4, twoByFour), { row: 0, col: 0 });
  assert.deepEqual(logicalIndexToVisual(11, twoByFour), { row: 1, col: 3 });
  assert.equal(logicalIndexToVisual(0, twoByFour), null);
});

test('recognizes the current 3x3 and 3x4 variant layouts', () => {
  const threeByThree = createBoardViewport(boardFromHex('100f101f213fffff'), true);
  assert.equal(boardViewportSignature(threeByThree), '0:0:3:3');
  assert.deepEqual(threeByThree.visibleIndices, [0, 1, 2, 4, 5, 6, 8, 9, 10]);

  const threeByFour = createBoardViewport(boardFromHex('1230e210ee11ffff'), true);
  assert.equal(boardViewportSignature(threeByFour), '0:0:3:4');
  assert.deepEqual(threeByFour.visibleIndices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
});

test('does not crop classic boards or irregular internal walls', () => {
  const classic = createBoardViewport(boardFromHex('ffff001010210000'), false);
  assert.equal(boardViewportSignature(classic), '0:0:4:4');

  const irregular = boardFromHex('000f000000000000');
  const viewport = createBoardViewport(irregular, true);
  assert.equal(boardViewportSignature(viewport), '0:0:4:4');
  assert.equal(viewport.visibleIndices.length, 16);
});

test('keeps visible tiles square for rectangular viewports', () => {
  const layout = createBoardViewportLayout({ rows: 2, cols: 4 });
  const physicalTileWidth = layout.widthPercent * layout.tileWidthPercent / 100;
  const physicalTileHeight = layout.heightPercent * layout.tileHeightPercent / 100;
  assert.ok(Math.abs(physicalTileWidth - physicalTileHeight) < 1e-9);
  assert.equal(layout.widthPercent, 100);
  assert.ok(Math.abs(layout.heightPercent - 51.25) < 1e-9);
});
