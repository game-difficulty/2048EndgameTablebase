import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { threeByThreeTileStyle } from '../src/components/threeByThreeTileStyle.js';
const source = readFileSync(new URL('../public/verse-replay/app.js', import.meta.url), 'utf8');
const fn = source.slice(source.indexOf('  function updateBoardGeometry()'), source.indexOf('  const boardResizeObserver'));
test('replay 3x3 gap matches Play at mobile and desktop widths', () => {
  for (const width of [280, 343, 500, 720]) {
    const values = {};
    const board = { clientWidth: width, classList: { contains: () => true }, style: { setProperty: (key, value) => values[key] = value } };
    runInNewContext(fn + ';updateBoardGeometry();', { elements: { board } });
    const gap = parseFloat(values['--replay-board-gap']);
    const tile = (width - 4 * gap) / 3;
    assert.ok(Math.abs(gap - parseFloat(threeByThreeTileStyle(3, 3, tile)['--board-gap'])) < 1e-9);
    board.classList.contains = () => false;
    board.clientWidth = 800;
    runInNewContext(fn + ';updateBoardGeometry();', { elements: { board } });
    assert.equal(parseFloat(values['--replay-board-gap']), gap);
  }
});
