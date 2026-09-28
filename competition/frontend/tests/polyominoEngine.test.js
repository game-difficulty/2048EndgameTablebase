import assert from 'node:assert/strict';
import test from 'node:test';
import { movePolyominoTiles, PolyominoGame } from '../src/projects/polyominoEngine.js';
import { nextRandom } from '../src/projects/randomStreams.js';

const tile = (id, value, ...cells) => ({ id, value, cells });

test('a polyomino spawn consumes one numeric ticket', () => {
  const game = new PolyominoGame({ rows: 4, cols: 4 }, { seed: 'poly-tickets' });
  const before = game.randomState;
  game.spawn();
  assert.equal(game.randomState, nextRandom(before));
});

test('64+64 makes one rigid two-cell 128 along the move axis', () => {
  const horizontal = movePolyominoTiles([tile('a', 64, 1), tile('b', 64, 2)], 'left');
  assert.deepEqual(horizontal.tiles.map(item => [item.value, item.cells]), [[128, [0, 1]]]);
  assert.equal(horizontal.score, 128);
  const vertical = movePolyominoTiles([tile('a', 64, 4), tile('b', 64, 8)], 'up');
  assert.deepEqual(vertical.tiles.map(item => [item.value, item.cells]), [[128, [0, 4]]]);
});

test('two 128s merge on two overlapping cells from the side', () => {
  const result = movePolyominoTiles([
    tile('top', 128, 0, 1), tile('bottom', 128, 4, 5),
  ], 'up');
  assert.deepEqual(result.tiles.map(item => [item.value, item.cells]), [[256, [0, 1]]]);
  assert.equal(result.score, 256);
});

test('parallel 128s align fully before merging instead of stopping at first overlap', () => {
  const horizontal = movePolyominoTiles([
    tile('front', 128, 0, 1), tile('back', 128, 2, 3),
  ], 'left');
  assert.deepEqual(horizontal.tiles.map(item => [item.value, item.cells]), [[256, [0, 1]]]);
  const vertical = movePolyominoTiles([
    tile('front', 128, 0, 4), tile('back', 128, 8, 12),
  ], 'up');
  assert.deepEqual(vertical.tiles.map(item => [item.value, item.cells]), [[256, [0, 4]]]);
});

test('offset 128s with one overlapping cell make a straight three-cell 256', () => {
  const result = movePolyominoTiles([
    tile('top', 128, 1, 2), tile('bottom', 128, 6, 7),
  ], 'up');
  assert.deepEqual(result.tiles.map(item => [item.value, item.cells]), [[256, [1, 2, 3]]]);
});

test('perpendicular 128s with one overlapping cell make an L-shaped 256', () => {
  const result = movePolyominoTiles([
    tile('horizontal', 128, 1, 2), tile('vertical', 128, 10, 14),
  ], 'up');
  assert.deepEqual(result.tiles.map(item => [item.value, item.cells]), [[256, [1, 2, 6]]]);
});

test('a rigid multi-cell tile cannot move if any covered cell is blocked', () => {
  const result = movePolyominoTiles([
    tile('blocker', 2, 1), tile('wide', 128, 4, 5),
  ], 'up');
  assert.equal(result.changed, false);
  assert.deepEqual(result.tiles.find(item => item.id === 'wide').cells, [4, 5]);
});

test('256s cannot merge and spawns use only uncovered cells', () => {
  const blocked = movePolyominoTiles([
    tile('top', 256, 0, 1), tile('bottom', 256, 4, 5),
  ], 'up');
  assert.equal(blocked.changed, false);
  const game = new PolyominoGame({ id: 'test', rows: 4, cols: 4, spawn4Rate: 0 }, { seed: 'large-test' });
  game.tiles = [tile('shape', 256, 0, 1, 4)];
  for (let index = 0; index < 13; index += 1) game.spawn();
  const occupied = game.tiles.flatMap(item => item.cells);
  assert.equal(new Set(occupied).size, 16);
  assert.equal(game.spawn(), null);
});
