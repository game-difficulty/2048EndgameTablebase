import test from 'node:test';
import assert from 'node:assert/strict';
import { moveCargoBoard, CARGO_SHAPES } from '../src/projects/cargoEngine.js';
import { movePolyominoTiles } from '../src/projects/polyominoEngine.js';
import { slideSpecialTiles } from '../src/projects/practiceSpecialEngine.js';
import { moveBoard } from '../src/projects/engine.js';
import { settleRigidTiles, shiftCells } from '../src/projects/rigidMovement.js';

const empty = () => Array(16).fill(0);
const tile = (id, value, cells) => ({ id, value, cells });
const geometry = tiles => tiles.map(t => [t.value, t.cells]).sort((a, b) => a[0] - b[0]);
function unique(tiles) {
  const cells = tiles.flatMap(t => t.cells);
  assert.equal(cells.length, new Set(cells).size, 'objects must never overlap');
}

test('number follows the cargo in the same swipe and the transition reaches its final cell', () => {
  const board = empty(); board[0] = 2;
  const result = moveCargoBoard(board, { id: 'cargo', shape: 0, row: 0, col: 1 }, 'right');
  assert.equal(result.cargo.col, 2);
  assert.equal(result.board[1], 2);
  assert.deepEqual(result.movements, [{ from: 0, to: 1, value: 2, merged: false }]);
});

test('cargo moves after a number merge, and trailing numbers follow without merging twice', () => {
  const board = empty();
  board[0] = 2; board[1] = 2; board[2] = 4;
  const result = moveCargoBoard(board, { id: 'cargo', shape: 0, row: 1, col: 1 }, 'right');
  assert.deepEqual(result.board.slice(0, 4), [0, 0, 4, 4]);
  assert.equal(result.gained, 4);
  const blocked = empty(); blocked[2] = 2; blocked[3] = 2; blocked[0] = 4;
  const moved = moveCargoBoard(blocked, { id: 'cargo', shape: 3, row: 0, col: 0 }, 'right');
  // The two twos make space ahead; the number behind follows the rigid L.
  assert.equal(moved.gained, 4);
  assert.equal(moved.cargo.col, 1);
  assert.equal(moved.board[1], 4);
  assert.equal(moveCargoBoard(moved.board, moved.cargo, 'right').cargoMoved, false);
});

test('all cargo shapes finish free following motion in every direction in one operation', () => {
  for (let shape = 0; shape < CARGO_SHAPES.length; shape++) {
    const covered = new Set(CARGO_SHAPES[shape].cells.map(([r, c]) => (r + 1) * 4 + c + 1));
    for (let cell = 0; cell < 16; cell++) {
      if (covered.has(cell)) continue;
      for (const direction of ['up', 'right', 'down', 'left']) {
        const board = empty(); board[cell] = 2;
        const result = moveCargoBoard(board, { id: 'cargo', shape, row: 1, col: 1 }, direction);
        assert.equal(moveCargoBoard(result.board, result.cargo, direction).changed, false);
        const occupied = CARGO_SHAPES[shape].cells.map(([r, c]) => [r + result.cargo.row, c + result.cargo.col]);
        for (const [r, c] of occupied) if (r >= 0 && r < 4) assert.equal(result.board[r * 4 + c], 0);
      }
    }
  }
});

test('L concavity cannot cross an unprocessed number, regardless of IDs or input order', () => {
  for (const id of ['a', 'z']) {
    const input = [tile(id, 256, [1, 2, 6]), tile('b', 2, [7]), tile('block', 8, [8]), tile('edge', 4, [9])];
    for (const order of [input, [...input].reverse()]) {
      const result = movePolyominoTiles(order, 'right', 4, 5);
      assert.equal(result.changed, false);
      assert.deepEqual(geometry(result.tiles), geometry(input));
      unique(result.tiles);
    }
  }
});

test('L follows a number ahead and lets a trailing number follow it', () => {
  const result = movePolyominoTiles([tile('a', 256, [1, 2, 6]), tile('b', 2, [7]), tile('rear', 4, [0])], 'right', 4, 5);
  assert.deepEqual(geometry(result.tiles), [[2, [9]], [4, [2]], [256, [3, 4, 8]]]);
  unique(result.tiles);
  assert.equal(movePolyominoTiles(result.tiles, 'right', 4, 5).changed, false);
});

test('bonded domino follows blockers in both rows, while trailing number follows domino', () => {
  const result = slideSpecialTiles([
    { id: 'pair', kind: 'pair-double', value: 0, cells: [1, 5] },
    { id: 'rear', kind: 'number', value: 2, cells: [0] },
    { id: 'front', kind: 'number', value: 4, cells: [6] },
  ], 'right');
  assert.deepEqual(geometry(result.tiles), [[0, [2, 6]], [2, [1]], [4, [7]]]);
  unique(result.tiles);
});

test('mutual movement dependencies translate together, unless an external boundary blocks them', () => {
  // Synthetic shapes exercise the fixed point independently of project rules.
  const input = [tile('a', 1, [0, 5]), tile('b', 2, [1, 4])];
  const run = tiles => settleRigidTiles(tiles, 'right', { cols: 4, step: t => {
    const cells = shiftCells(t.cells, 'right', 4, 4); return cells && { cells };
  } });
  const moved = run(input);
  assert.deepEqual(geometry(moved.tiles), [[1, [2, 7]], [2, [3, 6]]]);
  assert.equal(run(moved.tiles).changed, false);
});

test('single-cell movement, scores and merge limits agree with ordinary 2048', () => {
  const values = [0, 2, 4, 8];
  for (let code = 0; code < 256; code++) {
    const board = empty();
    for (let i = 0; i < 4; i++) board[i] = values[(code >> (i * 2)) & 3];
    for (const direction of ['left', 'right']) {
      const expected = moveBoard(board, { rows: 4, cols: 4 }, direction);
      const cargo = moveCargoBoard(board, null, direction);
      assert.deepEqual(cargo.board, expected.board);
      assert.equal(cargo.gained, expected.score);
      const poly = movePolyominoTiles(board.flatMap((value, cell) => value ? [tile(String(cell), value, [cell])] : []), direction);
      const after = empty(); for (const t of poly.tiles) after[t.cells[0]] = t.value;
      assert.deepEqual(after, expected.board);
      assert.equal(poly.score, expected.score);
      assert.equal(poly.changed, expected.changed);
    }
  }
});
