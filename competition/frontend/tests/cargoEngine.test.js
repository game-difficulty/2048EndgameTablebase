import test from 'node:test';
import assert from 'node:assert/strict';
import { CARGO_SHAPES, CARGO_SHAPE_GROUPS, FIRST_CARGO_MOVE, CargoGame, hasCargoMove, moveCargoBoard } from '../src/projects/cargoEngine.js';
import { nextRandom } from '../src/projects/randomStreams.js';

const empty = () => Array(16).fill(0);
const cargo = (shape, row, col = 1) => ({ id: 'cargo-0', shape, row, col });

test('cargo shapes have a separate stream from numeric spawns', () => {
  const first = new CargoGame({ id: 'cargo' }, { seed: 'shared-cargo' });
  const second = new CargoGame({ id: 'cargo' }, { seed: 'shared-cargo' });
  const numericBefore = first.randomState;
  assert.equal(first.nextCargo().shape, second.nextCargo().shape);
  assert.equal(first.randomState, numericBefore);
  first.spawnNumber();
  assert.equal(first.nextCargo().shape, second.nextCargo().shape);
  const before = first.randomState;
  first.board = Array(16).fill(2);
  assert.equal(first.spawnNumber(), null);
  assert.equal(first.randomState, nextRandom(before));
});

test('all six cargo variants fit the two-column entrance and exit', () => {
  assert.equal(CARGO_SHAPES.length, 6);
  for (let shape = 0; shape < CARGO_SHAPES.length; shape += 1) {
    const start = cargo(shape, -2);
    const blocked = empty();
    blocked[13] = 2;
    blocked[14] = 4;
    const first = moveCargoBoard(blocked, start, 'down');
    assert.equal(first.cargo.row, 1);
    assert.equal(moveCargoBoard(empty(), cargo(shape, -1), 'up').cargoMoved, false);
    const exit = moveCargoBoard(empty(), start, 'down');
    assert.equal(exit.delivered, true);
    assert.equal(exit.cargo.row, shape === 1 ? 3 : 4);
  }
});

test('numeric tiles remain on the board and block an occupied cargo destination', () => {
  const board = empty();
  board[4] = 2;
  const inside = cargo(0, 0, 1);
  const left = moveCargoBoard(board, inside, 'left');
  assert.equal(left.cargoMoved, false);
  assert.equal(left.board[4], 2);

  const shifted = moveCargoBoard(empty(), cargo(0, 0, 1), 'left');
  assert.equal(shifted.cargo.col, 0);
  assert.equal(moveCargoBoard(empty(), cargo(0, 0, 2), 'left').cargo.col, 0);
  assert.equal(moveCargoBoard(empty(), cargo(0, 2, 1), 'up').cargo.row, 0);
  assert.equal(moveCargoBoard(empty(), cargo(0, 3, 1), 'left').cargoMoved, false);
  assert.equal(moveCargoBoard(empty(), cargo(0, 3, 1), 'up').cargoMoved, false);
});

test('a partly exited vertical domino can shift sideways within the outlet', () => {
  const board = [
    2, 8, 2, 4,
    0, 4, 16, 2,
    2, 4, 16, 4,
    0, 0, 0, 128,
  ];
  const piece = cargo(5, 3, 0); // (3,1), (4,1); can align with outlet column 2.
  const aligned = moveCargoBoard(board, piece, 'right');
  assert.equal(aligned.changed, true);
  assert.equal(aligned.cargoMoved, true);
  assert.equal(aligned.cargo.col, 1);
  assert.equal(aligned.board[15], 128);
  assert.equal(moveCargoBoard(aligned.board, aligned.cargo, 'up').cargoMoved, false);
  assert.equal(moveCargoBoard(aligned.board, aligned.cargo, 'down').delivered, true);
  assert.equal(hasCargoMove(board, piece), true);

  const game = new CargoGame({ id: 'cargo' }, { seed: 'partly-exited' });
  game.board = board.slice();
  game.cargo = piece;
  game.moves = 89;
  const action = game.move('right');
  assert.equal(action.changed, true);
  assert.equal(action.snapshot.cargo.col, 1);
  assert.equal(action.snapshot.finished, false);
});

test('four equally weighted families contain only the requested variants', () => {
  assert.deepEqual(CARGO_SHAPE_GROUPS, [[3, 4], [1], [2, 5], [0]]);
  assert.deepEqual(CARGO_SHAPES[1].cells, [[1, 0], [1, 1]]);
  assert.deepEqual(CARGO_SHAPES[2].cells, [[0, 0], [1, 0]]);
  assert.deepEqual(CARGO_SHAPES[5].cells, [[0, 1], [1, 1]]);
  for (const shape of CARGO_SHAPE_GROUPS[0]) {
    assert.equal(CARGO_SHAPES[shape].cells.filter(([row]) => row === 0).length, 1);
    assert.equal(CARGO_SHAPES[shape].cells.filter(([row]) => row === 1).length, 2);
  }
  const game = new CargoGame({ id: 'cargo' }, { seed: 'shape-families' });
  const originalNumeric = game.randomState;
  const counts = [0, 0, 0, 0], variants = Array(6).fill(0);
  for (let i = 0; i < 16000; i++) {
    const groupState = nextRandom(game.shapeState);
    const variantState = nextRandom(groupState);
    const groupIndex = Math.floor(groupState / 0x100000000 * 4);
    const group = CARGO_SHAPE_GROUPS[groupIndex];
    const next = game.nextCargo();
    assert.equal(next.shape, group[Math.floor(variantState / 0x100000000 * group.length)]);
    assert.equal(game.shapeState, variantState);
    counts[groupIndex]++; variants[next.shape]++;
  }
  assert.equal(game.randomState, originalNumeric);
  for (const count of counts) assert.ok(count > 3600 && count < 4400);
  for (const index of [2, 3, 4, 5]) assert.ok(variants[index] > 1700 && variants[index] < 2300);
});

test('same seed keeps cargo sequence identical despite different numeric draws, and restores by shape cursor', () => {
  const a = new CargoGame({}, { seed: 'two-players' });
  const b = new CargoGame({}, { seed: 'two-players' });
  for (let i = 0; i < 100; i++) {
    for (let j = 0; j < i % 7; j++) a.nextSpawnTicket();
    assert.equal(a.nextCargo().shape, b.nextCargo().shape);
  }
  const restored = new CargoGame({}, { seed: 'two-players' });
  restored.shapeState = a.shapeState;
  restored.nextCargoId = a.nextCargoId;
  for (let i = 0; i < 20; i++) assert.deepEqual(restored.nextCargo(), a.nextCargo());
});

test('horizontal domino uses actual cells for entry, lateral movement and delivery', () => {
  // Anchor remains at -1 but both actual cells have entered row 0.
  const entered = cargo(1, -1, 1);
  const left = moveCargoBoard(empty(), entered, 'left');
  assert.equal(left.cargo.col, 0);
  assert.equal(moveCargoBoard(empty(), entered, 'up').changed, false);
  const exit = moveCargoBoard(empty(), entered, 'down');
  assert.equal(exit.delivered, true);
  assert.equal(exit.cargo.row, 3); // Both actual cells are already in outlet row 4.
  assert.equal(moveCargoBoard(empty(), exit.cargo, 'down').changed, false);
  assert.equal(moveCargoBoard(empty(), cargo(5, 0, 1), 'left').cargo.col, -1);
});

test('one delivery adds one point and immediately stages the next cargo', () => {
  const game = new CargoGame({ id: 'cargo' }, { seed: 'delivery' });
  game.board = empty();
  game.cargo = cargo(0, 3);
  game.nextCargoId = 1;
  const result = game.move('down');
  assert.equal(result.changed, true);
  assert.equal(result.snapshot.score, 1);
  assert.equal(result.snapshot.cargo.row, -2);
  assert.equal(result.snapshot.transition.cargoExit.row, 4);
  assert.notEqual(result.snapshot.cargo.id, result.snapshot.transition.cargoExit.id);
});

test('the first cargo appears after ten effective opening moves', () => {
  const game = new CargoGame({ id: 'cargo' }, { seed: 'first-cargo-ten' });
  assert.equal(game.snapshot().cargo, null);
  assert.equal(game.move('diagonal').changed, false);
  assert.equal(game.snapshot().moves, 0);
  for (let step = 1; step <= FIRST_CARGO_MOVE; step += 1) {
    const direction = ['down', 'left', 'up', 'right']
      .find(candidate => moveCargoBoard(game.board, game.cargo, candidate).changed);
    assert.ok(direction);
    const result = game.move(direction);
    assert.equal(result.changed, true);
    assert.equal(result.snapshot.moves, step);
    assert.equal(result.snapshot.cargo == null, step < FIRST_CARGO_MOVE);
  }
  assert.equal(game.snapshot().cargo.row, -2);
  assert.equal(game.snapshot().score, 0);
});

test('a dead opening board ends before its first cargo appears', () => {
  const game = new CargoGame({ id: 'cargo' }, { seed: 'death-search' });
  game.moves = 8;
  game.board = [8, 32, 32, 16, 4, 16, 0, 4, 16, 32, 4, 16, 32, 2, 32, 4];
  const result = game.move('down');
  assert.equal(result.snapshot.moves, 9);
  assert.equal(result.snapshot.cargo, null);
  assert.equal(result.snapshot.outcome, 'no_moves');
});

test('cargo remains playable beyond ten minutes', () => {
  const game = new CargoGame({ id: 'cargo' }, { seed: 'deadline' });
  game.startedAt -= 1200000;
  assert.ok(game.snapshot().elapsedMs >= 1200000);
  assert.ok(['down', 'left', 'up', 'right'].some(direction => game.move(direction).changed));
  assert.equal(game.finished, false);
  assert.equal(game.elapsed(game.startedAt+1200000),1200000);
  assert.equal(game.snapshot().remainingMs, null);
});

test('death requires both number tiles and cargo to have no effective move', () => {
  const full = Array.from({ length: 16 }, (_value, index) =>
    (Math.floor(index / 4) + index % 4) % 2 ? 4 : 2);
  assert.equal(hasCargoMove(full, cargo(0, -2)), false);
  assert.equal(hasCargoMove(empty(), cargo(0, -2)), true);
});

test('death freezes elapsed time and keeps the existing finish state', () => {
  const game = new CargoGame({ id: 'cargo' }, { seed: 'death-search' });
  game.board = [8, 32, 32, 16, 4, 16, 0, 4, 16, 32, 4, 16, 32, 2, 32, 4];
  game.cargo = cargo(0, -2);
  const result = game.move('down');
  assert.equal(result.changed, true);
  assert.equal(result.snapshot.finished, true);
  assert.equal(result.snapshot.outcome, 'no_moves');
  assert.equal(result.snapshot.remainingMs, null);
  const elapsed = result.snapshot.elapsedMs;
  assert.equal(game.elapsed(game.startedAt + 1200000), elapsed);
  assert.equal(game.snapshot().outcome, 'no_moves');
  assert.equal(game.move('left').changed, false);
});
