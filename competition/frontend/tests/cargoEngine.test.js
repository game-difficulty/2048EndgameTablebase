import test from 'node:test';
import assert from 'node:assert/strict';
import { CARGO_SHAPES, FIRST_CARGO_MOVE, CargoGame, hasCargoMove, moveCargoBoard } from '../src/projects/cargoEngine.js';
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

test('all five cargo shapes fit the two-column entrance and exit', () => {
  assert.equal(CARGO_SHAPES.length, 5);
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
    assert.equal(exit.cargo.row, 4);
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

test('a partly exited L can shift right to align with the outlet', () => {
  const board = [
    2, 8, 2, 4,
    0, 4, 16, 2,
    2, 4, 16, 4,
    0, 0, 0, 128,
  ];
  const piece = cargo(2, 3, 0); // (3,0), (3,1), (4,1), as pictured.
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
