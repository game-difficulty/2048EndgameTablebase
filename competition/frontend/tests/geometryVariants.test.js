import test from 'node:test';
import assert from 'node:assert/strict';
import { PROJECT_BY_ORDER } from '../src/projects/catalog.js';
import { AftershockGame, LookBackGame, shiftAftershock } from '../src/projects/geometryVariants.js';
import { nextRandom, seed32 } from '../src/projects/randomStreams.js';
import { moveBoard } from '../src/projects/engine.js';

function shape(state) {
  return Array.from({ length: state.rows }, (_, row) =>
    state.board.slice(row * state.cols, (row + 1) * state.cols)
      .map(value => value === -1 ? '_' : 'x').join(''));
}

test('aftershock preserves the exact original-axis geometry in the 3×3 example', () => {
  const first = shiftAftershock(Array(9).fill(0), 3, 3, 0, 0, 'row', 1, 1);
  assert.deepEqual(shape(first), ['xxx_', '_xxx', 'xxx_']);
  const second = shiftAftershock(first.board, first.rows, first.cols, first.originRow, first.originCol, 'col', 1, -1);
  assert.deepEqual(shape(second), ['_x__', 'xxx_', '_xxx', 'x_x_']);
  assert.equal(second.originRow, -1);
  const third = shiftAftershock(second.board, second.rows, second.cols, second.originRow, second.originCol, 'row', 2, -1);
  assert.deepEqual(shape(third), ['__x__', '_xxx_', '__xxx', 'x_x__']);
  assert.equal(third.originCol, -1);
  assert.equal(third.board.filter(value => value !== -1).length, 9);
});

test('an empty original row may be drawn and shifted without altering any cell', () => {
  let state = { board: Array(16).fill(0), rows: 4, cols: 4, originRow: 0, originCol: 0 };
  for (let col = 0; col < 4; col += 1) {
    state = shiftAftershock(state.board, state.rows, state.cols, state.originRow, state.originCol, 'col', col, 1);
  }
  assert.equal(state.originRow, 1); // Original y=0 is now entirely empty.
  const shifted = shiftAftershock(state.board, state.rows, state.cols, state.originRow, state.originCol, 'row', 0, -1);
  assert.deepEqual(shifted.board, state.board);
  assert.equal(shifted.originRow, 1);
  assert.ok(shifted.cells.every(cell => cell.fromRow === cell.toRow && cell.fromCol === cell.toCol));
});

test('disconnected regions cannot slide or merge across a missing cell, and isolated empties do not prevent death', () => {
  const board = [2,-1,2,2];
  const moved = moveBoard(board, {rows:1,cols:4}, 'left');
  assert.deepEqual(moved.board, [2,-1,4,0]);
  const game = new AftershockGame(PROJECT_BY_ORDER[19], {seed:'isolated'});
  game.board = [2,-1,0]; game.rows = 1; game.cols = 3;
  game.settleOutcome();
  assert.equal(game.outcome, 'no_moves');
});

test('multiple large merges draw exactly one shared quake, independent of the numeric stream', () => {
  const a = new AftershockGame(PROJECT_BY_ORDER[19], {seed:'stream-audit',side:'yellow'});
  const b = new AftershockGame(PROJECT_BY_ORDER[19], {seed:'stream-audit',side:'white'});
  b.nextSpawnTicket();
  const before = a.quakeState;
  for (const game of [a,b]) game.board = [128,128,256,256,...Array(12).fill(0)];
  a.move('left'); b.move('left');
  assert.deepEqual(a.transition.quake, b.transition.quake);
  assert.equal(a.quakeState, nextRandom(nextRandom(nextRandom(before))));
  assert.equal(a.score, 768);
  assert.equal(a.transition.kind, 'reshape');
  assert.notEqual(a.randomState, b.randomState);
});

test('one large merge causes one quake from its own shared stream before spawning', () => {
  const project = PROJECT_BY_ORDER[19];
  const yellow = new AftershockGame(project, { seed: 'shared-shock', side: 'yellow' });
  const white = new AftershockGame(project, { seed: 'shared-shock', side: 'white' });
  for (const game of [yellow, white]) {
    game.board = [128,128,0,0, 2,0,0,0, ...Array(8).fill(0)];
    game.quakeDraw = (() => { const values = [.1, .375, .75]; return () => values.shift(); })();
  }
  const a = yellow.move('left').snapshot;
  const b = white.move('left').snapshot;
  assert.equal(a.transition.quake.axis, 'row');
  assert.equal(a.transition.quake.line, 1);
  assert.equal(a.transition.quake.step, 1);
  assert.equal(a.rows, 4);
  assert.equal(a.cols, 5);
  assert.equal(a.board.filter(value => value !== -1).length, 16);
  assert.equal(a.score, 256);
  assert.deepEqual(a.board, b.board);
  assert.equal(a.transition.quake.cells.length, 16);
});

test('a merge below 256 never draws the aftershock stream', () => {
  const game = new AftershockGame(PROJECT_BY_ORDER[19], { seed: 'small-merge' });
  game.board = [64,64,0,0, ...Array(12).fill(0)];
  const before = game.quakeState;
  const result = game.move('left').snapshot;
  assert.equal(result.score, 128);
  assert.equal(result.transition.quake, undefined);
  assert.equal(game.quakeState, before);
});

test('look back ignores ineffective input, then restores all game metrics and spawns twice', () => {
  let seed;
  for (let index = 0; index < 10000; index += 1) {
    const candidate = `lookback-${index}`;
    const first = nextRandom(seed32(`${candidate}:lookback:0`));
    const second = nextRandom(first);
    if (first / 0x100000000 >= .05 && second / 0x100000000 < .05) { seed = candidate; break; }
  }
  assert.ok(seed);
  const game = new LookBackGame(PROJECT_BY_ORDER[20], { seed });
  game.board = [2,0,0,0, ...Array(8).fill(0)];
  const state = game.lookBackState;
  assert.equal(game.move('left').changed, false);
  assert.equal(game.lookBackState, state);
  assert.equal(game.move('right').changed, true);
  assert.equal(game.moves, 1);
  const randomAfterForward = game.randomState;
  const reverted = game.move('left').snapshot;
  assert.equal(reverted.transition.kind, 'lookback');
  assert.equal(reverted.moves, 0);
  assert.equal(reverted.score, 0);
  assert.equal(reverted.board.filter(value => value > 0).length, 3);
  assert.equal(reverted.transition.spawns.length, 2);
  assert.notEqual(game.randomState, randomAfterForward);
});

test('look back restores score and move counters, preserves clocks and RNG cursors, and caps spawns at free cells', () => {
  const game = new LookBackGame(PROJECT_BY_ORDER[20], {seed:'undo-audit'});
  game.board = [0,2,4,8,16,32,64,128,256,512,1024,2048];
  game.score = 120; game.moves = 7;
  const previous = game.historyEntry();
  game.history = [previous];
  game.board = [2,0,0,0,...Array(8).fill(0)]; game.score = 200; game.moves = 8;
  game.lookBackState = 1; // The next chance draw is below 5%.
  const spawnCursor = game.randomState, started = game.startedAt;
  const result = game.move('right').snapshot;
  assert.equal(result.transition.kind, 'lookback');
  assert.equal(result.score, 120); assert.equal(result.moves, 7);
  assert.equal(result.transition.spawns.length, 1);
  assert.equal(game.randomState, nextRandom(spawnCursor));
  assert.equal(game.lookBackState, nextRandom(1));
  assert.equal(game.startedAt, started);
  assert.deepEqual(result.board.slice(1), previous.board.slice(1));
});
