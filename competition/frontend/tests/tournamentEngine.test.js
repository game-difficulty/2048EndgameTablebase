import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createHardShape,
  cutsOffCorner,
  hasMoveWithSeals,
  ISLAND,
  maxPlayableRectangle,
  moveBoard,
  moveBoardWithSeals,
  TournamentGame,
  WALL,
} from '../src/projects/engine.js';
import { nextRandom, ticketFloat } from '../src/projects/randomStreams.js';
import { PROJECT_BY_ORDER } from '../src/projects/catalog.js';

const mirror = { rows: 4, cols: 4, mirrorPortals: true, unmergeable: 64, spawn4Rate: .1 };

test('mirror movement exits the outer edge and enters the opposite portal', () => {
  const before = [2,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0];
  const result = moveBoard(before, mirror, 'left');
  assert.deepEqual(result.board.slice(0, 4), [0,0,2,0]);
  assert.deepEqual(result.movements[0], { from: 0, to: 2, value: 2, merged: false });
});

test('64 and 256 bricks do not merge', () => {
  const sixtyFour = moveBoard([0,0,64,64, ...Array(12).fill(0)], mirror, 'left');
  assert.deepEqual(sixtyFour.board.slice(0, 4), [0,0,64,64]);
  assert.equal(sixtyFour.score, 0);

  const large = { rows: 5, cols: 5, unmergeable: 256 };
  const twoFiftySix = moveBoard([256,256,0,0,0, ...Array(20).fill(0)], large, 'left');
  assert.deepEqual(twoFiftySix.board.slice(0, 5), [256,256,0,0,0]);
  assert.equal(twoFiftySix.score, 0);
});

test('a wall splits a movement line and never moves', () => {
  const project = { rows: 3, cols: 3 };
  const result = moveBoard([2,0,WALL, 0,0,0, 0,0,0], project, 'right');
  assert.deepEqual(result.board.slice(0, 3), [0,2,WALL]);
});

test('sealed cells keep their number and split movement without receiving spawns', () => {
  const project = { rows: 4, cols: 4, sealEveryMoves: 100 };
  const board = [2, 4, 2, 0, ...Array(12).fill(0)];
  const moved = moveBoardWithSeals(board, project, 'right', [1]);
  assert.deepEqual(moved.board.slice(0, 4), [2, 4, 0, 2]);
  assert.equal(moved.movements.some(item => item.from === 1), false);

  const game = new TournamentGame(project, { seed: 'sealed-spawn' });
  game.sealedCells = [0];
  game.board = [0, ...Array(15).fill(2)];
  assert.equal(game.randomSpawn(game.board), null);
});

test('opening seals are selected before either initial number spawns', () => {
  const project = { id: 'seal', rows: 4, cols: 4, sealEveryMoves: 100, sealCount: 3 };
  for (let sample = 0; sample < 32; sample += 1) {
    const game = new TournamentGame(project, { seed: `opening-seals-${sample}` });
    const snapshot = game.snapshot();
    assert.equal(snapshot.sealRound, 1);
    assert.equal(snapshot.sealedCells.length, 3);
    assert.equal(new Set(snapshot.sealedCells).size, 3);
    assert.deepEqual(snapshot.transition.seals.sealed, snapshot.sealedCells);
    assert.equal(snapshot.transition.spawns.length, 2);
    for (const spawn of snapshot.transition.spawns) {
      assert.ok(!snapshot.sealedCells.includes(spawn.index));
    }
    for (const sealed of snapshot.sealedCells) assert.equal(snapshot.board[sealed], 0);
  }
});

test('three diagonal seals may not cut off any of the four corners', () => {
  for (const pattern of [[8, 5, 2], [1, 6, 11], [4, 9, 14], [7, 10, 13]]) {
    assert.equal(cutsOffCorner(pattern, 4, 4), true);
  }
  assert.equal(cutsOffCorner([2, 5, 9], 4, 4), false);
  assert.equal(cutsOffCorner([8, 5, 2], 5, 5), false);

  const project = { id: 'seal', rows: 4, cols: 4, sealEveryMoves: 100, sealCount: 3 };
  const game = new TournamentGame(project, { seed: 'corner-cutoff-reroll' });
  game.sealedCells = [];
  const draws = [.5, 5 / 15, 2 / 14, 0, 0, 0];
  game.sealRandom = () => draws.shift();
  const result = game.rotateSeals();
  assert.deepEqual(result.sealed, [0, 1, 2]);
  assert.equal(draws.length, 0);
});

test('the hundredth-move spawn excludes newly selected sealed cells', async () => {
  const project = { id: 'seal', rows: 4, cols: 4, sealEveryMoves: 100, sealCount: 3 };
  for (let sample = 0; sample < 32; sample += 1) {
    const game = new TournamentGame(project, { seed: `rotation-spawn-${sample}` });
    const opening = game.sealedCells.slice();
    const source = Array.from({ length: 16 }, (_value, index) => index).find(index =>
      index % 4 < 3 && !opening.includes(index) && !opening.includes(index + 1));
    game.board = Array(16).fill(0);
    game.board[source] = 2;
    game.moves = 99;
    const result = await game.move('right');
    assert.equal(result.snapshot.moves, 100);
    assert.ok(!result.snapshot.sealedCells.includes(result.snapshot.transition.spawn.index));
  }
});

test('every hundred effective moves releases old cells and seals three new ones', async () => {
  const project = { id: 'seal', rows: 4, cols: 4, sealEveryMoves: 100, sealCount: 3 };
  const game = new TournamentGame(project, { seed: 'seal-cycle' });
  const opening = game.sealedCells.slice();
  game.spawn = async () => null;
  game.board = Array(16).fill(0);
  const firstSource = Array.from({length: 16}, (_value, index) => index).find(index =>
    index % 4 < 3 && !opening.includes(index) && !opening.includes(index + 1));
  game.board[firstSource] = 2;
  game.moves = 99;
  const first = await game.move('right');
  assert.equal(first.snapshot.moves, 100);
  assert.equal(first.snapshot.sealRound, 2);
  assert.equal(first.snapshot.sealedCells.length, 3);
  assert.ok(first.snapshot.sealedCells.every(index => !opening.includes(index)));
  assert.deepEqual(first.snapshot.transition.seals.released, opening);
  assert.equal(first.snapshot.nextSealIn, 100);
  const previous = first.snapshot.sealedCells;

  game.board = Array(16).fill(0);
  const source = Array.from({length: 16}, (_value, index) => index).find(index =>
    index % 4 < 3 && !previous.includes(index) && !previous.includes(index + 1));
  game.board[source] = 2;
  game.moves = 199;
  game.finished = false;
  game.finishedAt = null;
  const second = await game.move('right');
  assert.equal(second.snapshot.moves, 200);
  assert.equal(second.snapshot.sealRound, 3);
  assert.equal(second.snapshot.sealedCells.length, 3);
  assert.ok(second.snapshot.sealedCells.every(index => !previous.includes(index)));
  assert.deepEqual(second.snapshot.transition.seals.released, previous);
});

test('a sealed empty cell cannot postpone a no-moves result', () => {
  const project = { rows: 4, cols: 4, sealEveryMoves: 100 };
  const game = new TournamentGame(project, { seed: 'sealed-death' });
  game.board = Array.from({length: 16}, (_value, index) => 2 ** (index + 1));
  game.board[0] = 0;
  game.sealedCells = [0];
  assert.equal(hasMoveWithSeals(game.board, project, game.sealedCells), false);
  game.settleOutcome();
  assert.equal(game.finished, true);
  assert.equal(game.outcome, 'no_moves');
});

test('islands merge only with islands and a whole adjacent group contracts to one', () => {
  const project = { rows: 1, cols: 5 };
  const result = moveBoard([ISLAND,ISLAND,ISLAND,2,0], project, 'left');
  assert.deepEqual(result.board, [ISLAND,2,0,0,0]);
  assert.equal(result.score, 0);
  assert.equal(result.movements.filter(item => item.value === ISLAND).length, 3);
});

test('hard isolated island uses a five-percent base special spawn', async () => {
  const game = new TournamentGame({ rows: 4, cols: 4, isolatedIsland: true }, { seed: 'island' });
  let ticket = 1;
  while (ticketFloat(ticket, 'special') >= .05 || ticketFloat(ticket, 'position') >= 1 / 16) ticket += 1;
  game.nextSpawnTicket = () => ticket;
  game.board = Array(16).fill(0);
  const spawn = await game.spawn(game.board);
  assert.deepEqual(spawn, { index: 0, value: ISLAND });
});

test('one spawn ticket per effective spawn, including special and empty-board attempts', async () => {
  const game = new TournamentGame({ rows: 4, cols: 4, isolatedIsland: true }, { seed: 'one-ticket' });
  game.board = Array(16).fill(0);
  let before = game.randomState;
  await game.spawn(game.board);
  assert.equal(game.randomState, nextRandom(before));
  game.board = Array(16).fill(2);
  before = game.randomState;
  assert.equal(await game.spawn(game.board), null);
  assert.equal(game.randomState, nextRandom(before));
});

test('seal and shape setup do not consume numeric spawn tickets', () => {
  const seal = new TournamentGame({ rows: 4, cols: 4, sealEveryMoves: 100 }, { seed: 'streams' });
  const matchingSeal = new TournamentGame({ rows: 4, cols: 4, sealEveryMoves: 100 }, { seed: 'streams' });
  const before = seal.randomState;
  matchingSeal.nextSpawnTicket();
  assert.deepEqual(seal.rotateSeals().sealed, matchingSeal.rotateSeals().sealed);
  assert.equal(seal.randomState, before);
  const shapeA = new TournamentGame({ rows: 6, cols: 6, shapeShifter: true, playableCells: 12 }, { seed: 'shape' });
  const shapeB = new TournamentGame({ rows: 6, cols: 6, shapeShifter: true, playableCells: 12 }, { seed: 'shape' });
  shapeB.nextSpawnTicket();
  assert.deepEqual(shapeA.emptyBoard(), shapeB.emptyBoard());
});

test('dice sides may differ while their numeric spawn stream stays shared', () => {
  const project = { rows: 3, cols: 3, diceWall: true };
  const yellow = new TournamentGame(project, { seed: 'same-seed', side: 'yellow' });
  const white = new TournamentGame(project, { seed: 'same-seed', side: 'white' });
  assert.equal(yellow.randomState, white.randomState);
  assert.equal(yellow.nextSpawnTicket(), white.nextSpawnTicket());
});

test('hard shape shifter crops twelve connected cells from a 6x6 source', () => {
  let state = 0x12345678;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  for (let sample = 0; sample < 200; sample += 1) {
    const shape = createHardShape(random, 12);
    const playable = shape.board.map((value,index) => value !== WALL ? index : -1).filter(index => index >= 0);
    assert.equal(playable.length, 12);
    assert.ok(shape.cols >= shape.rows);
    assert.ok(shape.rows <= 6 && shape.cols <= 6);
    for (const edge of [
      shape.board.slice(0, shape.cols),
      shape.board.slice(-shape.cols),
      Array.from({length: shape.rows}, (_, row) => shape.board[row * shape.cols]),
      Array.from({length: shape.rows}, (_, row) => shape.board[row * shape.cols + shape.cols - 1]),
    ]) assert.ok(edge.some(value => value !== WALL));
    const rectangle = maxPlayableRectangle(shape.board, shape.rows, shape.cols);
    assert.ok(rectangle >= 4 && rectangle <= 8);

    const reached = new Set([playable[0]]), queue = [playable[0]];
    while (queue.length) {
      const index = queue.shift(), row = Math.floor(index / shape.cols), col = index % shape.cols;
      for (const [nextRow,nextCol] of [[row-1,col],[row+1,col],[row,col-1],[row,col+1]]) {
        const next = nextRow * shape.cols + nextCol;
        if (nextRow < 0 || nextRow >= shape.rows || nextCol < 0 || nextCol >= shape.cols || shape.board[next] === WALL || reached.has(next)) continue;
        reached.add(next); queue.push(next);
      }
    }
    assert.equal(reached.size, 12);
  }
});

test('EvilGen adapter uses adaptive depth and caps low-sum boards at 5', async () => {
  const depths = [];
  const fake = async (board, depth) => {
    depths.push(depth);
    return { index: board.indexOf(0), value: 2 };
  };
  const game = new TournamentGame({ rows: 4, cols: 4, evilSpawn: true }, { seed: 'depth', evilSpawn: fake });
  for (const emptyCount of [1, 2, 3, 5, 6]) {
    game.board = Array(16).fill(16);
    for (let index = 0; index < emptyCount; index += 1) game.board[index] = 0;
    await game.spawn(game.board);
  }
  game.board = Array(16).fill(4);
  game.board[0] = 0;
  await game.spawn(game.board);
  game.board = Array(16).fill(8);
  game.board[0] = 0;
  await game.spawn(game.board);
  assert.deepEqual(depths, [7, 6, 5, 5, 4, 5, 7]);
});

test('undo restores gameplay state without rewinding the RNG cursor', async () => {
  const project = { rows: 3, cols: 3, spawn4Rate: .1, allowUndo: true };
  const game = new TournamentGame(project, { seed: 'undo' });
  const before = game.snapshot();
  const randomStateBeforeMove = game.randomState;
  const direction = ['left','right','up','down'].find(item => moveBoard(before.board, project, item).changed);
  const moved = await game.move(direction);
  assert.equal(moved.changed, true);
  const randomStateAfterMove = game.randomState;
  assert.equal(game.undo(), true);
  const restored = game.snapshot();
  assert.deepEqual(restored.board, before.board);
  assert.equal(restored.score, before.score);
  assert.equal(restored.moves, before.moves);
  assert.equal(game.randomState, randomStateAfterMove);
  assert.notEqual(game.randomState, randomStateBeforeMove);
});

test('undo race remains recoverable after a no-move board before its target', () => {
  const project = { rows: 3, cols: 3, targetSum: 2044, allowUndo: true };
  const game = new TournamentGame(project, { seed: 'recoverable-death' });
  const previous = game.historyEntry();
  game.board = [2, 4, 8, 16, 32, 64, 128, 256, 512];
  game.score = 4096;
  game.moves = 100;
  game.history = [previous];

  game.settleOutcome();

  assert.equal(game.finished, false);
  assert.equal(game.outcome, null);
  assert.equal(game.snapshot().canUndo, true);
  assert.equal(game.undo(), true);
  assert.deepEqual(game.board, previous.board);
});

test('pure-2 race accepts a board sum above 1022, unlike the exact 2044 race', () => {
  const pure2 = new TournamentGame(PROJECT_BY_ORDER[4], { seed: 'pure2-threshold' });
  pure2.board = [1024, 0, 0, 0, 0, 0, 0, 0, 0];
  assert.equal(pure2.targetReached(), true);
  pure2.board[0] = 512;
  assert.equal(pure2.targetReached(), false);
  const grand = new TournamentGame(PROJECT_BY_ORDER[5], { seed: 'grand-exact' });
  grand.board = [2048, 0, 0, 0, 0, 0, 0, 0, 0];
  assert.equal(grand.targetReached(), false);
});
