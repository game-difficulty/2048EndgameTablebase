import test from 'node:test';
import assert from 'node:assert/strict';
import { PROJECT_BY_ORDER } from '../src/projects/catalog.js';
import { moveHeavyBoard, PracticeScoreVariantGame } from '../src/projects/practiceScoreVariants.js';

test('all three score variants register as separate 4×4 practice projects', () => {
  for (const order of [16, 17, 18]) {
    const project = PROJECT_BY_ORDER[order];
    assert.equal(project.rows, 4);
    assert.equal(project.cols, 4);
    assert.equal(project.practicePath, `/practice/${order}`);
  }
});

test('full load fails as soon as the spawn creates a thirteenth tile', () => {
  const game = new PracticeScoreVariantGame(PROJECT_BY_ORDER[16], { seed: 'capacity' });
  game.board = [2,4,8,16, 32,64,128,256, 512,1024,2048,0, 2,0,0,0];
  const { snapshot } = game.move('right');
  assert.equal(snapshot.board.filter(value => value > 0).length, 13);
  assert.equal(snapshot.finished, true);
  assert.equal(snapshot.outcome, 'tile_limit');
});

test('heavy tiles block the prohibited axis but other tiles still slide and merge', () => {
  const project = PROJECT_BY_ORDER[17];
  const board = [256,0,0,0, 2,0,0,0, 512,0,0,0, 1024,0,0,0];
  assert.equal(moveHeavyBoard(board, project, 'up').board[0], 256);
  assert.equal(moveHeavyBoard(board, project, 'down').board[8], 512);
  assert.equal(moveHeavyBoard(board, project, 'right').board[8], 512);
  assert.equal(moveHeavyBoard(board, project, 'left').board[12], 1024);
  assert.equal(moveHeavyBoard(board, project, 'right').board[3], 256);

  const merge = moveHeavyBoard([2,2,0,256, ...Array(12).fill(0)], project, 'right');
  assert.deepEqual(merge.board.slice(0, 4), [0,0,4,256]);
  assert.equal(merge.score, 4);
});

test('heavy tile death check does not mistake an unreachable empty cell for a legal move', () => {
  const game = new PracticeScoreVariantGame(PROJECT_BY_ORDER[17], { seed: 'heavy-death' });
  game.board = [1024,0,0,0, ...Array(12).fill(0)];
  game.settleOutcome();
  assert.equal(game.outcome, 'no_moves');
});

test('fission countdown is assigned at merge and hidden within 16–40 effective moves', () => {
  const game = new PracticeScoreVariantGame(PROJECT_BY_ORDER[18], { seed: 'fission-timer' });
  game.board = [512,512,0,0, ...Array(12).fill(0)];
  const { snapshot } = game.move('left');
  const timer = game.fissionTimers.get(0);
  assert.ok(timer.remaining >= 16 && timer.remaining <= 40);
  assert.equal(snapshot.score, 1024);
  assert.equal(snapshot.transition.fission, undefined);
});

test('one due fission replaces the turn spawn and exposes its animation event', () => {
  const game = new PracticeScoreVariantGame(PROJECT_BY_ORDER[18], { seed: 'fission-due' });
  game.board = [1024,0,0,0, 0,2,0,0, ...Array(8).fill(0)];
  game.fissionTimers = new Map([[0, { sequence: 0, remaining: 1 }]]);
  game.fissionSequence = 1;
  const { snapshot } = game.move('left');
  assert.equal(snapshot.transition.spawn, null);
  assert.deepEqual(snapshot.transition.fission, {
    index: 0, spawnedIndex: snapshot.transition.fission.spawnedIndex,
    parent: 1024, value: 512,
  });
  assert.equal(snapshot.board[0], 512);
  assert.equal(snapshot.board[snapshot.transition.fission.spawnedIndex], 512);
  assert.equal(snapshot.board.filter(value => value > 0).length, 3);
});

test('simultaneously due blocks split in creation order, at most once per move', () => {
  const game = new PracticeScoreVariantGame(PROJECT_BY_ORDER[18], { seed: 'fission-queue' });
  game.board = [1024,0,0,0, 1024,0,0,0, 0,2,0,0, 0,0,0,0];
  game.fissionTimers = new Map([
    [0, { sequence: 0, remaining: 1 }],
    [4, { sequence: 1, remaining: 1 }],
  ]);
  game.fissionSequence = 2;
  const first = game.move('right').snapshot;
  assert.equal(first.transition.fission.index, 3);
  assert.equal(first.board.filter(value => value === 1024).length, 1);
  assert.equal(game.fissionTimers.get(7).remaining, 0);
  const second = game.move('left').snapshot;
  assert.equal(second.transition.fission.index, 4);
  assert.equal(second.board[4], 512);
  assert.equal(game.fissionTimers.has(4), false);
});
