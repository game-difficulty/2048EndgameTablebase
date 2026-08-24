import assert from 'node:assert/strict';
import test from 'node:test';

import { buildOptimisticTesterLastStep } from '../src/features/tester/engine/testerOptimisticFeedback.js';

test('optimistic tester feedback exposes the completed step without a server round trip', () => {
  const board = [2, 4, ...new Array(14).fill(0)];
  const lastStep = buildOptimisticTesterLastStep({
    board,
    results: { right: 0.9, left: 0.72, down: null, up: null },
    dtype: 'float64',
    direction: 'right',
    goodnessOfFit: 0.8,
  });

  assert.equal(lastStep.best_move, 'right');
  assert.equal(lastStep.evaluation, 'Perfect!');
  assert.equal(lastStep.loss, 0);
  assert.equal(lastStep.goodness_of_fit, 0.8);
  assert.deepEqual(lastStep.board, board);
  assert.notEqual(lastStep.board, board);
});

test('optimistic tester feedback matches the server ratio evaluation', () => {
  const lastStep = buildOptimisticTesterLastStep({
    board: new Array(16).fill(0),
    results: { right: 1, left: 0.8 },
    dtype: 'float64',
    direction: 'left',
    goodnessOfFit: 0.5,
  });

  assert.equal(lastStep.best_move, 'right');
  assert.equal(lastStep.evaluation, 'Blunder!');
  assert.ok(Math.abs(lastStep.loss - 0.2) < 1e-12);
  assert.ok(Math.abs(lastStep.goodness_of_fit - 0.4) < 1e-12);
});
