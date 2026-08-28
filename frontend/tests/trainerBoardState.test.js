import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildTrainerBoardEdit,
  normalizeTrainerBoardHex,
  transformTrainerBoard,
  previousDistinctTrainerHistoryState,
} from '../src/features/trainer/engine/trainerBoardState.js';

test('normalizes editable trainer board hex without losing leading zeroes', () => {
  assert.equal(normalizeTrainerBoardHex('0x123a'), '000000000000123a');
  assert.equal(normalizeTrainerBoardHex('xyz'), null);
  assert.equal(normalizeTrainerBoardHex('1'.repeat(17)), null);
});

test('trainer board transforms operate on visual rows and columns', () => {
  const board = Array.from({ length: 16 }, (_value, index) => index + 1);
  assert.deepEqual(transformTrainerBoard(board, 'UD').slice(0, 4), [13, 14, 15, 16]);
  assert.deepEqual(transformTrainerBoard(board, 'LR').slice(0, 4), [4, 3, 2, 1]);
  assert.deepEqual(transformTrainerBoard(board, 'R90').slice(0, 4), [13, 9, 5, 1]);
  assert.deepEqual(transformTrainerBoard(board, 'L90').slice(0, 4), [4, 8, 12, 16]);
  assert.equal(transformTrainerBoard(board, 'bad'), null);
});

test('builds palette edits locally without mutating the current board', () => {
  const current = Array(16).fill(0);
  const edit = buildTrainerBoardEdit(current, 1, 2, 8);

  assert.equal(current[6], 0);
  assert.equal(edit.board[6], 8);
  assert.equal(edit.boardHex, '0000003000000000');
});

test('undo prediction skips duplicate current states and keeps move alignment', () => {
  const state = previousDistinctTrainerHistoryState(
    ['1', '2', '3', '3'],
    [null, 'left', 'spawn', 'spawn'],
    '3',
  );

  assert.deepEqual(state, {
    boardHex: '0000000000000002',
    history: ['0000000000000001', '0000000000000002'],
    moves: [null, 'left'],
    lastMove: 'left',
  });
});

test('undo prediction returns null when no distinct previous board exists', () => {
  assert.equal(previousDistinctTrainerHistoryState(['1'], [null], '1'), null);
  assert.equal(previousDistinctTrainerHistoryState(['2', '2'], [null, 'spawn'], '2'), null);
});
