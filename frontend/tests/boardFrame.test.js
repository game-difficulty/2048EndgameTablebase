import assert from 'node:assert/strict';
import test from 'node:test';

import {
  boardFrameRenderMode,
  createBoardFrame,
  createSnapshotBoardFrame,
  createTransitionBoardFrame,
} from '../src/components/boardFrame.js';

const emptyBoard = () => new Array(16).fill(0);
const moveMetadata = () => ({
  direction: 'left',
  slide_distances: [1, ...new Array(15).fill(0)],
  pop_positions: new Array(16).fill(0),
  appear_tile: { index: 15, value: 2 },
});

test('a continuous move animates from the already settled visual board', () => {
  const fromBoard = emptyBoard();
  fromBoard[1] = 2;
  const toBoard = emptyBoard();
  toBoard[0] = 2;
  toBoard[15] = 2;
  const frame = createBoardFrame({
    revision: 1,
    kind: 'move',
    fromBoard,
    toBoard,
    metadata: moveMetadata(),
  });
  assert.equal(boardFrameRenderMode(fromBoard, frame), 'animate');
});

test('a discontinuous move snaps to its target instead of redrawing its source', () => {
  const settledBoard = emptyBoard();
  settledBoard[5] = 4;
  const fromBoard = emptyBoard();
  fromBoard[1] = 2;
  const toBoard = emptyBoard();
  toBoard[0] = 2;
  const frame = createBoardFrame({
    revision: 2,
    kind: 'move',
    fromBoard,
    toBoard,
    metadata: moveMetadata(),
  });
  assert.equal(boardFrameRenderMode(settledBoard, frame), 'snapshot');
  assert.deepEqual(frame.toBoard, toBoard);
});

test('snapshots never enter the animation path', () => {
  const board = emptyBoard();
  board[3] = 8;
  const frame = createSnapshotBoardFrame(3, board);
  assert.equal(boardFrameRenderMode(board, frame), 'snapshot');
});

test('transition frames clone their boards and reject incomplete metadata', () => {
  const fromBoard = emptyBoard();
  const toBoard = emptyBoard();
  fromBoard[1] = 2;
  toBoard[0] = 2;
  const transition = {
    fromBoard,
    toBoard,
    metadata: { direction: 'left' },
  };
  const frame = createTransitionBoardFrame(4, transition, toBoard);
  fromBoard[1] = 16;
  toBoard[0] = 16;
  assert.equal(frame.kind, 'snapshot');
  assert.equal(frame.toBoard[0], 2);
});
