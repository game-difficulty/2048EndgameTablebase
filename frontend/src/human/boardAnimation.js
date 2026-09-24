import { createBoardFrame } from '../components/boardFrame.js';
import { move } from './engine.js';

// Main-site animation frames use a 4-column logical grid. Empty padding is outside
// the explicit viewport, so 32768 remains an ordinary tile, never a variant wall.
export function paddedBoard(board, cols) {
  const padded = Array(16).fill(0);
  board.forEach((value, index) => { padded[Math.floor(index / cols) * 4 + index % cols] = value; });
  return padded;
}
export function humanBoardFrame(revision, board, rows, cols, transition) {
  const toBoard = paddedBoard(board, cols);
  if (!transition || transition.toBoard.length !== board.length || !board.every((v,i) => v === transition.toBoard[i])) {
    return createBoardFrame({ revision, toBoard });
  }
  const from = transition.fromBoard;
  let metadata = null;
  if (Number.isInteger(transition.direction)) {
    const moved = move(from, rows, cols, transition.direction);
    metadata = { ...moved.metadata, slide_distances: paddedBoard(moved.metadata.slide_distances, cols), pop_positions: paddedBoard(moved.metadata.pop_positions, cols) };
    const spawned = board.findIndex((value, i) => value !== moved.board[i]);
    if (spawned >= 0) metadata.appear_tile = { index: Math.floor(spawned / cols) * 4 + spawned % cols, value: board[spawned] };
  } else if (transition.spawn != null) {
    const i = transition.spawn;
    metadata = { appear_tile: { index: Math.floor(i / cols) * 4 + i % cols, value: board[i] } };
  }
  return createBoardFrame({ revision, kind: 'move', fromBoard: paddedBoard(from, cols), toBoard, metadata });
}
