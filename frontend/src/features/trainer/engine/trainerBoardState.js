import { boardHex, encodeBoard } from '../../replay/engine/replayTransition.js';

export function normalizeTrainerBoardHex(value) {
  const normalized = String(value || '').trim().replace(/^0x/i, '').toLowerCase();
  if (!/^[0-9a-f]{1,16}$/.test(normalized)) return null;
  return normalized.padStart(16, '0');
}

export function buildTrainerBoardEdit(board, row, col, nextValue) {
  const normalizedRow = Number(row);
  const normalizedCol = Number(col);
  if (
    !Array.isArray(board)
    || board.length < 16
    || !Number.isInteger(normalizedRow)
    || !Number.isInteger(normalizedCol)
    || normalizedRow < 0
    || normalizedRow > 3
    || normalizedCol < 0
    || normalizedCol > 3
  ) {
    return null;
  }

  const nextBoard = board.slice(0, 16);
  nextBoard[normalizedRow * 4 + normalizedCol] = Number(nextValue) || 0;
  return {
    board: nextBoard,
    boardHex: boardHex(encodeBoard(nextBoard)),
  };
}

export function transformTrainerBoard(board, type) {
  if (!Array.isArray(board) || board.length < 16) return null;
  const normalizedType = String(type || '').toUpperCase();
  if (!['UD', 'LR', 'RL', 'R90', 'L90'].includes(normalizedType)) return null;
  const transformed = new Array(16).fill(0);
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      let sourceRow = row;
      let sourceCol = col;
      if (normalizedType === 'UD') sourceRow = 3 - row;
      else if (normalizedType === 'LR' || normalizedType === 'RL') sourceCol = 3 - col;
      else if (normalizedType === 'R90') {
        sourceRow = 3 - col;
        sourceCol = row;
      } else {
        sourceRow = col;
        sourceCol = 3 - row;
      }
      transformed[row * 4 + col] = Number(board[sourceRow * 4 + sourceCol]) || 0;
    }
  }
  return transformed;
}

export function previousDistinctTrainerHistoryState(history, moves, currentBoardHex) {
  if (!Array.isArray(history) || history.length <= 1) return null;
  const normalizedHistory = history.map(normalizeTrainerBoardHex);
  if (normalizedHistory.some((entry) => !entry)) return null;

  const current = normalizeTrainerBoardHex(currentBoardHex);
  let index = normalizedHistory.length - 2;
  while (index > 0 && normalizedHistory[index] === current) index -= 1;
  const target = normalizedHistory[index];
  if (!target || target === current) return null;

  const normalizedMoves = Array.isArray(moves) ? moves : [];
  return {
    boardHex: target,
    history: normalizedHistory.slice(0, index + 1),
    moves: normalizedMoves.slice(0, index + 1),
    lastMove: normalizedMoves[index] ?? null,
  };
}
