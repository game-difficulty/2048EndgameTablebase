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
