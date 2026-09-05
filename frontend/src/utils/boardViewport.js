const LOGICAL_BOARD_SIZE = 4;
const LOGICAL_CELL_COUNT = LOGICAL_BOARD_SIZE * LOGICAL_BOARD_SIZE;

const fullBoardViewport = () => ({
  rows: LOGICAL_BOARD_SIZE,
  cols: LOGICAL_BOARD_SIZE,
  rowStart: 0,
  colStart: 0,
  visibleIndices: Array.from({ length: LOGICAL_CELL_COUNT }, (_, index) => index),
});

export function createBoardViewport(board, isVariant) {
  if (!isVariant || !Array.isArray(board) || board.length < LOGICAL_CELL_COUNT) {
    return fullBoardViewport();
  }

  const isWall = (row, col) => Number(board[row * LOGICAL_BOARD_SIZE + col]) === 32768;
  const visibleRows = [];
  const visibleCols = [];

  for (let row = 0; row < LOGICAL_BOARD_SIZE; row += 1) {
    if (Array.from({ length: LOGICAL_BOARD_SIZE }, (_, col) => col).some((col) => !isWall(row, col))) {
      visibleRows.push(row);
    }
  }
  for (let col = 0; col < LOGICAL_BOARD_SIZE; col += 1) {
    if (Array.from({ length: LOGICAL_BOARD_SIZE }, (_, row) => row).some((row) => !isWall(row, col))) {
      visibleCols.push(col);
    }
  }

  if (!visibleRows.length || !visibleCols.length) {
    return fullBoardViewport();
  }

  const rowStart = visibleRows[0];
  const rowEnd = visibleRows[visibleRows.length - 1];
  const colStart = visibleCols[0];
  const colEnd = visibleCols[visibleCols.length - 1];
  const rows = rowEnd - rowStart + 1;
  const cols = colEnd - colStart + 1;

  // Crop only complete wall borders. Irregular/internal walls remain visible.
  const outsideBoundsAreWalls = Array.from({ length: LOGICAL_CELL_COUNT }, (_, index) => index)
    .every((index) => {
      const row = Math.floor(index / LOGICAL_BOARD_SIZE);
      const col = index % LOGICAL_BOARD_SIZE;
      const outside = row < rowStart || row > rowEnd || col < colStart || col > colEnd;
      return !outside || isWall(row, col);
    });
  if (!outsideBoundsAreWalls) {
    return fullBoardViewport();
  }

  const visibleIndices = [];
  for (let row = rowStart; row <= rowEnd; row += 1) {
    for (let col = colStart; col <= colEnd; col += 1) {
      visibleIndices.push(row * LOGICAL_BOARD_SIZE + col);
    }
  }

  return { rows, cols, rowStart, colStart, visibleIndices };
}

export function logicalIndexToVisual(index, viewport) {
  const logicalIndex = Number(index);
  if (!Number.isInteger(logicalIndex) || logicalIndex < 0 || logicalIndex >= LOGICAL_CELL_COUNT) {
    return null;
  }
  const row = Math.floor(logicalIndex / LOGICAL_BOARD_SIZE) - Number(viewport?.rowStart || 0);
  const col = logicalIndex % LOGICAL_BOARD_SIZE - Number(viewport?.colStart || 0);
  if (row < 0 || row >= Number(viewport?.rows || 4) || col < 0 || col >= Number(viewport?.cols || 4)) {
    return null;
  }
  return { row, col };
}
