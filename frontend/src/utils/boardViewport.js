const LOGICAL_BOARD_SIZE = 4;
const LOGICAL_CELL_COUNT = LOGICAL_BOARD_SIZE * LOGICAL_BOARD_SIZE;
const DEFAULT_PADDING_RATIO = 0.025;
const DEFAULT_GAP_RATIO = 0.025;

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

  // Only complete outer wall borders are cropped. Internal walls stay visible.
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

export function boardViewportSignature(viewport) {
  return [
    Number(viewport?.rowStart || 0),
    Number(viewport?.colStart || 0),
    Number(viewport?.rows || LOGICAL_BOARD_SIZE),
    Number(viewport?.cols || LOGICAL_BOARD_SIZE),
  ].join(':');
}

export function logicalIndexToVisual(index, viewport) {
  const logicalIndex = Number(index);
  if (!Number.isInteger(logicalIndex) || logicalIndex < 0 || logicalIndex >= LOGICAL_CELL_COUNT) {
    return null;
  }
  const row = Math.floor(logicalIndex / LOGICAL_BOARD_SIZE) - Number(viewport?.rowStart || 0);
  const col = logicalIndex % LOGICAL_BOARD_SIZE - Number(viewport?.colStart || 0);
  if (row < 0 || row >= Number(viewport?.rows || LOGICAL_BOARD_SIZE)
    || col < 0 || col >= Number(viewport?.cols || LOGICAL_BOARD_SIZE)) {
    return null;
  }
  return { row, col };
}

export function createBoardViewportLayout(
  viewport,
  { paddingRatio = DEFAULT_PADDING_RATIO, gapRatio = DEFAULT_GAP_RATIO } = {},
) {
  const rows = Math.max(1, Number(viewport?.rows || LOGICAL_BOARD_SIZE));
  const cols = Math.max(1, Number(viewport?.cols || LOGICAL_BOARD_SIZE));
  const padding = Math.max(0, Number(paddingRatio) || 0);
  const gap = Math.max(0, Number(gapRatio) || 0);
  const tileByWidth = (1 - padding * 2 - gap * (cols - 1)) / cols;
  const tileByHeight = (1 - padding * 2 - gap * (rows - 1)) / rows;
  const tile = Math.max(0, Math.min(tileByWidth, tileByHeight));
  const width = padding * 2 + tile * cols + gap * (cols - 1);
  const height = padding * 2 + tile * rows + gap * (rows - 1);
  const asPercent = (value, total = 1) => (total > 0 ? value / total * 100 : 0);

  return {
    widthPercent: width * 100,
    heightPercent: height * 100,
    leftPercent: (1 - width) * 50,
    topPercent: (1 - height) * 50,
    paddingXPercent: asPercent(padding, width),
    paddingYPercent: asPercent(padding, height),
    gapXPercent: asPercent(gap, width),
    gapYPercent: asPercent(gap, height),
    tileWidthPercent: asPercent(tile, width),
    tileHeightPercent: asPercent(tile, height),
  };
}
