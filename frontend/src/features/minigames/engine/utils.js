export const POWERUP_KEYS = ['bomb', 'glove', 'twist'];
export const SPAWN_RATE4 = 0.1;

export function clampInteger(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.trunc(numeric) : fallback;
}

export function cloneBoard(board) {
  return (Array.isArray(board) ? board : []).map((row) =>
    Array.isArray(row) ? row.map((value) => clampInteger(value)) : []
  );
}

export function createBoard(rows = 4, cols = 4, fill = 0) {
  return Array.from({ length: rows }, () => new Array(cols).fill(fill));
}

export function boardShape(board) {
  const rows = Array.isArray(board) ? board.length : 0;
  const cols = rows > 0 && Array.isArray(board[0]) ? board[0].length : 0;
  return { rows, cols };
}

export function flattenBoard(board) {
  return cloneBoard(board).flat();
}

export function boardFromFlat(values, rows = 4, cols = 4) {
  const flat = Array.isArray(values) ? values : [];
  return Array.from({ length: rows }, (_row, row) =>
    Array.from({ length: cols }, (_col, col) => clampInteger(flat[row * cols + col]))
  );
}

export function boardsEqual(left, right) {
  const leftShape = boardShape(left);
  const rightShape = boardShape(right);
  if (leftShape.rows !== rightShape.rows || leftShape.cols !== rightShape.cols) {
    return false;
  }
  for (let row = 0; row < leftShape.rows; row += 1) {
    for (let col = 0; col < leftShape.cols; col += 1) {
      if (clampInteger(left[row][col]) !== clampInteger(right[row][col])) {
        return false;
      }
    }
  }
  return true;
}

export function positiveMax(board) {
  return flattenBoard(board).reduce((maxValue, value) => Math.max(maxValue, value > 0 ? value : 0), 0);
}

export function countCells(board, predicate) {
  return flattenBoard(board).reduce((count, value, index) => count + (predicate(value, index) ? 1 : 0), 0);
}

export function emptyPositions(board) {
  const { rows, cols } = boardShape(board);
  const positions = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      if (clampInteger(board[row][col]) === 0) {
        positions.push([row, col]);
      }
    }
  }
  return positions;
}

const randomIndex = (count, rng = null) => {
  if (rng && typeof rng.chooseIndex === 'function') return rng.chooseIndex(count);
  if (rng && typeof rng.randomIndex === 'function') return rng.randomIndex(count);
  if (rng && typeof rng.nextFloat === 'function') return Math.floor(rng.nextFloat() * count);
  if (rng && typeof rng.random === 'function') return Math.floor(rng.random() * count);
  return Math.floor(Math.random() * count);
};

export function randomChoice(items, rng = null) {
  if (!Array.isArray(items) || items.length === 0) {
    return null;
  }
  return items[randomIndex(items.length, rng)];
}

export function randomSample(items, count, rng = null) {
  const pool = Array.isArray(items) ? items.slice() : [];
  const result = [];
  while (pool.length > 0 && result.length < count) {
    const index = randomIndex(pool.length, rng);
    result.push(pool.splice(index, 1)[0]);
  }
  return result;
}

export function exponentToTileValue(exponent) {
  const numeric = clampInteger(exponent);
  return numeric > 0 ? 2 ** numeric : 0;
}

export function formatCompactTile(value) {
  const numeric = Math.round(Number(value) || 0);
  if (!numeric) return '';
  return numeric < 1000 ? String(numeric) : `${Math.floor(numeric / 1000)}k`;
}

export function normalizeDirection(direction) {
  const normalized = String(direction || '').trim().toLowerCase();
  return ['left', 'right', 'up', 'down'].includes(normalized) ? normalized : '';
}

export function directionToCode(direction) {
  return {
    left: 1,
    right: 2,
    up: 3,
    down: 4,
  }[normalizeDirection(direction)] || 0;
}

export function codeToDirection(code) {
  return {
    1: 'left',
    2: 'right',
    3: 'up',
    4: 'down',
  }[clampInteger(code)] || '';
}

export function sanitizeExtra(value) {
  if (value == null) return value;
  return JSON.parse(JSON.stringify(value));
}
