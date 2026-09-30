import { DIRECTIONS, moveBoard, TournamentGame, WALL } from './engine.js';
import { nextRandom, seed32 } from './randomStreams.js';

const cellKey = (row, col) => `${row},${col}`;

// Keep the original 4×4 axes fixed. Only the visible bounding rectangle is
// rebased; its top-left corner is never used when drawing a quake ticket.
export function shiftAftershock(board, rows, cols, originRow, originCol, axis, line, step) {
  const cells = [];
  for (let index = 0; index < board.length; index += 1) {
    if (board[index] === WALL) continue;
    let row = originRow + Math.floor(index / cols);
    let col = originCol + index % cols;
    const from = { row, col, index };
    if (axis === 'row' && row === line) col += step;
    if (axis === 'col' && col === line) row += step;
    cells.push({ from, row, col, value: board[index] });
  }
  const minRow = Math.min(...cells.map(cell => cell.row));
  const maxRow = Math.max(...cells.map(cell => cell.row));
  const minCol = Math.min(...cells.map(cell => cell.col));
  const maxCol = Math.max(...cells.map(cell => cell.col));
  const nextRows = maxRow - minRow + 1;
  const nextCols = maxCol - minCol + 1;
  const nextBoard = Array(nextRows * nextCols).fill(WALL);
  const occupied = new Set();
  for (const cell of cells) {
    const key = cellKey(cell.row, cell.col);
    if (occupied.has(key)) throw new Error('余震格子坐标发生重叠。');
    occupied.add(key);
    cell.to = (cell.row - minRow) * nextCols + cell.col - minCol;
    nextBoard[cell.to] = cell.value;
  }
  return {
    board: nextBoard, rows: nextRows, cols: nextCols,
    originRow: minRow, originCol: minCol,
    cells: cells.map(({ from, row, col, to, value }) => ({
      from: from.index, to, fromRow: from.row, fromCol: from.col,
      toRow: row, toCol: col, value,
    })),
  };
}

export class AftershockGame extends TournamentGame {
  reset(increment = true) {
    super.reset(increment);
    this.originRow = 0;
    this.originCol = 0;
    this.quakeState = seed32(`${this.seed}:aftershock:${this.restartCount}`);
    this.lastQuake = null;
    return this.snapshot();
  }

  quakeDraw() {
    this.quakeState = nextRandom(this.quakeState);
    return this.quakeState / 0x100000000;
  }

  spawnAfterMove(moved) {
    this.lastQuake = null;
    if (moved.movements.some(item => item.merged && item.value * 2 >= 256)) {
      const axis = this.quakeDraw() < .5 ? 'row' : 'col';
      const line = Math.floor(this.quakeDraw() * 4);
      const step = this.quakeDraw() < .5 ? -1 : 1;
      const before = this.board.slice();
      const { rows, cols, originRow, originCol } = this;
      const changed = shiftAftershock(before, rows, cols, originRow, originCol, axis, line, step);
      this.board = changed.board;
      this.rows = changed.rows;
      this.cols = changed.cols;
      this.originRow = changed.originRow;
      this.originCol = changed.originCol;
      this.lastQuake = {
        axis, line, step, before, rows, cols, originRow, originCol,
        cells: changed.cells,
      };
    }
    return super.spawnAfterMove(moved);
  }

  // Older spectators understand the final irregular board but not a changing
  // coordinate frame. A distinct event prevents them animating old indices.
  moveTransitionExtras() { return this.lastQuake ? { kind: 'reshape', quake: this.lastQuake } : {}; }

  settleOutcome() {
    // Empty cells can be isolated, so a non-full board is not proof of a move.
    if (!DIRECTIONS.some(direction => moveBoard(this.board, this, direction).changed)) {
      this.finished = true;
      this.outcome = 'no_moves';
      this.finishedAt ??= performance.now();
    }
  }

  snapshot() { return { ...super.snapshot(), originRow: this.originRow ?? 0, originCol: this.originCol ?? 0 }; }
}

export class LookBackGame extends TournamentGame {
  reset(increment = true) {
    const snapshot = super.reset(increment);
    this.lookBackState = seed32(`${this.seed}:lookback:${this.restartCount}`);
    return snapshot;
  }

  move(direction) {
    if (this.finished || !DIRECTIONS.includes(direction)) return { changed: false, snapshot: this.snapshot() };
    // An ineffective input neither draws a chance ticket nor changes history.
    if (!moveBoard(this.board, this, direction).changed) return { changed: false, snapshot: this.snapshot() };
    this.lookBackState = nextRandom(this.lookBackState);
    if (this.lookBackState / 0x100000000 < .05 && this.history.length) {
      const before = this.board.slice();
      const previous = this.history.pop();
      Object.assign(this, previous);
      const spawns = [];
      for (let count = 0; count < 2 && this.board.includes(0); count += 1) {
        const spawn = this.spawn(this.board);
        if (spawn) spawns.push(spawn);
      }
      this.finished = false;
      this.outcome = null;
      this.finishedAt = null;
      this.revision += 1;
      this.transition = { kind: 'lookback', before, spawns };
      this.settleOutcome();
      return { changed: true, snapshot: this.snapshot() };
    }
    this.history.push(this.historyEntry());
    return super.move(direction);
  }
}
