import { evilSpawn as wasmEvilSpawn } from './evilSpawn.js';
import { nextRandom, randomStream, seed32, ticketFloat } from './randomStreams.js';

export const WALL = -1;
export const ISLAND = -3;
export const DIRECTIONS = Object.freeze(['up', 'right', 'down', 'left']);

const clone = value => JSON.parse(JSON.stringify(value));
const boardSum = board => board.reduce((sum, value) => sum + Math.max(0, value), 0);

function paths(project, direction) {
  const { rows, cols, mirrorPortals } = project;
  if (mirrorPortals) {
    const order = ['left', 'up'].includes(direction) ? [2, 3, 0, 1] : [1, 0, 3, 2];
    return ['left', 'right'].includes(direction)
      ? Array.from({ length: rows }, (_, row) => order.map(col => row * cols + col))
      : Array.from({ length: cols }, (_, col) => order.map(row => row * cols + col));
  }
  if (direction === 'left') return Array.from({ length: rows }, (_, row) => Array.from({ length: cols }, (_v, col) => row * cols + col));
  if (direction === 'right') return Array.from({ length: rows }, (_, row) => Array.from({ length: cols }, (_v, offset) => row * cols + cols - 1 - offset));
  if (direction === 'up') return Array.from({ length: cols }, (_, col) => Array.from({ length: rows }, (_v, row) => row * cols + col));
  return Array.from({ length: cols }, (_, col) => Array.from({ length: rows }, (_v, offset) => (rows - 1 - offset) * cols + col));
}

function largestRectangleArea(heights) {
  const stack = [];
  let largest = 0;
  const extended = heights.concat(0);
  for (let index = 0; index < extended.length; index += 1) {
    while (stack.length && extended[index] < extended[stack.at(-1)]) {
      const height = extended[stack.pop()];
      const width = stack.length ? index - stack.at(-1) - 1 : index;
      largest = Math.max(largest, height * width);
    }
    stack.push(index);
  }
  return largest;
}

export function maxPlayableRectangle(board, rows, cols) {
  const heights = Array(cols).fill(0);
  let largest = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      heights[col] = board[row * cols + col] === 0 ? heights[col] + 1 : 0;
    }
    largest = Math.max(largest, largestRectangleArea(heights));
  }
  return largest;
}

function connectedShape(size, count, random) {
  const board = Array(size * size).fill(WALL);
  const key = (row, col) => `${row},${col}`;
  const visited = new Set(), remaining = new Set();
  let stack = [[Math.floor(random() * size), Math.floor(random() * size)]];
  while (visited.size < count) {
    if (!stack.length) {
      const choices = Array.from(remaining);
      const chosen = choices[Math.floor(random() * choices.length)];
      remaining.delete(chosen);
      stack = [chosen.split(',').map(Number)];
    }
    const [row, col] = stack.pop();
    if (row < 0 || row >= size || col < 0 || col >= size || visited.has(key(row,col))) continue;
    visited.add(key(row,col));
    board[row * size + col] = 0;
    const neighbors = [[row - 1,col],[row + 1,col],[row,col - 1],[row,col + 1]];
    const randomized = [];
    while (neighbors.length) randomized.push(neighbors.splice(Math.floor(random() * neighbors.length), 1)[0]);
    for (const [nextRow,nextCol] of randomized) {
      if (nextRow < 0 || nextRow >= size || nextCol < 0 || nextCol >= size || visited.has(key(nextRow,nextCol))) continue;
      if (random() > .5) stack.push([nextRow,nextCol]);
      else remaining.add(key(nextRow,nextCol));
    }
  }
  return board;
}

function cropAndOrientShape(board, size) {
  const playable = board.map((value, index) => value === 0 ? index : -1).filter(index => index >= 0);
  const sourceRows = playable.map(index => Math.floor(index / size));
  const sourceCols = playable.map(index => index % size);
  const minRow = Math.min(...sourceRows), maxRow = Math.max(...sourceRows);
  const minCol = Math.min(...sourceCols), maxCol = Math.max(...sourceCols);
  const rows = maxRow - minRow + 1, cols = maxCol - minCol + 1;
  const cropped = Array.from({ length: rows * cols }, (_value, index) => {
    const row = Math.floor(index / cols) + minRow;
    const col = index % cols + minCol;
    return board[row * size + col];
  });
  if (cols >= rows) return { board: cropped, rows, cols };
  return {
    board: Array.from({ length: rows * cols }, (_value, index) => {
      const row = Math.floor(index / rows), col = index % rows;
      return cropped[col * cols + row];
    }),
    rows: cols,
    cols: rows,
  };
}

export function createHardShape(random, playableCells = 12, generationSize = 6) {
  for (let attempt = 0; attempt < 4096; attempt += 1) {
    const board = connectedShape(generationSize, playableCells, random);
    const rectangle = maxPlayableRectangle(board, generationSize, generationSize);
    if (rectangle >= 4 && rectangle <= 8) return cropAndOrientShape(board, generationSize);
  }
  throw new Error('Could not generate a valid hard shape.');
}

function segments(path, board) {
  const groups = [];
  let current = [];
  for (const index of path) {
    if (board[index] === WALL) {
      if (current.length) groups.push(current);
      current = [];
    } else current.push(index);
  }
  if (current.length) groups.push(current);
  return groups;
}

export function moveBoard(board, project, direction) {
  const after = board.map(value => value === WALL ? WALL : 0);
  const movements = [];
  let score = 0;
  for (const path of paths(project, direction)) {
    for (const segment of segments(path, board)) {
      const entries = segment.filter(index => board[index] > 0 || board[index] === ISLAND).map(index => ({ index, value: board[index] }));
      let source = 0;
      let target = 0;
      while (source < entries.length) {
        const entry = entries[source];
        if (entry.value === ISLAND) {
          let groupEnd = source + 1;
          while (groupEnd < entries.length && entries[groupEnd].value === ISLAND) groupEnd += 1;
          const merged = groupEnd - source > 1;
          const destination = segment[target++];
          after[destination] = ISLAND;
          for (let index = source; index < groupEnd; index += 1) {
            movements.push({ from: entries[index].index, to: destination, value: ISLAND, merged });
          }
          source = groupEnd;
          continue;
        }
        const shouldMerge = source + 1 < entries.length
          && entries[source + 1].value === entry.value
          && entry.value !== project.unmergeable;
        const destination = segment[target++];
        if (shouldMerge) {
          after[destination] = entry.value * 2;
          score += entry.value * 2;
          movements.push(
            { from: entry.index, to: destination, value: entry.value, merged: true },
            { from: entries[source + 1].index, to: destination, value: entry.value, merged: true },
          );
          source += 2;
        } else {
          after[destination] = entry.value;
          movements.push({ from: entry.index, to: destination, value: entry.value, merged: false });
          source += 1;
        }
      }
    }
  }
  const changed = after.some((value, index) => value !== board[index]);
  return { board: after, score, changed, movements };
}

export function hasMove(board, project) {
  if (board.includes(0)) return true;
  return DIRECTIONS.some(direction => moveBoard(board, project, direction).changed);
}

export function moveBoardWithSeals(board, project, direction, sealedCells) {
  if (!sealedCells?.length) return moveBoard(board, project, direction);
  const sealed = new Set(sealedCells);
  const masked = board.map((value, index) => sealed.has(index) ? WALL : value);
  const moved = moveBoard(masked, project, direction);
  const restored = moved.board.map((value, index) => sealed.has(index) ? board[index] : value);
  return {
    ...moved,
    board: restored,
    changed: restored.some((value, index) => value !== board[index]),
  };
}

export function hasMoveWithSeals(board, project, sealedCells) {
  return DIRECTIONS.some(direction => moveBoardWithSeals(board, project, direction, sealedCells).changed);
}

// On a 4x4 board, each of these three-cell diagonals cuts a corner's three
// playable cells off from the other ten. The four sets are rotations of
// a3/b2/c1 (indices 8/5/2).
const CORNER_CUTOFF_SEALS = [
  [2, 5, 8],
  [1, 6, 11],
  [4, 9, 14],
  [7, 10, 13],
];

export function cutsOffCorner(sealedCells, rows, cols) {
  if (rows !== 4 || cols !== 4 || sealedCells.length < 3) return false;
  const sealed = new Set(sealedCells);
  return CORNER_CUTOFF_SEALS.some(pattern => pattern.every(index => sealed.has(index)));
}

export class TournamentGame {
  constructor(project, { seed = `${Date.now()}-${Math.random()}`, side = 'solo', evilSpawn = wasmEvilSpawn } = {}) {
    this.project = project;
    this.seed = String(seed);
    this.side = side;
    this.randomState = seed32(`${this.seed}:spawn`);
    this.evilSpawn = evilSpawn;
    this.revision = 0;
    this.restartCount = 0;
    this.history = [];
    this.rows = Number(project.rows || 4);
    this.cols = Number(project.cols || 4);
    this.reset(false);
  }

  nextSpawnTicket() {
    this.randomState = nextRandom(this.randomState);
    return this.randomState;
  }

  sealRandom() {
    this.sealState = nextRandom(this.sealState);
    return this.sealState / 0x100000000;
  }

  emptyBoard() {
    let board;
    if (this.project.shapeShifter) {
      const shape = createHardShape(
        randomStream(this.seed, `shape:${this.restartCount}`),
        this.project.playableCells || 12,
        this.project.shapeGenerationSize || 6,
      );
      board = shape.board;
      this.rows = shape.rows;
      this.cols = shape.cols;
    } else {
      this.rows = Number(this.project.rows || 4);
      this.cols = Number(this.project.cols || 4);
      board = Array(this.rows * this.cols).fill(0);
    }
    this.dice = null;
    this.wallIndex = null;
    if (this.project.diceWall) {
      const diceRandom = randomStream(this.seed, `dice:${this.side}:${this.restartCount}`);
      this.dice = Math.floor(diceRandom() * 6) + 1;
      const { rows, cols } = this;
      const corners = [0, cols - 1, (rows - 1) * cols, rows * cols - 1];
      const edges = board.map((_v, index) => index).filter(index => !corners.includes(index)
        && (index < cols || index >= (rows - 1) * cols || [0, cols - 1].includes(index % cols)));
      const candidates = this.dice <= 3 ? corners : this.dice <= 5 ? edges : [Math.floor(rows / 2) * cols + Math.floor(cols / 2)];
      this.wallIndex = candidates[Math.floor(diceRandom() * candidates.length)];
      board[this.wallIndex] = WALL;
    }
    return board;
  }

  randomSpawn(board, ticket = this.nextSpawnTicket()) {
    const sealed = new Set(this.sealedCells);
    const empty = board.map((_value, index) => index).filter(index => board[index] === 0 && !sealed.has(index));
    if (!empty.length) return null;
    const index = empty[Math.floor(ticketFloat(ticket, 'position') * empty.length)];
    const value = ticketFloat(ticket, 'value') < (this.project.spawn4Rate ?? .1) ? 4 : 2;
    board[index] = value;
    return { index, value };
  }

  spawn(board) {
    const ticket = this.nextSpawnTicket();
    if (this.project.isolatedIsland) {
      const islandChance = .05 - board.filter(value => value === ISLAND).length * .02;
      if (ticketFloat(ticket, 'special') < islandChance) {
        const empty = board.map((_value, index) => index).filter(index => board[index] === 0);
        if (!empty.length) return null;
        const index = empty[Math.floor(ticketFloat(ticket, 'position') * empty.length)];
        board[index] = ISLAND;
        return { index, value: ISLAND };
      }
    }
    if (!this.project.evilSpawn) return this.randomSpawn(board, ticket);
    if (!board.includes(0)) return null;
    return this.spawnEvil(board, ticket);
  }

  async spawnEvil(board, ticket) {
    const emptyCount = board.filter(value => value === 0).length;
    const boardSum = board.reduce((sum, value) => sum + (value > 0 ? value : 0), 0);
    const adaptiveDepth = emptyCount <= 1 ? 7 : emptyCount <= 2 ? 6 : emptyCount <= 5 ? 5 : 4;
    const depth = boardSum < 120 ? Math.min(adaptiveDepth, 5) : adaptiveDepth;
    try {
      const spawn = await this.evilSpawn(board, depth, ticket);
      board[spawn.index] = spawn.value;
      return spawn;
    } catch (error) {
      this.aiError = error instanceof Error ? error.message : String(error);
      if (this.project.strictEvilSpawn) throw error;
      return this.randomSpawn(board, ticket);
    }
  }

  reset(increment = true) {
    if (increment) this.restartCount += 1;
    this.sealState = seed32(`${this.seed}:seal:${this.restartCount}`);
    this.board = this.emptyBoard();
    this.sealedCells = [];
    this.sealRound = 0;
    const openingSeals = this.project.sealEveryMoves ? this.rotateSeals() : null;
    const first = this.randomSpawn(this.board);
    const second = this.randomSpawn(this.board);
    this.score = 0;
    this.moves = 0;
    this.finished = false;
    this.outcome = null;
    this.history = [];
    this.startedAt = performance.now();
    this.finishedAt = null;
    this.revision += 1;
    this.transition = { kind: increment ? 'restart' : 'initial', spawns: [first, second], seals: openingSeals };
    return this.snapshot();
  }

  elapsed(now = performance.now()) {
    return Math.max(0, (this.finishedAt ?? now) - this.startedAt);
  }

  targetReached() {
    if (this.project.targetSum != null) {
      const sum = boardSum(this.board);
      return this.project.targetAtLeast ? sum >= this.project.targetSum : sum === this.project.targetSum;
    }
    if (this.project.targetTile != null) return this.board.filter(value => value === this.project.targetTile).length >= this.project.targetCount;
    return false;
  }

  settleOutcome() {
    if (this.targetReached()) {
      this.finished = true;
      this.outcome = 'target_reached';
    } else if (
      !this.project.allowUndo
      && !(this.project.sealEveryMoves
        ? hasMoveWithSeals(this.board, { ...this.project, rows: this.rows, cols: this.cols }, this.sealedCells)
        : hasMove(this.board, { ...this.project, rows: this.rows, cols: this.cols }))
    ) {
      this.finished = true;
      this.outcome = 'no_moves';
    }
    if (this.finished && this.finishedAt == null) this.finishedAt = performance.now();
  }

  historyEntry() {
    return {
      board: this.board.slice(), score: this.score, moves: this.moves,
      // Undo rewinds the board state, not the random stream. Otherwise replaying
      // the same move after an undo would produce the exact same spawn forever.
      dice: this.dice, wallIndex: this.wallIndex,
      sealedCells: this.sealedCells.slice(), sealRound: this.sealRound,
    };
  }

  rotateSeals() {
    const released = this.sealedCells.slice();
    const previous = new Set(released);
    const candidates = this.board.map((_value, index) => index)
      .filter(index => this.board[index] !== WALL && !previous.has(index));
    const sealCount = Math.min(this.project.sealCount || 3, candidates.length);
    const draw = () => {
      const available = candidates.slice();
      const selection = [];
      for (let count = 0; count < sealCount; count += 1) {
        selection.push(available.splice(Math.floor(this.sealRandom() * available.length), 1)[0]);
      }
      return selection;
    };
    let sealed = draw();
    for (let attempt = 0; attempt < 32 && cutsOffCorner(sealed, this.rows, this.cols); attempt += 1) {
      sealed = draw();
    }
    // A bounded fallback also handles a pathological deterministic random
    // source that keeps drawing the same forbidden combination.
    if (cutsOffCorner(sealed, this.rows, this.cols)) {
      const replacement = candidates.find(index =>
        !sealed.slice(0, -1).includes(index)
        && !cutsOffCorner([...sealed.slice(0, -1), index], this.rows, this.cols));
      if (replacement != null) sealed[sealed.length - 1] = replacement;
    }
    this.sealedCells = sealed.sort((a, b) => a - b);
    this.sealRound += 1;
    return { released, sealed: this.sealedCells.slice() };
  }

  move(direction) {
    if (this.finished || !DIRECTIONS.includes(direction)) return { changed: false, snapshot: this.snapshot() };
    const rules = { ...this.project, rows: this.rows, cols: this.cols };
    const moved = this.project.sealEveryMoves
      ? moveBoardWithSeals(this.board, rules, direction, this.sealedCells)
      : moveBoard(this.board, rules, direction);
    if (!moved.changed) return { changed: false, snapshot: this.snapshot() };
    if (this.project.allowUndo) this.history.push(this.historyEntry());
    const before = this.board.slice();
    this.board = moved.board;
    this.score += moved.score;
    this.moves += 1;
    const interval = Number(this.project.sealEveryMoves || 0);
    const seals = interval > 0 && this.moves % interval === 0 ? this.rotateSeals() : null;
    const finish = spawn => {
      this.revision += 1;
      this.transition = { kind: 'move', direction, before, movements: moved.movements, spawn, seals };
      this.settleOutcome();
      return { changed: true, snapshot: this.snapshot() };
    };
    const spawned = this.spawn(this.board);
    return spawned?.then ? spawned.then(finish) : finish(spawned);
  }

  undo() {
    if (!this.project.allowUndo || !this.history.length || this.finished) return false;
    const previous = this.history.pop();
    Object.assign(this, previous);
    this.revision += 1;
    this.transition = { kind: 'undo' };
    return true;
  }

  snapshot() {
    return {
      projectId: this.project.id,
      rows: this.rows,
      cols: this.cols,
      board: this.board.slice(),
      score: this.score,
      boardSum: boardSum(this.board),
      moves: this.moves,
      revision: this.revision,
      transition: clone(this.transition),
      finished: this.finished,
      outcome: this.outcome,
      elapsedMs: this.elapsed(),
      dice: this.dice,
      wallIndex: this.wallIndex,
      restartCount: this.restartCount,
      sealedCells: this.sealedCells.slice(),
      sealRound: this.sealRound,
      nextSealIn: this.project.sealEveryMoves
        ? this.project.sealEveryMoves - (this.moves % this.project.sealEveryMoves)
        : null,
      canUndo: this.project.allowUndo && this.history.length > 0,
      aiError: this.aiError || '',
      targetCount: this.project.targetTile ? this.board.filter(value => value === this.project.targetTile).length : null,
    };
  }
}

export function formatElapsed(milliseconds) {
  const value = Math.max(0, Number(milliseconds) || 0);
  const minutes = Math.floor(value / 60000);
  const seconds = Math.floor(value / 1000) % 60;
  const centiseconds = Math.floor(value / 10) % 100;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(centiseconds).padStart(2, '0')}`;
}
