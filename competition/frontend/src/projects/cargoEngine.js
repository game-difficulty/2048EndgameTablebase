import { moveBoard, WALL } from './engine.js';
import { nextRandom, seed32, ticketFloat } from './randomStreams.js';

export const CARGO_LIMIT_MS = 10 * 60 * 1000;
export const FIRST_CARGO_MOVE = 10;
export const CARGO_SHAPES = Object.freeze([
  Object.freeze({ key: 'square', name: '2×2', cells: [[0, 0], [0, 1], [1, 0], [1, 1]] }),
  Object.freeze({ key: 'l0', name: 'L', cells: [[0, 0], [0, 1], [1, 0]] }),
  Object.freeze({ key: 'l1', name: 'L', cells: [[0, 0], [0, 1], [1, 1]] }),
  Object.freeze({ key: 'l2', name: 'L', cells: [[0, 1], [1, 0], [1, 1]] }),
  Object.freeze({ key: 'l3', name: 'L', cells: [[0, 0], [1, 0], [1, 1]] }),
]);

const VECTORS = Object.freeze({ up: [-1, 0], right: [0, 1], down: [1, 0], left: [0, -1] });
const copyCargo = cargo => cargo && { ...cargo };
const tileCells = cargo => CARGO_SHAPES[cargo.shape].cells.map(([dr, dc]) => [cargo.row + dr, cargo.col + dc]);

function cargoBoardCells(cargo) {
  if (!cargo) return [];
  return tileCells(cargo).filter(([row]) => row >= 0 && row < 4).map(([row, col]) => row * 4 + col);
}

function canShiftCargo(cargo, direction, board) {
  if (!cargo || !VECTORS[direction]) return false;
  if (cargo.row < 0 && direction !== 'down') return false;
  // Once a piece crosses the outlet it cannot be pulled back, but it may
  // still slide sideways to align its protruding cells with the two-cell exit.
  if (cargo.row >= 3 && direction === 'up') return false;
  const [dr, dc] = VECTORS[direction];
  const next = { ...cargo, row: cargo.row + dr, col: cargo.col + dc };
  if (next.row < 0 && direction !== 'down') return false;
  const cells = tileCells(next);
  if (cells.some(([, col]) => col < 0 || col >= 4)) return false;
  if (cells.some(([row, col]) => row >= 4 && ![1, 2].includes(col))) return false;
  return cells.every(([row, col]) => row < 0 || row >= 4 || board[row * 4 + col] === 0);
}

export function moveCargoBoard(board, cargo, direction) {
  if (!VECTORS[direction]) return { changed: false, board, cargo, movements: [], delivered: false };
  const occupied = new Set(cargoBoardCells(cargo));
  const masked = board.map((value, index) => occupied.has(index) ? WALL : value);
  const moved = moveBoard(masked, { rows: 4, cols: 4 }, direction);
  const nextBoard = moved.board.map(value => value === WALL ? 0 : value);
  const [dr, dc] = VECTORS[direction];
  let nextCargo = cargo;
  while (canShiftCargo(nextCargo, direction, nextBoard)) {
    nextCargo = { ...nextCargo, row: nextCargo.row + dr, col: nextCargo.col + dc };
    if (nextCargo.row >= 4) break; // Fully through the outlet; never slide beyond the delivery point.
  }
  const cargoMoved = nextCargo !== cargo;
  return {
    changed: moved.changed || cargoMoved,
    board: nextBoard,
    cargo: nextCargo,
    movements: moved.movements,
    gained: moved.score,
    cargoMoved,
    delivered: cargoMoved && nextCargo.row >= 4,
  };
}

export function hasCargoMove(board, cargo) {
  return Object.keys(VECTORS).some(direction => moveCargoBoard(board, cargo, direction).changed);
}

export class CargoGame {
  constructor(project, { seed = `${Date.now()}-${Math.random()}` } = {}) {
    this.project = project;
    this.seed = String(seed);
    this.randomState = seed32(`${this.seed}:spawn`);
    this.shapeState = seed32(`${this.seed}:cargo-shape`);
    this.revision = 0;
    this.nextCargoId = 0;
    this.reset(false);
  }

  nextSpawnTicket() {
    this.randomState = nextRandom(this.randomState);
    return this.randomState;
  }

  spawnNumber() {
    const ticket = this.nextSpawnTicket();
    const occupied = new Set(cargoBoardCells(this.cargo));
    const empty = this.board.map((_value, index) => index).filter(index => this.board[index] === 0 && !occupied.has(index));
    if (!empty.length) return null;
    const index = empty[Math.floor(ticketFloat(ticket, 'position') * empty.length)];
    const value = ticketFloat(ticket, 'value') < .1 ? 4 : 2;
    this.board[index] = value;
    return { index, value };
  }

  nextCargo() {
    this.shapeState = nextRandom(this.shapeState);
    return { id: `cargo-${this.nextCargoId++}`, shape: this.shapeState % CARGO_SHAPES.length, row: -2, col: 1 };
  }

  reset(increment = true) {
    this.board = Array(16).fill(0);
    this.cargo = null;
    const first = this.spawnNumber();
    const second = this.spawnNumber();
    this.score = 0;
    this.moves = 0;
    this.finished = false;
    this.outcome = null;
    this.startedAt = performance.now();
    this.finishedAt = null;
    this.revision += 1;
    this.transition = { kind: increment ? 'restart' : 'initial', spawns: [first, second] };
    return this.snapshot();
  }

  elapsed(now = performance.now()) {
    return Math.min(CARGO_LIMIT_MS, Math.max(0, (this.finishedAt ?? now) - this.startedAt));
  }

  expire(now = performance.now()) {
    if (!this.finished && now - this.startedAt >= CARGO_LIMIT_MS) {
      this.finished = true;
      this.outcome = 'time_limit';
      this.finishedAt = this.startedAt + CARGO_LIMIT_MS;
      this.revision += 1;
      this.transition = { kind: 'time_limit' };
    }
    return this.snapshot();
  }

  move(direction) {
    this.expire();
    if (this.finished) return { changed: false, snapshot: this.snapshot() };
    const before = this.board.slice();
    const cargoBefore = copyCargo(this.cargo);
    const moved = moveCargoBoard(this.board, this.cargo, direction);
    if (!moved.changed) return { changed: false, snapshot: this.snapshot() };
    this.board = moved.board;
    this.cargo = moved.delivered ? this.nextCargo() : moved.cargo;
    if (moved.delivered) this.score += 1;
    const spawn = this.spawnNumber();
    this.moves += 1;
    if (!this.cargo && this.moves >= FIRST_CARGO_MOVE) this.cargo = this.nextCargo();
    this.revision += 1;
    this.transition = {
      kind: 'move', direction, before, movements: moved.movements, spawn,
      cargoBefore, cargoMoved: moved.cargoMoved,
      cargoExit: moved.delivered ? copyCargo(moved.cargo) : null,
    };
    if (!hasCargoMove(this.board, this.cargo)) {
      this.finished = true;
      this.outcome = 'no_moves';
      this.finishedAt = performance.now();
    }
    return { changed: true, snapshot: this.snapshot() };
  }

  snapshot() {
    return {
      projectId: this.project.id, rows: 4, cols: 4, board: this.board.slice(),
      score: this.score, deliveries: this.score, moves: this.moves,
      boardSum: this.board.reduce((sum, value) => sum + value, 0),
      cargo: copyCargo(this.cargo), elapsedMs: this.elapsed(), limitMs: CARGO_LIMIT_MS,
      remainingMs: Math.max(0, CARGO_LIMIT_MS - this.elapsed()),
      finished: this.finished, outcome: this.outcome,
      revision: this.revision, transition: this.transition,
    };
  }
}
