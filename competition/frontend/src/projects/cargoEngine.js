import { settleRigidTiles, shiftCells } from './rigidMovement.js';
import { nextRandom, seed32, ticketFloat } from './randomStreams.js';
import { CARGO_SHAPES, CARGO_SHAPE_GROUPS } from '../../../shared/cargoShapes.mjs';
export { CARGO_SHAPES, CARGO_SHAPE_GROUPS } from '../../../shared/cargoShapes.mjs';

export const FIRST_CARGO_MOVE = 10;

const VECTORS = Object.freeze({ up: [-1, 0], right: [0, 1], down: [1, 0], left: [0, -1] });
const copyCargo = cargo => cargo && { ...cargo };
const tileCells = cargo => CARGO_SHAPES[cargo.shape].cells.map(([dr, dc]) => [cargo.row + dr, cargo.col + dc]);

function cargoBoardCells(cargo) {
  if (!cargo) return [];
  return tileCells(cargo).filter(([row]) => row >= 0 && row < 4).map(([row, col]) => row * 4 + col);
}

function canShiftCargo(cargo, direction) {
  if (!cargo || !VECTORS[direction]) return false;
  const currentCells = tileCells(cargo);
  if (currentCells.every(([row]) => row >= 4)) return false;
  if (currentCells.some(([row]) => row < 0) && direction !== 'down') return false;
  // Once a piece crosses the outlet it cannot be pulled back, but it may
  // still slide sideways to align its protruding cells with the two-cell exit.
  if (currentCells.some(([row]) => row >= 4) && direction === 'up') return false;
  const [dr, dc] = VECTORS[direction];
  const next = { ...cargo, row: cargo.row + dr, col: cargo.col + dc };
  const cells = tileCells(next);
  if (cells.some(([row]) => row < 0) && direction !== 'down') return false;
  if (cells.some(([, col]) => col < 0 || col >= 4)) return false;
  if (cells.some(([row, col]) => row >= 4 && ![1, 2].includes(col))) return false;
  return true;
}

export function moveCargoBoard(board, cargo, direction) {
  if (!VECTORS[direction]) return { changed: false, board, cargo, movements: [], delivered: false };
  const [dr, dc] = VECTORS[direction];
  const tiles = board.flatMap((value, index) => value > 0 ? [{ id: `number-${index}`, value, cells: [index] }] : []);
  if (cargo) tiles.push({ ...cargo, isCargo: true, cells: tileCells(cargo).map(([r, c]) => r * 4 + c) });
  const moved = settleRigidTiles(tiles, direction, {
    cols: 4,
    step: tile => {
      if (!tile.isCargo) {
        const cells = shiftCells(tile.cells, direction, 4, 4);
        return cells && { cells };
      }
      if (!canShiftCargo(tile, direction)) return null;
      return { row: tile.row + dr, col: tile.col + dc, cells: tile.cells.map(cell => cell + dr * 4 + dc) };
    },
    merge: (tile, target) => !tile.isCargo && !target.isCargo && tile.value === target.value
      ? { tile: { id: `merge-${target.id}-${tile.id}`, value: tile.value * 2, cells: target.cells.slice() }, score: tile.value * 2 } : null,
  });
  const nextBoard = Array(16).fill(0);
  for (const tile of moved.tiles) if (!tile.isCargo) nextBoard[tile.cells[0]] = tile.value;
  const resolvedCargo = moved.tiles.find(tile => tile.isCargo);
  const nextCargo = resolvedCargo ? { id: cargo.id, shape: cargo.shape, row: resolvedCargo.row, col: resolvedCargo.col } : null;
  const cargoMoved = Boolean(cargo && (nextCargo.row !== cargo.row || nextCargo.col !== cargo.col));
  const movements = moved.movements.filter(item => item.id !== cargo?.id).map(item => ({
    from: item.from[0], to: item.to[0], value: board[item.from[0]], merged: Boolean(item.mergeInto),
  }));
  return {
    changed: moved.changed || cargoMoved,
    board: nextBoard,
    cargo: nextCargo,
    movements,
    gained: moved.score,
    cargoMoved,
    delivered: cargoMoved && tileCells(nextCargo).every(([row]) => row >= 4),
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
    const group = CARGO_SHAPE_GROUPS[Math.floor(this.shapeState / 0x100000000 * CARGO_SHAPE_GROUPS.length)];
    // Always take two draws from the shape stream, even for single-variant
    // families. Neither numeric spawns nor player move counts affect it.
    this.shapeState = nextRandom(this.shapeState);
    const shape = group[Math.floor(this.shapeState / 0x100000000 * group.length)];
    return { id: `cargo-${this.nextCargoId++}`, shape, row: -2, col: 1 };
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
    return Math.max(0, (this.finishedAt ?? now) - this.startedAt);
  }

  move(direction) {
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
      cargo: copyCargo(this.cargo), elapsedMs: this.elapsed(), limitMs: null,
      remainingMs: null,
      finished: this.finished, outcome: this.outcome,
      revision: this.revision, transition: this.transition,
    };
  }
}
