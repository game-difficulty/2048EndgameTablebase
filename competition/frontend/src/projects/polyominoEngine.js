// Multi-cell tiles are one rigid object. A covered cell is not an independent
// tile and never receives a spawn. The normal numeric board engine intentionally
// stays separate from this shape-aware movement model.
import { nextRandom, seed32, ticketFloat } from './randomStreams.js';
import { settleRigidTiles } from './rigidMovement.js';
const DIRECTIONS = Object.freeze({
  up: [-1, 0], right: [0, 1], down: [1, 0], left: [0, -1],
});

const copyTile = tile => ({ id: tile.id, value: tile.value, cells: tile.cells.slice() });
const sortedCells = cells => [...new Set(cells)].sort((a, b) => a - b);

function shifted(cells, direction, rows, cols) {
  const [dr, dc] = DIRECTIONS[direction];
  const next = [];
  for (const cell of cells) {
    const row = Math.floor(cell / cols) + dr;
    const col = cell % cols + dc;
    if (row < 0 || row >= rows || col < 0 || col >= cols) return null;
    next.push(row * cols + col);
  }
  return sortedCells(next);
}

function mergeCells(value, movingCells, candidate, targetCells) {
  if (value === 128) return sortedCells([...candidate, ...targetCells]);
  if (value === 64) return sortedCells([...movingCells, ...targetCells]);
  return targetCells.slice();
}

export function movePolyominoTiles(tiles, direction, rows = 4, cols = 4) {
  if (!DIRECTIONS[direction]) return { changed: false, tiles: tiles.map(copyTile), score: 0, movements: [], merges: [] };
  const result = settleRigidTiles(tiles.map(copyTile), direction, {
    cols,
    step: tile => {
      const cells = shifted(tile.cells, direction, rows, cols);
      return cells && { cells };
    },
    merge: (tile, target, candidate) => {
      if (target.value !== tile.value || tile.value >= 256) return null;
      const intersection = candidate.filter(cell => target.cells.includes(cell));
      let mergePosition = candidate;
      if (tile.value === 128 && intersection.length === 1) {
        const aligned = shifted(candidate, direction, rows, cols);
        // Two parallel dominoes meet at one cell before reaching full
        // alignment. Prefer the two-cell overlap when that position exists.
        if (aligned?.every(cell => target.cells.includes(cell))) mergePosition = aligned;
      }
      const resultCells = mergeCells(tile.value, tile.cells, mergePosition, target.cells);
      // An ordinary merge occupies one cell, 64+64 keeps both source cells,
      // and 128+128 takes the union at the first overlapping position.
      return { tile: {
        id: `merge-${target.id}-${tile.id}`,
        value: tile.value * 2,
        cells: resultCells,
      }, to: mergePosition, score: tile.value * 2 };
    },
  });
  return { ...result, tiles: result.tiles.map(copyTile), merges: result.merges.map(item => ({ ...item, tile: copyTile(item.tile) })) };
}

export function hasPolyominoMove(tiles, rows = 4, cols = 4) {
  return Object.keys(DIRECTIONS).some(direction => movePolyominoTiles(tiles, direction, rows, cols).changed);
}

export class PolyominoGame {
  constructor(project, { seed = `${Date.now()}-${Math.random()}` } = {}) {
    this.project = project;
    this.seed = String(seed);
    this.randomState = seed32(`${this.seed}:spawn`);
    this.rows = Number(project.rows || 4);
    this.cols = Number(project.cols || 4);
    this.revision = 0;
    this.restartCount = 0;
    this.nextTileId = 0;
    this.reset(false);
  }

  nextSpawnTicket() {
    this.randomState = nextRandom(this.randomState);
    return this.randomState;
  }

  spawn() {
    const ticket = this.nextSpawnTicket();
    const occupied = new Set(this.tiles.flatMap(tile => tile.cells));
    const empty = Array.from({ length: this.rows * this.cols }, (_value, index) => index)
      .filter(index => !occupied.has(index));
    if (!empty.length) return null;
    const cell = empty[Math.floor(ticketFloat(ticket, 'position') * empty.length)];
    const value = ticketFloat(ticket, 'value') < Number(this.project.spawn4Rate ?? .1) ? 4 : 2;
    const tile = { id: `tile-${this.nextTileId++}`, value, cells: [cell] };
    this.tiles.push(tile);
    return copyTile(tile);
  }

  reset(increment = true) {
    if (increment) this.restartCount += 1;
    this.tiles = [];
    const first = this.spawn();
    const second = this.spawn();
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
    const before = this.tiles.map(copyTile);
    const result = movePolyominoTiles(before, direction, this.rows, this.cols);
    if (!result.changed) return { changed: false, snapshot: this.snapshot() };
    this.tiles = result.tiles;
    this.score += result.score;
    this.moves += 1;
    const spawn = this.spawn();
    this.revision += 1;
    this.transition = {
      kind: 'move', before, movements: result.movements,
      merges: result.merges, spawn,
    };
    if (!hasPolyominoMove(this.tiles, this.rows, this.cols)) {
      this.finished = true;
      this.outcome = 'no_moves';
      this.finishedAt = performance.now();
    }
    return { changed: true, snapshot: this.snapshot() };
  }

  snapshot() {
    const board = Array(this.rows * this.cols).fill(0);
    for (const tile of this.tiles) for (const cell of tile.cells) board[cell] = tile.value;
    return {
      projectId: this.project.id,
      rows: this.rows, cols: this.cols, board,
      tiles: this.tiles.map(copyTile),
      score: this.score, boardSum: this.tiles.reduce((sum, tile) => sum + tile.value, 0),
      moves: this.moves, revision: this.revision,
      transition: JSON.parse(JSON.stringify(this.transition)),
      finished: this.finished, outcome: this.outcome,
      elapsedMs: this.elapsed(), restartCount: this.restartCount,
    };
  }
}
