// Practice-only special-tile rules. The ordinary competition engine remains
// untouched; numeric tiles, specials, dominoes and walls share one occupancy
// model so every accepted swipe is resolved as a single board transaction.
import { nextRandom, seed32, ticketFloat } from './randomStreams.js';
import { settleRigidTiles } from './rigidMovement.js';

const VECTORS = { up: [-1, 0], right: [0, 1], down: [1, 0], left: [0, -1] };
const clone = tile => ({ ...tile, cells: tile.cells.slice() });
const copy = tiles => tiles.map(clone);
const sameCells = (a, b) => a.length === b.length && a.every((cell, index) => cell === b[index]);
const adjacent = (a, b, cols) => a.cells.some(cell => b.cells.some(other =>
  Math.abs(Math.floor(cell / cols) - Math.floor(other / cols)) + Math.abs(cell % cols - other % cols) === 1));

function shift(cells, direction, rows, cols) {
  const [dr, dc] = VECTORS[direction];
  const result = [];
  for (const cell of cells) {
    const r = Math.floor(cell / cols) + dr, c = cell % cols + dc;
    if (r < 0 || c < 0 || r >= rows || c >= cols) return null;
    result.push(r * cols + c);
  }
  return result.sort((a, b) => a - b);
}

function leading(tile, direction, cols) {
  const coords = tile.cells.map(cell => direction === 'up' || direction === 'down' ? Math.floor(cell / cols) : cell % cols);
  return direction === 'up' || direction === 'left' ? Math.min(...coords) : -Math.max(...coords);
}

function mergeable(tile, target) {
  if (target.merged) return false;
  if (tile.kind === 'number' && target.kind === 'number') return tile.value === target.value;
  if (tile.kind === 'bomb' && target.kind === 'bomb') return true;
  return (tile.kind === 'chemical-a' || tile.kind === 'chemical-b')
    && (target.kind === 'chemical-a' || target.kind === 'chemical-b');
}

export function slideSpecialTiles(tiles, direction, rows = 4, cols = 4) {
  if (!VECTORS[direction]) return { changed: false, tiles: copy(tiles), score: 0, movements: [], merges: [], removals: [] };
  // Bonded pieces use the same complete-occupancy solver as cargo and growing
  // tiles. Single-cell chemical/bomb reactions retain their existing rules.
  if (tiles.some(tile => tile.kind === 'pair-double' || tile.kind === 'pair-single')) {
    const result = settleRigidTiles(copy(tiles), direction, {
      cols,
      step: tile => {
        const cells = tile.kind !== 'wall' && shift(tile.cells, direction, rows, cols);
        return cells ? { cells } : null;
      },
      merge: (tile, target) => tile.kind === 'number' && target.kind === 'number' && tile.value === target.value
        ? { tile: { id: `merge-${target.id}-${tile.id}`, kind: 'number', value: tile.value * 2, cells: target.cells.slice() }, score: tile.value * 2 } : null,
    });
    const clean = tile => { const value = clone(tile); delete value.merged; return value; };
    return { ...result, tiles: result.tiles.map(clean), merges: result.merges.map(item => ({ ...item, tile: clean(item.tile) })) };
  }
  const settled = new Map(), occupied = new Map(), movements = [], merges = [], removals = [];
  const add = tile => { settled.set(tile.id, tile); for (const cell of tile.cells) occupied.set(cell, tile); };
  const remove = tile => { settled.delete(tile.id); for (const cell of tile.cells) occupied.delete(cell); };
  for (const tile of tiles.filter(item => item.kind === 'wall')) add(clone(tile));
  const ordered = tiles.filter(item => item.kind !== 'wall').map(clone)
    .sort((a, b) => leading(a, direction, cols) - leading(b, direction, cols) || String(a.id).localeCompare(String(b.id)));
  let score = 0, changed = false;
  for (const tile of ordered) {
    let cells = tile.cells.slice(), consumed = false;
    while (true) {
      const candidate = shift(cells, direction, rows, cols);
      if (!candidate) break;
      const blockers = [...new Set(candidate.map(cell => occupied.get(cell)).filter(Boolean))];
      if (!blockers.length) { cells = candidate; continue; }
      if (blockers.length !== 1 || !mergeable(tile, blockers[0])) break;
      const target = blockers[0];
      const mergeId = `merge-${target.id}-${tile.id}`;
      movements.push({ id: tile.id, from: tile.cells.slice(), to: candidate, mergeInto: mergeId });
      const targetMove = movements.find(item => item.id === target.id);
      if (targetMove) targetMove.mergeInto = mergeId;
      remove(target);
      if (tile.kind === 'number') {
        const merged = { id: mergeId, kind: 'number', value: tile.value * 2, cells: target.cells.slice(), merged: true };
        add(merged); merges.push({ tile: clone(merged), sources: [target.id, tile.id] });
        score += merged.value;
      } else if (tile.kind === 'bomb') {
        // The collision consumes one movement, even if the target slid first.
        const countdown = (target.baseCountdown ?? target.countdown) + tile.countdown - 1;
        const merged = countdown <= 0
          ? { id: mergeId, kind: 'wall', value: -1, cells: target.cells.slice(), merged: true }
          : { id: mergeId, kind: 'bomb', value: 0, cells: target.cells.slice(), countdown, baseCountdown: countdown, merged: true };
        add(merged); merges.push({ tile: clone(merged), sources: [target.id, tile.id] });
      } else if (tile.kind !== target.kind) {
        const wall = { id: mergeId, kind: 'wall', value: -1, cells: target.cells.slice(), merged: true };
        add(wall); merges.push({ tile: clone(wall), sources: [target.id, tile.id] });
      } else {
        removals.push(target.id, tile.id);
      }
      changed = consumed = true;
      break;
    }
    if (consumed) continue;
    const moved = !sameCells(tile.cells, cells);
    if (moved) changed = true;
    const result = { ...tile, cells };
    if (tile.kind === 'bomb' && moved) {
      result.countdown -= 1;
      if (result.countdown <= 0) {
        result.id = `wall-${tile.id}`;
        result.kind = 'wall'; result.value = -1; delete result.countdown;
        merges.push({ tile: clone(result), sources: [tile.id] });
        removals.push(tile.id);
      }
    }
    add(result);
    movements.push({ id: tile.id, from: tile.cells.slice(), to: cells });
  }
  return { changed, tiles: [...settled.values()].map(tile => {
    const clean = clone(tile);
    delete clean.merged;
    if (clean.kind === 'bomb') clean.baseCountdown = clean.countdown;
    return clean;
  }), score, movements, merges, removals };
}

export class PracticeSpecialGame {
  constructor(project, { seed = `${Date.now()}-${Math.random()}` } = {}) {
    this.project = project;
    this.seed = String(seed);
    this.rows = Number(project.rows || 4);
    this.cols = Number(project.cols || 4);
    this.randomState = seed32(`${this.seed}:spawn`);
    if (project.specialRule === 'chemical') this.chemicalColorState = seed32(`${this.seed}:chemical-color`);
    this.nextTileId = 0;
    this.revision = 0;
    this.restartCount = 0;
    this.reset(false);
  }

  id() { return `special-${this.nextTileId++}`; }
  nextTicket() { this.randomState = nextRandom(this.randomState); return this.randomState; }
  specialSpawnChance() {
    const count = this.tiles.filter(tile => this.project.specialRule === 'pair'
      ? tile.kind === 'pair-single' || tile.kind === 'pair-double'
      : this.project.specialRule === 'chemical'
        ? tile.kind === 'chemical-a' || tile.kind === 'chemical-b'
        : tile.kind === 'bomb').length;
    return Math.max(0, Number(this.project.specialSpawnRate ?? .05) - count * .02);
  }
  spawn() {
    const ticket = this.nextTicket();
    const occupied = new Set(this.tiles.flatMap(tile => tile.cells));
    const empty = Array.from({ length: this.rows * this.cols }, (_, index) => index).filter(cell => !occupied.has(cell));
    if (!empty.length) return null;
    const cell = empty[Math.floor(ticketFloat(ticket, 'position') * empty.length)];
    const special = ticketFloat(ticket, 'special') < this.specialSpawnChance();
    let tile;
    if (!special) {
      tile = { id: this.id(), kind: 'number', value: ticketFloat(ticket, 'value') < Number(this.project.spawn4Rate ?? .1) ? 4 : 2, cells: [cell] };
    } else if (this.project.specialRule === 'pair') {
      tile = { id: this.id(), kind: 'pair-single', value: 0, cells: [cell] };
    } else if (this.project.specialRule === 'chemical') {
      // Advance only when a chemical actually spawns. Its ordinal color stays
      // shared even when the two players spawn chemicals on different moves.
      this.chemicalColorState = nextRandom(this.chemicalColorState);
      tile = { id: this.id(), kind: this.chemicalColorState / 0x100000000 < .5 ? 'chemical-a' : 'chemical-b', value: 0, cells: [cell] };
    } else {
      // The value channel of this spawn ticket is unused by bombs. Reusing it
      // selects an inclusive integer without advancing the shared spawn RNG.
      const minimum = Number(this.project.bombCountdownMin ?? 12);
      const maximum = Number(this.project.bombCountdownMax ?? 32);
      const countdown = minimum + Math.floor(ticketFloat(ticket, 'value') * (maximum - minimum + 1));
      tile = { id: this.id(), kind: 'bomb', value: 0, cells: [cell], countdown, baseCountdown: countdown };
    }
    this.tiles.push(tile);
    return clone(tile);
  }

  // Only pair-bonding reacts to adjacency. Chemicals react on collision during
  // the slide, so two unlike neighbors do not turn into two separate walls.
  react(transition) {
    if (this.project.specialRule === 'pair') {
      while (true) {
        const singles = this.tiles.filter(tile => tile.kind === 'pair-single').sort((a, b) => a.cells[0] - b.cells[0]);
        const pair = singles.flatMap((a, index) => singles.slice(index + 1).map(b => [a, b]))
          .find(([a, b]) => adjacent(a, b, this.cols));
        if (!pair) break;
        const [a, b] = pair;
        const oldPairs = this.tiles.filter(tile => tile.kind === 'pair-double');
        const result = { id: this.id(), kind: 'pair-double', value: 0, cells: [a.cells[0], b.cells[0]].sort((x, y) => x - y) };
        this.tiles = this.tiles.filter(tile => tile !== a && tile !== b && tile.kind !== 'pair-double');
        this.tiles.push(result);
        transition.removals.push(...oldPairs.map(tile => tile.id));
        transition.merges.push({ tile: clone(result), sources: [a.id, b.id] });
        transition.changed = true;
      }
    }
  }

  reset(increment = true) {
    if (increment) this.restartCount += 1;
    this.tiles = [];
    this.score = 0; this.moves = 0; this.finished = false; this.outcome = null;
    this.startedAt = performance.now(); this.finishedAt = null;
    const first = this.spawn(), second = this.spawn();
    const transition = { kind: increment ? 'restart' : 'initial', merges: [], removals: [], changed: false };
    this.react(transition);
    this.revision += 1;
    this.transition = { kind: transition.kind, spawns: [first, second] };
    this.checkFinished();
    return this.snapshot();
  }

  elapsed(now = performance.now()) { return Math.max(0, (this.finishedAt ?? now) - this.startedAt); }
  checkFinished() {
    if (Object.keys(VECTORS).some(direction => slideSpecialTiles(this.tiles, direction, this.rows, this.cols).changed)) return;
    this.finished = true; this.outcome = 'no_moves'; this.finishedAt = performance.now();
  }

  move(direction) {
    if (this.finished) return { changed: false, snapshot: this.snapshot() };
    const before = copy(this.tiles);
    const result = slideSpecialTiles(before, direction, this.rows, this.cols);
    if (!result.changed) return { changed: false, snapshot: this.snapshot() };
    this.tiles = result.tiles;
    this.score += result.score;
    this.moves += 1;
    const transition = { kind: 'move', before, movements: result.movements, merges: result.merges,
      removals: result.removals, changed: true, spawn: null };
    this.react(transition);
    transition.spawn = this.spawn();
    this.react(transition);
    // A spawn involved in an immediate reaction is rendered as the resulting
    // pair/wall instead of briefly showing a tile that never existed alone.
    if (transition.spawn && !this.tiles.some(tile => tile.id === transition.spawn.id)) transition.spawn = null;
    const surviving = new Set(this.tiles.map(tile => tile.id));
    transition.merges = transition.merges.filter(item => surviving.has(item.tile.id));
    this.revision += 1;
    this.transition = transition;
    this.checkFinished();
    return { changed: true, snapshot: this.snapshot() };
  }

  snapshot() {
    const board = Array(this.rows * this.cols).fill(0);
    for (const tile of this.tiles) for (const cell of tile.cells) board[cell] = tile.value || (tile.kind === 'wall' ? -1 : -2);
    return {
      projectId: this.project.id, rows: this.rows, cols: this.cols, board,
      tiles: copy(this.tiles), score: this.score,
      boardSum: this.tiles.filter(tile => tile.kind === 'number').reduce((sum, tile) => sum + tile.value, 0),
      moves: this.moves, revision: this.revision, transition: JSON.parse(JSON.stringify(this.transition)),
      finished: this.finished, outcome: this.outcome, elapsedMs: this.elapsed(), restartCount: this.restartCount,
    };
  }
}
