import { Xoshiro128StarStar } from '../utils/xoshiro128.js';

export const VARIANTS = { '4x4': [4, 4], '3x4': [3, 4], '2x4': [2, 4], '3x3': [3, 3] };
// event.code mapping matches the main site's useTrainerSession.handleKeydown.
export const DIRECTIONS = { ArrowUp: 0, KeyW: 0, KeyK: 0, ArrowRight: 1, KeyD: 1, KeyL: 1, ArrowDown: 2, KeyS: 2, KeyJ: 2, ArrowLeft: 3, KeyA: 3, KeyH: 3 };
export const MAX_MOVES = 200000;
// Record all milestones; viewport height only decides how many are displayed.
export const NODE_TILES = Array.from({ length: 27 }, (_, i) => 2 ** (i + 5));
export const clone = (value) => JSON.parse(JSON.stringify(value));

export function move(board, rows, cols, direction) {
  if (![0, 1, 2, 3].includes(direction)) throw new Error('invalid_direction');
  const result = [...board]; let score = 0;
  const slide = Array(board.length).fill(0), pops = Array(board.length).fill(0);
  const horizontal = direction === 1 || direction === 3;
  for (let line = 0; line < (horizontal ? rows : cols); line++) {
    const positions = Array.from({ length: horizontal ? cols : rows }, (_, i) => horizontal ? line * cols + i : i * cols + line);
    if (direction === 1 || direction === 2) positions.reverse();
    const source = positions.filter(i => board[i]);
    const values = source.map(i => board[i]); const merged = [];
    for (let i = 0; i < values.length; i++) {
      let value = values[i];
      const destination = positions[merged.length];
      slide[source[i]] = Math.abs(positions.indexOf(source[i]) - merged.length);
      if (value === values[i + 1]) {
        slide[source[i + 1]] = Math.abs(positions.indexOf(source[i + 1]) - merged.length);
        pops[destination] = 1; value *= 2; score += value; i++;
      }
      if (value > 2 ** 31) throw new Error('tile_limit');
      merged.push(value);
    }
    positions.forEach((p, i) => { result[p] = merged[i] || 0; });
  }
  return { board: result, score, changed: result.some((v, i) => v !== board[i]),
    metadata: { direction: ['up','right','down','left'][direction], slide_distances: slide, pop_positions: pops } };
}

export function randomSpawn(board, random = Math.random) {
  const empty = board.flatMap((v, i) => v ? [] : [i]);
  if (!empty.length) return null;
  const index = empty[Math.floor(random() * empty.length)]; const value = random() < .1 ? 4 : 2;
  board[index] = value; return { index, value };
}

export function seededSpawn(board, rng) {
  const empty = board.flatMap((v, i) => v ? [] : [i]);
  if (!empty.length) throw new Error('no_spawn_space');
  const index = empty[rng.chooseIndex(empty.length)]; const value = rng.nextFloat() < .1 ? 4 : 2;
  board[index] = value; return { index, value };
}

export const hex = (bytes) => Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
export const unhex = (value) => Uint8Array.from(value.match(/../g) || [], b => parseInt(b, 16));
export async function sha256(bytes) { return hex(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))); }
export function eventBytes(events) {
  const bytes = new Uint8Array(events.length * 5); const view = new DataView(bytes.buffer);
  events.forEach((e, i) => { view.setUint8(i * 5, e[0]); view.setUint32(i * 5 + 1, e[1], true); });
  return bytes;
}
export async function eventHash(previous, event) {
  const bytes = new Uint8Array(37); bytes.set(unhex(previous)); bytes.set(eventBytes([event]), 32);
  return sha256(bytes);
}
export function initialState(runId, variant, seed) {
  const [rows, cols] = VARIANTS[variant]; const rng = Xoshiro128StarStar.fromSeedHex(seed);
  const board = Array(rows * cols).fill(0); seededSpawn(board, rng); seededSpawn(board, rng);
  return { board, rng: rng.exportState(), seq: 0, score: 0, elapsed: 0, nodes: {} };
}
export async function initialHash(id, variant, seed) {
  return sha256(new TextEncoder().encode(`HPR1|${id}|${variant}|${seed}`));
}
export function isOver(board, variant) { return [0, 1, 2, 3].every(d => !move(board, ...VARIANTS[variant], d).changed); }

export function nextMove(run, direction, delta, nodes = NODE_TILES) {
  if (run.seq >= MAX_MOVES) throw new Error('move_limit');
  const moved = move(run.board, ...VARIANTS[run.variant], direction);
  if (!moved.changed) return null;
  const rng = new Xoshiro128StarStar(run.rng);
  const spawn = seededSpawn(moved.board, rng);
  const event = [direction | (spawn.index << 2) | (spawn.value === 4 ? 64 : 0), delta];
  const state = { ...run, board: moved.board, score: run.score + moved.score, rng: rng.exportState(),
    seq: run.seq + 1, elapsed: run.elapsed + delta, nodes: { ...run.nodes }, splitTimes: { ...(run.splitTimes || {}) } };
  for (const tile of nodes) if (state.board.includes(tile) && !state.nodes[tile]) state.nodes[tile] = { seq: state.seq, elapsed: state.elapsed };
  if (Array.isArray(run.timerSplits)) {
    for (const expression of run.timerSplits) if (!state.splitTimes[expression] && splitReached(state.board, expression)) state.splitTimes[expression] = { seq: state.seq, elapsed: state.elapsed };
  }
  return { state, event };
}

function splitReached(board, expression) {
  const requested = String(expression).split('+').map(Number); const counts = new Map();
  board.forEach(value => counts.set(value, (counts.get(value) || 0) + 1));
  for (const value of requested) {
    if (counts.get(value)) counts.set(value, counts.get(value) - 1);
    else if ([...counts].some(([tile, count]) => count > 0 && tile > value)) return true;
    else return false;
  }
  return true;
}

export function rebuildTimerSplitTimes(runId, variant, seed, events, timerSplits) {
  let state = { ...initialState(runId, variant, seed), variant, timerSplits, splitTimes: {} };
  for (const event of events) {
    const next = nextMove(state, event[0] & 3, event[1]);
    if (!next || next.event[0] !== event[0]) throw new Error('invalid_replay_move');
    state = next.state;
  }
  return state.splitTimes;
}

export function parseReplay(buffer) {
  const bytes = new Uint8Array(buffer); const view = new DataView(buffer);
  const magic = new TextDecoder().decode(bytes.slice(0, 4));
  if (bytes.length < 8 || !['HPR1', 'HPR2'].includes(magic)) throw new Error('invalid_replay');
  const size = view.getUint32(4, true);
  if (size > 65536 || size + 8 > bytes.length || (bytes.length - 8 - size) % 5) throw new Error('invalid_replay');
  const header = JSON.parse(new TextDecoder().decode(bytes.slice(8, 8 + size)));
  const version = magic === 'HPR2' ? 2 : 1;
  if (header.version !== version || header.rules_version !== 1 || !VARIANTS[header.variant]) throw new Error('unsupported_replay');
  const count = (bytes.length - 8 - size) / 5, start = 8 + size;
  if (count > MAX_MOVES) throw new Error('move_limit');
  const events = [];
  if (version === 1) {
    for (let offset = start; offset < bytes.length; offset += 5) events.push([view.getUint8(offset), view.getUint32(offset + 1, true)]);
  } else {
    for (let i = 0; i < count; i++) events.push([bytes[start + i], bytes[start + count + i]
      + bytes[start + count * 2 + i] * 256 + bytes[start + count * 3 + i] * 65536 + bytes[start + count * 4 + i] * 16777216]);
  }
  return { header, events };
}

export function buildReplay(replay) {
  const { header, events } = replay;
  const snapshots = new Map(); let state = { ...initialState(header.run_id, header.variant, header.seed), variant: header.variant };
  snapshots.set(0, clone(state));
  const step = (current, e) => {
    const next = nextMove(current, e[0] & 3, e[1]);
    if (!next || next.event[0] !== e[0]) throw new Error('invalid_replay_move');
    return next.state;
  };
  events.forEach((e, i) => { state = step(state, e); if ((i + 1) % 256 === 0) snapshots.set(i + 1, clone(state)); });
  return { ...replay, total: events.length, final: state, seek(raw) {
    const index = Math.max(0, Math.min(events.length, Math.trunc(raw)));
    const start = Math.floor(index / 256) * 256; let result = clone(snapshots.get(start));
    for (let i = start; i < index; i++) result = step(result, events[i]);
    return result;
  } };
}
