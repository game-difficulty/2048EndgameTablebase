import { applyLiveStep } from '../liveEngine.js';
import { createSnapshotBoardFrame } from '../../components/boardFrame.js';

const fail = () => { throw Error('snapshot_required'); };
const integer = (n, max = 0xffffffff) => Number.isInteger(n) && n >= 0 && n <= max;
const blankFrame = () => createSnapshotBoardFrame('empty', Array(16).fill(0));

export function emptyMultiState() {
  return { epoch: null, seq: 0, sources: {}, slots: Array.from({ length: 3 }, (_, lane) =>
    ({ lane, generation: 0, status: 'recovering', run: null })), frames: [blankFrame(), blankFrame(), blankFrame()] };
}

function validateSlot(slot) {
  if (!slot || !integer(slot.lane, 2) || !integer(slot.generation)
    || !['running', 'recovering', 'ended'].includes(slot.status)) fail();
  const r = slot.run;
  if (r && (!r.run_id || !integer(r.seq, 100000) || !Number.isSafeInteger(r.score) || r.score < 0
    || !Number.isSafeInteger(r.elapsed_ms) || r.elapsed_ms < 0 || !Array.isArray(r.board) || r.board.length !== 16
    || r.board.some(v => !Number.isSafeInteger(v) || v < 0 || (v !== 0 && (v < 2 || !Number.isInteger(Math.log2(v))))))) fail();
  return { ...slot, run: r ? { ...r, board: [...r.board], nodes: {} } : null };
}

export function syncMultiFrames(state) {
  return { ...state, frames: state.slots.map(s => createSnapshotBoardFrame(
    `${s.run?.run_id}:${s.run?.seq}:sync`, s.run?.board || Array(16).fill(0))) };
}

export function receiveMultiJson(state, event) {
  if (event.type === 'snapshot') {
    if (!event.stream_epoch || !integer(event.content_seq) || !Array.isArray(event.lanes) || event.lanes.length !== 3) fail();
    const slots = event.lanes.map(validateSlot).sort((a, b) => a.lane - b.lane);
    if (slots.some((s, i) => s.lane !== i)) fail();
    const sources = { ...event.sources };
    if (slots.some(s => s.run && typeof sources[s.run.source_id] !== 'string')) fail();
    return syncMultiFrames({ epoch: event.stream_epoch, seq: event.content_seq, sources, slots, batch: event.batch || null });
  }
  if (!['batch', 'lane_start', 'lane_end', 'lane_status', 'dictionary'].includes(event.type)) return state;
  if (event.stream_epoch !== state.epoch || !integer(event.content_seq)) fail();
  if (event.content_seq <= state.seq) return state;
  if (event.content_seq !== state.seq + 1) fail();
  const next = { ...state, seq: event.content_seq, slots: [...state.slots], frames: [...state.frames] };
  if (event.type === 'batch') {
    next.batch = event.batch;
  } else if (event.type === 'dictionary') {
    const entries = Object.entries(event.sources || {});
    if (!entries.length || entries.some(([id, name]) => !integer(Number(id), 65535) || typeof name !== 'string' || name.length > 80
      || (Object.hasOwn(state.sources, id) && state.sources[id] !== name))) fail();
    next.sources = { ...state.sources, ...event.sources };
  } else if (event.type === 'lane_status') {
    if (!integer(event.lane, 2) || event.generation !== state.slots[event.lane].generation
      || !['running', 'recovering'].includes(event.status)) fail();
    next.slots[event.lane] = { ...state.slots[event.lane], status: event.status };
  } else {
    const slot = validateSlot(event.slot), previous = state.slots[slot.lane];
    if (!slot.run || typeof state.sources[slot.run.source_id] !== 'string') fail();
    if (event.type === 'lane_start' && (slot.generation !== previous.generation + 1 || slot.run.seq !== 0)) fail();
    if (event.type === 'lane_end' && (slot.generation !== previous.generation || slot.run.run_id !== previous.run?.run_id
      || slot.run.seq !== previous.run.seq || slot.run.score !== previous.run.score
      || slot.run.board.some((v, i) => v !== previous.run.board[i]))) fail();
    next.slots[slot.lane] = slot;
    next.frames[slot.lane] = createSnapshotBoardFrame(`${slot.run.run_id}:${slot.run.seq}:sync`, slot.run.board);
  }
  return next;
}

export function receiveMultiBatch(state, packet, transitions) {
  if (!state.epoch || !(packet instanceof ArrayBuffer) || packet.byteLength < 7 || packet.byteLength > 4096) fail();
  const view = new DataView(packet);
  if (view.getUint8(0) !== 0x21) fail();
  let sequence = view.getUint32(1, true), offset = 5, count = 0;
  if (!sequence || sequence > state.seq + 1) fail();
  const next = { ...state, slots: [...state.slots], frames: [...state.frames] };
  const accepted = [];
  function varint(max) {
    let value = 0;
    for (let index = 0; index < 4; index++) {
      if (offset >= view.byteLength) fail();
      const byte = view.getUint8(offset++);
      value += (byte & 127) * 2 ** (index * 7);
      if (byte < 128) {
        if (value > max || (index && byte === 0)) fail();
        return value;
      }
    }
    fail();
  }
  while (offset < view.byteLength) {
    if (++count > 64 || sequence > 0xffffffff) fail();
    const control = view.getUint8(offset++);
    const sourceEvent = (control & 3) === 3;
    let lane = control & 3;
    if (sourceEvent) {
      if (control !== 3 || offset >= view.byteLength) fail();
      lane = view.getUint8(offset++);
      if (lane > 2) fail();
    }
    const value = varint(sourceEvent ? 65535 : 7_200_001);
    if (sequence > state.seq) {
      if (sequence !== next.seq + 1) fail();
      const slot = next.slots[lane];
      if (!slot.run || slot.run.ended_at) fail();
      if (sourceEvent) {
        if (typeof next.sources[value] !== 'string') fail();
        next.slots[lane] = { ...slot, run: { ...slot.run, source_id: value, source: next.sources[value] } };
      } else {
        const step = new ArrayBuffer(9), bytes = new DataView(step);
        bytes.setUint32(0, slot.run.seq + 1, true);
        bytes.setUint32(4, Math.floor(value / 2), true);
        bytes.setUint8(8, ((control >> 2) & 3) | ((control >> 4) << 2) | ((value & 1) << 6));
        const applied = applyLiveStep(slot.run, step);
        next.slots[lane] = { ...slot, run: applied.run };
        next.frames[lane] = applied.frame;
        if (transitions) accepted.push({ lane, frame: applied.frame });
      }
      next.seq = sequence;
    }
    sequence++;
  }
  if (transitions) transitions.push(...accepted);
  return next; // Nothing mutates state or the playback buffer until the entire batch validates.
}
