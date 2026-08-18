import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  PERFORMANCE_EVALUATIONS,
  analyzeReplay,
} from '../src/features/replay/engine/replayAnalysis.js';
import { ReplayController } from '../src/features/replay/engine/replayController.js';
import {
  ReplayFormatError,
  parseRplArrayBuffer,
} from '../src/features/replay/engine/rplParser.js';
import {
  buildStepTransition,
  decodeBoard,
  encodeBoard,
  transitionMatchesNextSnapshot,
} from '../src/features/replay/engine/replayTransition.js';

const SENTINEL_RATES = [666666666, 233333333, 314159265, 987654321];

function writeUint64(view, offset, value) {
  const normalized = BigInt(value);
  view.setUint32(offset, Number(normalized & 0xffffffffn), true);
  view.setUint32(offset + 4, Number((normalized >> 32n) & 0xffffffffn), true);
}

function buildReplayBuffer(records, terminalBoard = 0n) {
  const buffer = new ArrayBuffer((records.length + 1) * 25);
  const view = new DataView(buffer);
  records.forEach((record, index) => {
    const offset = index * 25;
    writeUint64(view, offset, record.board);
    view.setUint8(offset + 8, record.change);
    (record.rates || [0, 0, 0, 0]).forEach((value, direction) => {
      view.setUint32(offset + 9 + direction * 4, value, true);
    });
  });
  const sentinelOffset = records.length * 25;
  writeUint64(view, sentinelOffset, terminalBoard);
  view.setUint8(sentinelOffset + 8, 88);
  SENTINEL_RATES.forEach((value, direction) => {
    view.setUint32(sentinelOffset + 9 + direction * 4, value, true);
  });
  return buffer;
}

function change(move, spawnIndex, spawnExponent) {
  return ((move & 0b11) << 5) | ((spawnIndex & 0b1111) << 1) | ((spawnExponent - 1) & 1);
}

test('parses the fixed binary format and terminal board', () => {
  const buffer = buildReplayBuffer([
    { board: 0x0210030129ab4cden, change: change(1, 0, 1), rates: [1, 2, 3, 4] },
  ], 0x1021003129ab4cden);
  const replay = parseRplArrayBuffer(buffer);
  assert.equal(replay.moveCount, 1);
  assert.equal(replay.boards[0], 0x0210030129ab4cden);
  assert.equal(replay.terminalBoard, 0x1021003129ab4cden);
  assert.deepEqual(Array.from(replay.rates), [1, 2, 3, 4]);
});

test('rejects missing sentinel and malformed record sizes', () => {
  assert.throws(() => parseRplArrayBuffer(new ArrayBuffer(25)), ReplayFormatError);
  assert.throws(() => parseRplArrayBuffer(new ArrayBuffer(51)), ReplayFormatError);
});

test('analysis preserves forced moves without scoring them', () => {
  const replay = parseRplArrayBuffer(buildReplayBuffer([
    { board: 1n, change: change(1, 0, 1), rates: [3_000_000_000, 2_250_000_000, 0, 0] },
    { board: 2n, change: change(0, 1, 1), rates: [4_000_000_000, 0, 0, 0] },
  ]));
  const analysis = analyzeReplay(replay);
  assert.deepEqual(Array.from(analysis.losses), [0.75, 1]);
  assert.deepEqual(Array.from(analysis.forced), [0, 1]);
  assert.equal(analysis.summary.total_moves, 1);
  assert.equal(analysis.summary.final_gof, 0.75);
  assert.equal(analysis.summary.counts['Blunder!'], 1);
});

test('continuous steps animate while discontinuities use snapshots', () => {
  const continuous = {
    moveCount: 2,
    boards: BigUint64Array.from([0x0210030129ab4cden, 0x1021003129ab4cden]),
    changes: Uint8Array.from([change(1, 0, 1), 0]),
    rates: new Uint32Array(8),
    terminalBoard: null,
  };
  assert.equal(transitionMatchesNextSnapshot(continuous, 0), true);
  const analysis = analyzeReplay(continuous);
  const controller = new ReplayController({ replay: continuous, analysis });
  assert.equal(controller.step(1).animation.direction, 'right');

  const discontinuous = {
    ...continuous,
    boards: BigUint64Array.from([0x0244157829ab1cden, 0x4211167829ab1cden]),
    changes: Uint8Array.from([change(2, 12, 2), 0]),
  };
  assert.equal(transitionMatchesNextSnapshot(discontinuous, 0), false);
  const discontinuousController = new ReplayController({
    replay: discontinuous,
    analysis: analyzeReplay(discontinuous),
  });
  const state = discontinuousController.step(1);
  assert.deepEqual(state.animation, {});
  assert.equal(state.hex_str, '4211167829ab1cde');
});

test('classic capped 32k merge matches the Python replay transition', () => {
  const board = [32768, 0, 32768, 2, ...new Array(12).fill(0)];
  const replay = {
    moveCount: 1,
    boards: BigUint64Array.from([encodeBoard(board)]),
    changes: Uint8Array.from([change(0, 15, 1)]),
    rates: new Uint32Array(4),
    terminalBoard: null,
  };
  const transition = buildStepTransition(replay, 0, false);
  assert.deepEqual(decodeBoard(transition.nextBoardEncoded), [
    32768, 2, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 2,
  ]);
  assert.equal(transition.metadata.pop_positions[0], 1);
});

test('variant walls split lines and 16k tiles never merge', () => {
  const wallReplay = {
    moveCount: 1,
    boards: BigUint64Array.from([encodeBoard([2, 2, 32768, 0, ...new Array(12).fill(0)])]),
    changes: Uint8Array.from([change(1, 15, 1)]),
    rates: new Uint32Array(4),
    terminalBoard: null,
  };
  const wallTransition = buildStepTransition(wallReplay, 0, true);
  assert.deepEqual(decodeBoard(wallTransition.nextBoardEncoded).slice(0, 4), [0, 4, 32768, 0]);
  assert.equal(wallTransition.metadata.pop_positions[1], 1);

  const cappedReplay = {
    ...wallReplay,
    boards: BigUint64Array.from([encodeBoard([16384, 16384, 0, 0, ...new Array(12).fill(0)])]),
  };
  const cappedTransition = buildStepTransition(cappedReplay, 0, true);
  assert.deepEqual(decodeBoard(cappedTransition.nextBoardEncoded).slice(0, 4), [0, 0, 16384, 16384]);
  assert.equal(cappedTransition.metadata.pop_positions.some(Boolean), false);
});

test('frontend performance thresholds match the shared JSON config', async () => {
  const raw = await readFile(new URL('../../docs_and_configs/performance_evaluations.json', import.meta.url), 'utf8');
  const config = JSON.parse(raw);
  assert.deepEqual(PERFORMANCE_EVALUATIONS, config.evaluations);
});
