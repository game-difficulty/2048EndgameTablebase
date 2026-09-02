import assert from 'node:assert/strict';
import test from 'node:test';

import { analyzeReplay } from '../src/features/replay/engine/replayAnalysis.js';
import { parseRplArrayBuffer } from '../src/features/replay/engine/rplParser.js';
import {
  applyTesterLocalMove,
  createTesterLocalSession,
  encodeTesterReplay,
  testerReplayFilename,
} from '../src/features/tester/engine/testerLocalSession.js';

const rolls = (...values) => {
  let index = 0;
  return () => values[index++] ?? 0;
};

test('tester move updates board, metrics and replay without a server state', () => {
  const initial = createTesterLocalSession({
    board: [2, 2, ...new Array(14).fill(0)],
    performanceLabels: ['Perfect!', 'Blunder!'],
  });
  const moved = applyTesterLocalMove(initial, {
    direction: 'left',
    results: { left: 0.9, right: 0.8, up: null, down: null },
    dtype: 'float64',
    spawnRate4: 0.1,
    randomSource: rolls(0, 0.9),
  });
  assert.equal(moved.accepted, true);
  assert.equal(moved.session.practice.revision, 1);
  assert.equal(moved.session.metrics.combo, 1);
  assert.equal(moved.session.metrics.score, 4);
  assert.equal(moved.session.records.length, 1);

  const replay = parseRplArrayBuffer(encodeTesterReplay(moved.session));
  assert.equal(replay.moveCount, 1);
  assert.equal(replay.changes[0] >> 5, 0);
  assert.equal(replay.terminalBoard > 0n, true);
});

test('tester rejects a move when results are not bound to the current step', () => {
  const initial = createTesterLocalSession({
    board: [2, 2, ...new Array(14).fill(0)],
  });
  const moved = applyTesterLocalMove(initial, {
    direction: 'left',
    results: {},
    dtype: 'float64',
    randomSource: rolls(0, 0),
  });
  assert.equal(moved.accepted, false);
  assert.equal(moved.reason, 'results_required');
  assert.equal(moved.session, initial);
});

test('tester filename and replay analysis use the same goodness formula', () => {
  const initial = createTesterLocalSession({
    board: [2, ...new Array(15).fill(0)],
    performanceLabels: ['Perfect!', 'Terrible!'],
  });
  const moved = applyTesterLocalMove(initial, {
    direction: 'right',
    results: { left: 1, right: 0.5, up: null, down: null },
    dtype: 'float64',
    spawnRate4: 0.1,
    randomSource: rolls(0, 0),
  });
  assert.equal(moved.accepted, true);

  const replay = parseRplArrayBuffer(encodeTesterReplay(moved.session));
  const analysis = analyzeReplay(replay);
  assert.equal(moved.session.metrics.goodness_of_fit, 0.5);
  assert.equal(analysis.summary.final_gof, 0.5);
  assert.equal(testerReplayFilename('L3_128', analysis.summary.final_gof), 'L3_128_0.5000.rpl');
});
