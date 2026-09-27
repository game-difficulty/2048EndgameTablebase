import test from 'node:test';
import assert from 'node:assert/strict';

import { nextMove } from '../src/human/engine.js';
import { normalizeTimerSplits, parseTimerSplit, timerSplitReached, timerSplitRow } from '../src/human/timerSplits.js';

test('combined timing milestones validate and use the final tile as the indented label', () => {
  assert.deepEqual(parseTimerSplit('512+256').values, [512, 256]);
  assert.deepEqual(timerSplitRow('512+256'), { key: '512+256', tile: 256, depth: 1 });
  assert.throws(() => parseTimerSplit('256+512'), /invalid_timer_splits/);
  assert.throws(() => normalizeTimerSplits({ '4x4': ['3'] }), /invalid_timer_splits/);
});

test('combined milestone follows Verse-compatible board reach semantics', () => {
  assert.equal(timerSplitReached([512, 256, 0, 0], '512+256'), true);
  assert.equal(timerSplitReached([1024, 0, 0, 0], '512+256'), true);
  assert.equal(timerSplitReached([512, 128, 0, 0], '512+256'), false);
});

test('moves record custom split times separately from verified standard nodes', () => {
  const run = { variant: '2x4', board: [256,256,256,0,0,0,0,0], score: 0, seq: 0, elapsed: 0,
    rng: [1,2,3,4], nodes: {}, timerSplits: ['512+256'], splitTimes: {} };
  const result = nextMove(run, 3, 125);
  assert.equal(result.state.splitTimes['512+256'].elapsed, 125);
  assert.equal(result.state.nodes[512].elapsed, 125);
  assert.equal(result.state.nodes[256].elapsed, 125);
  assert.equal(Object.hasOwn(result.state.nodes, '512+256'), false);
});
