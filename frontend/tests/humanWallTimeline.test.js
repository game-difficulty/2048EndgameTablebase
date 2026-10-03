import test from 'node:test';
import assert from 'node:assert/strict';
import { recordMoveTime, moveInstants } from '../src/human/wallTimeline.js';

const stamp = 1791000000000;
test('absolute instants survive closing, long delays and clock reversal', () => {
  let run = { seq: 0, wallTimeline: { version: 1, anchors: [], started_at_ms: stamp - 1000 } };
  const times = [stamp, stamp + 100, stamp + 86400000, stamp + 86400000 - 500, stamp + 0x100000000 + 86400000];
  const events = [];
  for (const time of times) {
    const value = recordMoveTime(run, time);
    events.push([0, value.delta]);
    run = JSON.parse(JSON.stringify({ ...run, seq: run.seq + 1, wallTimeline: value.timeline, lastActionAt: time }));
  }
  assert.deepEqual(moveInstants(run.wallTimeline, events), times);
  assert.equal(run.wallTimeline.anchors.length, 3);
});

test('upgraded old games do not invent absolute instants for earlier moves', () => {
  const { delta, timeline } = recordMoveTime({ seq: 2, lastActionAt: stamp - 10 }, stamp);
  assert.deepEqual(moveInstants(timeline, [[0, 0], [0, 20], [0, delta]]), [null, null, stamp]);
  assert.deepEqual(moveInstants(null, [[0, 0], [0, 10]]), [null, null]);
});
