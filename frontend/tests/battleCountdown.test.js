import assert from 'node:assert/strict';
import test from 'node:test';

import { battleCountdownState } from '../src/features/battle/core/battleCountdown.js';

const deadline = (seconds) => new Date(1_000_000 + seconds * 1000).toISOString();

test('countdown enters urgent and critical states at the intended thresholds', () => {
  assert.deepEqual(battleCountdownState({ deadline: deadline(6), now: 1_000_000, status: 'playing' }), {
    seconds: 6, urgent: false, critical: false, paused: false,
  });
  assert.deepEqual(battleCountdownState({ deadline: deadline(5), now: 1_000_000, status: 'playing' }), {
    seconds: 5, urgent: true, critical: false, paused: false,
  });
  assert.deepEqual(battleCountdownState({ deadline: deadline(2), now: 1_000_000, status: 'playing' }), {
    seconds: 2, urgent: true, critical: true, paused: false,
  });
});

test('correction pauses urgency and finished players have no countdown', () => {
  assert.deepEqual(battleCountdownState({
    deadline: deadline(1), now: 1_000_000, status: 'playing', correcting: true, pausedSeconds: 90,
  }), {
    seconds: 90, urgent: false, critical: false, paused: true,
  });
  assert.equal(battleCountdownState({ status: 'completed' }).seconds, null);
});

test('server resolution hides the stale local deadline until the next step is ready', () => {
  assert.deepEqual(battleCountdownState({
    deadline: deadline(1), now: 1_000_000, status: 'playing', resolving: true,
  }), {
    seconds: null, urgent: false, critical: false, paused: true,
  });
});
