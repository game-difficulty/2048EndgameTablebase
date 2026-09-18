import test from 'node:test';
import assert from 'node:assert/strict';
import { canConnectLive, backgroundExpired } from '../src/live/pipPolicy.js';

test('PiP permits background reconnects and bypasses the normal three-minute cutoff', () => {
  assert.equal(canConnectLive(true, true), true);
  assert.equal(backgroundExpired(true, true, 180000, 600000), false);
});
test('ordinary background pages retain the existing timeout and reconnect policy', () => {
  assert.equal(canConnectLive(true, false), false);
  assert.equal(canConnectLive(false, false), true);
  assert.equal(backgroundExpired(true, false, 180000, 179999), false);
  assert.equal(backgroundExpired(true, false, 180000, 180000), true);
  assert.equal(backgroundExpired(false, false, 180000, 600000), false);
  assert.equal(backgroundExpired(true, false, 0, 600000), false);
});
