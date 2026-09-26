import test from 'node:test';
import assert from 'node:assert/strict';
import { SPEED_REFRESH_MS, addSpeedSample, countSpeedSamples } from '../src/human/speedMetrics.js';

test('speed display refreshes at ten hertz', () => {
  assert.equal(SPEED_REFRESH_MS, 100);
});

test('speed samples appear immediately and expire after one second', () => {
  let samples = addSpeedSample([], 10_000);
  assert.equal(countSpeedSamples(samples, 10_000), 1);
  samples = addSpeedSample(samples, 10_200);
  assert.equal(countSpeedSamples(samples, 10_200), 2);
  assert.equal(countSpeedSamples(samples, 10_999), 2);
  assert.equal(countSpeedSamples(samples, 11_000), 1);
  assert.equal(countSpeedSamples(samples, 11_200), 0);
});

test('recording a new sample discards expired data and bounds storage', () => {
  let samples = Array.from({ length: 150 }, (_, index) => index);
  samples = addSpeedSample(samples, 2_000);
  assert.deepEqual(samples, [2_000]);
  for (let index = 1; index <= 150; index++) samples = addSpeedSample(samples, 2_000 + index);
  assert.equal(samples.length, 100);
  assert.equal(samples.at(-1), 2_150);
});
