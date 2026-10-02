import test from 'node:test';
import assert from 'node:assert/strict';
import { ServerClock } from '../src/serverClock.js';

test('late network samples cannot freeze the clock and offline time continues', () => {
  let local = 0;
  const clock = new ServerClock({ now: () => local, wall: () => 100000 });
  clock.observe(new Date(200000).toISOString());
  local = 3000;
  assert.equal(clock.now(), 203000);
  // A snapshot captured two seconds ago must not re-anchor at its arrival.
  clock.observe(new Date(201000).toISOString());
  assert.equal(clock.now(), 203000);
  local = 4000;
  clock.observe(new Date(200000).toISOString());
  assert.equal(clock.now(), 204000);
  local = 64000;
  assert.equal(clock.now(), 264000);
  clock.observe(new Date(265000).toISOString());
  assert.equal(clock.now(), 265000);
});

test('a changing device wall clock does not change the stopwatch', () => {
  let local = 0, wall = 100000;
  const clock = new ServerClock({ now: () => local, wall: () => wall });
  clock.observe(new Date(wall).toISOString());
  wall -= 60000; local = 500;
  assert.equal(clock.now(), 100500);
});
