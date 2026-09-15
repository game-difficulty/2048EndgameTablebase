import test from 'node:test';
import assert from 'node:assert/strict';
import { LikeFeedback } from '../src/live/likeFeedback.js';
import { BUILTIN_TRACKS, nextTrackIndex, audioFileKey } from '../src/live/playlist.js';

test('rapid clicks accumulate while one request is in flight, without double-counting broadcasts', () => {
  const likes = new LikeFeedback();
  likes.update(10);
  for (let i = 0; i < 5; i++) assert.equal(likes.begin(1000), true);
  assert.equal(likes.count, 15);
  assert.equal(likes.takeBatch(), 5);
  assert.equal(likes.begin(1100), true);
  assert.equal(likes.begin(1200), true);
  assert.equal(likes.takeBatch(), 0);
  likes.update(15);
  assert.equal(likes.count, 17);
  likes.finish(15);
  assert.equal(likes.count, 17);
  assert.equal(likes.takeBatch(), 2);
  likes.finish(17);
  likes.update(10);
  assert.equal(likes.count, 17);
  assert.equal(likes.pending, false);
});
test('rejected like rolls back without losing newer totals', () => {
  const likes = new LikeFeedback();
  likes.update(20); likes.begin(); likes.takeBatch(); likes.reject();
  assert.equal(likes.count, 20);
  likes.begin(); likes.takeBatch(); likes.begin(); likes.update(23); likes.reject();
  assert.equal(likes.count, 24);
  assert.equal(likes.takeBatch(), 1);
  likes.finish(24);
  assert.equal(likes.count, 24);
});
test('client bounds rapid clicks to twenty per minute without growing the queue', () => {
  const likes = new LikeFeedback();
  for (let i = 0; i < 20; i++) assert.equal(likes.begin(1000), true);
  assert.equal(likes.begin(1001), false);
  assert.equal(likes.count, 20);
  assert.equal(likes.takeBatch(), 20);
  likes.finish(20);
  assert.equal(likes.begin(60999), false);
  assert.equal(likes.begin(61000), true);
});
test('playlist wraps forward and backward; built-in tracks work without file selection', () => {
  assert.ok(BUILTIN_TRACKS.length >= 2);
  assert.equal(nextTrackIndex(1, 2), 0);
  assert.equal(nextTrackIndex(0, 2, -1), 1);
  assert.equal(nextTrackIndex(0, 0), 0);
  assert.notEqual(audioFileKey({name:'a',size:2,lastModified:0}), audioFileKey({name:'a',size:3,lastModified:0}));
});
