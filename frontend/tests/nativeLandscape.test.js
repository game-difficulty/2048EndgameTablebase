import test from 'node:test';
import assert from 'node:assert/strict';
import { requestNativeLandscape } from '../src/utils/nativeLandscape.js';

test('unsupported browsers do not enter fullscreen', async () => {
  assert.equal(await requestNativeLandscape({ documentElement: { requestFullscreen() { assert.fail(); } } }, {}), false);
});

test('requests native fullscreen and landscape in order', async () => {
  const calls = [];
  const doc = { documentElement: { async requestFullscreen() { calls.push('fullscreen'); } } };
  assert.equal(await requestNativeLandscape(doc, { async lock(value) { calls.push(value); } }), true);
  assert.deepEqual(calls, ['fullscreen', 'landscape']);
});

test('failed lock restores fullscreen entered by this request', async () => {
  const doc = { documentElement: { async requestFullscreen() { doc.fullscreenElement = {}; } }, async exitFullscreen() { doc.fullscreenElement = null; } };
  assert.equal(await requestNativeLandscape(doc, { async lock() { throw Error('unsupported'); } }), false);
  assert.equal(doc.fullscreenElement, null);
});

test('failed lock does not exit an existing fullscreen session', async () => {
  const doc = { fullscreenElement: {}, async exitFullscreen() { assert.fail(); } };
  assert.equal(await requestNativeLandscape(doc, { async lock() { throw Error('unsupported'); } }), false);
});

test('denied fullscreen is handled without throwing', async () => {
  const doc = { documentElement: { async requestFullscreen() { throw Error('denied'); } } };
  assert.equal(await requestNativeLandscape(doc, { async lock() { assert.fail(); } }), false);
});
