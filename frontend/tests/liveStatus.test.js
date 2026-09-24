import test from 'node:test';
import assert from 'node:assert/strict';
import { createLiveStatusPoller } from '../src/services/live/liveStatus.js';

const flush = () => new Promise(resolve => setImmediate(resolve));
function setup(request, hidden = false) {
  const doc = new EventTarget();
  doc.hidden = hidden;
  const timers = new Map(), values = [];
  let id = 0;
  const stop = createLiveStatusPoller({
    url: '/api/live/status', fetch: request, document: doc,
    onChange: value => values.push(value),
    setTimeout: (fn, ms) => { timers.set(++id, { fn, ms }); return id; },
    clearTimeout: key => timers.delete(key),
  });
  function tick(ms) {
    const entry = [...timers].find(([, value]) => value.ms === ms);
    assert.ok(entry, `expected ${ms}ms timer`);
    timers.delete(entry[0]);
    entry[1].fn();
  }
  return { doc, timers, values, stop, tick };
}
const response = online => ({ ok: true, json: async () => ({ online }) });

test('polls lightweight status every 30 seconds; offline and failures clear animation', async () => {
  const results = [response(true), response(false), response('true'), { ok: false }];
  let calls = 0;
  const state = setup(async (url, options) => {
    assert.equal(url, '/api/live/status');
    assert.equal(options.cache, 'no-store');
    return results[calls++];
  });
  await flush();
  for (let i = 0; i < 3; i++) { state.tick(30000); await flush(); }
  assert.deepEqual(state.values, [true, false, false, false]);
  state.stop();
  assert.equal(state.timers.size, 0);
});

test('hidden pages do not poll; returning refreshes and ignores stale responses', async () => {
  const pending = [];
  const state = setup((url, options) => new Promise(resolve => pending.push({ resolve, options })), true);
  assert.equal(pending.length, 0);
  state.doc.hidden = false;
  state.doc.dispatchEvent(new Event('visibilitychange'));
  state.doc.hidden = true;
  state.doc.dispatchEvent(new Event('visibilitychange'));
  assert.equal(pending[0].options.signal.aborted, true);
  state.doc.hidden = false;
  state.doc.dispatchEvent(new Event('visibilitychange'));
  pending[1].resolve(response(false));
  await flush();
  pending[0].resolve(response(true));
  await flush();
  assert.equal(state.values.at(-1), false);
  assert.equal([...state.timers.values()].filter(item => item.ms === 30000).length, 1);
  state.stop();
});

test('timeout aborts a stalled request and retries; stopped requests cannot update', async () => {
  let calls = 0;
  const state = setup((url, { signal }) => {
    calls++;
    return new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new Error('aborted'))));
  });
  state.tick(5000);
  await flush();
  assert.deepEqual(state.values, [false]);
  state.tick(30000);
  assert.equal(calls, 2);
  state.stop();
  await flush();
  assert.deepEqual(state.values, [false]);
  assert.equal(state.timers.size, 0);
});
