import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createLocalStorageStore } from '../src/services/storage/localStorageStore.js';
import { shouldRefreshLiveClock } from '../src/live/displayClock.js';

test('write-only persistence serializes once and preserves an independent saved snapshot', () => {
  const previous = globalThis.window;
  const saved = new Map();
  globalThis.window = { localStorage: { setItem: (k,v) => saved.set(k,v), getItem: k => saved.get(k), removeItem: k => saved.delete(k) } };
  try {
    const store = createLocalStorageStore({ key: 'write-only' });
    let serialized = 0;
    const records = [1, 2];
    const data = { records, toJSON() { serialized++; return { records }; } };
    assert.equal(store.write(data, { returnSnapshot: false }), undefined);
    assert.equal(serialized, 1);
    records.push(3);
    assert.deepEqual(store.read(), { records: [1,2] });
    const returned = store.write({ records });
    returned.records.push(4);
    assert.deepEqual(store.read(), { records: [1,2,3] });
  } finally { globalThis.window = previous; }
});

test('hidden display clocks stop except for active document or video PiP', () => {
  const doc = { hidden: true };
  assert.equal(shouldRefreshLiveClock(false, doc, {}), false);
  assert.equal(shouldRefreshLiveClock(true, doc, {}), true);
  assert.equal(shouldRefreshLiveClock(false, doc, { documentPictureInPicture: { window: { closed: false } } }), true);
  assert.equal(shouldRefreshLiveClock(false, doc, { documentPictureInPicture: { window: { closed: true } } }), false);
  assert.equal(shouldRefreshLiveClock(false, { ...doc, pictureInPictureElement: {} }, {}), true);
  assert.equal(shouldRefreshLiveClock(false, { hidden: false }, {}), true);
});

test('late ranked polling and heartbeat responses cannot restart timers after disposal', async () => {
  const source = readFileSync(new URL('../src/features/gamer/composables/useGamerSession.js', import.meta.url), 'utf8');
  const heartbeat = source.slice(source.indexOf('  const heartbeatCurrentRankedRun ='), source.indexOf('  const cancelRankedStart ='));
  const poll = source.slice(source.indexOf('  const pollRankedRun ='), source.indexOf('  const submitCompletedRankedRun ='));
  for (const method of ['pollRankedRun', 'heartbeatCurrentRankedRun']) {
    for (const rejectRequest of [false, true]) {
      let resolve, reject, scheduled = 0, updated = 0;
      const response = new Promise((yes, no) => { resolve = yes; reject = no; });
      const context = vm.createContext({
        disposed: false, ranked: { value: { runId: 'run', leaseToken: 'lease', status: 'pending' } },
        rankedRunLock: { isHeld: () => true }, rankedPollTimer: null, rankedHeartbeatTimer: null,
        clearRankedPollTimer() {}, clearRankedHeartbeatTimer() {}, loseRankedOwnership() {},
        heartbeatRankedRun: () => response, fetchRankedRun: () => response,
        updateRanked: () => updated++, applyRankedServerStatus: () => updated++,
        window: { setTimeout() { scheduled++; } },
      });
      vm.runInContext(heartbeat + poll + '\nglobalThis.api = { pollRankedRun, heartbeatCurrentRankedRun };', context);
      const pending = context.api[method]();
      context.disposed = true;
      if (rejectRequest) reject(new Error('offline'));
      else resolve({ status: method === 'pollRankedRun' ? 'validating' : 'active' });
      await pending;
      assert.equal(scheduled, 0, method);
      assert.equal(updated, 0, method);
    }
  }
});

test('lobby deduplicates requests, aborts on hiding/unmount and ignores stale responses', async () => {
  const source = readFileSync(new URL('../src/live/HumanLiveLobby.vue', import.meta.url), 'utf8')
    .split('<script setup>')[1].split('</script>')[0].replace(/^import .*;\r?$/gm, '');
  const requests = [], timeouts = new Map();
  let mounted, unmounted, id = 0;
  const doc = { hidden: false, documentElement: { dataset: {} }, addEventListener() {}, removeEventListener() {} };
  const context = vm.createContext({
    document: doc, navigator: { language: 'zh-CN' }, AbortController, ref: value => ({ value }), watch() {},
    liveLanguage: () => 'zh', saveLiveLanguage() {},
    onMounted: callback => { mounted = callback; }, onUnmounted: callback => { unmounted = callback; },
    setInterval: () => 1, clearInterval() {},
    setTimeout: callback => { timeouts.set(++id, callback); return id; }, clearTimeout: key => timeouts.delete(key),
    fetch: (url, options) => new Promise(resolve => requests.push({ options, resolve })),
  });
  vm.runInContext(source + '\nglobalThis.api = { load, visibility, rooms, error };', context);
  mounted();
  await context.api.load();
  assert.equal(requests.length, 1);
  doc.hidden = true; context.api.visibility();
  assert.equal(requests[0].options.signal.aborted, true);
  doc.hidden = false; context.api.visibility();
  assert.equal(requests.length, 2);
  requests[1].resolve({ ok: true, json: async () => ({ rooms: ['new'] }) });
  await new Promise(resolve => setImmediate(resolve));
  requests[0].resolve({ ok: true, json: async () => ({ rooms: ['stale'] }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(context.api.rooms.value[0], 'new');
  const pending = context.api.load();
  for (const timeout of timeouts.values()) timeout();
  assert.equal(requests[2].options.signal.aborted, true);
  unmounted();
  requests[2].resolve({ ok: true, json: async () => ({ rooms: ['after unmount'] }) });
  await pending;
  assert.equal(context.api.rooms.value[0], 'new');
  assert.equal(timeouts.size, 0);
  await context.api.load();
  assert.equal(requests.length, 3);
});
