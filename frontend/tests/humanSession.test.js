import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { ref } from 'vue';
import * as engine from '../src/human/engine.js';
import { needsReplayUpload } from '../src/human/archivePolicy.js';

// Exercise the real session state machine with deferred network replies. No sleep,
// database, or production API is needed to reproduce acknowledgement races.
let f;
const copy = value => value == null ? value : structuredClone(value);
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
mock.module('../src/human/storage.js', { namedExports: {
  browserId: async () => 'browser-fixture',
  acquireSlot: async () => () => {},
  meta: async (key, value) => value === undefined ? f.meta.get(key) : f.meta.set(key, value),
  readRun: async id => copy(f.runs.get(id)),
  readEvents: async id => copy(f.events.get(id) || []),
  pendingArchives: async () => {
    if (f.pendingError) throw f.pendingError;
    return [...f.runs.values()].filter(r => r.reason && !r.archived).map(copy);
  },
  saveRun: async (run, { event, expectedSeq } = {}) => {
    if (event) {
      assert.equal(f.runs.get(run.id)?.seq, expectedSeq);
      const events = f.events.get(run.id) || []; events.push(copy(event)); f.events.set(run.id, events);
    }
    f.runs.set(run.id, copy(run));
  },
} });
function receipt(server) {
  return { permit: server.monitored ? 'fixture-signed-permit' : '', run_id: server.id, seq: server.seq, epoch: server.epoch, monitored: server.monitored,
    status: server.status, eligibility: 'eligible', server_time: 100, permit_until: server.monitored ? 112 : 0 };
}
async function network(action, operation) {
  f.requests.push(action);
  const hold = f.holds.get(action);
  if (hold) { f.holds.delete(action); hold.entered.resolve(); await hold.promise; }
  if (f.failures.has(action)) throw f.failures.get(action);
  return operation();
}
mock.module('../src/human/client.js', { namedExports: {
  getStatus: async run => network('status', () => receipt(f.server.get(run.id))),
  json: async (path, { body } = {}) => {
    if (path === '/api/human/runs') {
      const id = `fixture-${f.server.size}`;
      f.server.set(id, { id, seq: 0, epoch: 1, monitored: false, status: 'active' });
      return { run_id: id, seed: '00000001000000020000000300000004', threshold: f.threshold, epoch: 1 };
    }
    const [, id, action] = path.match(/runs\/([^/]+)\/(.+)$/);
    return network(action, () => {
      const s = f.server.get(id);
      assert.equal(body.epoch, s.epoch);
      if (action === 'writer') s.epoch++;
      if (action === 'online-check') assert.equal(body.permit, 'fixture-signed-permit');
      return receipt(s);
    });
  },
  upload: async (run, events, browser, writer, action, status) => network(action, () => {
    const s = f.server.get(run.id);
    f.uploads.push({ action, run: copy(run), count: events.length, status: copy(status) });
    if (action === 'reentry' && run.seq < s.seq) throw Object.assign(new Error('rollback_detected'), { code: 'rollback_detected' });
    assert.equal(events.length, run.seq);
    assert.equal(status.seq, s.seq);
    s.seq = run.seq;
    if (action === 'monitor' || action === 'reentry') s.monitored = true;
    if (action === 'append') assert.equal(run.permit, 'fixture-signed-permit');
    if (action === 'seal') s.status = 'sealed';
    return receipt(s);
  }),
} });
const { useHumanSession } = await import('../src/human/session.js');

const settle = async () => { for (let i = 0; i < 20; i++) await new Promise(resolve => setImmediate(resolve)); };
function hold(action) {
  const pending = { ...deferred(), entered: deferred() }; f.holds.set(action, pending); return pending;
}
async function setup(t, threshold = 8) {
  f = { threshold, meta: new Map(), runs: new Map(), events: new Map(), server: new Map(), holds: new Map(), failures: new Map(), uploads: [], requests: [], clock: 100, intervals: [] };
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { onLine: true } });
  globalThis.document = { hidden: false, addEventListener() {}, removeEventListener() {} };
  globalThis.window = { addEventListener() {}, removeEventListener() {} };
  mock.method(performance, 'now', () => f.clock);
  mock.method(globalThis, 'setInterval', callback => { f.intervals.push(callback); return 1; });
  mock.method(globalThis, 'clearInterval', () => {});
  const session = useHumanSession(ref({ id: 1 }), ref({ variants: [] }));
  await session.activate(); session.start();
  t.after(() => { session.stop(); mock.restoreAll(); });
  return session;
}
async function move(session) {
  const before = session.run.value.seq;
  const direction = [3, 2, 1, 0].find(d => engine.nextMove(session.run.value, d, 1));
  assert.notEqual(direction, undefined, 'fixture has a legal move');
  await session.play(direction);
  assert.equal(session.run.value.seq, before + 1, 'input must remain available');
}
async function cross(session) {
  while (session.run.value.score <= f.threshold) await move(session);
}

test('first checkpoint is frozen; delayed receipt preserves newer moves and local events', async t => {
  const s = await setup(t), pending = hold('monitor');
  await cross(s); await pending.entered.promise;
  const crossing = s.run.value.seq;
  assert.equal(s.busy.value, false); assert.equal(s.gate.value, 'ready');
  for (let i = 0; i < 5; i++) await move(s);
  const newer = copy(s.run.value);
  pending.resolve(); await settle();
  assert.equal(f.uploads[0].run.seq, crossing);
  assert.equal(s.run.value.serverSeq, crossing);
  for (const key of ['board', 'rng', 'seq', 'score', 'hash']) assert.deepEqual(s.run.value[key], newer[key]);
  assert.equal(f.events.get(newer.id).length, newer.seq);
});

test('periodic upload and heartbeat do not hold input busy or replace the board', async t => {
  const s = await setup(t); await cross(s); await settle();
  await move(s);
  // Heartbeats renew the 12-second permit while the upload waits for 20 seconds.
  for (const elapsed of [5000, 5000, 5000, 4999]) {
    f.clock += elapsed; await f.intervals[0](); await settle();
    assert.equal(f.uploads.filter(item => item.action === 'append').length, 0);
  }
  const append = hold('append'); f.clock += 1;
  await f.intervals[0](); await append.entered.promise;
  const uploadedSeq = s.run.value.seq;
  await move(s); append.resolve(); await settle();
  assert.equal(s.run.value.serverSeq, uploadedSeq); assert.equal(s.run.value.seq, uploadedSeq + 1);
  const heartbeat = hold('online-check'); f.clock += 5001;
  await f.intervals[0](); await heartbeat.entered.promise;
  await move(s); heartbeat.resolve(); await settle();
  assert.equal(s.run.value.seq, uploadedSeq + 2); assert.equal(s.gate.value, 'ready');
});

test('idle high-score games do not upload; 32 pending moves upload without waiting for the timer', async t => {
  const s = await setup(t); await cross(s); await settle();
  for (let i = 0; i < 5; i++) {
    f.clock += 5000; await f.intervals[0](); await settle();
  }
  assert.equal(f.uploads.filter(item => item.action === 'append').length, 0);
  // The first move after the idle period is uploaded on the next periodic check.
  await move(s); await f.intervals[0](); await settle();
  assert.equal(f.uploads.filter(item => item.action === 'append').length, 1);
  const acknowledged = s.run.value.serverSeq;
  for (let i = 0; i < 31; i++) await move(s);
  assert.equal(f.uploads.filter(item => item.action === 'append').length, 1);
  await move(s); await settle();
  assert.equal(f.uploads.filter(item => item.action === 'append').length, 2);
  assert.equal(s.run.value.serverSeq, acknowledged + 32);
});

test('background reentry freezes pre-check progress while accepting fresh moves', async t => {
  const s = await setup(t); await cross(s); await settle();
  const before = s.run.value.seq, pending = hold('reentry');
  s.pause(); const checking = s.resume(); await pending.entered.promise;
  assert.equal(s.gate.value, 'ready'); await move(s);
  pending.resolve(); await checking; await settle();
  assert.equal(s.run.value.seq, before + 1);
  assert.equal(f.uploads.at(-1).run.seq, before);
});

test('moves during reentry cannot disguise a previously rolled-back local save', async t => {
  const s = await setup(t); await cross(s); await settle();
  const before = s.run.value.seq;
  f.server.get(s.run.value.id).seq = before + 1;
  const pending = hold('reentry'); s.pause(); const checking = s.resume();
  await pending.entered.promise;
  await move(s); await move(s);
  pending.resolve(); await checking; await settle();
  assert.equal(s.gate.value, 'rejected'); assert.equal(s.run.value.seq, before + 2);
});

test('old response after variant switch cannot mutate the new game', async t => {
  const s = await setup(t), pending = hold('monitor');
  await cross(s); await pending.entered.promise;
  await s.activate('3x3'); const next = copy(s.run.value);
  pending.resolve(); await settle();
  assert.deepEqual(s.run.value, next); assert.equal(s.gate.value, 'ready');
});

test('historical replay sealing runs in background after restart', async t => {
  const s = await setup(t); await cross(s); await settle();
  const old = s.run.value.id, pending = hold('seal');
  await s.restart(); await pending.entered.promise;
  assert.notEqual(s.run.value.id, old); await move(s);
  const newer = copy(s.run.value); pending.resolve(); await settle();
  assert.deepEqual(s.run.value, newer); assert.equal(f.runs.get(old).archived, true);
});

test('archive retry submits the missing first checkpoint before its newer tail', async t => {
  const s = await setup(t); await cross(s); await settle(); await move(s);
  // Reproduce a lost first request: local has progressed, server retained no moves.
  const old = s.run.value.id, server = f.server.get(old);
  server.seq = 0; server.monitored = false;
  f.uploads.length = 0;
  await s.restart(); await settle();
  assert.deepEqual(f.uploads.map(item => item.action), ['monitor', 'seal']);
  assert.equal(f.uploads[0].run.seq, f.runs.get(old).firstOverSeq);
  assert.equal(f.runs.get(old).archived, true);
});

test('offline and expired server connection still block high-score input', async t => {
  const s = await setup(t); await cross(s); await settle();
  const before = s.run.value.seq;
  navigator.onLine = false; await s.play(3);
  assert.equal(s.run.value.seq, before); assert.equal(s.gate.value, 'network');
  navigator.onLine = true; await s.retry(); await settle();
  f.clock += 12001; await s.play(3);
  assert.equal(s.run.value.seq, before); assert.equal(s.gate.value, 'network');
});

async function finish(session) {
  for (let i = 0; i < 2000 && !session.run.value.reason; i++) await move(session);
  assert.equal(session.run.value.reason, 'game_over');
  await settle();
}

test('failed death sealing reports the correct replay after restart and does not repeatedly alert', async t => {
  const s = await setup(t, 1000000);
  f.failures.set('seal', Object.assign(new Error('HTTP_500'), { status: 500 }));
  await finish(s);
  assert.equal(s.archiveFailures.value.length, 1);
  const failure = s.archiveFailures.value[0], id = failure.run.id;
  assert.equal(failure.run.seq, f.events.get(id).length);
  assert.equal(s.run.value.archived, undefined);
  await s.flushArchives(); assert.equal(s.archiveFailures.value.length, 1);
  await s.restart(); await settle(); await move(s);
  assert.notEqual(s.run.value.id, id);
  assert.deepEqual(await s.failedReplayEvents(failure), f.events.get(id));
  s.dismissArchiveFailure(id); await s.flushArchives();
  assert.equal(s.archiveFailures.value.length, 0);
  f.failures.clear(); await s.flushArchives();
  assert.equal(f.runs.get(id).archived, true);
});

test('offline death alerts without a request; successful later upload removes the warning', async t => {
  const s = await setup(t, 1000000);
  navigator.onLine = false; await finish(s);
  assert.equal(f.uploads.length, 0);
  assert.equal(s.archiveFailures.value.length, 1);
  navigator.onLine = true; await s.flushArchives();
  assert.equal(s.run.value.archived, true);
  assert.equal(s.archiveFailures.value.length, 0);
});

test('status/authentication rejection also warns about the finished replay', async t => {
  const s = await setup(t, 1000000);
  f.failures.set('status', Object.assign(new Error('HTTP_401'), { status: 401 }));
  await finish(s);
  assert.equal(f.uploads.length, 0);
  assert.equal(s.archiveFailures.value[0].run.id, s.run.value.id);
});

test('local archive enumeration failure retains an exportable in-memory death replay', async t => {
  const s = await setup(t, 1000000);
  f.pendingError = new Error('storage_failed');
  await finish(s);
  const failure = s.archiveFailures.value[0];
  assert.deepEqual(await s.failedReplayEvents(failure), f.events.get(s.run.value.id));
});

test('high-score restart upload failures remain silent, including subsequent offline and storage failures', async t => {
  const s = await setup(t); await cross(s); await settle();
  f.requests.length = 0;
  f.failures.set('seal', new TypeError('Failed to fetch'));
  await s.restart(); await settle();
  assert.ok(f.requests.includes('seal'));
  assert.equal(s.archiveFailures.value.length, 0);
  assert.equal(s.archiveNotice.value, '');
  navigator.onLine = false; await s.flushArchives();
  assert.equal(s.archiveNotice.value, '');
  navigator.onLine = true; f.pendingError = new Error('storage_failed'); await s.flushArchives();
  assert.equal(s.archiveNotice.value, '');
  assert.equal(s.archiveFailures.value.length, 0);
});

test('low-score restarts stay local and historical retry makes no status or upload requests', async t => {
  const s = await setup(t, 1000000); await move(s);
  const old = s.run.value.id, recorded = copy(f.events.get(old));
  f.requests.length = 0;
  await s.restart(); await settle(); await s.flushArchives();
  assert.deepEqual(f.requests, []);
  assert.deepEqual(f.events.get(old), recorded);
  assert.equal(f.runs.get(old).reason, 'restarted');
  assert.equal(f.runs.get(old).archived, undefined);
  assert.equal(s.archiveNotice.value, '');
});

for (const [variant, threshold] of [['4x4', 360000], ['3x4', 36000], ['3x3', 7200], ['2x4', 4000]]) {
  test(`${variant} non-natural archives require strictly greater than the high-score threshold`, () => {
    for (const reason of ['restarted', 'abandoned', 'interrupted']) {
      for (const score of [0, threshold - 1, threshold, threshold + 1]) {
        assert.equal(needsReplayUpload({ variant, reason, score, threshold }), score > threshold);
      }
    }
    assert.equal(needsReplayUpload({ variant, reason: 'game_over', score: 0, threshold }), true);
    assert.equal(needsReplayUpload({ variant, reason: 'game_over', guest: true, score: threshold + 1, threshold }), false);
    assert.equal(needsReplayUpload({ variant, reason: 'game_over', archived: true, score: threshold + 1, threshold }), false);
  });
}
