import test from 'node:test';
import assert from 'node:assert/strict';
import { createPersonalRecordsSync } from '../src/features/minigames/services/personalRecordsSync.js';

const payload = (id, score = 100, tier = 2, tile = 11) => ({ user_id: id, records: [
  { game_id: 'ice-age', difficulty: 1, best_score: score, trophy_tier: tier, best_tile_exp: tile },
] });
function fixture() {
  const cache = new Map();
  const requests = [];
  let clock = 0;
  const sync = createPersonalRecordsSync({
    fetchRecords: () => new Promise((resolve, reject) => requests.push({ resolve, reject })),
    readCache: (id) => cache.get(id),
    writeCache: (id, value) => cache.set(id, value),
    onChange: () => {}, now: () => clock,
  });
  return { sync, cache, requests, advance: (ms) => { clock += ms; } };
}

test('account records replace cached values, including downward corrections and removals', async () => {
  const f = fixture();
  f.cache.set(1, payload(1, 999, 4, 15));
  f.sync.setUser(1);
  assert.equal(f.sync.read().summaries['ice-age:1'].bestScore, 999);
  const task = f.sync.refresh(); f.requests[0].resolve(payload(1)); await task;
  assert.deepEqual(f.sync.read().summaries['ice-age:1'], {
    bestScore: 100, trophy: 2, highestExp: 11, highestTile: 2048,
  });
  const remove = f.sync.refresh({ force: true });
  f.requests[1].resolve({ user_id: 1, records: [] }); await remove;
  assert.deepEqual(f.sync.read().summaries, {});
});

test('other accounts and anonymous local records never get merged into an account', async () => {
  const f = fixture();
  f.cache.set(1, payload(1, 111)); f.cache.set(2, payload(2, 222));
  f.sync.setUser(1); const old = f.sync.refresh();
  f.sync.setUser(2); const current = f.sync.refresh();
  f.requests[1].resolve(payload(2, 333)); await current;
  f.requests[0].resolve(payload(1, 999)); await old;
  assert.equal(f.sync.read().userId, 2);
  assert.equal(f.sync.read().summaries['ice-age:1'].bestScore, 333);
  assert.equal(f.cache.get(1).records[0].best_score, 111);
  f.sync.setUser(null);
  assert.deepEqual(f.sync.read().summaries, {});
  await f.sync.refresh(); assert.equal(f.requests.length, 2);
});

test('network errors and malformed responses keep the last confirmed record', async () => {
  const f = fixture(); f.cache.set(1, payload(1)); f.sync.setUser(1);
  let task = f.sync.refresh(); f.requests[0].reject(new Error('offline')); await task;
  assert.equal(f.sync.read().summaries['ice-age:1'].bestScore, 100);
  task = f.sync.refresh(); f.requests[1].resolve(payload(2, 999)); await task;
  assert.equal(f.sync.read().summaries['ice-age:1'].bestScore, 100);
  assert.equal(f.cache.get(1).user_id, 1);
});

test('foreground refresh is limited to one success per minute and force requests are coalesced', async () => {
  const f = fixture(); f.sync.setUser(1);
  const first = f.sync.refresh();
  assert.equal(f.sync.refresh(), first);
  f.requests[0].resolve(payload(1)); await first;
  await f.sync.refresh(); assert.equal(f.requests.length, 1);
  f.advance(60_000);
  const next = f.sync.refresh(); f.requests[1].resolve(payload(1, 200)); await next;
  const pending = f.sync.refresh({ force: true });
  f.sync.refresh({ force: true }); f.sync.refresh({ force: true });
  f.requests[2].resolve(payload(1, 300)); await Promise.resolve();
  assert.equal(f.requests.length, 4);
  f.requests[3].resolve(payload(1, 400)); await pending;
  assert.equal(f.sync.read().summaries['ice-age:1'].bestScore, 400);
});

test('logout invalidates an in-flight refresh even when the same account logs back in', async () => {
  const f = fixture(); f.sync.setUser(1); const old = f.sync.refresh();
  f.sync.setUser(null); f.sync.setUser(1);
  const current = f.sync.refresh();
  f.requests[0].resolve(payload(1, 999)); await old;
  assert.equal(f.sync.read().loaded, false);
  f.requests[1].resolve(payload(1, 123)); await current;
  assert.equal(f.sync.read().summaries['ice-age:1'].bestScore, 123);
});
