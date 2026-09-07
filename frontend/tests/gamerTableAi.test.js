import assert from 'node:assert/strict';
import test from 'node:test';
import { TableAiCache } from '../src/features/gamer/services/tableAiCache.js';
import { TableDispatcher } from '../src/features/gamer/engine/tableDispatcher.js';
import { createOrdinaryRng, planGamerSpawn, copySpawnRng } from '../src/features/gamer/engine/gamerSpawn.js';

const body = (step = 0) => ({ catalog_version: 'v1', full_pattern: 'L3_256',
  board_codes: [step, ...Array(15).fill(0)], rng_state: [step + 1, 2, 3, 4],
  spawn_rate4: .1, difficulty: 0, random_only: false });
const result = (step) => ({ ...body(step), type: 'result', results: { left: .9 }, dtype: 'float64' });
const tick = () => new Promise((resolve) => setImmediate(resolve));

function controlledTransport() {
  const requests = [];
  const transport = (request, { signal, onResult }) => new Promise((resolve, reject) => {
    requests.push({ request, onResult, resolve, reject });
    signal.addEventListener('abort', () => reject(new Error('Aborted')), { once: true });
  });
  return { requests, transport };
}

test('current result and subsequent nodes can be consumed before the four-node stream finishes', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const first = cache.lookup(body(0));
  assert.equal(requests[0].request.steps, 4);
  requests[0].onResult(result(0));
  assert.deepEqual(await first, result(0));
  const second = cache.lookup(body(1));
  requests[0].onResult(result(1));
  assert.deepEqual(await second, result(1));
  assert.equal(requests.length, 1);
  requests[0].onResult(result(2));
  requests[0].onResult(result(3));
  requests[0].resolve();
  await tick();
  assert.equal(requests.length, 2, 'stream completion refills when only two nodes remain ahead');
  await cache.lookup(body(1));
  assert.equal(requests.length, 2);
  assert.deepEqual(requests[1].request.board_codes, body(3).board_codes);
  assert.equal(requests[1].request.advance_first, true);
  cache.clear(); await tick();
});

test('consuming the stream tail refills without waiting for a new cache miss', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  for (let step = 0; step < 4; step += 1) {
    const pending = cache.lookup(body(step));
    requests[0].onResult(result(step));
    await pending;
  }
  requests[0].resolve(); await tick();
  assert.equal(requests.length, 2);
  assert.equal(requests[1].request.advance_first, true);
  assert.deepEqual(requests[1].request.board_codes, body(3).board_codes);
  const next = cache.lookup(body(4));
  requests[1].onResult(result(4));
  await next;
  assert.equal(requests.length, 2);
  cache.clear(); await tick();
});

test('pausing cancels speculative work but preserves already purchased results', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const pending = cache.lookup(body());
  requests[0].onResult(result(0)); await pending;
  cache.cancelPrefetch(); await tick();
  assert.deepEqual(cache.get(body()), result(0));
  assert.equal(requests.length, 1);
  assert.equal(cache.active, null);
  cache.clear();
  assert.equal(cache.get(body()), null);
});

test('negative results and completed lookahead do not produce endless background batches', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const pending = cache.lookup(body());
  requests[0].onResult({ ...result(0), results: { left: null } });
  requests[0].resolve(); await pending; await tick();
  assert.equal(requests.length, 1);
  cache.clear();
});

test('probing a different candidate retains the previous table route tail', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const first = cache.lookup(body());
  for (let step = 0; step < 4; step += 1) requests[0].onResult(result(step));
  requests[0].resolve(); await first; await tick();
  const other = cache.lookup({ ...body(1), full_pattern: 'free11_1024' });
  requests[1].onResult({ ...result(1), full_pattern: 'free11_1024', results: { left: null } });
  requests[1].resolve(); await other; await tick();
  await cache.lookup(body(1));
  assert.equal(requests.length, 3);
  assert.deepEqual(requests[2].request.board_codes, body(3).board_codes);
  assert.equal(requests[2].request.advance_first, true);
  cache.clear(); await tick();
});

test('clear cancels old work and late responses never enter another account cache', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const pending = cache.lookup(body());
  cache.clear();
  requests[0].onResult(result(0));
  await assert.rejects(pending);
  assert.equal(cache.entries.size, 0);
});

test('a cached decision followed by EvilGen sends no speculative request', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const request = { ...body(), difficulty: 100 };
  const pending = cache.lookup(request);
  requests[0].onResult(result(0)); requests[0].resolve();
  await pending; await tick();
  await cache.lookup(request);
  assert.equal(requests.length, 1);
});

test('cache is memory-only, bounded to 256 entries and expires after five minutes', async () => {
  const { requests, transport } = controlledTransport();
  let now = 0;
  const cache = new TableAiCache({ transport, now: () => now });
  const pending = cache.lookup(body());
  for (let step = 0; step < 260; step += 1) requests[0].onResult(result(step));
  requests[0].resolve();
  await assert.rejects(pending);
  assert.equal(cache.entries.size, 256);
  assert.equal(cache.get(body(0)), null);
  assert.ok(cache.get(body(259)));
  now = 300001;
  assert.equal(cache.get(body(259)), null);
});

test('predicting a spawn never advances the live RNG; ordinary branches can be reseeded', () => {
  const rng = createOrdinaryRng();
  const state = rng.exportState();
  const clone = copySpawnRng(rng);
  const values = [2, 0, 4, 0, ...Array(12).fill(0)];
  assert.deepEqual(planGamerSpawn(values, rng), planGamerSpawn(values, clone));
  assert.deepEqual(rng.exportState(), state);
  assert.notDeepEqual(createOrdinaryRng().exportState(), state);
  const evil = planGamerSpawn(values, rng, { difficulty: 100 });
  assert.equal(evil.evil, true);
  assert.equal(evil.spawn, null);
});

test('variant descriptors and incompatible spawn rates never become Gamer candidates', () => {
  const tables = [
    { pattern: '3x3', fullPattern: '3x3_1024', target: '1024', spawnRate: .1, ai: { compatible: false } },
    { pattern: 'L3', fullPattern: 'L3_256', target: '256', spawnRate: .2,
      ai: { compatible: true, policy_version: 1, large_tiles: 7, free_tiles: 0 } },
  ];
  const dispatcher = new TableDispatcher(tables, .1);
  assert.equal(dispatcher.tables.length, 0);
});

test('table cooldown lasts twenty resets and certain endgames return to search', () => {
  const dispatcher = new TableDispatcher();
  dispatcher.reset([128,128,...Array(14).fill(0)]);
  const candidate = { table: { target:'256',fullPattern:'L3_256' },type:1 };
  assert.equal(dispatcher.accept(candidate, { results: { left:1 },dtype:'uint32' }), 'AI');
  for (let step=0; step<19; step+=1) dispatcher.reset(dispatcher.board);
  assert.equal(dispatcher.cooldowns.has('L3_256'),true);
  dispatcher.reset(dispatcher.board);
  assert.equal(dispatcher.cooldowns.has('L3_256'),false);
});
