import assert from 'node:assert/strict';
import test from 'node:test';

import {
  TablebaseResultCache,
  TABLEBASE_RESULT_CACHE_MAX_ENTRIES,
  TABLEBASE_RESULT_CACHE_TTL_MS,
  installTablebaseResultCacheAuthListener,
  normalizeTablebaseResult,
  tablebaseResultCache,
} from '../src/services/tablebases/tablebaseResultCache.js';
import {
  fetchTablebaseCatalog,
  getCatalogVersion,
} from '../src/services/tablebases/catalogClient.js';

const key = (boardHex, catalogVersion = 'v1') => ({
  catalogVersion,
  fullPattern: 'L3_256',
  boardHex,
});

test('uses the production capacity and TTL defaults', () => {
  assert.equal(TABLEBASE_RESULT_CACHE_MAX_ENTRIES, 256);
  assert.equal(TABLEBASE_RESULT_CACHE_TTL_MS, 5 * 60 * 1000);
});

test('normalizes positive and negative server results', () => {
  assert.deepEqual(normalizeTablebaseResult({
    dtype: 'uint32',
    results: { LEFT: 0.75, right: null, down: Number.NaN },
  }), {
    found: true,
    dtype: 'uint32',
    results: { left: 0.75, right: null, down: null },
  });
  assert.deepEqual(normalizeTablebaseResult({
    found: false,
    dtype: 'uint64',
    results: { left: 0.1 },
  }), {
    found: false,
    dtype: 'uint64',
    results: {},
  });
});

test('expires entries after the five-minute TTL', () => {
  let now = 0;
  const cache = new TablebaseResultCache({ now: () => now });
  cache.set(key('1'), { results: { left: 0.5 }, dtype: 'uint32' });
  now = 5 * 60 * 1000 - 1;
  assert.equal(cache.get(key('1'))?.results.left, 0.5);
  now = 5 * 60 * 1000;
  assert.equal(cache.get(key('1')), null);
});

test('evicts the least recently used entry', () => {
  const cache = new TablebaseResultCache({ maxEntries: 2 });
  cache.set(key('1'), { results: { left: 1 } });
  cache.set(key('2'), { results: { left: 2 } });
  cache.get(key('1'));
  cache.set(key('3'), { results: { left: 3 } });
  assert.equal(cache.get(key('2')), null);
  assert.equal(cache.get(key('1'))?.results.left, 1);
  assert.equal(cache.get(key('3'))?.results.left, 3);
});

test('clear removes positive and cached negative entries', () => {
  const cache = new TablebaseResultCache();
  cache.set(key('1'), { results: { left: 1 } });
  cache.set(key('2'), { found: false, results: {} });
  assert.equal(cache.get(key('2'))?.found, false);
  cache.clear();
  assert.equal(cache.get(key('1')), null);
  assert.equal(cache.get(key('2')), null);
});

test('switching full patterns clears the previous tablebase scope', () => {
  const cache = new TablebaseResultCache();
  cache.set(key('1'), { results: { left: 1 } });
  cache.set({ ...key('2'), fullPattern: 'free10_256' }, { results: { right: 2 } });
  assert.equal(cache.get(key('1')), null);
});

test('auth-changed clears the shared cache', () => {
  const events = new EventTarget();
  installTablebaseResultCacheAuthListener(events);
  tablebaseResultCache.set(key('auth'), { results: { up: 0.25 } });
  events.dispatchEvent(new Event('auth-changed'));
  assert.equal(tablebaseResultCache.get(key('auth')), null);
});

test('catalog version changes clear cache and preserve array callers', async (context) => {
  const originalFetch = globalThis.fetch;
  const originalWindow = globalThis.window;
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalWindow === undefined) {
      delete globalThis.window;
    } else {
      globalThis.window = originalWindow;
    }
  });
  globalThis.window = { location: { href: 'https://2048tables.online/' } };

  const payloads = [
    { catalog_version: 'catalog-a', tables: [{ pattern: 'L3', target: 256, full_pattern: 'L3_256' }] },
    { catalog_version: 'catalog-b', tables: [{ pattern: 'L3', target: 256, full_pattern: 'L3_256' }] },
  ];
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => payloads.shift(),
  });

  const first = await fetchTablebaseCatalog();
  assert.equal(Array.isArray(first), true);
  assert.equal(first.catalogVersion, 'catalog-a');
  assert.equal(Object.keys(first).includes('catalogVersion'), false);
  tablebaseResultCache.set(key('version', getCatalogVersion()), { results: { left: 0.8 } });

  const second = await fetchTablebaseCatalog();
  assert.equal(second.catalogVersion, 'catalog-b');
  assert.equal(getCatalogVersion(), 'catalog-b');
  assert.equal(tablebaseResultCache.get(key('version', 'catalog-a')), null);
});
