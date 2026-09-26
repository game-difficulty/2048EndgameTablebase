import assert from "node:assert/strict";
import test from "node:test";
import { VARIANTS, decodeVerseHistorySegment, encodeVerseHistorySegmentV2 } from "./verse-history-codec.mjs";
import { INCREMENTAL_CUTOFF_MS, refreshVariant } from "./refresh-verse-history-api.mjs";

const variant = VARIANTS[0];
const oldBase = INCREMENTAL_CUTOFF_MS - 2 * 86400000;
const newBase = INCREMENTAL_CUTOFF_MS + 86400000;
const board = Array.from({ length: variant.rows }, (_, row) =>
  Array.from({ length: variant.columns }, (_, column) => row === 0 && column === 0 ? 2 : 0));
const makeRecord = (id, playedAtMilliseconds, score = id * 4) =>
  ({ id, playedAtMilliseconds, score, tileExponents: board.flat().map((tile) => tile ? 1 : 0) });
const old = Array.from({ length: 53 }, (_, index) => makeRecord(index + 1, oldBase + index * 60000));
const newGames = [makeRecord(54, newBase), makeRecord(55, newBase)];
const current = [...old, ...newGames];

function cache(records) {
  return Buffer.from(encodeVerseHistorySegmentV2({ variant: variant.name,
    collectedAtMilliseconds: INCREMENTAL_CUTOFF_MS, pageDeclared: records.length,
    rawRead: records.length, maximumScore: Math.max(0, ...records.map((row) => row.score)),
    passMask: 1, records }));
}

function fixtureFetcher(records, calls) {
  return async (url) => {
    const query = new URL(url).searchParams;
    calls.push({ sort: query.get("sort"), desc: query.get("desc"), page: Number(query.get("page")) });
    const descending = query.get("desc") === "true";
    const sorted = [...records].sort((a, b) => query.get("sort") === "date"
      ? (a.playedAtMilliseconds - b.playedAtMilliseconds || a.id - b.id) * (descending ? -1 : 1)
      : (a.score - b.score || a.id - b.id) * (descending ? -1 : 1));
    const page = Number(query.get("page"));
    const games = sorted.slice((page - 1) * 50, page * 50).map((record) => ({
      id: record.id, score: record.score, variant: variant.name,
      played_at: new Date(record.playedAtMilliseconds).toISOString(), board,
    }));
    return { ok: true, json: async () => ({ totalGames: records.length,
      hs: Math.max(0, ...records.map((row) => row.score)), games }) };
  };
}

test("cached complete history fetches only newest date page through September 24 and merges by id", async () => {
  const calls = [];
  const result = await refreshVariant("player", variant, cache(old), {
    fetchImpl: fixtureFetcher(current, calls), delayMs: 0, retries: 0,
    nowMs: newBase + 1000,
  });
  const decoded = decodeVerseHistorySegment(result.bytes);
  assert.equal(result.added, 2);
  assert.equal(result.fallback, false);
  assert.equal(decoded.counts.unique, 55);
  assert.equal(decoded.counts.page_declared, 55);
  assert.deepEqual(calls, [{ sort: "date", desc: "true", page: 1 }]);
  assert.deepEqual(decoded.records.slice(-2).map((record) => record.id), [54, 55]);
});

test("count gap outside the cutoff triggers a full recheck without losing cached records", async () => {
  const calls = [];
  const result = await refreshVariant("player", variant, cache(old.filter((row) => row.id !== 1)), {
    fetchImpl: fixtureFetcher(current, calls), delayMs: 0, retries: 0,
  });
  assert.equal(result.fallback, true);
  assert.equal(result.added, 3);
  assert.equal(decodeVerseHistorySegment(result.bytes).counts.unique, 55);
  assert.ok(calls.some((call) => call.page === 2));
});

test("a conflicting cached game id is rejected instead of overwritten", async () => {
  const cached = cache([makeRecord(54, newBase, 999)]);
  await assert.rejects(refreshVariant("player", variant, cached, {
    fetchImpl: fixtureFetcher(current, []), delayMs: 0, retries: 0,
  }), /verse_record_conflict:54/);
});

test("a stable remote count decrease retains removed cached ids after a full recheck", async () => {
  const calls = [];
  const remote = old.slice(4);
  const result = await refreshVariant("player", variant, cache(old), {
    fetchImpl: fixtureFetcher(remote, calls), delayMs: 0, retries: 0,
  });
  const decoded = decodeVerseHistorySegment(result.bytes);
  assert.equal(result.fallback, true);
  assert.equal(result.count, old.length);
  assert.equal(result.added, 0);
  assert.equal(result.remoteTotal, remote.length);
  assert.equal(result.remoteRemoved, 4);
  assert.deepEqual(result.remoteRemovedIds, [1, 2, 3, 4]);
  assert.equal(decoded.counts.page_declared, old.length);
  assert.deepEqual(decoded.records.map((record) => record.id), old.map((record) => record.id));
  assert.ok(calls.some((call) => call.sort === "date" && call.desc === "true"));
});

test("September 24 boundary is included and older page entries are not lost", async () => {
  const before = makeRecord(1, INCREMENTAL_CUTOFF_MS - 1);
  const boundary = makeRecord(2, INCREMENTAL_CUTOFF_MS);
  const result = await refreshVariant("player", variant, cache([before]), {
    fetchImpl: fixtureFetcher([before, boundary], []), delayMs: 0, retries: 0,
  });
  assert.equal(result.added, 1);
  assert.deepEqual(decodeVerseHistorySegment(result.bytes).records.map((row) => row.id), [1, 2]);
});

test("unstable date pages trigger a full recheck instead of publishing a partial merge", async () => {
  const calls = [];
  let pageTwoSeen = false;
  const recentOnly = current.map((row) => ({ ...row,
    playedAtMilliseconds: newBase + row.id * 1000 }));
  const stable = fixtureFetcher(recentOnly, calls);
  const fetchImpl = async (url) => {
    const response = await stable(url);
    const query = new URL(url).searchParams;
    if (!pageTwoSeen && query.get("sort") === "date" && query.get("desc") === "true"
        && query.get("page") === "2") {
      pageTwoSeen = true;
      return { ok: true, json: async () => ({ ...(await response.json()), totalGames: 56 }) };
    }
    return response;
  };
  const result = await refreshVariant("player", variant, cache(recentOnly.slice(0, 53)), {
    fetchImpl, delayMs: 0, retries: 0,
  });
  assert.equal(result.count, 55);
  assert.equal(result.fallback, true);
  assert.ok(calls.some((call) => call.page === 2));
});
