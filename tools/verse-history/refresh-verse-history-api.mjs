import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { VARIANTS, decodeVerseHistorySegment, encodeVerseHistorySegmentV2 } from "./verse-history-codec.mjs";

const ENDPOINT = "https://backend.2048verse.com/leaderboard/user";
const PAGE_SIZE = 50;
const PASSES = [
  { sort: "date", desc: true, bit: 1 },
  { sort: "date", desc: false, bit: 2 },
  { sort: "score", desc: true, bit: 4 },
  { sort: "score", desc: false, bit: 8 },
];
// Include the whole September 24 calendar day in the project's UTC+8 timezone.
export const INCREMENTAL_CUTOFF_MS = Date.parse("2026-09-23T16:00:00.000Z");

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const signature = (record) => `${record.playedAtMilliseconds}|${record.score}|${record.tileExponents.join(",")}`;

function normalized(game, variant) {
  if (!Number.isSafeInteger(game.id) || game.id < 0 ||
      !Number.isSafeInteger(game.score) || game.score < 0 || game.variant !== variant.name) {
    throw new Error("verse_game_invalid");
  }
  const playedAtMilliseconds = Date.parse(game.played_at);
  if (!Number.isSafeInteger(playedAtMilliseconds) || playedAtMilliseconds < 0 ||
      !Array.isArray(game.board) || game.board.length !== variant.rows ||
      game.board.some((row) => !Array.isArray(row) || row.length !== variant.columns)) {
    throw new Error("verse_game_invalid");
  }
  const tileExponents = game.board.flat().map((tile) => {
    if (tile === 0) return 0;
    const exponent = Math.log2(tile);
    if (!Number.isInteger(exponent) || exponent < 1 || exponent > 31) throw new Error("verse_tile_invalid");
    return exponent;
  });
  return { id: game.id, playedAtMilliseconds, score: game.score, tileExponents };
}

function fromCache(record) {
  return { id: record.id, playedAtMilliseconds: record.played_at_milliseconds,
    score: record.score, tileExponents: record.tile_exponents };
}

function merge(records, record) {
  const prior = records.get(record.id);
  if (prior && signature(prior) !== signature(record)) throw new Error(`verse_record_conflict:${record.id}`);
  records.set(record.id, record);
}

function expectedPageLength(total, page) {
  return Math.min(PAGE_SIZE, Math.max(0, total - (page - 1) * PAGE_SIZE));
}

async function collectPass(username, variant, pass, options, records, stopAtCutoff = false) {
  let total = null, highScore = null, rawRead = 0, previousTime = null;
  const url = new URL(ENDPOINT);
  for (let page = 1; total === null || page <= Math.max(1, Math.ceil(total / PAGE_SIZE)); page += 1) {
    url.search = new URLSearchParams({ username, sort: pass.sort, variant: variant.name,
      page: String(page), desc: String(pass.desc) });
    let payload;
    for (let attempt = 0; attempt <= options.retries; attempt += 1) {
      if (options.delayMs && options.lastRequestAt) {
        const interval = total >= 5000 ? 1000 : total >= 1000 ? 750 : total >= 250 ? 500 : options.delayMs;
        await pause(Math.max(0, Math.max(options.delayMs, interval) - (Date.now() - options.lastRequestAt)));
      }
      try {
        const response = await options.fetchImpl(url, { headers: { accept: "application/json" },
          signal: AbortSignal.timeout(30000) });
        if (response.ok) { payload = await response.json(); break; }
        if (response.status < 500 && response.status !== 429) throw new Error(`verse_http_${response.status}`);
        if (attempt === options.retries) throw new Error(`verse_http_${response.status}`);
        const retryAfter = Number(response.headers.get("retry-after"));
        await pause(Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : 1000 * 2 ** attempt);
      } catch (error) {
        if (attempt === options.retries || /^verse_http_4(?!29)/.test(error.message)) throw error;
        await pause(1000 * 2 ** attempt);
      } finally {
        options.lastRequestAt = Date.now();
      }
    }
    if (!Number.isSafeInteger(payload?.totalGames) || payload.totalGames < 0 ||
        !Number.isSafeInteger(payload.hs) || payload.hs < 0 || !Array.isArray(payload.games)) {
      throw new Error("verse_page_invalid");
    }
    if (total === null) { total = payload.totalGames; highScore = payload.hs; }
    if (payload.totalGames !== total || payload.hs !== highScore ||
        payload.games.length !== expectedPageLength(total, page)) throw new Error("verse_page_drift");
    let crossedCutoff = false;
    for (const game of payload.games) {
      const record = normalized(game, variant);
      if (pass.sort === "date" && previousTime !== null &&
          (pass.desc ? record.playedAtMilliseconds > previousTime : record.playedAtMilliseconds < previousTime)) {
        throw new Error("verse_date_order_changed");
      }
      previousTime = record.playedAtMilliseconds;
      if (record.playedAtMilliseconds < options.cutoffMs) crossedCutoff = true;
      merge(records, record);
    }
    rawRead += payload.games.length;
    if (stopAtCutoff && crossedCutoff) break;
  }
  return { total, highScore, rawRead };
}

function certified(records, total, highScore) {
  return records.size === total &&
    [...records.values()].reduce((best, row) => Math.max(best, row.score), 0) === highScore;
}

export async function refreshVariant(username, variant, cachedBytes, config = {}) {
  const cached = decodeVerseHistorySegment(cachedBytes);
  if (cached.format !== "VHS2" || cached.variant !== variant.name ||
      cached.counts.page_declared !== cached.counts.unique) throw new Error("verse_cache_incomplete");
  const cache = new Map();
  for (const row of cached.records) merge(cache, fromCache(row));
  const options = { fetchImpl: config.fetchImpl || fetch, delayMs: config.delayMs ?? 500,
    retries: config.retries ?? 4, cutoffMs: config.cutoffMs ?? INCREMENTAL_CUTOFF_MS,
    lastRequestAt: 0 };
  const merged = new Map(cache);
  let recent = null;
  try {
    recent = await collectPass(username, variant, PASSES[0], options, merged, true);
  } catch (error) {
    if (!['verse_page_drift', 'verse_date_order_changed'].includes(error.message)) throw error;
  }
  let records = merged, rawRead = cache.size + (recent?.rawRead || 0);
  let passMask = cached.pass_mask | 1;
  let fallback = false;
  let remoteRemovedIds = [];
  if (!recent || !certified(merged, recent.total, recent.highScore)) {
    fallback = true;
    const remoteRecords = new Map();
    rawRead = cache.size;
    passMask = 0;
    let expectedTotal = recent?.total ?? null;
    let highScore = recent?.highScore ?? null;
    for (const pass of PASSES) {
      const full = await collectPass(username, variant, pass, options, remoteRecords);
      if (expectedTotal === null) { expectedTotal = full.total; highScore = full.highScore; }
      if (full.total !== expectedTotal || full.highScore !== highScore) {
        throw new Error("verse_page_drift");
      }
      rawRead += full.rawRead;
      passMask |= pass.bit;
      if (remoteRecords.size >= expectedTotal) break;
    }
    if (!certified(remoteRecords, expectedTotal, highScore)) throw new Error("verse_full_recheck_conflict");
    records = new Map(cache);
    for (const record of remoteRecords.values()) merge(records, record);
    remoteRemovedIds = [...cache.keys()].filter((id) => !remoteRecords.has(id)).sort((a, b) => a - b);
    recent = { total: expectedTotal, highScore, rawRead: recent?.rawRead || 0 };
  }
  const maximumScore = [...records.values()].reduce((best, row) => Math.max(best, row.score), 0);
  const bytes = Buffer.from(encodeVerseHistorySegmentV2({ variant: variant.name,
    collectedAtMilliseconds: config.nowMs ?? Date.now(), pageDeclared: records.size,
    rawRead, maximumScore, passMask, records: [...records.values()] }));
  const verified = decodeVerseHistorySegment(bytes);
  if (verified.format !== "VHS2" || verified.counts.unique !== records.size) throw new Error("verse_refresh_invalid");
  return { bytes, count: records.size, added: records.size - cache.size,
    remoteTotal: recent.total, remoteRemoved: remoteRemovedIds.length, remoteRemovedIds,
    fallback, pagesRead: Math.ceil(recent.rawRead / PAGE_SIZE) };
}

function directoryFor(root, username) {
  const matches = fs.readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.toLowerCase() === username.toLowerCase());
  if (matches.length !== 1) throw new Error("verse_cache_directory_missing_or_ambiguous");
  return path.join(root, matches[0].name);
}

async function main() {
  const args = process.argv.slice(2);
  const values = Object.fromEntries(Array.from({ length: args.length / 2 }, (_, index) =>
    [args[index * 2], args[index * 2 + 1]]));
  const username = values["--username"];
  const root = values["--output-root"];
  if (args.length % 2 || Object.keys(values).some((key) => !["--username", "--output-root", "--delay-ms", "--retries"].includes(key)) ||
      !/^[A-Za-z0-9_-]{1,64}$/.test(username || "") || !root) throw new Error("verse_refresh_arguments_invalid");
  const delayMs = Number(values["--delay-ms"] ?? 500);
  const retries = Number(values["--retries"] ?? 4);
  if (!Number.isInteger(delayMs) || delayMs < 0 || delayMs > 10000 ||
      !Number.isInteger(retries) || retries < 0 || retries > 8) throw new Error("verse_refresh_arguments_invalid");
  const directory = directoryFor(path.resolve(root), username);
  const prepared = [];
  for (const variant of VARIANTS) {
    const file = path.join(directory, `${variant.name}.vhs`);
    const result = await refreshVariant(username, variant, fs.readFileSync(file), { delayMs, retries });
    prepared.push({ file, ...result });
    process.stderr.write(`${variant.name}: ${result.count} records, +${result.added}, ${result.pagesRead} recent pages${result.fallback ? ", full recheck" : ""}${result.remoteRemoved ? `, ${result.remoteRemoved} retained records removed remotely` : ""}\n`);
  }
  const staged = [];
  try {
    for (const [index, item] of prepared.entries()) {
      const temporary = `${item.file}.refresh-${process.pid}-${index}`;
      fs.writeFileSync(temporary, item.bytes);
      staged.push({ temporary, file: item.file });
    }
    for (const item of staged) fs.renameSync(item.temporary, item.file);
  } finally {
    for (const item of staged) fs.rmSync(item.temporary, { force: true });
  }
  process.stdout.write(`${JSON.stringify({ username, cutoff: new Date(INCREMENTAL_CUTOFF_MS).toISOString(),
    variants: prepared.map(({ file, count, added, remoteTotal, remoteRemoved,
      remoteRemovedIds, fallback }) => ({ file, count, added, remoteTotal,
      remoteRemoved, remoteRemovedIds, fallback })) })}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => { process.stderr.write(`${error.stack || error}\n`); process.exitCode = 1; });
}
