import fs from "node:fs";
import path from "node:path";
import {
  VARIANTS,
  decodeVerseHistorySegment,
  encodeVerseHistorySegmentV2,
} from "./verse-history-codec.mjs";

const ENDPOINT = "https://backend.2048verse.com/leaderboard/user";
const PAGE_SIZE = 50;
const STRATEGIES = Object.freeze([
  { name: "date_desc", sort: "date", desc: true, bit: 1 },
  { name: "date_asc", sort: "date", desc: false, bit: 2 },
  { name: "score_desc", sort: "score", desc: true, bit: 4 },
  { name: "score_asc", sort: "score", desc: false, bit: 8 },
]);
let lastRequestFinishedAt = 0;

function parseArguments(argv) {
  const options = { delayMs: 250, retries: 3, force: false };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--force") options.force = true;
    else if (argument === "--username") options.username = argv[++index];
    else if (argument === "--output-root") options.outputRoot = argv[++index];
    else if (argument === "--delay-ms") options.delayMs = Number(argv[++index]);
    else if (argument === "--retries") options.retries = Number(argv[++index]);
    else throw new Error(`Unknown argument: ${argument}`);
  }
  if (!/^[A-Za-z0-9_-]+$/.test(options.username ?? "")) throw new Error("A safe --username is required");
  if (!options.outputRoot) throw new Error("--output-root is required");
  if (!Number.isInteger(options.delayMs) || options.delayMs < 0 || options.delayMs > 10000) {
    throw new Error("--delay-ms must be an integer between 0 and 10000");
  }
  if (!Number.isInteger(options.retries) || options.retries < 0 || options.retries > 8) {
    throw new Error("--retries must be an integer between 0 and 8");
  }
  return options;
}

const pause = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

function adaptiveRequestInterval(baseDelayMs, expectedTotal) {
  if (!Number.isSafeInteger(expectedTotal)) return baseDelayMs;
  if (expectedTotal >= 5000) return Math.max(baseDelayMs, 1000);
  if (expectedTotal >= 1000) return Math.max(baseDelayMs, 750);
  if (expectedTotal >= 250) return Math.max(baseDelayMs, 500);
  return baseDelayMs;
}

async function waitForRequestSlot(intervalMs) {
  const elapsed = Date.now() - lastRequestFinishedAt;
  if (elapsed < intervalMs) await pause(intervalMs - elapsed);
}

async function fetchJson(url, options, intervalMs) {
  let lastError;
  for (let attempt = 0; attempt <= options.retries; attempt += 1) {
    try {
      await waitForRequestSlot(intervalMs);
      const response = await fetch(url, {
        headers: { accept: "application/json" },
        signal: AbortSignal.timeout(30000),
      });
      if (response.ok) return await response.json();
      if (response.status < 500 && response.status !== 429) {
        throw new Error(`HTTP ${response.status} for ${url}`);
      }
      const retryAfter = Number(response.headers.get("retry-after"));
      const waitMs = Number.isFinite(retryAfter) && retryAfter > 0
        ? retryAfter * 1000
        : 1000 * (2 ** attempt);
      lastError = new Error(`HTTP ${response.status} for ${url}`);
      if (attempt < options.retries) await pause(waitMs);
    } catch (error) {
      lastError = error;
      if (attempt < options.retries) await pause(1000 * (2 ** attempt));
    } finally {
      lastRequestFinishedAt = Date.now();
    }
  }
  throw lastError;
}

function buildUrl(username, variant, strategy, pageNumber) {
  const url = new URL(ENDPOINT);
  url.search = new URLSearchParams({
    username,
    sort: strategy.sort,
    variant,
    page: String(pageNumber),
    desc: String(strategy.desc),
  });
  return url;
}

function tileExponent(value) {
  if (value === 0) return 0;
  const exponent = Math.log2(value);
  if (!Number.isInteger(exponent) || exponent < 1 || exponent > 31) {
    throw new RangeError(`Invalid tile value from Verse: ${value}`);
  }
  return exponent;
}

function normalizeGame(game, variantDefinition) {
  if (!Number.isSafeInteger(game.id) || game.id < 0) throw new RangeError(`Invalid game id: ${game.id}`);
  if (!Number.isSafeInteger(game.score) || game.score < 0) throw new RangeError(`Invalid score for game ${game.id}`);
  const playedAtMilliseconds = Date.parse(game.played_at);
  if (!Number.isSafeInteger(playedAtMilliseconds) || playedAtMilliseconds < 0) {
    throw new RangeError(`Invalid played_at for game ${game.id}: ${game.played_at}`);
  }
  if (game.variant !== variantDefinition.name) {
    throw new Error(`Game ${game.id} reports variant ${game.variant}, expected ${variantDefinition.name}`);
  }
  if (!Array.isArray(game.board) || game.board.length !== variantDefinition.rows
      || game.board.some((row) => !Array.isArray(row) || row.length !== variantDefinition.columns)) {
    throw new RangeError(`Game ${game.id} has the wrong board dimensions`);
  }
  return {
    id: game.id,
    score: game.score,
    playedAtMilliseconds,
    tileExponents: game.board.flat().map(tileExponent),
  };
}

function signature(record) {
  return `${record.score}|${record.playedAtMilliseconds}|${record.tileExponents.join(",")}`;
}

function mergeGames(target, games, variantDefinition) {
  for (const game of games) {
    const record = normalizeGame(game, variantDefinition);
    const existing = target.get(record.id);
    if (existing && signature(existing) !== signature(record)) {
      throw new Error(`Game id ${record.id} returned conflicting contents`);
    }
    if (!existing) target.set(record.id, record);
  }
}

async function readPass(username, variantDefinition, strategy, expectedTotal, records, options) {
  let rawRead = 0;
  let maximumScore = null;
  let pageCount = null;
  for (let pageNumber = 1; pageCount === null || pageNumber <= pageCount; pageNumber += 1) {
    const intervalMs = adaptiveRequestInterval(options.delayMs, expectedTotal);
    const payload = await fetchJson(
      buildUrl(username, variantDefinition.name, strategy, pageNumber), options, intervalMs,
    );
    if (!Number.isSafeInteger(payload.totalGames) || payload.totalGames < 0) {
      throw new Error(`${variantDefinition.name} ${strategy.name} returned an invalid totalGames`);
    }
    if (expectedTotal !== null && payload.totalGames !== expectedTotal) {
      throw new Error(`${variantDefinition.name} totalGames changed from ${expectedTotal} to ${payload.totalGames}`);
    }
    if (!Array.isArray(payload.games)) throw new Error(`${variantDefinition.name} page ${pageNumber} has no games array`);
    if (pageNumber === 1) {
      expectedTotal = payload.totalGames;
      pageCount = Math.ceil(expectedTotal / PAGE_SIZE);
      maximumScore = payload.hs;
    }
    rawRead += payload.games.length;
    mergeGames(records, payload.games, variantDefinition);
    if (pageNumber % 10 === 0 || pageNumber === pageCount) {
      process.stderr.write(
        `${variantDefinition.name} ${strategy.name}: page ${pageNumber}/${pageCount}, raw ${rawRead}, unique ${records.size}/${expectedTotal}\n`,
      );
    }
  }
  if (rawRead !== expectedTotal) {
    throw new Error(`${variantDefinition.name} ${strategy.name} read ${rawRead} rows, expected ${expectedTotal}`);
  }
  return { expectedTotal, rawRead, maximumScore };
}

function isCompleteV2(filePath, expectedVariant) {
  if (!fs.existsSync(filePath)) return false;
  try {
    const decoded = decodeVerseHistorySegment(fs.readFileSync(filePath));
    return decoded.format === "VHS2"
      && decoded.variant === expectedVariant
      && decoded.counts.page_declared === decoded.counts.unique;
  } catch {
    return false;
  }
}

function writeVerifiedSegment(filePath, bytes, expectedVariant) {
  const decoded = decodeVerseHistorySegment(bytes);
  if (decoded.format !== "VHS2" || decoded.variant !== expectedVariant
      || decoded.counts.page_declared !== decoded.counts.unique) {
    throw new Error(`Refusing to save incomplete ${expectedVariant} segment`);
  }
  const temporaryPath = `${filePath}.tmp-${process.pid}`;
  fs.writeFileSync(temporaryPath, bytes);
  try {
    fs.rmSync(filePath, { force: true });
    fs.renameSync(temporaryPath, filePath);
  } finally {
    fs.rmSync(temporaryPath, { force: true });
  }
  return decoded;
}

async function collectVariant(username, variantDefinition, options) {
  const records = new Map();
  let expectedTotal = null;
  let maximumScore = null;
  let rawRead = 0;
  let passMask = 0;

  for (const strategy of STRATEGIES) {
    if (strategy !== STRATEGIES[0] && records.size >= expectedTotal) break;
    const result = await readPass(
      username, variantDefinition, strategy, expectedTotal, records, options,
    );
    expectedTotal = result.expectedTotal;
    maximumScore ??= result.maximumScore;
    rawRead += result.rawRead;
    passMask |= strategy.bit;
  }

  if (records.size !== expectedTotal) {
    throw new Error(
      `${variantDefinition.name} has ${records.size} unique ids after all passes, expected ${expectedTotal}`,
    );
  }
  const actualMaximum = [...records.values()].reduce((maximum, record) => Math.max(maximum, record.score), 0);
  if (actualMaximum !== maximumScore) {
    throw new Error(`${variantDefinition.name} maximum ${actualMaximum} does not match API value ${maximumScore}`);
  }
  return {
    bytes: encodeVerseHistorySegmentV2({
      variant: variantDefinition.name,
      collectedAtMilliseconds: Date.now(),
      pageDeclared: expectedTotal,
      rawRead,
      maximumScore,
      passMask,
      records: [...records.values()],
    }),
    passMask,
  };
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const targetDirectory = path.resolve(options.outputRoot, options.username);
  fs.mkdirSync(targetDirectory, { recursive: true });
  const summaries = {};

  for (const variantDefinition of VARIANTS) {
    const filePath = path.join(targetDirectory, `${variantDefinition.name}.vhs`);
    if (!options.force && isCompleteV2(filePath, variantDefinition.name)) {
      const decoded = decodeVerseHistorySegment(fs.readFileSync(filePath));
      summaries[variantDefinition.name] = { skipped: true, file: filePath, ...decoded.counts };
      process.stderr.write(`Resume: keeping complete VHS2 segment ${variantDefinition.name}\n`);
      continue;
    }
    const { bytes } = await collectVariant(options.username, variantDefinition, options);
    const decoded = writeVerifiedSegment(filePath, bytes, variantDefinition.name);
    summaries[variantDefinition.name] = {
      skipped: false,
      file: filePath,
      bytes: bytes.length,
      retrieval_passes: decoded.retrieval_passes,
      ...decoded.counts,
    };
    process.stderr.write(`Saved ${variantDefinition.name} VHS2 segment (${bytes.length} bytes)\n`);
  }
  process.stdout.write(`${JSON.stringify({ username: options.username, summaries }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
