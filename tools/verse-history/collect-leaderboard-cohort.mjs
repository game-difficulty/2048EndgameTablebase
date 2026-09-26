import fs from "node:fs";
import path from "node:path";

const ENDPOINT = "https://backend.2048verse.com/leaderboard/all";
const PAGE_SIZE = 50;
const SELECTIONS = Object.freeze([
  { variant: "4x4", limit: 600 },
  { variant: "3x4", limit: 500 },
  { variant: "3x3", limit: 500 },
  { variant: "2x4", limit: 500 },
]);

const pause = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

function parseArguments(argv) {
  const options = { delayMs: 750, retries: 4 };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--cohort-dir") options.cohortDir = argv[++index];
    else if (argument === "--delay-ms") options.delayMs = Number(argv[++index]);
    else if (argument === "--retries") options.retries = Number(argv[++index]);
    else throw new Error(`Unknown argument: ${argument}`);
  }
  if (!options.cohortDir) throw new Error("--cohort-dir is required");
  if (!Number.isInteger(options.delayMs) || options.delayMs < 250 || options.delayMs > 10000) {
    throw new Error("--delay-ms must be an integer between 250 and 10000");
  }
  return options;
}

async function fetchPage(variant, page, options) {
  const url = new URL(ENDPOINT);
  url.search = new URLSearchParams({ time: "all", variant, page: String(page) });
  let lastError;
  for (let attempt = 0; attempt <= options.retries; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: { accept: "application/json" },
        signal: AbortSignal.timeout(30000),
      });
      if (response.ok) {
        const payload = await response.json();
        if (!Array.isArray(payload.leaderboard)) throw new Error(`${variant} page ${page} has no leaderboard`);
        return payload.leaderboard;
      }
      if (response.status !== 429 && response.status < 500) throw new Error(`HTTP ${response.status} for ${url}`);
      const retryAfter = Number(response.headers.get("retry-after"));
      lastError = new Error(`HTTP ${response.status} for ${url}`);
      if (attempt < options.retries) {
        await pause(Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : 1000 * (2 ** attempt));
      }
    } catch (error) {
      lastError = error;
      if (attempt < options.retries) await pause(1000 * (2 ** attempt));
    }
  }
  throw lastError;
}

function csvCell(value) {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeAtomic(filePath, contents) {
  const temporaryPath = `${filePath}.tmp-${process.pid}`;
  fs.writeFileSync(temporaryPath, contents);
  fs.rmSync(filePath, { force: true });
  fs.renameSync(temporaryPath, filePath);
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const cohortDir = path.resolve(options.cohortDir);
  fs.mkdirSync(cohortDir, { recursive: true });
  const rankings = [];
  let requestCount = 0;

  for (const selection of SELECTIONS) {
    const pages = Math.ceil(selection.limit / PAGE_SIZE);
    let capturedRows = null;
    for (let captureAttempt = 1; captureAttempt <= 4 && !capturedRows; captureAttempt += 1) {
      const candidateRows = [];
      for (let page = 1; page <= pages; page += 1) {
        if (requestCount > 0) await pause(options.delayMs);
        const rows = await fetchPage(selection.variant, page, options);
        requestCount += 1;
        const remaining = selection.limit - candidateRows.length;
        for (const [index, row] of rows.slice(0, remaining).entries()) {
          if (typeof row.username !== "string" || !row.username) throw new Error(`${selection.variant} page ${page} has an invalid username`);
          if (!Number.isSafeInteger(row.score) || row.score < 0) throw new Error(`${selection.variant} page ${page} has an invalid score`);
          candidateRows.push({
            variant: selection.variant,
            rank: (page - 1) * PAGE_SIZE + index + 1,
            username: row.username,
            score: row.score,
            played_at: row.played_at,
          });
        }
        process.stderr.write(`${selection.variant} pass ${captureAttempt}: page ${page}/${pages}, ${Math.min(page * PAGE_SIZE, selection.limit)}/${selection.limit}\n`);
      }
      const uniqueNames = new Set(candidateRows.map(({ username }) => username.toLowerCase()));
      if (candidateRows.length === selection.limit) {
        if (uniqueNames.size < candidateRows.length) {
          process.stderr.write(
            `${selection.variant}: ${candidateRows.length} ranks contain ${uniqueNames.size} unique usernames; duplicates will be removed from the player queue\n`,
          );
        }
        capturedRows = candidateRows;
      } else if (captureAttempt < 4) {
        process.stderr.write(
          `${selection.variant} returned only ${candidateRows.length}/${selection.limit} ranks during pass ${captureAttempt}; retrying after 5s\n`,
        );
        await pause(5000);
      } else {
        throw new Error(
          `${selection.variant} remained incomplete after 4 passes: ${candidateRows.length}/${selection.limit} ranks`,
        );
      }
    }
    rankings.push(...capturedRows);
  }

  const playersByKey = new Map();
  for (const ranking of rankings) {
    const key = ranking.username.toLowerCase();
    let player = playersByKey.get(key);
    if (!player) {
      player = { username: ranking.username, username_key: key, selected_by: {} };
      playersByKey.set(key, player);
    }
    const existingSelection = player.selected_by[ranking.variant];
    if (!existingSelection || ranking.rank < existingSelection.rank) {
      player.selected_by[ranking.variant] = { rank: ranking.rank, score: ranking.score };
    }
  }
  const players = [...playersByKey.values()].sort((left, right) => {
    const leftBest = Math.min(...Object.values(left.selected_by).map(({ rank }) => rank));
    const rightBest = Math.min(...Object.values(right.selected_by).map(({ rank }) => rank));
    return leftBest - rightBest || left.username.localeCompare(right.username, "en", { sensitivity: "base" });
  });
  const collectedAt = new Date().toISOString();
  const variantStats = Object.fromEntries(SELECTIONS.map(({ variant, limit }) => {
    const rows = rankings.filter((row) => row.variant === variant);
    const uniquePlayers = new Set(rows.map(({ username }) => username.toLowerCase())).size;
    return [variant, { requested_ranks: limit, raw_ranks: rows.length, unique_players: uniquePlayers, duplicate_ranks: rows.length - uniquePlayers }];
  }));
  const manifest = {
    format: "VERSE_LEADERBOARD_COHORT_V1",
    collected_at: collectedAt,
    source: "https://2048verse.com/leaderboard/<variant>/all",
    selection: Object.fromEntries(SELECTIONS.map(({ variant, limit }) => [variant, limit])),
    variant_stats: variantStats,
    requests: requestCount,
    ranked_entries: rankings.length,
    unique_players: players.length,
    players,
  };
  writeAtomic(path.join(cohortDir, "cohort.json"), `${JSON.stringify(manifest, null, 2)}\n`);

  const rankingHeader = ["variant", "rank", "username", "score", "played_at_utc"];
  const rankingCsv = [rankingHeader, ...rankings.map((row) => [row.variant, row.rank, row.username, row.score, row.played_at])]
    .map((row) => row.map(csvCell).join(",")).join("\n");
  writeAtomic(path.join(cohortDir, "leaderboard-selection.csv"), `${rankingCsv}\n`);

  const selectionHeader = ["username", ...SELECTIONS.flatMap(({ variant }) => [`${variant}_rank`, `${variant}_leaderboard_score`])];
  const selectionCsv = [selectionHeader, ...players.map((player) => [
    player.username,
    ...SELECTIONS.flatMap(({ variant }) => [player.selected_by[variant]?.rank ?? "", player.selected_by[variant]?.score ?? ""]),
  ])].map((row) => row.map(csvCell).join(",")).join("\n");
  writeAtomic(path.join(cohortDir, "players-selected.csv"), `${selectionCsv}\n`);

  process.stdout.write(`${JSON.stringify({ cohortDir, collectedAt, rankedEntries: rankings.length, uniquePlayers: players.length, requestCount }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
