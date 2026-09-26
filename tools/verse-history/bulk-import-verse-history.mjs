import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { decodeVerseHistorySegment, VARIANTS } from "./verse-history-codec.mjs";

const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const IMPORTER = path.join(SCRIPT_DIRECTORY, "import-verse-history-api.mjs");
const pause = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

function parseArguments(argv) {
  const options = { delayMs: 500, retries: 4, betweenPlayersMs: 3000, limit: Infinity, dryRun: false };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--cohort") options.cohort = argv[++index];
    else if (argument === "--output-root") options.outputRoot = argv[++index];
    else if (argument === "--delay-ms") options.delayMs = Number(argv[++index]);
    else if (argument === "--between-players-ms") options.betweenPlayersMs = Number(argv[++index]);
    else if (argument === "--retries") options.retries = Number(argv[++index]);
    else if (argument === "--limit") options.limit = Number(argv[++index]);
    else if (argument === "--only") options.only = argv[++index];
    else if (argument === "--dry-run") options.dryRun = true;
    else throw new Error(`Unknown argument: ${argument}`);
  }
  if (!options.cohort || !options.outputRoot) throw new Error("--cohort and --output-root are required");
  if (!Number.isInteger(options.delayMs) || options.delayMs < 250 || options.delayMs > 10000) throw new Error("Invalid --delay-ms");
  if (!Number.isInteger(options.betweenPlayersMs) || options.betweenPlayersMs < 0 || options.betweenPlayersMs > 60000) throw new Error("Invalid --between-players-ms");
  if (!(options.limit === Infinity || (Number.isInteger(options.limit) && options.limit > 0))) throw new Error("Invalid --limit");
  return options;
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

function acquireBatchLock(cohortDirectory) {
  const lockPath = path.join(cohortDirectory, "bulk-import.lock");
  if (fs.existsSync(lockPath)) {
    let activePid = null;
    try { activePid = JSON.parse(fs.readFileSync(lockPath, "utf8")).pid; } catch { /* stale or damaged */ }
    if (Number.isInteger(activePid)) {
      try {
        process.kill(activePid, 0);
        throw new Error(`Another bulk import is already running with PID ${activePid}`);
      } catch (error) {
        if (error.code !== "ESRCH") throw error;
      }
    }
    fs.rmSync(lockPath, { force: true });
  }
  const descriptor = fs.openSync(lockPath, "wx");
  fs.writeFileSync(descriptor, `${JSON.stringify({ pid: process.pid, started_at: new Date().toISOString() })}\n`);
  fs.closeSync(descriptor);
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    fs.rmSync(lockPath, { force: true });
  };
  process.on("exit", release);
  return release;
}

function readPlayerSummary(outputRoot, username) {
  const result = { username };
  let totalGames = 0;
  for (const { name: variant } of VARIANTS) {
    const filePath = path.join(outputRoot, username, `${variant}.vhs`);
    if (!fs.existsSync(filePath)) throw new Error(`Missing ${filePath}`);
    const decoded = decodeVerseHistorySegment(fs.readFileSync(filePath));
    if (decoded.format !== "VHS2" || decoded.variant !== variant || decoded.counts.unique !== decoded.counts.page_declared) {
      throw new Error(`Incomplete or invalid ${filePath}`);
    }
    result[variant] = { total_games: decoded.counts.unique, high_score: decoded.maximum_score };
    totalGames += decoded.counts.unique;
  }
  result.total_games = totalGames;
  return result;
}

function writeSummaryTable(filePath, cohortPlayers, summaries, states) {
  const header = ["username", ...VARIANTS.flatMap(({ name }) => [`${name}_total_games`, `${name}_high_score`]), "total_games", "status", "updated_at_utc"];
  const rows = cohortPlayers.map(({ username }) => {
    const summary = summaries.get(username.toLowerCase());
    const state = states.get(username.toLowerCase());
    return [
      username,
      ...VARIANTS.flatMap(({ name }) => [summary?.[name]?.total_games ?? "", summary?.[name]?.high_score ?? ""]),
      summary?.total_games ?? "",
      state?.status ?? "pending",
      state?.updated_at ?? "",
    ];
  });
  writeAtomic(filePath, `${[header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n")}\n`);
}

function runImporter(username, outputRoot, options) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [
      IMPORTER,
      "--username", username,
      "--output-root", outputRoot,
      "--delay-ms", String(options.delayMs),
      "--retries", String(options.retries),
    ], { stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
    let stdout = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { process.stderr.write(`[${username}] ${chunk}`); });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolve(stdout);
      else reject(new Error(`Importer exited with code ${code}`));
    });
  });
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const cohortPath = path.resolve(options.cohort);
  const cohortDirectory = path.dirname(cohortPath);
  const outputRoot = path.resolve(options.outputRoot);
  const releaseLock = acquireBatchLock(cohortDirectory);
  const cohort = JSON.parse(fs.readFileSync(cohortPath, "utf8"));
  if (cohort.format !== "VERSE_LEADERBOARD_COHORT_V1" || !Array.isArray(cohort.players)) throw new Error("Unsupported cohort file");
  fs.mkdirSync(outputRoot, { recursive: true });
  let players = cohort.players;
  if (options.only) players = players.filter(({ username }) => username.toLowerCase() === options.only.toLowerCase());
  players = players.slice(0, options.limit);

  const statePath = path.join(cohortDirectory, "bulk-state.json");
  const tablePath = path.join(cohortDirectory, "players-history-summary.csv");
  let stateDocument = { format: "VERSE_BULK_STATE_V1", cohort: cohortPath, players: {} };
  if (fs.existsSync(statePath)) stateDocument = JSON.parse(fs.readFileSync(statePath, "utf8"));
  const states = new Map(Object.entries(stateDocument.players ?? {}));
  const summaries = new Map();
  for (const { username } of cohort.players) {
    try {
      const summary = readPlayerSummary(outputRoot, username);
      const key = username.toLowerCase();
      summaries.set(key, summary);
      states.set(key, {
        username,
        status: "complete",
        updated_at: states.get(key)?.updated_at ?? new Date().toISOString(),
        total_games: summary.total_games,
      });
    } catch { /* pending */ }
  }
  stateDocument.players = Object.fromEntries(states);
  writeAtomic(statePath, `${JSON.stringify(stateDocument, null, 2)}\n`);
  writeSummaryTable(tablePath, cohort.players, summaries, states);

  if (options.dryRun) {
    process.stdout.write(`${JSON.stringify({ players: players.length, outputRoot, statePath, tablePath, mode: "dry-run" }, null, 2)}\n`);
    releaseLock();
    return;
  }

  let attempted = 0;
  for (const [index, player] of players.entries()) {
    const key = player.username.toLowerCase();
    try {
      const existingSummary = readPlayerSummary(outputRoot, player.username);
      summaries.set(key, existingSummary);
      const updatedAt = states.get(key)?.updated_at ?? new Date().toISOString();
      states.set(key, {
        username: player.username,
        status: "complete",
        updated_at: updatedAt,
        total_games: existingSummary.total_games,
      });
      stateDocument.players = Object.fromEntries(states);
      writeAtomic(statePath, `${JSON.stringify(stateDocument, null, 2)}\n`);
      writeSummaryTable(tablePath, cohort.players, summaries, states);
      process.stderr.write(`Player ${index + 1}/${players.length}: ${player.username} already complete\n`);
      continue;
    } catch { /* incomplete player: resume through the per-variant importer */ }
    const startedAt = new Date().toISOString();
    states.set(key, { username: player.username, status: "running", started_at: startedAt, updated_at: startedAt });
    stateDocument.players = Object.fromEntries(states);
    writeAtomic(statePath, `${JSON.stringify(stateDocument, null, 2)}\n`);
    writeSummaryTable(tablePath, cohort.players, summaries, states);
    process.stderr.write(`Player ${index + 1}/${players.length}: ${player.username}\n`);
    attempted += 1;
    try {
      await runImporter(player.username, outputRoot, options);
      const summary = readPlayerSummary(outputRoot, player.username);
      summaries.set(key, summary);
      const updatedAt = new Date().toISOString();
      states.set(key, { username: player.username, status: "complete", started_at: startedAt, updated_at: updatedAt, total_games: summary.total_games });
    } catch (error) {
      const updatedAt = new Date().toISOString();
      states.set(key, { username: player.username, status: "failed", started_at: startedAt, updated_at: updatedAt, error: String(error.message ?? error) });
      process.stderr.write(`[${player.username}] FAILED: ${error.stack || error}\n`);
    }
    stateDocument.players = Object.fromEntries(states);
    writeAtomic(statePath, `${JSON.stringify(stateDocument, null, 2)}\n`);
    writeSummaryTable(tablePath, cohort.players, summaries, states);

    const totalGames = summaries.get(key)?.total_games ?? 0;
    const volumeCooldown = totalGames >= 5000 ? 10000 : totalGames >= 1000 ? 6000 : options.betweenPlayersMs;
    if (index + 1 < players.length) await pause(Math.max(options.betweenPlayersMs, volumeCooldown));
  }
  const complete = [...states.values()].filter(({ status }) => status === "complete").length;
  const failed = [...states.values()].filter(({ status }) => status === "failed").length;
  process.stdout.write(`${JSON.stringify({ attempted, complete, failed, statePath, tablePath }, null, 2)}\n`);
  releaseLock();
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
