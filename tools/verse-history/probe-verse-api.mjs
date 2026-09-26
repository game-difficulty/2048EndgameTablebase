const [, , username, variant = "3x3", sort = "score"] = process.argv;
if (!username) {
  process.stderr.write("Usage: node probe-verse-api.mjs <username> [variant] [score|date]\n");
  process.exit(2);
}

const endpoint = "https://backend.2048verse.com/leaderboard/user";
const pause = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function getPage(page) {
  const url = new URL(endpoint);
  url.search = new URLSearchParams({
    username,
    sort,
    variant,
    page: String(page),
    desc: "true",
  });
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Page ${page} returned HTTP ${response.status}`);
  return response.json();
}

const startedAt = Date.now();
const first = await getPage(1);
const expected = first.totalGames;
const pageCount = Math.ceil(expected / 50);
const games = [...first.games];

for (let page = 2; page <= pageCount; page += 1) {
  await pause(250);
  const payload = await getPage(page);
  if (payload.totalGames !== expected) {
    throw new Error(`Page ${page} total changed from ${expected} to ${payload.totalGames}`);
  }
  games.push(...payload.games);
  if (page % 10 === 0 || page === pageCount) {
    process.stderr.write(`page ${page}/${pageCount}, rows ${games.length}/${expected}\n`);
  }
}

const boardKey = (game) => game.board.flat().join(",");
const fullKey = (game) => `${game.score}|${game.played_at}|${boardKey(game)}`;
const secondKey = (game) => `${game.score}|${game.played_at.replace(/\.\d{3}Z$/, "Z")}|${boardKey(game)}`;
const uniqueIds = new Set(games.map((game) => game.id));
const uniqueFullRecords = new Set(games.map(fullKey));
const uniqueSecondRecords = new Set(games.map(secondKey));

process.stdout.write(`${JSON.stringify({
  username,
  variant,
  sort,
  expected,
  pages: pageCount,
  rows: games.length,
  unique_ids: uniqueIds.size,
  duplicate_ids: games.length - uniqueIds.size,
  unique_millisecond_records: uniqueFullRecords.size,
  duplicate_millisecond_records: games.length - uniqueFullRecords.size,
  unique_second_records: uniqueSecondRecords.size,
  duplicate_second_records: games.length - uniqueSecondRecords.size,
  elapsed_seconds: Number(((Date.now() - startedAt) / 1000).toFixed(3)),
}, null, 2)}\n`);
