import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const pageSource = await readFile(new URL('../src/human/HumanLeaderboardPage.vue', import.meta.url), 'utf8');
const appSource = await readFile(new URL('../src/human/HumanApp.vue', import.meta.url), 'utf8');
const cssSource = await readFile(new URL('../src/human/human.css', import.meta.url), 'utf8');
const viteSource = await readFile(new URL('../vite.config.js', import.meta.url), 'utf8');
const localAppSource = await readFile(new URL('../../backend/human_play/local_app.py', import.meta.url), 'utf8');

test('full leaderboards use a standalone route and fixed 50-row requests', () => {
  assert.match(appSource, /location\.pathname === '\/leaderboard'/);
  assert.match(appSource, /view === 'leaderboard'/);
  assert.match(pageSource, /page_size:'50'/);
  assert.doesNotMatch(pageSource, /limit=100/);
  assert.match(viteSource, /\/leaderboard/);
  assert.match(localAppSource, /@app\.get\('\/leaderboard'/);
});

test('all agreed leaderboard types and filters are present', () => {
  for (const kind of ['score', 'rating', 'rate32k', 'count', 'strength']) {
    assert.match(pageSource, new RegExp(`id:'${kind}'`));
  }
  assert.match(pageSource, /Players need at least ten 32K games/);
  assert.match(pageSource, /个人主页历史对局/);
  assert.match(pageSource, /started from Profile history/);
  assert.match(pageSource, /当前 B10 线/);
  assert.match(pageSource, /Current B10 line/);
  assert.doesNotMatch(pageSource, /基于 \$\{item\.game_count\} 局/);
  assert.match(pageSource, /filters\.period === 'week'/);
  assert.match(pageSource, /filters\.pattern/);
  assert.match(pageSource, /filters\.target/);
});

test('private strength score stays absent from the leaderboard view', () => {
  assert.doesNotMatch(pageSource, /weighted_score/);
  assert.match(pageSource, /item\.grade/);
  assert.match(pageSource, /item\.mean_goodness_of_fit/);
});

test('full leaderboard keeps long player names inside their column', () => {
  const rule = cssSource.match(/(?:^|\n)\.full-player \{([^}]*)\}/)?.[1] || '';
  assert.match(rule, /min-width:\s*0/);
  assert.match(rule, /overflow:\s*hidden/);
  assert.match(rule, /text-overflow:\s*ellipsis/);
  assert.match(rule, /white-space:\s*nowrap/);
});
