import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const appSource = await readFile(new URL('../src/human/HumanApp.vue', import.meta.url), 'utf8');
const cssSource = await readFile(new URL('../src/human/human.css', import.meta.url), 'utf8');

test('game sidebar leaderboard shows only username and score while preserving links', () => {
  const sidebar = appSource.slice(
    appSource.indexOf('<div class="rank-list"'),
    appSource.indexOf('<div v-if="boardLoading" class="ranking-status"'),
  );

  assert.doesNotMatch(sidebar, /最大块|回放 ↗|无回放/);
  assert.equal((sidebar.match(/@click="openPlayer\(/g) || []).length, 2);
  assert.equal((sidebar.match(/@click="openReplay\(/g) || []).length, 2);
  assert.equal((sidebar.match(/:disabled="![^\"]+\.has_replay"/g) || []).length, 2);
});

test('long leaderboard usernames shrink before truncating beside the non-shrinking score column', () => {
  const nameRule = cssSource.match(/(?:^|\n)\.rank-name \{([^}]*)\}/)?.[1] || '';
  const nameTextRule = cssSource.match(/(?:^|\n)\.rank-name strong \{([^}]*)\}/)?.[1] || '';
  const scoreRule = cssSource.match(/(?:^|\n)\.rank-score \{([^}]*)\}/)?.[1] || '';
  const compactScoreRule = cssSource.match(/\.ranking-body \.rank-score \{([^}]*)\}/)?.[1] || '';

  assert.match(nameRule, /flex:\s*1\s*;/);
  assert.match(nameRule, /min-width:\s*0\s*;/);
  assert.match(nameTextRule, /text-overflow:\s*ellipsis\s*;/);
  assert.match(nameTextRule, /white-space:\s*nowrap\s*;/);
  assert.match(scoreRule, /flex:\s*0\s+0\s+auto\s*;/);
  assert.match(compactScoreRule, /white-space:\s*nowrap\s*;/);
  assert.match(appSource, /v-fit-ranking-name/);
  assert.match(cssSource, /grid-template-columns:\s*minmax\(0, 210px\)\s+minmax\(0, 500px\)\s+minmax\(0, 273px\)/);
});
