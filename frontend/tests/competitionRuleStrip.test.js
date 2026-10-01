import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../src/live/content/CompetitionMatchContent.vue', import.meta.url), 'utf8');
const rule = selector => source.match(new RegExp(`${selector.replaceAll('.', '\\.')}\\{([^}]+)\\}`))[1];

test('broadcast rule footer uses readable text and wraps instead of truncating', () => {
  const strip = rule('.game-rule-strip');
  assert.match(strip, /font-size:24px/);
  assert.match(strip, /line-height:1\.5/);
  assert.match(strip, /grid-column:1\/-1/);
  const text = rule('.game-rule-strip span');
  assert.match(text, /white-space:normal/);
  assert.match(text, /overflow-wrap:anywhere/);
  assert.doesNotMatch(text, /ellipsis|overflow:hidden|nowrap/);
});
