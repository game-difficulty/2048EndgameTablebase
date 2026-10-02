import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('live cargo fits its full seven-row stage without height compression', () => {
  const live = readFileSync(new URL('../src/live/content/StreamProjectView.vue', import.meta.url), 'utf8');
  const rule = live.match(/\.stream-board-area>:deep\(\.cargo-stage\)\{([^}]+)\}/)?.[1];
  assert.ok(rule);
  assert.match(rule, /width:min\(100%,330px,calc\(100cqh \* 4 \/ 7\)\)/);
  assert.match(rule, /flex-shrink:0/);
  assert.doesNotMatch(rule, /(?:max-)?height:/);
  const board = readFileSync(new URL('../../competition/frontend/src/projects/CargoBoard.vue', import.meta.url), 'utf8');
  assert.match(board, /\.cargo-stage\{[^}]*aspect-ratio:4\/7/);
});
