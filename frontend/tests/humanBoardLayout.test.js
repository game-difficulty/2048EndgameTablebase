import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const cssSource = await readFile(new URL('../src/human/human.css', import.meta.url), 'utf8');
const compatSource = await readFile(new URL('../public/compat/render-compat.css', import.meta.url), 'utf8');
const boardSource = await readFile(new URL('../src/human/HumanBoard.vue', import.meta.url), 'utf8');

test('the play board keeps intrinsic square-tile geometry when a shared grid row grows', () => {
  const boardRule = cssSource.match(/(?:^|\n)\.human-board \{([^}]*)\}/)?.[1] || '';
  assert.match(boardRule, /width:\s*100%\s*;/);
  assert.match(boardRule, /align-self:\s*start\s*;/);
  assert.match(compatSource, /\.human-board \{[^}]*align-self:start/);
});

test('mobile layout also applies when a mobile Chrome tab restores a desktop layout viewport', () => {
  assert.match(cssSource, /@media \(max-width:\s*620px\),\s*\(max-device-width:\s*620px\)/);
});

test('the board suppresses delayed browser menus after a manual right-click spawn', () => {
  assert.match(boardSource, /@contextmenu\.prevent/);
  assert.match(boardSource, /@auxclick\.prevent/);
  assert.doesNotMatch(boardSource, /@contextmenu="editable/);
});
