import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const source = path => readFileSync(new URL(`../src/${path}`, import.meta.url), 'utf8');
const directions = { H: 'left', J: 'down', K: 'up', L: 'right' };

for (const page of ['Gamer', 'Trainer', 'Tester', 'Notebook']) {
  test(`${page} maps HJKL to left/down/up/right`, () => {
    const text = source(`features/${page.toLowerCase()}/composables/use${page}Session.js`);
    for (const [key, direction] of Object.entries(directions)) {
      if (page === 'Gamer' || page === 'Trainer') {
        assert.match(text, new RegExp(`Key${key}: '${direction}'`));
      } else {
        const action = page === 'Tester' ? `move('${direction}')` : `answerDirection('${direction[0].toUpperCase()}${direction.slice(1)}')`;
        const start = text.indexOf(`=== 'Key${key}'`);
        assert.ok(start >= 0);
        assert.ok(text.slice(start, text.indexOf('}', start)).includes(action));
      }
    }
  });
}

for (const path of [
  'features/minigames/composables/useMinigameSession.js',
  'features/battle/modes/goodness/useGoodnessMatch.js',
  'features/battle/modes/freeGoodness/useFreeGoodnessMatch.js',
]) {
  test(`${path} supports both cases without changing directions`, () => {
    const text = source(path);
    for (const [key, direction] of Object.entries(directions)) {
      for (const letter of [key, key.toLowerCase()]) assert.ok(text.includes(`${letter}: '${direction}'`));
    }
  });
}

test('app focus routing recognizes all HJKL keys', () => {
  const declaration = source('App.vue').match(/const BOARD_HOTKEYS = new Set\(\[([^\]]+)\]\)/)[1];
  for (const key of 'hjklHJKL') assert.ok(declaration.includes(`'${key}'`));
});
