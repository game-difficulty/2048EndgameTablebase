import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { defaultTileValueClass, resolvePracticeAppearance, tileLabelSize } from '../src/projects/practiceAppearance.js';

test('practice labels keep the main board size tiers across grid widths', () => {
  assert.equal(tileLabelSize(2, 4), '8.0000cqw');
  assert.equal(tileLabelSize(128, 4), '6.6667cqw');
  assert.equal(tileLabelSize(1024, 4), '5.3333cqw');
  assert.equal(tileLabelSize(16384, 4), '4.0000cqw');
  assert.equal(tileLabelSize(2, 3), '10.6667cqw');
  assert.equal(tileLabelSize(2, 5, 1.5), '9.6000cqw');
  assert.equal(tileLabelSize(128, 7), '3.8095cqw');
});

test('signed-in named and custom palettes resolve per-tile backgrounds', () => {
  assert.deepEqual(resolvePracticeAppearance(null), { tileStyles: {}, fontScale: 1 });
  const named = resolvePracticeAppearance({ theme: 'Classic', font_size_factor: 125 });
  assert.equal(named.fontScale, 1.25);
  assert.equal(named.tileStyles[2].backgroundColor, '#eee4da');
  const custom = resolvePracticeAppearance({ use_custom_theme: true, custom_colors: ['#123456'] });
  assert.equal(custom.tileStyles[2].backgroundColor, '#123456');
  assert.equal(custom.tileStyles[2].color, '#f9f6f2');
});

test('saved themes select the page mode and retain their text color', () => {
  const style = background => ({
    '--tile-background': background,
    '--tile-color': '#ffffff',
    '--tile-shadow-color': '#000000',
    '--tile-outline-color': '#111111',
  });
  const saved_theme = { light: { 2: style('#eeeeee') }, dark: { 2: style('#222222') } };
  const dark = resolvePracticeAppearance({ saved_theme }, 'dark');
  assert.equal(dark.tileStyles[2].backgroundColor, '#222222');
  assert.equal(dark.tileStyles[2].color, '#ffffff');
});

test('all practice boards assign a high-value color class without a saved theme', () => {
  for (const value of [512, 1024, 2048]) {
    assert.equal(defaultTileValueClass(value), `value-${value}`);
  }
  assert.equal(defaultTileValueClass(4096), 'value-2048');
  assert.equal(defaultTileValueClass(65536), 'value-2048');
  for (const component of ['PolyominoBoard.vue', 'TournamentBoard.vue']) {
    const source = readFileSync(new URL(`../src/projects/${component}`, import.meta.url), 'utf8');
    for (const value of [512, 1024, 2048]) {
      const rule = source.match(new RegExp(`\\.value-${value}(?:\\s|,|\\{)[^}]*`));
      assert.ok(rule, `${component} is missing a ${value} tile rule`);
      assert.match(rule[0], /(?:--tile-color|background):/, `${component} is missing the ${value} background`);
    }
  }
});
