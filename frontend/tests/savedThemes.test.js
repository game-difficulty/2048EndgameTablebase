import test from 'node:test';
import assert from 'node:assert/strict';

import { normalizeVthTheme, VTH_TILE_VALUES } from '../src/services/preferences/vthThemeFormat.js';

const style = { '--tile-color': '#ffffff', '--tile-background': '#123456', '--tile-shadow-color': '#00000000', '--tile-outline-color': '#ffffff22' };
const validTheme = () => ({ light: Object.fromEntries(VTH_TILE_VALUES.map(value => [value, { ...style }])) });

test('Verse single-mode export accepts an empty alternate mode and lowercase super', () => {
  for (const mode of ['dark', 'light']) {
    const other = mode === 'dark' ? 'light' : 'dark';
    const palette = validTheme().light;
    palette.super = { ...style };
    const normalized = normalizeVthTheme({ [mode]: palette, [other]: {} });
    assert.deepEqual(Object.keys(normalized), [mode]);
    assert.deepEqual(normalized[mode].Super, style);
    assert.equal(normalized[mode]['65536']['--tile-background'], '#123456');
    assert.deepEqual(normalizeVthTheme(JSON.parse(JSON.stringify(normalized))), normalized);
  }
});

test('empty, partial and invalid-color palettes still fail validation', () => {
  assert.throws(() => normalizeVthTheme({ light: {}, dark: {} }), /theme_tiles_missing/);
  assert.throws(() => normalizeVthTheme({ ...validTheme(), dark: { 2: style } }), /theme_tiles_missing/);
  const theme = validTheme();
  theme.light['65536']['--tile-background'] = 'url(evil)';
  assert.throws(() => normalizeVthTheme(theme), /invalid_theme_color/);
});

test('.vth normalization explicitly covers 2 through 65K', () => {
  const theme = normalizeVthTheme(validTheme());
  assert.equal(VTH_TILE_VALUES.length, 16);
  assert.equal(VTH_TILE_VALUES.at(-1), 65536);
  assert.equal(theme.light['65536']['--tile-background'], '#123456');
});

test('.vth normalization does not add support for 131K tiles', () => {
  const theme = validTheme(); theme.light['131072'] = { ...style };
  const normalized = normalizeVthTheme(theme);
  assert.equal(normalized.light['131072'], undefined);
});
