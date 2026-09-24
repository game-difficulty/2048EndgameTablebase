import test from 'node:test';
import assert from 'node:assert/strict';
import { getTileLabelStyle } from '../src/components/tileLabelStyle.js';

test('tile labels stay centered without digit-dependent offsets', () => {
  for (const value of [2, 16, 64, 128, 512, 1024, 8192, 16384, 65536]) {
    const style = getTileLabelStyle({ value });
    assert.equal(style.transform, undefined);
    assert.equal(style.lineHeight, 1);
    assert.equal(style.alignItems, 'center');
    assert.equal(style.justifyContent, 'center');
  }
});

test('centering preserves existing tile font sizes', () => {
  for (const [value, base, scale] of [
    [2, 'small, 2.5rem', 1.2], [64, 'small, 2.5rem', 1.2],
    [128, 'small, 2.5rem', 1], [512, 'small, 2.5rem', 1],
    [1024, 'medium, 2rem', 1], [16384, 'large, 1.5rem', 1],
  ]) {
    assert.equal(getTileLabelStyle({ value }).fontSize,
      `calc(var(--tile-label-${base}) * var(--tile-font-scale, 1) * ${scale})`);
  }
});
