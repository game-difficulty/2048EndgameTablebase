import test from 'node:test';
import assert from 'node:assert/strict';
import { fittedFontSize } from '../src/human/fitSingleLineText.js';

test('single-line fitting keeps short names at the default size', () => {
  assert.equal(fittedFontSize(13, 140, 100), 13);
});

test('single-line fitting shrinks long names before reaching the ellipsis floor', () => {
  assert.ok(fittedFontSize(13, 140, 200) < 13);
  assert.ok(fittedFontSize(13, 140, 200) > 6.5);
  assert.equal(fittedFontSize(13, 140, 1000), 6.5);
});
