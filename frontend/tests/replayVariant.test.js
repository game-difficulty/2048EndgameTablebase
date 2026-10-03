import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizePatternName, isVariantPattern } from '../src/utils/patternCategories.js';
import { patternFromReplayFilename, resolveReplayVariant } from '../src/utils/replayVariant.js';
import { createBoardViewport } from '../src/utils/boardViewport.js';

test('tile and sum targets resolve all built-in variants without catalog loading', () => {
  for (const pattern of ['2x4', '3x3', '3x4', '3x3free8', '3x4free9']) {
    for (const target of ['512', 'sum-1800']) {
      assert.equal(normalizePatternName(`${pattern}_${target}`), pattern);
      assert.equal(isVariantPattern(`${pattern}_${target}`), true);
    }
  }
  assert.equal(isVariantPattern('free10_sum-1800'), false);
  assert.equal(normalizePatternName('custom_pattern_sum-1800'), 'custom_pattern');
});

test('all replay sources resolve the same shape and repair legacy false flags', () => {
  const board = [0, 2, 2, 32768, 0, 0, 0, 32768, 0, 0, 0, 32768, 32768, 32768, 32768, 32768];
  for (const metadata of [
    { filename: '3x3_sum-1800_42.rpl' },
    { pattern: '3x3_sum-1800', useVariant: true },
    { pattern: '3x3_sum-1800', useVariant: false },
    { source: 'C:\\replays\\3x3_sum-1800.rpl' },
  ]) {
    const resolved = resolveReplayVariant(metadata);
    assert.equal(resolved.pattern, '3x3_sum-1800');
    assert.equal(resolved.useVariant, true);
    assert.equal(resolved.conflict, metadata.useVariant === false);
    const viewport = createBoardViewport(board, resolved.useVariant);
    assert.equal(viewport.rows, 3);
    assert.equal(viewport.cols, 3);
  }
});

test('unknown files do not auto-crop ordinary masked 4x4 boards', () => {
  const board = new Array(16).fill(32768);
  board[0] = 2;
  assert.equal(resolveReplayVariant({ filename: 'renamed.rpl' }).useVariant, false);
  assert.equal(createBoardViewport(board, resolveReplayVariant({ pattern: 'free10_512' }).useVariant).rows, 4);
  assert.equal(resolveReplayVariant({ filename: 'renamed.rpl', useVariant: true }).useVariant, true);
  assert.equal(resolveReplayVariant({ pattern: 'free10_512', useVariant: true }).conflict, true);
  assert.equal(patternFromReplayFilename('custom_pattern_512_42.rpl'), 'custom_pattern_512');
});
