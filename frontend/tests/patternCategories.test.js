import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { getPatternCategory, isVariantPattern } from '../src/utils/patternCategories.js';
import { groupTablebasePatternsByCategory, getCatalogTargetsForPattern } from '../src/services/tablebases/catalogClient.js';

test('cloud menu taxonomy agrees with every configured pattern', () => {
  const patterns = JSON.parse(readFileSync(new URL('../../docs_and_configs/patterns_config.json', import.meta.url), 'utf8'));
  const categories = { '10 space': 'space10', '12 space': 'space12', free: 'free', others: 'others', variant: 'variant' };
  for (const [pattern, config] of Object.entries(patterns)) {
    assert.equal(getPatternCategory(pattern), categories[config.category], pattern);
    assert.equal(getPatternCategory(`${pattern}_1024`), categories[config.category], `${pattern}_1024`);
  }
});

test('catalog groups only available patterns, deduplicates targets and preserves variant semantics', () => {
  const tables = [
    { pattern: 'L3', target: '256' }, { pattern: 'free10', target: '512' },
    { pattern: 'free10', target: '1024' }, { pattern: '444', target: '512' },
    { pattern: '3x4free9', target: '256' }, { pattern: 'future-pattern', target: '128' },
    { pattern: '', target: '128' },
  ];
  const groups = groupTablebasePatternsByCategory(tables);
  assert.deepEqual(groups, {
    free: ['free10'], space10: ['L3'], space12: ['444'],
    others: ['future-pattern'], variant: ['3x4free9'],
  });
  assert.deepEqual(getCatalogTargetsForPattern(tables, 'free10'), ['512', '1024']);
  assert.equal(isVariantPattern('3x4free9_256', groups), true);
  assert.equal(isVariantPattern('free10_1024', groups), false);
  assert.deepEqual(groupTablebasePatternsByCategory([{ pattern: 'free14' }]), { free: ['free14'] });
  assert.deepEqual(groupTablebasePatternsByCategory([]), {});
});
