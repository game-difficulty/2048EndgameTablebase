import assert from 'node:assert/strict';
import test from 'node:test';

import {
  loadLastAnalysisSelection,
  normalizeAnalysisSelection,
  saveLastAnalysisSelection,
} from '../src/human/analysisSelectionHistory.js';

const tables = [
  { pattern: 'L3', target: '512' },
  { pattern: 'L3', target: '1024' },
  { pattern: 'LL', target: '2048' },
];

function memoryStorage() {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
}

test('analysis selections are normalized against the current catalog', () => {
  assert.deepEqual(normalizeAnalysisSelection([
    { pattern: 'L3', target: '512' },
    { pattern: 'missing', target: '512' },
    { pattern: 'L3', target: '512' },
    { pattern: 'LL', target: '2048' },
  ], tables), [
    { pattern: 'L3', target: '512' },
    { pattern: 'LL', target: '2048' },
  ]);
});

test('last analysis selection is stored separately for each variant', () => {
  const storage = memoryStorage();
  saveLastAnalysisSelection('4x4', [{ pattern: 'L3', target: '1024' }], tables, storage);
  saveLastAnalysisSelection('3x4', [{ pattern: 'LL', target: '2048' }], tables, storage);
  assert.deepEqual(loadLastAnalysisSelection('4x4', tables, storage), [{ pattern: 'L3', target: '1024' }]);
  assert.deepEqual(loadLastAnalysisSelection('3x4', tables, storage), [{ pattern: 'LL', target: '2048' }]);
  assert.deepEqual(loadLastAnalysisSelection('3x3', tables, storage), []);
});
