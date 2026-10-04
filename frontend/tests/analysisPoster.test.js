import test from 'node:test';
import assert from 'node:assert/strict';
import { analysisPercent, analysisDuration, drawAnalysisPoster } from '../src/human/analysisPoster.js';

test('3x3 accuracy preserves differences at the highest thresholds', () => {
  assert.equal(analysisPercent(.9999976, 5), '99.99976%');
  assert.equal(analysisPercent(.9999990, 5), '99.99990%');
  assert.equal(analysisPercent(null, 5), '—');
  assert.equal(analysisPercent(.9657), '96.57%');
  assert.equal(analysisDuration(660000), '11:00');
  assert.equal(analysisDuration(195000), '3:15');
  assert.equal(analysisDuration(null), '—');
});

test('poster uses 3x3 metrics and readable target label while keeping 4x4 layout', async () => {
  const oldDocument = globalThis.document, oldStyle = globalThis.getComputedStyle;
  const labels = [];
  const ctx = new Proxy({
    measureText: value => ({ width: String(value).length * 10, actualBoundingBoxAscent: 20, actualBoundingBoxDescent: 0 }),
    fillText: value => labels.push(String(value)),
    createLinearGradient: () => ({ addColorStop() {} }),
    createRadialGradient: () => ({ addColorStop() {} }),
  }, { get: (target, key) => key in target ? target[key] : () => {} });
  const canvas = { getContext: () => ctx };
  globalThis.document = { fonts: { ready: Promise.resolve() }, createElement: () => canvas, documentElement: {} };
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => '1' });
  try {
    await drawAnalysisPoster({ canvas, language:'en', data: {
      pattern:'3x3',target:'sum-1790',run:{variant:'3x3',board:[1024,512,128,64,32,16,8,4,2],score:15000},
      aggregate:{mean_single_step_accuracy:.999999,perfect_rate:.93,max_combo:300,run_elapsed_ms:480000,run_board_sum:1790},grade:'X',
    } });
    assert.ok(labels.includes('3×3  ·  Board sum 1790'));
    assert.ok(labels.includes('99.99990%'));
    assert.ok(labels.includes('GEOMETRIC ACCURACY'));
    assert.ok(labels.includes('PERFECT'));
    assert.ok(labels.includes('X'));
    assert.ok(!labels.includes('AVERAGE FIT'));
    labels.length = 0;
    await drawAnalysisPoster({ canvas, language:'en', data:{run:{variant:'4x4',board:[]},aggregate:{mean_goodness_of_fit:.9657}} });
    assert.ok(labels.includes('96.57%'));
    assert.ok(labels.includes('AVERAGE FIT'));
    assert.ok(!labels.includes('GEOMETRIC ACCURACY'));
  } finally {
    if (oldDocument === undefined) delete globalThis.document; else globalThis.document = oldDocument;
    if (oldStyle === undefined) delete globalThis.getComputedStyle; else globalThis.getComputedStyle = oldStyle;
  }
});
