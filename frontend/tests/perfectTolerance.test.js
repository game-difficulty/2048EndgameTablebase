import assert from 'node:assert/strict';
import test from 'node:test';
import { perfectTolerance, isPerfectResult } from '../src/utils/perfectTolerance.js';
import { analyzeReplay } from '../src/features/replay/engine/replayAnalysis.js';
import { applyTesterLocalMove, createTesterLocalSession } from '../src/features/tester/engine/testerLocalSession.js';

for (const dtype of ['uint32', 'float32', '1-float32', 'uint64', 'float64', '1-float64']) {
  test(dtype + ' shares Perfect, combo and goodness policy', () => {
    const expected = dtype.includes('32');
    assert.equal(perfectTolerance(dtype), expected ? 3e-10 : 1e-14);
    assert.equal(isPerfectResult(.5 - 1e-12, .5, dtype), expected);
    const session = createTesterLocalSession({ board: [2, 2, ...Array(14).fill(0)] });
    const offset = dtype.startsWith('1-') ? -1 : 0;
    const moved = applyTesterLocalMove(session, {
      direction: 'right', results: { left: .5 + offset, right: .5 - 1e-12 + offset },
      dtype, spawnRate4: .1, randomSource: () => .9,
    });
    assert.equal(moved.accepted, true);
    assert.equal(moved.session.metrics.combo, Number(expected));
    assert.equal(moved.session.metrics.goodness_of_fit === 1, expected);
    assert.equal(moved.session.lastStep.evaluation === 'Perfect!', expected);
  });
}

test('dtype aliases and unknown fallback', () => {
  assert.equal(perfectTolerance('1-f32'), 3e-10);
  assert.equal(perfectTolerance('1-f64'), 1e-14);
  assert.equal(perfectTolerance('?'), 3e-10);
});

test('legacy uint32 replay compares normalized probabilities consistently', () => {
  const replay = { moveCount: 1, changes: Uint8Array.of(1 << 5),
    rates: Uint32Array.of(2_000_000_000, 1_999_999_999, 0, 0) };
  const result = analyzeReplay(replay);
  assert.equal(result.summary.final_gof, 1);
  assert.equal(result.summary.max_combo, 1);
  assert.equal(result.summary.counts['Perfect!'], 1);
});
