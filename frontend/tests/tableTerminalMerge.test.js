import assert from 'node:assert/strict';
import test from 'node:test';
import { TableDispatcher, missingImmediateMergeResult } from '../src/features/gamer/engine/tableDispatcher.js';

const board = code => [...code].map(c => c === '0' ? 0 : 2 ** parseInt(c, 16));
const current = board('e931a88f102d0013');
const candidate = { type: 1, table: { pattern: 'free11', n: 5, target: 512, fullPattern: 'free11_512' } };
const payload = (results, dtype = 'uint32') => ({ results, dtype, legal_moves_mask: 8 });

test('missing immediate goal result hands off without restrictions, cooldown, or another lookup', async () => {
  const d = new TableDispatcher();
  d.reset(current);
  d.aiSearchMoves = [4];
  const result = payload({ down: .978171212, left: 0, right: 0, up: null });
  assert.equal(d.accept(candidate, result), 'AI');
  assert.equal(d.aiSearchMoves, null);
  assert.equal(d.cooldowns.size, 0);
  d.candidates = () => [candidate, candidate];
  let calls = 0;
  assert.equal(await d.choose(async () => { calls++; return result; }), 'AI');
  assert.equal(calls, 1);
});

test('missing and nonpositive results, including failure-rate dtypes', () => {
  for (const left of [undefined, null, '', 0, NaN, Infinity, false]) {
    assert.equal(missingImmediateMergeResult(current, 512, payload({ left, right: .9 })), true);
  }
  assert.equal(missingImmediateMergeResult(current, 512, payload({ left: -.1, right: -.2 }, '1-float32')), false);
  assert.equal(missingImmediateMergeResult(current, 512, payload({ left: -1, right: -.2 }, '1-float32')), true);
});

test('ordinary zero or positive terminal result preserves table acceptance', () => {
  const d = new TableDispatcher();
  d.reset(board('e931a78f102d0013'));
  assert.equal(d.accept(candidate, payload({ down: .9, left: 0, right: 0 })), 'down');
  d.reset(current);
  assert.equal(d.accept(candidate, payload({ down: .98, left: .9, right: .8, up: 0 })), 'down');
});

test('detect real new merges with gaps or vertical movement, not existing targets or chain merges', () => {
  for (const code of ['8089000000000000', '8000000080009000', '9988000000000000']) {
    assert.equal(missingImmediateMergeResult(board(code), 512, payload({})), true);
  }
  for (const code of ['8189000000000000', '7789000000000000']) {
    assert.equal(missingImmediateMergeResult(board(code), 512, payload({})), false);
  }
});
