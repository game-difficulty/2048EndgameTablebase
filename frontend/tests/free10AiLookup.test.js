import assert from 'node:assert/strict';
import test from 'node:test';
import { TableDispatcher, maskLargeTiles } from '../src/features/gamer/engine/tableDispatcher.js';

function setup(pattern, target, smallTiles) {
  const board = [65536, 32768, 16384, 8192, 4096, 2048, ...smallTiles];
  while (board.length < 16) board.push(0);
  const dispatcher = new TableDispatcher();
  dispatcher.reset(board);
  const candidate = { table: { pattern, target: String(target), n: 6,
    fullPattern: `${pattern}_${target}`, structureRules: null }, type: 1 };
  dispatcher.candidates = () => [candidate];
  return { dispatcher, board, candidate };
}

test('free10 skips lookup below 32 for every target, including the low-sum mask fallback', async () => {
  for (const target of [128, 256, 512]) {
    for (const tiles of [[], [16, 4, 2], [16, 8], [16, 8, 4, 2]]) {
      const { dispatcher, board } = setup('free10', target, tiles);
      assert.equal(await dispatcher.choose(() => assert.fail('Must not query or consume cache')), 'AI');
      assert.deepEqual(dispatcher.board, board);
      assert.equal(dispatcher.cooldowns.size, 0);
    }
  }
});

test('free10 still queries at 32 and above with the same masked board', async () => {
  for (const tiles of [[16, 8, 8], [32, 2]]) {
    const { dispatcher, board } = setup('free10', 512, tiles);
    const calls = [];
    assert.equal(await dispatcher.choose(async (_, hex) => {
      calls.push(hex);
      return { results: { left: .9 }, dtype: 'float64' };
    }), 'left');
    assert.equal(calls.length, 1);
    assert.equal(maskLargeTiles(board, 6).filter(v => v !== 32768).reduce((a, b) => a + b, 0),
      tiles.reduce((a, b) => a + b, 0));
  }
});

test('skipping free10 continues with the next candidate without adding a cooldown', async () => {
  const { dispatcher, candidate } = setup('free10', 512, [16, 8, 4, 2]);
  const next = { ...candidate, table: { ...candidate.table, pattern: 'free11', fullPattern: 'free11_512' } };
  dispatcher.candidates = () => [candidate, next];
  const calls = [];
  assert.equal(await dispatcher.choose(async ({ table }) => {
    calls.push(table.fullPattern);
    return { results: { right: .9 }, dtype: 'float64' };
  }), 'right');
  assert.deepEqual(calls, ['free11_512']);
  assert.equal(dispatcher.cooldowns.size, 0);
});
