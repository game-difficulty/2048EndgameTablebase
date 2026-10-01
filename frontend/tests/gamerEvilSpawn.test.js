import assert from 'node:assert/strict';
import test from 'node:test';

import createEvilCore from '../public/wasm/evil_core.js';

test('seeded EvilGen randomizes only exact ties and keeps the legacy result', async () => {
  const module = await createEvilCore();
  const board = 0x123456789abcde0fn;
  const generator = new module.EvilGen(board);
  try {
    const choice = (seed) => {
      generator.reset_board(board);
      const result = seed === null
        ? generator.gen_new_num(5)
        : generator.gen_new_num_seeded(5, seed);
      return [Number(result[1]), Number(result[2])];
    };
    assert.deepEqual(choice(1), [14, 2]);
    assert.deepEqual(choice(2), [14, 1]);
    assert.deepEqual(choice(1), [14, 2]);
    assert.deepEqual(choice(null), [14, 1]);
  } finally {
    generator.delete();
  }
});
