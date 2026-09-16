import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import vm from 'node:vm';
import { canResolveLargePair } from '../src/features/gamer/engine/aiSearchBoard.js';

test('only two equal real large tiles enable search normalization', () => {
  for (const [tiles, expected] of [
    [[32768, 32768], true], [[65536, 65536], true],
    [[32768, 65536], false], [[65536, 32768], false],
    [[32768], false], [[], false], [[16384, 16384], false],
    [[32768, 32768, 65536], false],
  ]) assert.equal(canResolveLargePair([...tiles, ...Array(16 - tiles.length).fill(0)]), expected);
});

test('worker applies opt-in normalization for both creation and reset', async () => {
  const raw = 0x000f11171366224fn;
  const normalized = 0x000e11171366224en;
  let player;
  let conversions = 0;
  const core = {
    AIPlayer: class {
      constructor(board) { this.board = board; player = this; }
      reset_board(board) { this.board = board; }
    },
    resolve_32768_doubles(board) { assert.equal(board, raw); conversions++; return normalized; },
  };
  const context = vm.createContext({ createAICore: async () => core, self: {}, postMessage() {}, console: { log() {} } });
  const source = fs.readFileSync(new URL('../public/wasm/ai_worker.js', import.meta.url), 'utf8').replace(/^import .*;\r?$/gm, '');
  vm.runInContext(source, context);
  await Promise.resolve();
  vm.runInContext('runAI = (board, counts, hex) => { globalThis.original = { board, counts, hex }; };', context);
  for (const enabled of [true, false, true, undefined]) {
    context.self.onmessage({ data: { type: 'calculate', board_encoded: '000f11171366224f', resolve_large_pair: enabled } });
    assert.equal(player.board, enabled === true ? normalized : raw);
    assert.equal(context.original.counts[15], 2);
    assert.equal(context.original.hex, '000f11171366224f');
  }
  assert.equal(conversions, 2);
});
