import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createRankedInitialBoard,
  Xoshiro128StarStar,
} from '../src/features/gamer/engine/seededRng.js';
import {
  encodeRankedReplay,
  exactBoardCodes,
  RANKED_RECORD,
} from '../src/features/gamer/engine/rankedReplayEncoder.js';


const SEED = '00000001000000020000000300000004';

test('xoshiro128** matches the shared fixed vector', () => {
  const rng = Xoshiro128StarStar.fromSeedHex(SEED);
  assert.deepEqual(
    Array.from({ length: 8 }, () => rng.nextUint32()),
    [11520, 0, 5927040, 70819200, 2031721883, 1637235492, 1287239034, 3734860849],
  );
});

test('xoshiro state restoration permits individual zero words', () => {
  const rng = new Xoshiro128StarStar([1, 0, 2, 3]);
  assert.equal(Number.isInteger(rng.nextUint32()), true);
});

test('ranked initial tiles consume the deterministic RNG', () => {
  const initial = createRankedInitialBoard(SEED);
  assert.deepEqual(initial.initialTiles, [[0, 1], [1, 1]]);
  assert.deepEqual(initial.board.slice(0, 4), [4, 4, 0, 0]);
});

test('ranked initial tiles use the run-bound 4-spawn rate', () => {
  const allTwos = createRankedInitialBoard(SEED, 0);
  const allFours = createRankedInitialBoard(SEED, 1);
  assert.deepEqual(allTwos.initialTiles.map((tile) => tile[1]), [0, 0]);
  assert.deepEqual(allFours.initialTiles.map((tile) => tile[1]), [1, 1]);
});

test('ranked encoder emits the cross-language fixture', () => {
  const encoded = encodeRankedReplay({
    seedHex: SEED,
    rulesVersion: 1,
    initialTiles: [[0, 1], [1, 1]],
    records: [
      [RANKED_RECORD.DIFFICULTY, 0],
      [RANKED_RECORD.MOVE, 3, 2, 0, 123],
      [RANKED_RECORD.AI_USED],
      [RANKED_RECORD.END],
    ],
  });
  assert.equal(
    encoded,
    'REPLAY_v1RPL_B64_UlBMMUQAAhARg2QRAQAAAAEAAAACAAAAAwAAAASDAgRwb3cyg2UBAAt7g2YAhMwwCkc=',
  );
});

test('exact final board codes preserve tiles above 32768', () => {
  assert.deepEqual(exactBoardCodes([0, 2, 32768, 65536, ...new Array(12).fill(0)]).slice(0, 4), [0, 1, 15, 16]);
});
