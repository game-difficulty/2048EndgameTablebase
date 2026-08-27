import { Xoshiro128StarStar } from '../../../utils/xoshiro128.js';

export { Xoshiro128StarStar };

export function randomSpawnWithRng(values, rng, spawnRate4 = 0.1) {
  const empty = values
    .map((value, index) => (Number(value) === 0 ? index : null))
    .filter((index) => index !== null);
  if (!empty.length) return null;
  return {
    index: empty[rng.chooseIndex(empty.length)],
    value: rng.nextFloat() < spawnRate4 ? 4 : 2,
  };
}

export function createRankedInitialBoard(seedHex, spawnRate4 = 0.1) {
  const rng = Xoshiro128StarStar.fromSeedHex(seedHex);
  const board = new Array(16).fill(0);
  const initialTiles = [];
  for (let index = 0; index < 2; index += 1) {
    const spawn = randomSpawnWithRng(board, rng, spawnRate4);
    board[spawn.index] = spawn.value;
    initialTiles.push([spawn.index, spawn.value === 4 ? 1 : 0]);
  }
  return { board, initialTiles, rng };
}
