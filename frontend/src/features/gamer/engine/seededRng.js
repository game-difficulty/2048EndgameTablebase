const UINT32_RANGE = 0x100000000;

const rotateLeft = (value, shift) => (
  ((value << shift) | (value >>> (32 - shift))) >>> 0
);

export class Xoshiro128StarStar {
  constructor(state) {
    const normalized = Array.isArray(state)
      ? state.map((value) => Number(value) >>> 0)
      : [];
    if (normalized.length !== 4 || normalized.every((value) => value === 0)) {
      throw new Error('Invalid ranked RNG state.');
    }
    this.state = normalized;
  }

  static fromSeedHex(seedHex) {
    const normalized = String(seedHex || '').trim().toLowerCase();
    if (!/^[0-9a-f]{32}$/u.test(normalized)) {
      throw new Error('Invalid ranked seed.');
    }
    return new Xoshiro128StarStar(
      [0, 8, 16, 24].map((offset) => Number.parseInt(normalized.slice(offset, offset + 8), 16) >>> 0),
    );
  }

  nextUint32() {
    let [s0, s1, s2, s3] = this.state;
    const result = Math.imul(rotateLeft(Math.imul(s1, 5) >>> 0, 7), 9) >>> 0;
    const temporary = (s1 << 9) >>> 0;
    s2 = (s2 ^ s0) >>> 0;
    s3 = (s3 ^ s1) >>> 0;
    s1 = (s1 ^ s2) >>> 0;
    s0 = (s0 ^ s3) >>> 0;
    s2 = (s2 ^ temporary) >>> 0;
    s3 = rotateLeft(s3, 11);
    this.state = [s0, s1, s2, s3];
    return result;
  }

  nextFloat() {
    return this.nextUint32() / UINT32_RANGE;
  }

  chooseIndex(count) {
    if (!Number.isInteger(count) || count <= 0) throw new Error('Invalid ranked choice size.');
    return this.nextUint32() % count;
  }

  exportState() {
    return [...this.state];
  }
}

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

export function createRankedInitialBoard(seedHex) {
  const rng = Xoshiro128StarStar.fromSeedHex(seedHex);
  const board = new Array(16).fill(0);
  const initialTiles = [];
  for (let index = 0; index < 2; index += 1) {
    const spawn = randomSpawnWithRng(board, rng);
    board[spawn.index] = spawn.value;
    initialTiles.push([spawn.index, spawn.value === 4 ? 1 : 0]);
  }
  return { board, initialTiles, rng };
}
