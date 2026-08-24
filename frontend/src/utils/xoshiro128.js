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
      throw new Error('Invalid RNG state.');
    }
    this.state = normalized;
  }

  static fromSeedHex(seedHex) {
    const normalized = String(seedHex || '').trim().toLowerCase();
    if (!/^[0-9a-f]{32}$/u.test(normalized)) {
      throw new Error('Invalid RNG seed.');
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
    if (!Number.isInteger(count) || count <= 0) throw new Error('Invalid RNG choice size.');
    return this.nextUint32() % count;
  }

  exportState() {
    return [...this.state];
  }
}

export function createRandomXoshiroState() {
  const state = new Uint32Array(4);
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(state);
  } else {
    for (let index = 0; index < state.length; index += 1) {
      state[index] = Math.floor(Math.random() * UINT32_RANGE) >>> 0;
    }
  }
  if (state.every((value) => value === 0)) state[0] = 1;
  return Array.from(state, (value) => Number(value) >>> 0);
}
