const REPLAY_PREFIX = 'REPLAY_v1RPL_B64_';
const RECORD_UNDO1 = 128;
const RECORD_UNDON = 129;
const RECORD_EXT = 131;
const RECORD_END = 132;
const EXT_RULESET = 2;
const EXT_RANKED_METADATA = 100;
const EXT_DIFFICULTY_CHANGE = 101;
const EXT_AI_USED = 102;

export const RANKED_RULES_VERSION = 1;
export const MAX_RANKED_RECORD_BYTES = 500 * 1024;
export const MAX_RANKED_MOVES = 100000;

export const RANKED_RECORD = Object.freeze({
  MOVE: 0,
  DIFFICULTY: 1,
  AI_USED: 2,
  UNDO: 3,
  END: 4,
});

export const DIRECTION_TO_NEXT_CODE = Object.freeze({
  up: 0,
  right: 1,
  down: 2,
  left: 3,
});

export const encodeUleb128 = (rawValue) => {
  let value = Math.max(0, Math.floor(Number(rawValue) || 0));
  const bytes = [];
  do {
    const byte = value % 128;
    value = Math.floor(value / 128);
    bytes.push(byte | (value ? 0x80 : 0));
  } while (value);
  return bytes;
};

const encodeExtension = (type, payload) => [
  RECORD_EXT,
  ...encodeUleb128(type),
  ...encodeUleb128(payload.length),
  ...payload,
];

const seedBytes = (seedHex) => {
  if (!/^[0-9a-f]{32}$/iu.test(String(seedHex || ''))) throw new Error('Invalid ranked seed.');
  return Array.from({ length: 16 }, (_unused, index) => (
    Number.parseInt(seedHex.slice(index * 2, index * 2 + 2), 16)
  ));
};

const crcTable = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) ? (0xedb88320 ^ (value >>> 1)) : (value >>> 1);
    }
    table[index] = value >>> 0;
  }
  return table;
})();

export const crc32 = (bytes) => {
  let value = 0xffffffff;
  for (const byte of bytes) value = crcTable[(value ^ byte) & 0xff] ^ (value >>> 8);
  return (value ^ 0xffffffff) >>> 0;
};

export const bytesToBase64 = (bytes) => {
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
};

export function encodeRankedReplay({ seedHex, rulesVersion, initialTiles, records }) {
  const bytes = [0x52, 0x50, 0x4c, 0x31, 0x44, 0, 2];
  if (!Array.isArray(initialTiles) || initialTiles.length !== 2) {
    throw new Error('Ranked replay requires two initial tiles.');
  }
  for (const [cellIndex, valueBit] of initialTiles) {
    bytes.push((Number(cellIndex) & 0x0f) | ((Number(valueBit) & 1) << 4));
  }
  bytes.push(...encodeExtension(
    EXT_RANKED_METADATA,
    [Number(rulesVersion) & 0xff, ...seedBytes(seedHex)],
  ));
  bytes.push(...encodeExtension(EXT_RULESET, Array.from(new TextEncoder().encode('pow2'))));

  for (const record of records) {
    if (!Array.isArray(record)) throw new Error('Invalid ranked record.');
    if (record[0] === RANKED_RECORD.MOVE) {
      const direction = Number(record[1]);
      const spawnIndex = Number(record[2]);
      const valueBit = Number(record[3]);
      bytes.push(direction | (spawnIndex << 2) | (valueBit << 6));
      bytes.push(...encodeUleb128(record[4]));
    } else if (record[0] === RANKED_RECORD.DIFFICULTY) {
      bytes.push(...encodeExtension(EXT_DIFFICULTY_CHANGE, [Number(record[1])]));
    } else if (record[0] === RANKED_RECORD.AI_USED) {
      bytes.push(...encodeExtension(EXT_AI_USED, []));
    } else if (record[0] === RANKED_RECORD.UNDO) {
      const count = Math.max(1, Number(record[1]) || 1);
      bytes.push(count === 1 ? RECORD_UNDO1 : RECORD_UNDON);
      if (count > 1) bytes.push(...encodeUleb128(count));
      bytes.push(...encodeUleb128(record[2]));
    } else if (record[0] === RANKED_RECORD.END) {
      bytes.push(RECORD_END);
    } else {
      throw new Error('Unsupported ranked record.');
    }
    if (bytes.length > MAX_RANKED_RECORD_BYTES) throw new Error('Ranked replay is too large.');
  }
  const checksum = crc32(bytes);
  bytes.push(checksum & 0xff, (checksum >>> 8) & 0xff, (checksum >>> 16) & 0xff, checksum >>> 24);
  if (bytes.length > MAX_RANKED_RECORD_BYTES) throw new Error('Ranked replay is too large.');
  return REPLAY_PREFIX + bytesToBase64(Uint8Array.from(bytes));
}

export function exactBoardCodes(board) {
  return board.map((rawValue) => {
    const value = Number(rawValue) || 0;
    if (!value) return 0;
    const exponent = Math.log2(value);
    if (!Number.isInteger(exponent) || exponent < 1 || exponent > 31) {
      throw new Error('The final board contains an unsupported tile.');
    }
    return exponent;
  });
}

export function rankedMoveCount(records) {
  return records.reduce((count, record) => count + (record?.[0] === RANKED_RECORD.MOVE ? 1 : 0), 0);
}
