export const MGO1_PREFIX = 'MINIGAME_v1MGO_B64_';
export const MGO1_FORMAT_VERSION = 1;
export const MAX_MGO1_BYTES = 256 * 1024;
export const MAX_MGO1_ACTIONS = 50000;

const MAGIC = Uint8Array.of(0x4d, 0x47, 0x4f, 0x31);
const HEADER_FIXED_BYTES = MAGIC.length + 1;
const HEADER_TAIL_BYTES = 1 + 1 + 1 + 16 + 16;
const CRC_BYTES = 4;
const MAX_ULEB_BYTES = 8;
const MAX_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
const UINT64_MAX = 0xffffffffffffffffn;

export const MGO1_TAG = Object.freeze({
  MOVE_UP: 0x00,
  MOVE_RIGHT: 0x01,
  MOVE_DOWN: 0x02,
  MOVE_LEFT: 0x03,
  BOMB: 0x10,
  GLOVE: 0x11,
  TWIST: 0x12,
  CUSTOM: 0x20,
  TICK: 0x30,
  DIGEST: 0x70,
  END: 0x7f,
});

export const MOVE_DIRECTION_TO_TAG = Object.freeze({
  up: MGO1_TAG.MOVE_UP,
  right: MGO1_TAG.MOVE_RIGHT,
  down: MGO1_TAG.MOVE_DOWN,
  left: MGO1_TAG.MOVE_LEFT,
});

export const MOVE_TAG_TO_DIRECTION = Object.freeze({
  [MGO1_TAG.MOVE_UP]: 'up',
  [MGO1_TAG.MOVE_RIGHT]: 'right',
  [MGO1_TAG.MOVE_DOWN]: 'down',
  [MGO1_TAG.MOVE_LEFT]: 'left',
});

const GAME_IDS = [
  'design-master-1',
  'mystery-merge-1',
  'column-chaos',
  'gravity-twist-1',
  'blitzkrieg',
  'tricky-tiles',
  'design-master-2',
  'shape-shifter',
  'ferris-wheel',
  'gravity-twist-2',
  'design-master-3',
  'mystery-merge-2',
  'ice-age',
  'isolated-island',
  'design-master-4',
  'endless-factorization',
  'endless-explosions',
  'endless-giftbox',
  'endless-hybrid',
  'endless-airraid',
];

export const MINIGAME_ID_TO_CODE = Object.freeze(Object.fromEntries(
  GAME_IDS.map((gameId, index) => [gameId, index + 1]),
));

export const MINIGAME_CODE_TO_ID = Object.freeze(Object.fromEntries(
  GAME_IDS.map((gameId, index) => [index + 1, gameId]),
));

const crcTable = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < table.length; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) ? (0xedb88320 ^ (value >>> 1)) : (value >>> 1);
    }
    table[index] = value >>> 0;
  }
  return table;
})();

export function crc32(bytes) {
  let value = 0xffffffff;
  for (const byte of bytes) value = crcTable[(value ^ byte) & 0xff] ^ (value >>> 8);
  return (value ^ 0xffffffff) >>> 0;
}

function assertInteger(value, name, minimum, maximum) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer from ${minimum} to ${maximum}.`);
  }
  return value;
}

function encodeUleb128(rawValue, name = 'ULEB128 value') {
  assertInteger(rawValue, name, 0, Number.MAX_SAFE_INTEGER);
  let value = BigInt(rawValue);
  const bytes = [];
  do {
    let byte = Number(value & 0x7fn);
    value >>= 7n;
    if (value) byte |= 0x80;
    bytes.push(byte);
  } while (value);
  return bytes;
}

function decodeUleb128(bytes, offset, limit, name) {
  let value = 0n;
  let shift = 0n;
  for (let index = 0; index < MAX_ULEB_BYTES; index += 1) {
    if (offset >= limit) throw new Error(`Truncated ${name}.`);
    const byte = bytes[offset];
    offset += 1;
    value |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      if (index > 0 && (byte & 0x7f) === 0) throw new Error(`Non-canonical ${name}.`);
      if (value > MAX_SAFE_BIGINT) throw new Error(`${name} exceeds the safe integer range.`);
      return { value: Number(value), offset };
    }
    shift += 7n;
  }
  throw new Error(`${name} is too long.`);
}

function hexToBytes(rawValue, byteLength, name) {
  if (rawValue instanceof Uint8Array) {
    if (rawValue.length !== byteLength) throw new Error(`${name} must contain ${byteLength} bytes.`);
    return Uint8Array.from(rawValue);
  }
  const value = String(rawValue || '').replaceAll('-', '').toLowerCase();
  if (!new RegExp(`^[0-9a-f]{${byteLength * 2}}$`, 'u').test(value)) {
    throw new Error(`${name} must contain ${byteLength * 2} hexadecimal digits.`);
  }
  return Uint8Array.from({ length: byteLength }, (_unused, index) => (
    Number.parseInt(value.slice(index * 2, index * 2 + 2), 16)
  ));
}

function parseRunId(runId) {
  const raw = String(runId || '').toLowerCase();
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/u.test(raw)) {
    throw new Error('runId must be a valid UUID.');
  }
  return hexToBytes(raw, 16, 'runId');
}

function bytesToHex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

function bytesToUuid(bytes) {
  const hex = bytesToHex(bytes);
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function uint64Value(rawValue, name) {
  let value;
  try {
    value = BigInt(rawValue);
  } catch {
    throw new Error(`${name} must be an unsigned 64-bit integer.`);
  }
  if (value < 0n || value > UINT64_MAX) {
    throw new Error(`${name} must be an unsigned 64-bit integer.`);
  }
  return value;
}

function appendUint64Le(target, rawValue) {
  let value = uint64Value(rawValue, 'digest');
  for (let index = 0; index < 8; index += 1) {
    target.push(Number(value & 0xffn));
    value >>= 8n;
  }
}

function readUint64Le(bytes, offset) {
  let value = 0n;
  for (let index = 7; index >= 0; index -= 1) value = (value << 8n) | BigInt(bytes[offset + index]);
  return value;
}

function encodeAction(action) {
  if (!action || typeof action !== 'object') throw new Error('Each action must be an object.');
  const deltaMs = action.deltaMs ?? 0;
  const encoded = [];
  let tag;

  if (action.type === 'move') {
    tag = MOVE_DIRECTION_TO_TAG[String(action.direction || '').toLowerCase()];
    if (tag === undefined) throw new Error('Move direction is invalid.');
  } else if (action.type === 'bomb') {
    tag = MGO1_TAG.BOMB;
  } else if (action.type === 'glove') {
    tag = MGO1_TAG.GLOVE;
  } else if (action.type === 'twist') {
    tag = MGO1_TAG.TWIST;
  } else if (action.type === 'custom') {
    tag = MGO1_TAG.CUSTOM;
  } else if (action.type === 'tick') {
    tag = MGO1_TAG.TICK;
  } else if (action.type === 'digest') {
    tag = MGO1_TAG.DIGEST;
  } else if (action.type === 'end') {
    tag = MGO1_TAG.END;
  } else {
    throw new Error(`Unsupported MGO1 action type: ${String(action.type)}.`);
  }

  encoded.push(tag, ...encodeUleb128(deltaMs, 'deltaMs'));
  if (tag === MGO1_TAG.BOMB || tag === MGO1_TAG.TWIST) {
    encoded.push(assertInteger(action.index, 'index', 0, 255));
  } else if (tag === MGO1_TAG.GLOVE) {
    encoded.push(
      assertInteger(action.source, 'source', 0, 255),
      assertInteger(action.target, 'target', 0, 255),
    );
  } else if (tag === MGO1_TAG.CUSTOM) {
    encoded.push(
      assertInteger(action.actionId, 'actionId', 0, 255),
      assertInteger(action.phase, 'phase', 0, 255),
    );
  } else if (tag === MGO1_TAG.DIGEST) {
    appendUint64Le(encoded, action.digest);
  } else if (tag === MGO1_TAG.END) {
    encoded.push(assertInteger(action.reason, 'reason', 0, 255));
  }
  return encoded;
}

function decodeAction(bytes, offset, limit) {
  const tag = bytes[offset];
  offset += 1;
  const deltaResult = decodeUleb128(bytes, offset, limit, 'action deltaMs');
  const deltaMs = deltaResult.value;
  offset = deltaResult.offset;
  const requireBytes = (count) => {
    if (offset + count > limit) throw new Error('Truncated MGO1 action payload.');
  };

  if (Object.hasOwn(MOVE_TAG_TO_DIRECTION, tag)) {
    return { action: { type: 'move', direction: MOVE_TAG_TO_DIRECTION[tag], deltaMs }, offset };
  }
  if (tag === MGO1_TAG.BOMB || tag === MGO1_TAG.TWIST) {
    requireBytes(1);
    const action = { type: tag === MGO1_TAG.BOMB ? 'bomb' : 'twist', index: bytes[offset], deltaMs };
    return { action, offset: offset + 1 };
  }
  if (tag === MGO1_TAG.GLOVE) {
    requireBytes(2);
    return {
      action: { type: 'glove', source: bytes[offset], target: bytes[offset + 1], deltaMs },
      offset: offset + 2,
    };
  }
  if (tag === MGO1_TAG.CUSTOM) {
    requireBytes(2);
    return {
      action: { type: 'custom', actionId: bytes[offset], phase: bytes[offset + 1], deltaMs },
      offset: offset + 2,
    };
  }
  if (tag === MGO1_TAG.TICK) return { action: { type: 'tick', deltaMs }, offset };
  if (tag === MGO1_TAG.DIGEST) {
    requireBytes(8);
    return { action: { type: 'digest', digest: readUint64Le(bytes, offset), deltaMs }, offset: offset + 8 };
  }
  if (tag === MGO1_TAG.END) {
    requireBytes(1);
    return { action: { type: 'end', reason: bytes[offset], deltaMs }, offset: offset + 1 };
  }
  throw new Error(`Unknown MGO1 action tag 0x${tag.toString(16).padStart(2, '0')}.`);
}

function bytesToBase64(bytes) {
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function base64ToBytes(rawValue) {
  const value = String(rawValue || '');
  if (!value || value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/u.test(value)) {
    throw new Error('MGO1 payload is not valid Base64.');
  }
  let binary;
  try {
    binary = atob(value);
  } catch {
    throw new Error('MGO1 payload is not valid Base64.');
  }
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  if (bytesToBase64(bytes) !== value) throw new Error('MGO1 payload uses non-canonical Base64.');
  return bytes;
}

function appendCrc(bytes) {
  const checksum = crc32(bytes);
  return Uint8Array.from([
    ...bytes,
    checksum & 0xff,
    (checksum >>> 8) & 0xff,
    (checksum >>> 16) & 0xff,
    checksum >>> 24,
  ]);
}

function storedCrc32(bytes) {
  const offset = bytes.length - CRC_BYTES;
  return (
    bytes[offset]
    | (bytes[offset + 1] << 8)
    | (bytes[offset + 2] << 16)
    | (bytes[offset + 3] << 24)
  ) >>> 0;
}

export function encodeMgo1({
  rulesVersion,
  gameId,
  difficulty,
  flags = 0,
  runId,
  seed,
  seedHex,
  actions = [],
}) {
  const gameCode = MINIGAME_ID_TO_CODE[gameId];
  if (!gameCode) throw new Error(`Unknown minigame id: ${String(gameId)}.`);
  if (!Array.isArray(actions)) throw new Error('actions must be an array.');
  if (actions.length > MAX_MGO1_ACTIONS) throw new Error('MGO1 action count exceeds the limit.');

  const bytes = [
    ...MAGIC,
    MGO1_FORMAT_VERSION,
    ...encodeUleb128(rulesVersion, 'rulesVersion'),
    gameCode,
    assertInteger(difficulty, 'difficulty', 0, 255),
    assertInteger(flags, 'flags', 0, 255),
    ...parseRunId(runId),
    ...hexToBytes(seedHex ?? seed, 16, 'seed'),
  ];

  let ended = false;
  for (const action of actions) {
    if (ended) throw new Error('MGO1 end must be the final action.');
    bytes.push(...encodeAction(action));
    ended = action?.type === 'end';
    if (bytes.length + CRC_BYTES > MAX_MGO1_BYTES) throw new Error('MGO1 payload exceeds the size limit.');
  }

  const encodedBytes = appendCrc(bytes);
  if (encodedBytes.length > MAX_MGO1_BYTES) throw new Error('MGO1 payload exceeds the size limit.');
  return MGO1_PREFIX + bytesToBase64(encodedBytes);
}

export function decodeMgo1(encoded) {
  if (!String(encoded || '').startsWith(MGO1_PREFIX)) throw new Error('Invalid MGO1 prefix.');
  const encodedPayload = String(encoded).slice(MGO1_PREFIX.length);
  if (encodedPayload.length > Math.ceil(MAX_MGO1_BYTES / 3) * 4) {
    throw new Error('MGO1 payload exceeds the size limit.');
  }
  const bytes = base64ToBytes(encodedPayload);
  const minimumBytes = HEADER_FIXED_BYTES + 1 + HEADER_TAIL_BYTES + CRC_BYTES;
  if (bytes.length < minimumBytes) throw new Error('MGO1 payload is too short.');
  if (bytes.length > MAX_MGO1_BYTES) throw new Error('MGO1 payload exceeds the size limit.');

  const contentLimit = bytes.length - CRC_BYTES;
  if (crc32(bytes.subarray(0, contentLimit)) !== storedCrc32(bytes)) throw new Error('MGO1 CRC mismatch.');
  for (let index = 0; index < MAGIC.length; index += 1) {
    if (bytes[index] !== MAGIC[index]) throw new Error('Invalid MGO1 magic.');
  }
  if (bytes[MAGIC.length] !== MGO1_FORMAT_VERSION) throw new Error('Unsupported MGO1 format version.');

  let offset = HEADER_FIXED_BYTES;
  const rulesResult = decodeUleb128(bytes, offset, contentLimit, 'rulesVersion');
  const rulesVersion = rulesResult.value;
  offset = rulesResult.offset;
  if (offset + HEADER_TAIL_BYTES > contentLimit) throw new Error('Truncated MGO1 header.');

  const gameCode = bytes[offset];
  const gameId = MINIGAME_CODE_TO_ID[gameCode];
  if (!gameId) throw new Error(`Unknown minigame code: ${gameCode}.`);
  const difficulty = bytes[offset + 1];
  const flags = bytes[offset + 2];
  offset += 3;
  const runId = bytesToUuid(bytes.subarray(offset, offset + 16));
  offset += 16;
  const seedHex = bytesToHex(bytes.subarray(offset, offset + 16));
  offset += 16;

  const actions = [];
  let ended = false;
  while (offset < contentLimit) {
    if (actions.length >= MAX_MGO1_ACTIONS) throw new Error('MGO1 action count exceeds the limit.');
    if (ended) throw new Error('MGO1 end must be the final action.');
    const decoded = decodeAction(bytes, offset, contentLimit);
    actions.push(decoded.action);
    offset = decoded.offset;
    ended = decoded.action.type === 'end';
  }

  return {
    formatVersion: MGO1_FORMAT_VERSION,
    rulesVersion,
    gameId,
    gameCode,
    difficulty,
    flags,
    runId,
    seedHex,
    actions,
  };
}

function canonicalJson(value, seen = new Set()) {
  if (value === null) return 'null';
  if (typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('Canonical state cannot contain a non-finite number.');
    return JSON.stringify(Object.is(value, -0) ? 0 : value);
  }
  if (typeof value === 'bigint') return `{"$bigint":${JSON.stringify(value.toString())}}`;
  if (Array.isArray(value)) {
    if (seen.has(value)) throw new Error('Canonical state cannot contain cycles.');
    seen.add(value);
    const result = `[${value.map((item) => canonicalJson(item, seen)).join(',')}]`;
    seen.delete(value);
    return result;
  }
  if (typeof value === 'object') {
    if (seen.has(value)) throw new Error('Canonical state cannot contain cycles.');
    seen.add(value);
    const keys = Object.keys(value).sort();
    const result = `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key], seen)}`).join(',')}}`;
    seen.delete(value);
    return result;
  }
  throw new Error(`Unsupported canonical state value: ${typeof value}.`);
}

export function canonicalStateDigest(state) {
  const bytes = new TextEncoder().encode(canonicalJson(state));
  let digest = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    digest ^= BigInt(byte);
    digest = BigInt.asUintN(64, digest * 0x100000001b3n);
  }
  return digest;
}

export const encode = encodeMgo1;
export const decode = decodeMgo1;
