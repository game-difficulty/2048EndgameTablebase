import assert from 'node:assert/strict';
import test from 'node:test';

import { MINIGAME_REGISTRY } from '../src/features/minigames/engine/registry.js';
import {
  canonicalStateDigest,
  crc32,
  decode,
  decodeMgo1,
  encode,
  encodeMgo1,
  MAX_MGO1_ACTIONS,
  MGO1_PREFIX,
  MINIGAME_CODE_TO_ID,
  MINIGAME_ID_TO_CODE,
} from '../src/features/minigames/protocol/index.js';

const RUN_ID = '123e4567-e89b-42d3-a456-426614174000';
const SEED = '00112233445566778899aabbccddeeff';

const ALL_ACTIONS = [
  { type: 'move', direction: 'up', deltaMs: 0 },
  { type: 'move', direction: 'right', deltaMs: 1 },
  { type: 'move', direction: 'down', deltaMs: 127 },
  { type: 'move', direction: 'left', deltaMs: 128 },
  { type: 'bomb', index: 15, deltaMs: 250 },
  { type: 'glove', source: 2, target: 13, deltaMs: 16384 },
  { type: 'twist', index: 5, deltaMs: 9 },
  { type: 'custom', actionId: 200, phase: 3, deltaMs: 4 },
  { type: 'tick', deltaMs: 1000 },
  { type: 'digest', digest: 0x0123456789abcdefn, deltaMs: 7 },
  { type: 'end', reason: 2, deltaMs: 11 },
];

function basePayload(overrides = {}) {
  return {
    rulesVersion: 300,
    gameId: 'tricky-tiles',
    difficulty: 1,
    flags: 0xa5,
    runId: RUN_ID,
    seedHex: SEED,
    actions: ALL_ACTIONS,
    ...overrides,
  };
}

function encodedBytes(encoded) {
  return Uint8Array.from(atob(encoded.slice(MGO1_PREFIX.length)), (character) => character.charCodeAt(0));
}

function encodeRawBytes(bytes) {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return MGO1_PREFIX + btoa(binary);
}

function replaceCrc(content) {
  const checksum = crc32(content);
  return Uint8Array.from([
    ...content,
    checksum & 0xff,
    (checksum >>> 8) & 0xff,
    (checksum >>> 16) & 0xff,
    checksum >>> 24,
  ]);
}

test('MGO1 roundtrip preserves the header and every opcode', () => {
  assert.equal(encode, encodeMgo1);
  assert.equal(decode, decodeMgo1);
  const decoded = decode(encode(basePayload()));
  assert.deepEqual(decoded, {
    formatVersion: 1,
    rulesVersion: 300,
    gameId: 'tricky-tiles',
    gameCode: MINIGAME_ID_TO_CODE['tricky-tiles'],
    difficulty: 1,
    flags: 0xa5,
    runId: RUN_ID,
    seedHex: SEED,
    actions: ALL_ACTIONS,
  });
});

test('all 20 registry ids have stable unique uint8 codes', () => {
  const expectedIds = [
    'design-master-1', 'mystery-merge-1', 'column-chaos', 'gravity-twist-1',
    'blitzkrieg', 'tricky-tiles', 'design-master-2', 'shape-shifter',
    'ferris-wheel', 'gravity-twist-2', 'design-master-3', 'mystery-merge-2',
    'ice-age', 'isolated-island', 'design-master-4', 'endless-factorization',
    'endless-explosions', 'endless-giftbox', 'endless-hybrid', 'endless-airraid',
  ];
  assert.deepEqual(Object.keys(MINIGAME_ID_TO_CODE), expectedIds);
  assert.deepEqual(MINIGAME_REGISTRY.map(({ id }) => id), expectedIds);
  for (const [index, gameId] of expectedIds.entries()) {
    assert.equal(MINIGAME_ID_TO_CODE[gameId], index + 1);
    assert.equal(MINIGAME_CODE_TO_ID[index + 1], gameId);
  }
});

test('CRC rejects payload tampering', () => {
  const bytes = encodedBytes(encodeMgo1(basePayload()));
  bytes[10] ^= 0x01;
  assert.throws(() => decodeMgo1(encodeRawBytes(bytes)), /CRC mismatch/u);
});

test('malformed and non-canonical ULEB128 values are rejected', () => {
  const good = encodedBytes(encodeMgo1(basePayload({ rulesVersion: 1, actions: [] })));
  const content = good.subarray(0, good.length - 4);

  const unterminated = Uint8Array.from([
    ...content.subarray(0, 5),
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ...content.subarray(6),
  ]);
  assert.throws(() => decodeMgo1(encodeRawBytes(replaceCrc(unterminated))), /rulesVersion is too long/u);

  const nonCanonical = Uint8Array.from([
    ...content.subarray(0, 5),
    0x81, 0x00,
    ...content.subarray(6),
  ]);
  assert.throws(() => decodeMgo1(encodeRawBytes(replaceCrc(nonCanonical))), /Non-canonical rulesVersion/u);
});

test('UUID and seed are bound into the encoded header', () => {
  const original = encodeMgo1(basePayload({ actions: [] }));
  const otherRun = encodeMgo1(basePayload({
    actions: [],
    runId: '123e4567-e89b-42d3-a456-426614174001',
  }));
  const otherSeed = encodeMgo1(basePayload({
    actions: [],
    seedHex: 'ffeeddccbbaa99887766554433221100',
  }));
  assert.notEqual(original, otherRun);
  assert.notEqual(original, otherSeed);
  assert.equal(decodeMgo1(otherRun).runId, '123e4567-e89b-42d3-a456-426614174001');
  assert.equal(decodeMgo1(otherSeed).seedHex, 'ffeeddccbbaa99887766554433221100');
  assert.throws(() => encodeMgo1(basePayload({ runId: 'not-a-uuid' })), /valid UUID/u);
  assert.throws(() => encodeMgo1(basePayload({ seedHex: 'abcd' })), /32 hexadecimal digits/u);
});

test('action and payload validation rejects unsafe streams', () => {
  assert.throws(
    () => encodeMgo1(basePayload({ actions: new Array(MAX_MGO1_ACTIONS + 1).fill({ type: 'tick' }) })),
    /action count/u,
  );
  assert.throws(
    () => encodeMgo1(basePayload({ actions: [{ type: 'bomb', index: 256 }] })),
    /index must be an integer/u,
  );
  assert.throws(
    () => encodeMgo1(basePayload({ actions: [{ type: 'end', reason: 0 }, { type: 'tick' }] })),
    /end must be the final action/u,
  );
  assert.throws(
    () => encodeMgo1(basePayload({
      actions: new Array(30000).fill(null).map(() => ({ type: 'digest', digest: 0n })),
    })),
    /size limit/u,
  );
});

test('decoder rejects an unknown opcode with a valid CRC', () => {
  const good = encodedBytes(encodeMgo1(basePayload({ rulesVersion: 1, actions: [] })));
  const content = Uint8Array.from([...good.subarray(0, good.length - 4), 0x55, 0x00]);
  assert.throws(() => decodeMgo1(encodeRawBytes(replaceCrc(content))), /Unknown MGO1 action tag 0x55/u);
});

test('canonical state digest is key-order independent and changes with state', () => {
  const first = canonicalStateDigest({ score: 8, board: [0, 2, 2], powerups: { bomb: 1, glove: 0 } });
  const reordered = canonicalStateDigest({ powerups: { glove: 0, bomb: 1 }, board: [0, 2, 2], score: 8 });
  const changed = canonicalStateDigest({ score: 12, board: [0, 2, 2], powerups: { bomb: 1, glove: 0 } });
  assert.equal(first, reordered);
  assert.notEqual(first, changed);
  assert.equal(first >= 0n && first <= 0xffffffffffffffffn, true);
});
