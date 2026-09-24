import test from 'node:test';
import assert from 'node:assert/strict';
import { gunzipSync } from 'node:zlib';
import { decodeReceipt, uploadBody } from '../src/human/wire.js';

test('compact receipts preserve active, archived and rejected state and the exact permit expiry', () => {
  const token = '1790170012123.signature';
  const value = decodeReceipt([2, 123, 4, 4, 1790170000123, token]);
  assert.deepEqual(value, { seq: 123, epoch: 4, status: 'active', monitored: true, eligibility: 'eligible',
    server_time: 1790170000.123, permit_until: 1790170012.123, permit: token });
  assert.equal(decodeReceipt([2, 123, 4, 14, 1790170000123, '']).status, 'sealed');
  assert.equal(decodeReceipt([2, 123, 4, 14, 1790170000123, '']).eligibility, 'disqualified');
  assert.equal(decodeReceipt([2, 123, 4, 1, 1790170000123, '']).status, 'pending_archive');
  const old = { seq: 2, prefix_hash: 'legacy' };
  assert.equal(decodeReceipt(old), old);
  assert.throws(() => decodeReceipt([2, 1, 2, 3, 4, '']));
});

test('20-move batches remain uncompressed with no encoding header', async () => {
  const raw = new Uint8Array(100);
  const result = await uploadBody(raw);
  assert.equal(result.body, raw); assert.deepEqual(result.headers, {});
});

test('large upload byte planes preserve all 32 timestamp bits through native gzip', async () => {
  const raw = new Uint8Array(50000), view = new DataView(raw.buffer), count = raw.length / 5;
  for (let i = 0; i < count; i++) { view.setUint8(i * 5, i % 128); view.setUint32(i * 5 + 1, i % 2 ? 0xffffffff : 913, true); }
  const result = await uploadBody(raw);
  assert.equal(result.headers['Content-Encoding'], 'gzip');
  assert.equal(result.headers['X-Human-Layout'], 'planes5');
  assert.ok(result.body.length < raw.length);
  const packed = gunzipSync(result.body), restored = new Uint8Array(raw.length);
  for (let i = 0; i < count; i++) for (let lane = 0; lane < 5; lane++) restored[i * 5 + lane] = packed[lane * count + i];
  assert.deepEqual(restored, raw);
});

test('browsers without CompressionStream retain the original binary protocol', async () => {
  const original = globalThis.CompressionStream;
  try {
    globalThis.CompressionStream = undefined;
    const raw = new Uint8Array(5000), result = await uploadBody(raw);
    assert.equal(result.body, raw); assert.deepEqual(result.headers, {});
  } finally { globalThis.CompressionStream = original; }
});
