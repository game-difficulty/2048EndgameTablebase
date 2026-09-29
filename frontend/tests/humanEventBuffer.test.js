import test from 'node:test';
import assert from 'node:assert/strict';
import { EventBuffer, eventUploadSlice } from '../src/human/eventBuffer.js';

const hash = value => value.toString(16).padStart(64, '0');

test('compact human event buffer preserves events, hashes and wire bytes', () => {
  const events = Array.from({ length: 100 }, (_, index) => [index & 255, index * 1000, hash(index + 1)]);
  const buffer = new EventBuffer(events);
  assert.equal(buffer.length, events.length);
  assert.deepEqual(buffer.at(17), events[17]);
  assert.deepEqual(buffer.replayAt(17), events[17].slice(0, 2));
  assert.equal(buffer.hashAt(17), events[17][2]);
  assert.equal(buffer.countCodeMask(64), events.filter(event => event[0] & 64).length);
  assert.deepEqual(buffer.slice(98), events.slice(98));
  const bytes = buffer.bytes(99); const view = new DataView(bytes.buffer);
  assert.equal(bytes.length, 5); assert.equal(view.getUint8(0), events[99][0]); assert.equal(view.getUint32(1, true), events[99][1]);
});

test('compact snapshots are independent and keep long games bounded', () => {
  const buffer = new EventBuffer();
  for (let index = 0; index < 100000; index++) buffer.push([index & 255, index, hash(index + 1)]);
  const snapshot = buffer.clone(); buffer.pop(); buffer.push([7, 9, hash(200001)]);
  assert.equal(snapshot.length, 100000);
  assert.notDeepEqual(buffer.at(-1), snapshot.at(-1));
  assert.ok(buffer.allocatedBytes < 8 * 1024 * 1024, `allocated ${buffer.allocatedBytes} bytes`);
});

test('compact and legacy events produce identical upload tails and prefix proofs', () => {
  const events = Array.from({ length: 40 }, (_, index) => [index | 64, index * 17, hash(index + 1)]);
  const compact = eventUploadSlice(new EventBuffer(events), 31);
  const legacy = eventUploadSlice(events, 31);
  assert.deepEqual(compact.bytes, legacy.bytes);
  assert.equal(compact.prefix, legacy.prefix);
});
