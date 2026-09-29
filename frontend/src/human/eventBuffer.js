import { eventBytes } from './engine.js';

const HASH_BYTES = 32;
const HEX = Array.from({ length: 256 }, (_, value) => value.toString(16).padStart(2, '0'));

function decodeHash(value, target, offset) {
  if (typeof value !== 'string' || !/^[0-9a-f]{64}$/i.test(value)) throw new Error('invalid_event_hash');
  for (let index = 0; index < HASH_BYTES; index++) {
    const byte = Number.parseInt(value.slice(index * 2, index * 2 + 2), 16);
    if (!Number.isFinite(byte)) throw new Error('invalid_event_hash');
    target[offset + index] = byte;
  }
}

function encodeHash(source, offset) {
  let value = '';
  for (let index = 0; index < HASH_BYTES; index++) value += HEX[source[offset + index]];
  return value;
}

export class EventBuffer {
  constructor(events = []) {
    this.length = 0;
    this.capacity = Math.max(16, events.length);
    this.codes = new Uint8Array(this.capacity);
    this.deltas = new Uint32Array(this.capacity);
    this.hashes = new Uint8Array(this.capacity * HASH_BYTES);
    for (const event of events) this.push(event);
  }

  ensure(required) {
    if (required <= this.capacity) return;
    let capacity = this.capacity;
    while (capacity < required) capacity *= 2;
    const codes = new Uint8Array(capacity); codes.set(this.codes);
    const deltas = new Uint32Array(capacity); deltas.set(this.deltas);
    const hashes = new Uint8Array(capacity * HASH_BYTES); hashes.set(this.hashes);
    this.capacity = capacity; this.codes = codes; this.deltas = deltas; this.hashes = hashes;
  }

  push(event) {
    const code = Number(event?.[0]), delta = Number(event?.[1]);
    if (!Number.isInteger(code) || code < 0 || code > 255 || !Number.isInteger(delta) || delta < 0 || delta > 0xffffffff) {
      throw new Error('invalid_event');
    }
    this.ensure(this.length + 1);
    this.codes[this.length] = code; this.deltas[this.length] = delta;
    decodeHash(event[2], this.hashes, this.length * HASH_BYTES);
    this.length += 1;
    return this.length;
  }

  pop() {
    if (!this.length) return undefined;
    const event = this.at(this.length - 1); this.length -= 1; return event;
  }

  at(index) {
    const resolved = index < 0 ? this.length + index : index;
    if (resolved < 0 || resolved >= this.length) return undefined;
    return [this.codes[resolved], this.deltas[resolved], encodeHash(this.hashes, resolved * HASH_BYTES)];
  }

  replayAt(index) {
    const resolved = index < 0 ? this.length + index : index;
    return resolved < 0 || resolved >= this.length ? undefined : [this.codes[resolved], this.deltas[resolved]];
  }

  countCodeMask(mask) {
    let count = 0;
    for (let index = 0; index < this.length; index++) if (this.codes[index] & mask) count += 1;
    return count;
  }

  hashAt(index) {
    const resolved = index < 0 ? this.length + index : index;
    return resolved < 0 || resolved >= this.length ? undefined : encodeHash(this.hashes, resolved * HASH_BYTES);
  }

  slice(start = 0, end = this.length) {
    const from = Math.max(0, start < 0 ? this.length + start : start);
    const to = Math.max(from, Math.min(this.length, end < 0 ? this.length + end : end));
    return Array.from({ length: to - from }, (_, index) => this.at(from + index));
  }

  clone(end = this.length) {
    const count = Math.max(0, Math.min(this.length, end));
    const result = new EventBuffer(); result.ensure(count); result.length = count;
    result.codes.set(this.codes.subarray(0, count));
    result.deltas.set(this.deltas.subarray(0, count));
    result.hashes.set(this.hashes.subarray(0, count * HASH_BYTES));
    return result;
  }

  forEach(callback) { for (let index = 0; index < this.length; index++) callback(this.at(index), index, this); }
  map(callback) { return Array.from({ length: this.length }, (_, index) => callback(this.at(index), index, this)); }
  filter(callback) {
    const result = [];
    for (let index = 0; index < this.length; index++) { const event = this.at(index); if (callback(event, index, this)) result.push(event); }
    return result;
  }

  bytes(start = 0) {
    const from = Math.max(0, Math.min(this.length, start));
    const bytes = new Uint8Array((this.length - from) * 5); const view = new DataView(bytes.buffer);
    for (let index = from; index < this.length; index++) {
      const offset = (index - from) * 5;
      view.setUint8(offset, this.codes[index]); view.setUint32(offset + 1, this.deltas[index], true);
    }
    return bytes;
  }

  get allocatedBytes() { return this.codes.byteLength + this.deltas.byteLength + this.hashes.byteLength; }
  *[Symbol.iterator]() { for (let index = 0; index < this.length; index++) yield this.at(index); }
}

export function eventUploadSlice(events, start) {
  return {
    bytes: typeof events.bytes === 'function' ? events.bytes(start) : eventBytes(events.slice(start)),
    prefix: typeof events.hashAt === 'function' ? events.hashAt(start - 1) : events[start - 1]?.[2],
  };
}
