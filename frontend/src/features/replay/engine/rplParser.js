export const RPL_RECORD_BYTES = 25;
export const MAX_RPL_BYTES = 500 * 1024;

const SENTINEL = Object.freeze({
  change: 88,
  rates: [666666666, 233333333, 314159265, 987654321],
});

export class ReplayFormatError extends Error {
  constructor(message, code = 'INVALID_REPLAY') {
    super(message);
    this.name = 'ReplayFormatError';
    this.code = code;
  }
}

function readUint64LE(view, offset) {
  const low = BigInt(view.getUint32(offset, true));
  const high = BigInt(view.getUint32(offset + 4, true));
  return (high << 32n) | low;
}

function sourceBuffer(source) {
  if (source instanceof ArrayBuffer) return source;
  if (ArrayBuffer.isView(source)) {
    return source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
  }
  throw new ReplayFormatError('Replay data is not a binary buffer.');
}

export function parseRplArrayBuffer(source, { maxBytes = MAX_RPL_BYTES } = {}) {
  const buffer = sourceBuffer(source);
  if (buffer.byteLength <= 0) {
    throw new ReplayFormatError('Replay file is empty.', 'EMPTY_REPLAY');
  }
  if (buffer.byteLength > maxBytes) {
    throw new ReplayFormatError('Replay file exceeds the size limit.', 'REPLAY_TOO_LARGE');
  }
  if (buffer.byteLength % RPL_RECORD_BYTES !== 0) {
    throw new ReplayFormatError('Replay file has an invalid record size.');
  }

  const storedCount = buffer.byteLength / RPL_RECORD_BYTES;
  if (storedCount < 2) {
    throw new ReplayFormatError('Replay file contains no moves.');
  }

  const view = new DataView(buffer);
  const sentinelOffset = (storedCount - 1) * RPL_RECORD_BYTES;
  const sentinelRates = [0, 1, 2, 3].map((index) => (
    view.getUint32(sentinelOffset + 9 + index * 4, true)
  ));
  if (
    view.getUint8(sentinelOffset + 8) !== SENTINEL.change
    || sentinelRates.some((value, index) => value !== SENTINEL.rates[index])
  ) {
    throw new ReplayFormatError('Replay file sentinel is missing or corrupted.');
  }

  const moveCount = storedCount - 1;
  const boards = new BigUint64Array(moveCount);
  const changes = new Uint8Array(moveCount);
  const rates = new Uint32Array(moveCount * 4);
  for (let index = 0; index < moveCount; index += 1) {
    const offset = index * RPL_RECORD_BYTES;
    boards[index] = readUint64LE(view, offset);
    changes[index] = view.getUint8(offset + 8);
    for (let direction = 0; direction < 4; direction += 1) {
      rates[index * 4 + direction] = view.getUint32(offset + 9 + direction * 4, true);
    }
  }

  const terminalValue = readUint64LE(view, sentinelOffset);
  return {
    boards,
    changes,
    rates,
    terminalBoard: terminalValue === 0n ? null : terminalValue,
    moveCount,
    byteLength: buffer.byteLength,
  };
}
