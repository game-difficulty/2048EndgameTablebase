export const TRAINER_ROUTE_RECORD_BYTES = 17;
export const TRAINER_ROUTE_RATE_SCALE = 4_000_000_000;
export const TRAINER_ROUTE_DIRECTIONS = Object.freeze(['up', 'down', 'left', 'right']);

const DEFAULT_MAX_ROUTE_BYTES = 2 * 1024 * 1024;

export class BattleRouteFormatError extends Error {
  constructor(message) {
    super(message);
    this.name = 'BattleRouteFormatError';
  }
}

function asByteView(source) {
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  if (typeof SharedArrayBuffer !== 'undefined' && source instanceof SharedArrayBuffer) {
    return new Uint8Array(source);
  }
  if (ArrayBuffer.isView(source)) {
    return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  }
  throw new TypeError('battle_route_binary_required');
}

function readRecordRates(view, offset) {
  return [0, 1, 2, 3].map((index) => view.getUint32(offset + 1 + index * 4, true));
}

export function decodeTrainerRouteChange(encoded) {
  const value = Number(encoded) & 0xff;
  return {
    directionCode: value & 0b11,
    direction: TRAINER_ROUTE_DIRECTIONS[value & 0b11],
    spawnIndex: (value >> 2) & 0b1111,
    spawnExponent: ((value >> 6) & 0b1) + 1,
    spawnValue: 2 ** (((value >> 6) & 0b1) + 1),
  };
}

export function routeStepAt(route, index) {
  const step = Math.trunc(Number(index));
  if (!route || step < 0 || step >= Number(route.moveCount || 0)) return null;
  const offset = step * 4;
  return {
    index: step,
    change: route.changes[step],
    ...decodeTrainerRouteChange(route.changes[step]),
    rates: route.rates.slice(offset, offset + 4),
  };
}

export function parseTrainerBattleRoute(source, {
  maxBytes = DEFAULT_MAX_ROUTE_BYTES,
} = {}) {
  const bytes = asByteView(source);
  const byteLimit = Math.max(TRAINER_ROUTE_RECORD_BYTES, Math.trunc(Number(maxBytes) || 0));
  if (bytes.byteLength === 0) throw new BattleRouteFormatError('battle_route_empty');
  if (bytes.byteLength > byteLimit) throw new BattleRouteFormatError('battle_route_too_large');
  if (bytes.byteLength % TRAINER_ROUTE_RECORD_BYTES !== 0) {
    throw new BattleRouteFormatError('battle_route_invalid_size');
  }

  // Copy first so a caller cannot mutate an active route through a shared view.
  const owned = Uint8Array.from(bytes);
  const view = new DataView(owned.buffer);
  const recordCount = owned.byteLength / TRAINER_ROUTE_RECORD_BYTES;
  if (view.getUint8(0) !== 0) throw new BattleRouteFormatError('battle_route_invalid_header');

  const boardParts = readRecordRates(view, 0);
  if (boardParts.some((part) => part > 0xffff)) {
    throw new BattleRouteFormatError('battle_route_invalid_board');
  }
  const initialBoard = boardParts.reduce(
    (board, part, index) => board | (BigInt(part) << BigInt(index * 16)),
    0n,
  );

  const moveCount = recordCount - 1;
  const changes = new Uint8Array(moveCount);
  const rates = new Uint32Array(moveCount * 4);
  for (let index = 0; index < moveCount; index += 1) {
    const offset = (index + 1) * TRAINER_ROUTE_RECORD_BYTES;
    const change = view.getUint8(offset);
    if ((change & 0x80) !== 0) throw new BattleRouteFormatError('battle_route_invalid_change');
    changes[index] = change;
    for (let direction = 0; direction < 4; direction += 1) {
      const rate = view.getUint32(offset + 1 + direction * 4, true);
      if (rate > TRAINER_ROUTE_RATE_SCALE) {
        throw new BattleRouteFormatError('battle_route_invalid_rate');
      }
      rates[index * 4 + direction] = rate;
    }
  }

  return {
    format: 'trainer-route-v1',
    byteLength: owned.byteLength,
    recordCount,
    moveCount,
    initialBoard,
    changes,
    rates,
  };
}
