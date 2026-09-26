import fs from "node:fs";
import { pathToFileURL } from "node:url";

export const VARIANTS = Object.freeze([
  { name: "4x4", columns: 4, rows: 4 },
  { name: "3x4", columns: 4, rows: 3 },
  { name: "3x3", columns: 3, rows: 3 },
  { name: "2x4", columns: 4, rows: 2 },
]);

const HISTORY_MAGIC = Uint8Array.from([0x56, 0x48, 0x52, 0x31]); // VHR1
const SEGMENT_V1_MAGIC = Uint8Array.from([0x56, 0x48, 0x53, 0x31]); // VHS1
const SEGMENT_V2_MAGIC = Uint8Array.from([0x56, 0x48, 0x53, 0x32]); // VHS2
const PASS_NAMES = Object.freeze(["date_desc", "date_asc", "score_desc", "score_asc"]);

function writeVarUint(output, value) {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`Invalid unsigned integer: ${value}`);
  do {
    let byte = value % 128;
    value = Math.floor(value / 128);
    if (value > 0) byte += 128;
    output.push(byte);
  } while (value > 0);
}

function writeUint32(output, value) {
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new RangeError(`Value is outside uint32: ${value}`);
  }
  output.push(value & 255, Math.floor(value / 256) & 255, Math.floor(value / 65536) & 255, Math.floor(value / 16777216) & 255);
}

function packBoard(output, exponents, cellCount) {
  if (!Array.isArray(exponents) || exponents.length !== cellCount) {
    throw new RangeError(`Expected ${cellCount} tile exponents`);
  }
  let buffer = 0;
  let bitCount = 0;
  for (const exponent of exponents) {
    if (!Number.isInteger(exponent) || exponent < 0 || exponent > 31) {
      throw new RangeError(`Tile exponent is outside 0..31: ${exponent}`);
    }
    buffer += exponent * (2 ** bitCount);
    bitCount += 5;
    while (bitCount >= 8) {
      output.push(buffer & 255);
      buffer = Math.floor(buffer / 256);
      bitCount -= 8;
    }
  }
  if (bitCount > 0) output.push(buffer & 255);
}

function parseLocalIso(value) {
  const match = String(value).match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})$/);
  if (!match) throw new TypeError(`Invalid local timestamp: ${value}`);
  const seconds = Math.floor(Date.UTC(...match.slice(1).map(Number).map((part, index) => index === 1 ? part - 1 : part)) / 1000);
  if (seconds < 0 || seconds > 0xffffffff) throw new RangeError(`Timestamp is outside VHR1 range: ${value}`);
  return seconds;
}

function formatLocalSeconds(seconds) {
  return new Date(seconds * 1000).toISOString().slice(0, 19);
}

function boardCodeToExponents(code, expectedCells) {
  if (typeof code !== "string" || code.length !== expectedCells) {
    throw new RangeError(`Expected a ${expectedCells}-character board code`);
  }
  return [...code].map((character) => {
    const exponent = Number.parseInt(character, 36);
    if (!Number.isInteger(exponent) || exponent < 0 || exponent > 31) {
      throw new RangeError(`Invalid tile exponent character: ${character}`);
    }
    return exponent;
  });
}

function normalizeRecord(record, cellCount) {
  const playedAtSeconds = record.playedAtSeconds ?? parseLocalIso(
    record.played_at_local ?? record.played_at?.local_iso,
  );
  const tileExponents = record.tileExponents
    ?? record.final_board?.tile_exponents
    ?? boardCodeToExponents(record.final_board?.encoding, cellCount);
  return { score: record.score, playedAtSeconds, tileExponents };
}

export function encodeVerseHistory(input) {
  const output = [...HISTORY_MAGIC];
  const collectedAtSeconds = input.collectedAtSeconds
    ?? Math.floor(new Date(input.collected_at ?? Date.now()).getTime() / 1000);
  writeUint32(output, collectedAtSeconds);

  for (const variant of VARIANTS) {
    const source = input.variants?.[variant.name];
    const sourceRecords = Array.isArray(source) ? source : source?.records ?? [];
    const records = sourceRecords
      .map((record) => normalizeRecord(record, variant.columns * variant.rows))
      .sort((left, right) => left.playedAtSeconds - right.playedAtSeconds || right.score - left.score);
    writeVarUint(output, records.length);
    let previousTime = 0;
    for (let index = 0; index < records.length; index += 1) {
      const record = records[index];
      if (index === 0) writeUint32(output, record.playedAtSeconds);
      else writeVarUint(output, record.playedAtSeconds - previousTime);
      writeVarUint(output, record.score);
      packBoard(output, record.tileExponents, variant.columns * variant.rows);
      previousTime = record.playedAtSeconds;
    }
  }
  return Uint8Array.from(output);
}

export function encodeVerseHistorySegment(input) {
  const variantIndex = VARIANTS.findIndex(({ name }) => name === input.variant);
  if (variantIndex < 0) throw new TypeError(`Unknown variant: ${input.variant}`);
  const variant = VARIANTS[variantIndex];
  const records = (input.records ?? [])
    .map((record) => normalizeRecord(record, variant.columns * variant.rows))
    .sort((left, right) => left.playedAtSeconds - right.playedAtSeconds || right.score - left.score);
  const pageDeclared = input.pageDeclared;
  const rawRead = input.rawRead;
  if (!Number.isSafeInteger(pageDeclared) || pageDeclared < 0) throw new RangeError("Invalid pageDeclared count");
  if (!Number.isSafeInteger(rawRead) || rawRead < records.length) throw new RangeError("Invalid rawRead count");

  const output = [...SEGMENT_V1_MAGIC, variantIndex];
  const collectedAtSeconds = input.collectedAtSeconds
    ?? Math.floor(new Date(input.collected_at ?? Date.now()).getTime() / 1000);
  writeUint32(output, collectedAtSeconds);
  writeVarUint(output, pageDeclared);
  writeVarUint(output, rawRead);
  writeVarUint(output, records.length);
  writeVarUint(output, input.maximumScore ?? records.reduce((maximum, record) => Math.max(maximum, record.score), 0));
  let previousTime = 0;
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (index === 0) writeUint32(output, record.playedAtSeconds);
    else writeVarUint(output, record.playedAtSeconds - previousTime);
    writeVarUint(output, record.score);
    packBoard(output, record.tileExponents, variant.columns * variant.rows);
    previousTime = record.playedAtSeconds;
  }
  return Uint8Array.from(output);
}

export function encodeVerseHistorySegmentV2(input) {
  const variantIndex = VARIANTS.findIndex(({ name }) => name === input.variant);
  if (variantIndex < 0) throw new TypeError(`Unknown variant: ${input.variant}`);
  const variant = VARIANTS[variantIndex];
  const records = (input.records ?? []).map((record) => {
    if (!Number.isSafeInteger(record.id) || record.id < 0) throw new RangeError(`Invalid game id: ${record.id}`);
    if (!Number.isSafeInteger(record.playedAtMilliseconds) || record.playedAtMilliseconds < 0) {
      throw new RangeError(`Invalid UTC millisecond timestamp: ${record.playedAtMilliseconds}`);
    }
    const normalized = normalizeRecord({ ...record, playedAtSeconds: 0 }, variant.columns * variant.rows);
    return {
      id: record.id,
      playedAtMilliseconds: record.playedAtMilliseconds,
      score: normalized.score,
      tileExponents: normalized.tileExponents,
    };
  }).sort((left, right) =>
    left.playedAtMilliseconds - right.playedAtMilliseconds || left.id - right.id,
  );
  const pageDeclared = input.pageDeclared;
  const rawRead = input.rawRead;
  if (!Number.isSafeInteger(pageDeclared) || pageDeclared < 0) throw new RangeError("Invalid pageDeclared count");
  if (!Number.isSafeInteger(rawRead) || rawRead < records.length) throw new RangeError("Invalid rawRead count");
  if (!Number.isInteger(input.passMask) || input.passMask < 1 || input.passMask > 15) {
    throw new RangeError(`Invalid retrieval pass mask: ${input.passMask}`);
  }

  const output = [...SEGMENT_V2_MAGIC, variantIndex, input.passMask];
  const collectedAtMilliseconds = input.collectedAtMilliseconds ?? Date.now();
  writeVarUint(output, collectedAtMilliseconds);
  writeVarUint(output, pageDeclared);
  writeVarUint(output, rawRead);
  writeVarUint(output, records.length);
  writeVarUint(output, input.maximumScore ?? records.reduce((maximum, record) => Math.max(maximum, record.score), 0));
  let previousTime = 0;
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (index === 0) writeVarUint(output, record.playedAtMilliseconds);
    else writeVarUint(output, record.playedAtMilliseconds - previousTime);
    writeVarUint(output, record.id);
    writeVarUint(output, record.score);
    packBoard(output, record.tileExponents, variant.columns * variant.rows);
    previousTime = record.playedAtMilliseconds;
  }
  return Uint8Array.from(output);
}

class Reader {
  constructor(bytes) {
    this.bytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    this.offset = 0;
  }

  readByte() {
    if (this.offset >= this.bytes.length) throw new RangeError("Unexpected end of VHR file");
    return this.bytes[this.offset++];
  }

  readUint32() {
    return this.readByte() + this.readByte() * 256 + this.readByte() * 65536 + this.readByte() * 16777216;
  }

  readVarUint() {
    let result = 0;
    let multiplier = 1;
    for (let count = 0; count < 8; count += 1) {
      const byte = this.readByte();
      result += (byte & 127) * multiplier;
      if (byte < 128) {
        if (!Number.isSafeInteger(result)) throw new RangeError("VHR integer exceeds JavaScript safe integer range");
        return result;
      }
      multiplier *= 128;
    }
    throw new RangeError("Invalid VHR variable-length integer");
  }

  readBoard(cellCount) {
    const byteCount = Math.ceil(cellCount * 5 / 8);
    const exponents = [];
    let buffer = 0;
    let bitCount = 0;
    for (let index = 0; index < byteCount; index += 1) {
      buffer += this.readByte() * (2 ** bitCount);
      bitCount += 8;
      while (bitCount >= 5 && exponents.length < cellCount) {
        exponents.push(buffer & 31);
        buffer = Math.floor(buffer / 32);
        bitCount -= 5;
      }
    }
    return exponents;
  }
}

export function decodeVerseHistory(bytes) {
  const reader = new Reader(bytes);
  for (const expected of HISTORY_MAGIC) {
    if (reader.readByte() !== expected) throw new TypeError("Not a VHR1 history file");
  }
  const collectedAtSeconds = reader.readUint32();
  const variants = {};
  for (const variant of VARIANTS) {
    const recordCount = reader.readVarUint();
    const records = [];
    let previousTime = 0;
    for (let index = 0; index < recordCount; index += 1) {
      const playedAtSeconds = index === 0 ? reader.readUint32() : previousTime + reader.readVarUint();
      const score = reader.readVarUint();
      const tileExponents = reader.readBoard(variant.columns * variant.rows);
      records.push({
        score,
        played_at_local: formatLocalSeconds(playedAtSeconds),
        tile_exponents: tileExponents,
      });
      previousTime = playedAtSeconds;
    }
    variants[variant.name] = { record_count: records.length, records };
  }
  if (reader.offset !== reader.bytes.length) {
    throw new RangeError(`VHR file has ${reader.bytes.length - reader.offset} trailing bytes`);
  }
  return {
    format: "VHR1",
    collected_at: new Date(collectedAtSeconds * 1000).toISOString(),
    ordering: "played_at_local_ascending",
    timezone: null,
    variants,
  };
}

export function decodeVerseHistorySegment(bytes) {
  const source = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const magic = String.fromCharCode(...source.subarray(0, 4));
  if (magic === "VHS2") return decodeVerseHistorySegmentV2(source);
  const reader = new Reader(bytes);
  for (const expected of SEGMENT_V1_MAGIC) {
    if (reader.readByte() !== expected) throw new TypeError("Not a VHS1 history segment");
  }
  const variantIndex = reader.readByte();
  const variant = VARIANTS[variantIndex];
  if (!variant) throw new RangeError(`Unknown VHS1 variant id: ${variantIndex}`);
  const collectedAtSeconds = reader.readUint32();
  const pageDeclared = reader.readVarUint();
  const rawRead = reader.readVarUint();
  const uniqueCount = reader.readVarUint();
  const maximumScore = reader.readVarUint();
  const records = [];
  let previousTime = 0;
  for (let index = 0; index < uniqueCount; index += 1) {
    const playedAtSeconds = index === 0 ? reader.readUint32() : previousTime + reader.readVarUint();
    const score = reader.readVarUint();
    const tileExponents = reader.readBoard(variant.columns * variant.rows);
    records.push({
      score,
      played_at_local: formatLocalSeconds(playedAtSeconds),
      tile_exponents: tileExponents,
    });
    previousTime = playedAtSeconds;
  }
  if (reader.offset !== reader.bytes.length) {
    throw new RangeError(`VHS file has ${reader.bytes.length - reader.offset} trailing bytes`);
  }
  if (rawRead < uniqueCount) throw new RangeError("VHS raw count is smaller than its unique count");
  const actualMaximum = records.reduce((maximum, record) => Math.max(maximum, record.score), 0);
  if (actualMaximum !== maximumScore) {
    throw new RangeError(`VHS maximum score ${maximumScore} does not match record maximum ${actualMaximum}`);
  }
  const uniqueKeys = new Set(records.map((record) =>
    `${record.score}|${record.played_at_local}|${record.tile_exponents.join(",")}`,
  ));
  if (uniqueKeys.size !== uniqueCount) throw new RangeError("VHS record body contains duplicate entries");
  return {
    format: "VHS1",
    variant: variant.name,
    collected_at: new Date(collectedAtSeconds * 1000).toISOString(),
    ordering: "played_at_local_ascending",
    timezone: null,
    counts: {
      page_declared: pageDeclared,
      raw_read: rawRead,
      unique: uniqueCount,
      duplicates: rawRead - uniqueCount,
    },
    maximum_score: maximumScore,
    records,
  };
}

export function decodeVerseHistorySegmentV2(bytes) {
  const reader = new Reader(bytes);
  for (const expected of SEGMENT_V2_MAGIC) {
    if (reader.readByte() !== expected) throw new TypeError("Not a VHS2 history segment");
  }
  const variantIndex = reader.readByte();
  const variant = VARIANTS[variantIndex];
  if (!variant) throw new RangeError(`Unknown VHS2 variant id: ${variantIndex}`);
  const passMask = reader.readByte();
  if (passMask < 1 || passMask > 15) throw new RangeError(`Invalid VHS2 retrieval pass mask: ${passMask}`);
  const collectedAtMilliseconds = reader.readVarUint();
  const pageDeclared = reader.readVarUint();
  const rawRead = reader.readVarUint();
  const uniqueCount = reader.readVarUint();
  const maximumScore = reader.readVarUint();
  const records = [];
  let previousTime = 0;
  for (let index = 0; index < uniqueCount; index += 1) {
    const playedAtMilliseconds = index === 0 ? reader.readVarUint() : previousTime + reader.readVarUint();
    const id = reader.readVarUint();
    const score = reader.readVarUint();
    const tileExponents = reader.readBoard(variant.columns * variant.rows);
    records.push({
      id,
      score,
      played_at: new Date(playedAtMilliseconds).toISOString(),
      played_at_milliseconds: playedAtMilliseconds,
      tile_exponents: tileExponents,
    });
    previousTime = playedAtMilliseconds;
  }
  if (reader.offset !== reader.bytes.length) {
    throw new RangeError(`VHS2 file has ${reader.bytes.length - reader.offset} trailing bytes`);
  }
  if (rawRead < uniqueCount) throw new RangeError("VHS2 raw count is smaller than its unique count");
  const actualMaximum = records.reduce((maximum, record) => Math.max(maximum, record.score), 0);
  if (actualMaximum !== maximumScore) {
    throw new RangeError(`VHS2 maximum score ${maximumScore} does not match record maximum ${actualMaximum}`);
  }
  if (new Set(records.map((record) => record.id)).size !== uniqueCount) {
    throw new RangeError("VHS2 record body contains duplicate game ids");
  }
  return {
    format: "VHS2",
    variant: variant.name,
    collected_at: new Date(collectedAtMilliseconds).toISOString(),
    ordering: "played_at_utc_ascending",
    timestamp: "utc_milliseconds",
    retrieval_passes: PASS_NAMES.filter((_, index) => passMask & (1 << index)),
    pass_mask: passMask,
    counts: {
      page_declared: pageDeclared,
      raw_read: rawRead,
      unique: uniqueCount,
      duplicates: rawRead - uniqueCount,
    },
    maximum_score: maximumScore,
    records,
  };
}

function printSummary(filePath) {
  const bytes = fs.readFileSync(filePath);
  const magic = bytes.subarray(0, 4).toString("ascii");
  if (magic === "VHS1" || magic === "VHS2") {
    const decoded = decodeVerseHistorySegment(bytes);
    process.stdout.write(`${JSON.stringify({
      file: filePath,
      bytes: bytes.length,
      format: decoded.format,
      variant: decoded.variant,
      maximum_score: decoded.maximum_score,
      retrieval_passes: decoded.retrieval_passes,
      ...decoded.counts,
    }, null, 2)}\n`);
    return;
  }
  const decoded = decodeVerseHistory(bytes);
  const counts = Object.fromEntries(VARIANTS.map(({ name }) => [name, decoded.variants[name].record_count]));
  process.stdout.write(`${JSON.stringify({ file: filePath, bytes: bytes.length, ...counts }, null, 2)}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.argv[2]) {
    process.stderr.write("Usage: node verse-history-codec.mjs <history.vhr|variant.vhs>\n");
    process.exitCode = 2;
  } else {
    printSummary(process.argv[2]);
  }
}
