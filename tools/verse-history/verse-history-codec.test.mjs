import assert from "node:assert/strict";
import test from "node:test";
import {
  decodeVerseHistory,
  decodeVerseHistorySegment,
  encodeVerseHistory,
  encodeVerseHistorySegment,
  encodeVerseHistorySegmentV2,
} from "./verse-history-codec.mjs";

test("VHR1 round-trips 65536 and 131072 tiles without a uint64 board", () => {
  const board = [16, 17, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
  const encoded = encodeVerseHistory({
    collectedAtSeconds: 1_790_000_000,
    variants: {
      "4x4": [{ score: 999_999, playedAtSeconds: 1_780_000_001, tileExponents: board }],
    },
  });
  const decoded = decodeVerseHistory(encoded);
  assert.deepEqual(decoded.variants["4x4"].records[0].tile_exponents, board);
  assert.equal(decoded.variants["4x4"].records[0].tile_exponents[0], 16);
  assert.equal(2 ** decoded.variants["4x4"].records[0].tile_exponents[0], 65536);
  assert.equal(2 ** decoded.variants["4x4"].records[0].tile_exponents[1], 131072);
});

test("VHR1 stores variant identity once and preserves chronological records", () => {
  const encoded = encodeVerseHistory({
    collectedAtSeconds: 1_790_000_000,
    variants: {
      "2x4": [
        { score: 20, playedAtSeconds: 1_780_000_020, tileExponents: [1, 2, 3, 4, 5, 6, 7, 8] },
        { score: 10, playedAtSeconds: 1_780_000_010, tileExponents: [8, 7, 6, 5, 4, 3, 2, 1] },
      ],
    },
  });
  const decoded = decodeVerseHistory(encoded);
  assert.deepEqual(decoded.variants["2x4"].records.map((record) => record.score), [10, 20]);
  assert.equal(decoded.variants["4x4"].record_count, 0);
});

test("VHS1 preserves declared, raw and unique counts in a resumable variant segment", () => {
  const records = [
    { score: 20, playedAtSeconds: 1_780_000_020, tileExponents: [1, 2, 3, 4, 5, 6, 7, 8, 9] },
    { score: 10, playedAtSeconds: 1_780_000_010, tileExponents: [9, 8, 7, 6, 5, 4, 3, 2, 1] },
  ];
  const encoded = encodeVerseHistorySegment({
    variant: "3x3",
    collectedAtSeconds: 1_790_000_000,
    pageDeclared: 3,
    rawRead: 3,
    maximumScore: 20,
    records,
  });
  const decoded = decodeVerseHistorySegment(encoded);
  assert.deepEqual(decoded.counts, {
    page_declared: 3,
    raw_read: 3,
    unique: 2,
    duplicates: 1,
  });
  assert.equal(decoded.variant, "3x3");
  assert.deepEqual(decoded.records.map((record) => record.score), [10, 20]);
});

test("VHS2 preserves game ids and exact UTC milliseconds", () => {
  const encoded = encodeVerseHistorySegmentV2({
    variant: "4x4",
    collectedAtMilliseconds: 1_790_000_000_123,
    pageDeclared: 2,
    rawRead: 4,
    maximumScore: 999_999,
    passMask: 0b0011,
    records: [
      {
        id: 1_647_683,
        score: 999_999,
        playedAtMilliseconds: 1_780_000_000_651,
        tileExponents: [16, 17, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      },
      {
        id: 1_647_684,
        score: 888_888,
        playedAtMilliseconds: 1_780_000_000_652,
        tileExponents: [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 16, 17],
      },
    ],
  });
  const decoded = decodeVerseHistorySegment(encoded);
  assert.equal(decoded.format, "VHS2");
  assert.deepEqual(decoded.retrieval_passes, ["date_desc", "date_asc"]);
  assert.deepEqual(decoded.counts, {
    page_declared: 2,
    raw_read: 4,
    unique: 2,
    duplicates: 2,
  });
  assert.equal(decoded.records[0].id, 1_647_683);
  assert.equal(decoded.records[0].played_at_milliseconds, 1_780_000_000_651);
  assert.equal(decoded.records[0].tile_exponents[0], 16);
  assert.equal(decoded.records[0].tile_exponents[1], 17);
});
