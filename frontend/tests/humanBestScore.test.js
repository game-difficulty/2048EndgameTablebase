import test from 'node:test';
import assert from 'node:assert/strict';
import { activeBestScore } from '../src/human/bestScore.js';

test('formal play updates BEST immediately when the active score passes the stored record', () => {
  const bests = { '4x4': 1000, '3x4': 800 };
  assert.equal(activeBestScore(bests, { variant: '4x4', score: 999 }, '4x4'), 1000);
  assert.equal(activeBestScore(bests, { variant: '4x4', score: 1004 }, '4x4'), 1004);
});

test('one variant cannot change another variant BEST', () => {
  const bests = { '4x4': 1000, '3x4': 800 };
  assert.equal(activeBestScore(bests, { variant: '3x4', score: 1200 }, '4x4'), 1000);
  assert.equal(activeBestScore(bests, { variant: '3x4', score: 1200 }, '3x4'), 1200);
});

test('missing and malformed values fall back without lowering BEST', () => {
  assert.equal(activeBestScore({}, null, '4x4'), 0);
  assert.equal(activeBestScore({ '4x4': 500 }, { variant: '4x4', score: 'bad' }, '4x4'), 500);
});

test('a locally raised record survives a stale server refresh', () => {
  const local = { '4x4': 1200 };
  const server = { '4x4': 1000, '3x4': 700 };
  const merged = { ...local };
  for (const [key, value] of Object.entries(server)) {
    merged[key] = Math.max(Number(merged[key]) || 0, Number(value) || 0);
  }
  assert.deepEqual(merged, { '4x4': 1200, '3x4': 700 });
});
