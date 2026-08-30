import assert from 'node:assert/strict';
import test from 'node:test';

import {
  isBattleResultDraw,
  rankBattleResults,
} from '../src/features/battle/core/battleResultRanking.js';

test('completed players rank before unfinished players by goodness', () => {
  const ranked = rankBattleResults([
    { user_id: 1, status: 'timed_out', goodness_of_fit: 0.99 },
    { user_id: 2, status: 'completed', goodness_of_fit: 0.75 },
    { user_id: 3, status: 'completed', goodness_of_fit: 0.90 },
    { user_id: 4, status: 'disqualified', goodness_of_fit: 1 },
  ]);

  assert.deepEqual(ranked.map((result) => result.user_id), [3, 2, 1, 4]);
  assert.deepEqual(ranked.map((result) => result.rank), [1, 2, null, null]);
});

test('completed ties share rank while unfinished players never receive one', () => {
  const ranked = rankBattleResults([
    { user_id: 1, status: 'completed', goodness_of_fit: 0.8 },
    { user_id: 2, status: 'timed_out', goodness_of_fit: 0.9 },
    { user_id: 3, status: 'completed', goodness_of_fit: 0.8 },
    { user_id: 4, status: 'completed', goodness_of_fit: 0.6 },
  ]);

  assert.deepEqual(ranked.map((result) => result.rank), [1, 1, 3, null]);
  assert.equal(isBattleResultDraw(ranked), true);
});

test('unfinished players alone do not produce a draw or ranks', () => {
  const ranked = rankBattleResults([
    { user_id: 1, status: 'timed_out', goodness_of_fit: 0.5 },
    { user_id: 2, status: 'disqualified', goodness_of_fit: 0.5 },
  ]);

  assert.deepEqual(ranked.map((result) => result.rank), [null, null]);
  assert.equal(isBattleResultDraw(ranked), false);
});

test('live ranking includes active and completed players but leaves forfeits unranked', () => {
  const ranked = rankBattleResults([
    { user_id: 1, status: 'playing', goodness_of_fit: 0.70 },
    { user_id: 2, status: 'completed', goodness_of_fit: 0.90 },
    { user_id: 3, status: 'disconnected', goodness_of_fit: 0.80 },
    { user_id: 4, status: 'timed_out', goodness_of_fit: 1.00 },
    { user_id: 5, status: 'disqualified', goodness_of_fit: 0.95 },
  ], { mode: 'live' });

  assert.deepEqual(ranked.map((result) => result.user_id), [2, 3, 1, 4, 5]);
  assert.deepEqual(ranked.map((result) => result.rank), [1, 2, 3, null, null]);
});

test('open-route ranking puts fixed-step finishes before natural finishes and failures', () => {
  const ranked = rankBattleResults([
    {
      user_id: 1, status: 'completed', goodness_of_fit: 0.99, progress: 31,
      mode_data: { finish_class: 'natural' },
    },
    {
      user_id: 2, status: 'completed', goodness_of_fit: 0.80, progress: 64,
      mode_data: { finish_class: 'completed' },
    },
    {
      user_id: 3, status: 'completed', goodness_of_fit: 0.70, progress: 40,
      mode_data: { finish_class: 'natural' },
    },
    {
      user_id: 4, status: 'timed_out', goodness_of_fit: 1, progress: 63,
      mode_data: { finish_class: 'unranked' },
    },
  ], { battleMode: 'free_goodness' });

  assert.deepEqual(ranked.map((result) => result.user_id), [2, 3, 1, 4]);
  assert.deepEqual(ranked.map((result) => result.rank), [1, 2, 3, null]);
});

test('open-route natural finishes compare progress before average goodness', () => {
  const ranked = rankBattleResults([
    {
      user_id: 1, status: 'completed', goodness_of_fit: 0.95, progress: 20,
      mode_data: { finish_class: 'natural' },
    },
    {
      user_id: 2, status: 'completed', goodness_of_fit: 0.70, progress: 21,
      mode_data: { finish_class: 'natural' },
    },
  ], { battleMode: 'free_goodness' });

  assert.deepEqual(ranked.map((result) => result.user_id), [2, 1]);
});
