import assert from 'node:assert/strict';
import test from 'node:test';

import {
  EMPTY_PATTERN_CATEGORY,
  EMPTY_PATTERN_ID,
  isEmptyTrainerPattern,
  shouldDeferTrainerBoardSync,
  stripTrainerQueryPayload,
  withEmptyPatternGroup,
} from '../src/features/trainer/engine/trainerEmptyPattern.js';

test('adds the empty pattern as an independent menu category', () => {
  const source = [{ category: 'basic', patterns: ['L3'] }];
  const groups = withEmptyPatternGroup(source);

  assert.deepEqual(groups[0], {
    category: EMPTY_PATTERN_CATEGORY,
    patterns: [EMPTY_PATTERN_ID],
  });
  assert.deepEqual(groups.slice(1), source);
  assert.equal(isEmptyTrainerPattern(EMPTY_PATTERN_ID), true);
  assert.equal(isEmptyTrainerPattern('L3'), false);
  assert.equal(shouldDeferTrainerBoardSync(EMPTY_PATTERN_ID), true);
  assert.equal(shouldDeferTrainerBoardSync('L3'), false);
});

test('removes every tablebase query field from empty-pattern actions', () => {
  assert.deepEqual(stripTrainerQueryPayload({
    dir: 'left',
    board_hex: '1234000000000000',
    query_id: 'query-1',
    query_reason: 'auto',
    catalog_version: 'v1',
    full_pattern: 'L3_256',
    prefetch_rng: { state: 1 },
  }), {
    dir: 'left',
    board_hex: '1234000000000000',
  });
});
