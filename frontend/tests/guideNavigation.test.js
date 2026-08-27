import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createGuideTrainerJumpDetail,
  normalizeGuideBoardHex,
} from '../src/features/help/utils/guideNavigation.js';

test('pads partial guide boards at the bottom with f cells', () => {
  assert.equal(
    normalizeGuideBoardHex('001064127ff5', 3, 4),
    '001064127ff5ffff',
  );
  assert.equal(
    normalizeGuideBoardHex('13105521', 2, 4),
    '13105521ffffffff',
  );
});

test('keeps complete guide boards unchanged', () => {
  assert.equal(
    normalizeGuideBoardHex('210065327ff4ffff', 3, 4),
    '210065327ff4ffff',
  );
});

test('pads 3x3 variant guide boards with e on the right and f at the bottom', () => {
  assert.equal(
    normalizeGuideBoardHex('123456789', 3, 3, { right: 'e', bottom: 'f' }),
    '123e456e789effff',
  );
});

test('creates a Trainer jump without changing the current pattern', () => {
  const detail = createGuideTrainerJumpDetail({
    board_id: 'img_0230_b00',
    hex: '001064127ff5',
    visible_rows: 3,
    visible_cols: 4,
  }, '32768-dream-v3');

  assert.deepEqual(detail, {
    hex: '001064127ff5ffff',
    boardId: 'img_0230_b00',
    sourceDocumentId: '32768-dream-v3',
  });
  assert.equal(Object.hasOwn(detail, 'fullPattern'), false);
});

test('creates a pattern-aware Trainer jump for variant guides', () => {
  const detail = createGuideTrainerJumpDetail({
    board_id: 'img_0019_b00',
    hex: '23503802a',
    visible_rows: 3,
    visible_cols: 3,
    padding: { right: 'e', bottom: 'f' },
  }, '2048-34-variant-guide', { full_pattern: '3x4_4096' });

  assert.deepEqual(detail, {
    hex: '235e038e02aeffff',
    boardId: 'img_0019_b00',
    sourceDocumentId: '2048-34-variant-guide',
    fullPattern: '3x4_4096',
  });
});

test('rejects malformed or structurally inconsistent guide boards', () => {
  assert.equal(normalizeGuideBoardHex('12xz', 1, 4), null);
  assert.equal(normalizeGuideBoardHex('1234', 2, 4), null);
  assert.equal(createGuideTrainerJumpDetail({ hex: '' }), null);
});
