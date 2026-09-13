import assert from 'node:assert/strict';
import test from 'node:test';

import { correctionOverlayForResult } from '../src/features/battle/core/battleCorrection.js';
import { isBattlePlaybackStopped } from '../src/features/battle/core/battlePlaybackState.js';

test('forfeits and timeouts clear corrections, but completed certainty tails remain playable', () => {
  const mode_data = { correction: { selected_direction: 'left', standard_direction: 'up' } };
  for (const status of ['disqualified', 'forfeited', 'timed_out']) {
    assert.equal(isBattlePlaybackStopped({ status }), true);
    assert.equal(correctionOverlayForResult({ status, mode_data }), null);
  }
  for (const status of ['playing', 'completed']) {
    assert.equal(isBattlePlaybackStopped({ status }), false);
    assert.ok(correctionOverlayForResult({ status, mode_data }));
  }
});

test('normalizes a live opponent correction', () => {
  const visibleUntil = '2026-09-04T12:00:15+00:00';
  const overlay = correctionOverlayForResult({
    mode_data: {
      correction: {
        selected_direction: 'LEFT',
        standard_direction: 'down',
        goodness_drop: 0.125,
        previous_board_hex: '0000000000000011',
        previous_route_index: 7,
        visible_until: visibleUntil,
      },
    },
  }, Date.parse('2026-09-04T12:00:10+00:00'));

  assert.deepEqual(overlay, {
    selectedDirection: 'left',
    standardDirection: 'down',
    drop: 0.125,
    previousBoardHex: '0000000000000011',
    previousRouteIndex: 7,
    visibleUntil: Date.parse(visibleUntil),
  });
});

test('does not expose expired or malformed corrections', () => {
  assert.equal(correctionOverlayForResult({ mode_data: {} }), null);
  assert.equal(correctionOverlayForResult({
    mode_data: {
      correction: {
        selected_direction: 'left',
        standard_direction: 'up',
        visible_until: '2026-09-04T12:00:00+00:00',
      },
    },
  }, Date.parse('2026-09-04T12:00:01+00:00')), null);
});
