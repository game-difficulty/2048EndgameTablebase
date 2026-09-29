import test from 'node:test';
import assert from 'node:assert/strict';
import { ownDeadline, phaseSeconds, stageChange } from '../src/stageMotion.js';

const room = (status) => ({ room_code: 'ABC123', status });

test('only adjacent live snapshots trigger a transition', () => {
  assert.equal(stageChange(null, room('DRAW'), true), null);
  assert.equal(stageChange(room('READY_CHECK'), room('DRAW'), false), null);
  assert.equal(stageChange(room('READY_CHECK'), room('FIRST_PICK_BAN'), true), null);
  assert.equal(stageChange(room('DRAW'), { ...room('FIRST_PICK_BAN'), room_code: 'OTHER' }, true), null);
  assert.equal(stageChange(room('READY_CHECK'), room('DRAW'), true)?.kind, 'first-draw');
  assert.equal(stageChange(room('FIRST_PICK_BAN'), room('SECOND_PICK_BAN'), true)?.kind, 'pick-lock');
  assert.equal(stageChange(room('BLIND_PICK'), room('C_DRAW'), true)?.kind, 'c-draw');
  assert.equal(stageChange(room('GAME_B_READY'), room('GAME_B_PLAYING'), true)?.kind, 'game-start');
  assert.equal(stageChange(room('GAME_C_RESULT'), room('FINISHED'), true)?.kind, 'match-finished');
});

test('own simultaneous deadline never uses the opponents deadline', () => {
  const snapshot = {
    status: 'BLIND_PICK', me: { seat: { side: 'yellow' } },
    draft: { deadlines: { yellow: '2026-09-29T00:00:10Z', white: '2026-09-29T00:00:40Z' } },
  };
  assert.equal(ownDeadline(snapshot), '2026-09-29T00:00:10Z');
  assert.equal(phaseSeconds(ownDeadline(snapshot), Date.parse('2026-09-29T00:00:05Z')), 5);
  assert.equal(phaseSeconds(ownDeadline(snapshot), Date.parse('2026-09-29T00:00:11Z')), 0);
});
