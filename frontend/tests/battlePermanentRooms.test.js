import assert from 'node:assert/strict';
import test from 'node:test';

import {
  battleRoomSettingsDraft,
  buildBattleRoomSettingsPayload,
  compareBattleRooms,
  isPermanentBattleRoom,
  validateBattleRoomSettings,
} from '../src/features/battle/core/battleRoomSettings.js';

test('permanent Battle rooms sort before normal rooms', () => {
  const rooms = [
    { room_code: 'AAAAAA', lifecycle_kind: 'normal' },
    { room_code: 'CCCCCC', is_permanent: true },
    { room_code: 'BBBBBB', lifecycle_kind: 'permanent' },
  ].sort(compareBattleRooms);

  assert.deepEqual(rooms.map((room) => room.room_code), ['CCCCCC', 'BBBBBB', 'AAAAAA']);
  assert.equal(isPermanentBattleRoom(rooms[0]), true);
  assert.equal(isPermanentBattleRoom(rooms[2]), false);
});

test('free-goodness settings normalize the public room payload', () => {
  const draft = battleRoomSettingsDraft({
    mode_key: 'free_goodness',
    target: 512,
    step_timeout_seconds: 90,
    initial_board: 'ABCDEF0123456789',
    max_steps: 200,
    mode_settings: { score_step_limit: 180, ranking_min_steps: 120 },
  });

  assert.deepEqual(draft, {
    step_timeout_seconds: 90,
    initial_board: 'abcdef0123456789',
    score_step_limit: 180,
    ranking_min_steps: 120,
  });
});

test('free-goodness settings enforce board and step boundaries', () => {
  const room = { mode_key: 'free_goodness', target: 256 };
  const valid = {
    step_timeout_seconds: 95,
    initial_board: '0001012213ffffff',
    score_step_limit: 128,
    ranking_min_steps: 64,
  };

  assert.deepEqual(validateBattleRoomSettings(room, valid), { ok: true });
  assert.equal(validateBattleRoomSettings(room, { ...valid, step_timeout_seconds: 92 }).code, 'stepTimeout');
  assert.equal(validateBattleRoomSettings(room, { ...valid, initial_board: 'xyz' }).code, 'initialBoard');
  assert.equal(validateBattleRoomSettings(room, { ...valid, score_step_limit: 129 }).code, 'scoreStepLimit');
  assert.equal(validateBattleRoomSettings(room, { ...valid, ranking_min_steps: 129 }).code, 'rankingMinSteps');
});

test('settings payload includes revision and only mode-owned fields', () => {
  const shared = buildBattleRoomSettingsPayload(
    { mode_key: 'goodness' },
    { step_timeout_seconds: 120 },
    7,
  );
  assert.deepEqual(shared, { expected_revision: 7, step_timeout_seconds: 120 });

  const free = buildBattleRoomSettingsPayload(
    { mode_key: 'free_goodness' },
    {
      step_timeout_seconds: 60,
      initial_board: 'ABCDEF0123456789',
      score_step_limit: 200,
      ranking_min_steps: 100,
    },
    9,
  );
  assert.deepEqual(free, {
    expected_revision: 9,
    step_timeout_seconds: 60,
    initial_board: 'abcdef0123456789',
    score_step_limit: 200,
    ranking_min_steps: 100,
  });
});

test('only permanent goodness rooms can edit their initial board without free-mode step settings', () => {
  const room = {mode_key:'goodness', lifecycle_kind:'permanent', target:256};
  const draft = {step_timeout_seconds:90, initial_board:'011112221FFF3FFF'};
  assert.deepEqual(validateBattleRoomSettings(room,draft),{ok:true});
  assert.equal(validateBattleRoomSettings(room,{...draft,initial_board:'x'}).code,'initialBoard');
  assert.deepEqual(buildBattleRoomSettingsPayload(room,draft,2),{expected_revision:2,step_timeout_seconds:90,initial_board:'011112221fff3fff'});
  assert.deepEqual(buildBattleRoomSettingsPayload({...room,lifecycle_kind:'normal'},draft,2),{expected_revision:2,step_timeout_seconds:90});
});
