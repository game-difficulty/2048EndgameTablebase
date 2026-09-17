const HEX_BOARD_PATTERN = /^[0-9a-f]{16}$/u;

const integerInRange = (value, minimum, maximum) => (
  Number.isInteger(Number(value))
  && Number(value) >= minimum
  && Number(value) <= maximum
);

export function isPermanentBattleRoom(room) {
  return Boolean(room?.is_permanent || room?.lifecycle_kind === 'permanent');
}

export function canEditBattleInitialBoard(room) {
  return room?.mode_key === 'free_goodness' || (room?.mode_key === 'goodness' && isPermanentBattleRoom(room));
}

export function compareBattleRooms(left, right) {
  return Number(isPermanentBattleRoom(right)) - Number(isPermanentBattleRoom(left));
}

export function battleRoomSettingsDraft(room) {
  const modeSettings = room?.mode_settings || {};
  const targetCap = Math.max(1, Math.floor(Number(room?.target || 0) / 2));
  const scoreStepLimit = Number(
    modeSettings.score_step_limit
    ?? room?.max_steps
    ?? targetCap,
  );
  return {
    step_timeout_seconds: Number(room?.step_timeout_seconds || 90),
    initial_board: String(
      modeSettings.initial_board
      || room?.initial_board
      || room?.route?.initial_board
      || '',
    ).trim().toLowerCase(),
    score_step_limit: Number.isInteger(scoreStepLimit) && scoreStepLimit > 0
      ? scoreStepLimit
      : targetCap,
    ranking_min_steps: Number(
      modeSettings.ranking_min_steps
      ?? scoreStepLimit
      ?? targetCap,
    ),
  };
}

export function validateBattleRoomSettings(room, draft) {
  if (
    !integerInRange(draft?.step_timeout_seconds, 5, 600)
    || Number(draft.step_timeout_seconds) % 5 !== 0
  ) {
    return { ok: false, code: 'stepTimeout' };
  }

  if (!canEditBattleInitialBoard(room)) {
    return { ok: true };
  }

  const targetCap = Math.max(1, Math.floor(Number(room?.target || 0) / 2));
  const board = String(draft?.initial_board || '').trim().toLowerCase();
  if (!HEX_BOARD_PATTERN.test(board)) {
    return { ok: false, code: 'initialBoard' };
  }
  if (room?.mode_key !== 'free_goodness') return { ok: true };
  if (!integerInRange(draft?.score_step_limit, 1, targetCap)) {
    return { ok: false, code: 'scoreStepLimit' };
  }
  if (!integerInRange(draft?.ranking_min_steps, 1, Number(draft.score_step_limit))) {
    return { ok: false, code: 'rankingMinSteps' };
  }
  return { ok: true };
}

export function buildBattleRoomSettingsPayload(room, draft, expectedRevision) {
  const payload = {
    expected_revision: Number(expectedRevision),
    step_timeout_seconds: Number(draft.step_timeout_seconds),
  };
  if (canEditBattleInitialBoard(room)) {
    payload.initial_board = String(draft.initial_board).trim().toLowerCase();
  }
  if (String(room?.mode_key || 'goodness') === 'free_goodness') {
    payload.score_step_limit = Number(draft.score_step_limit);
    payload.ranking_min_steps = Number(draft.ranking_min_steps);
  }
  return payload;
}
