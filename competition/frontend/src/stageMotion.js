export const STAGE_ORDER = [
  'SEATING', 'READY_CHECK', 'DRAW', 'FIRST_PICK_BAN', 'SECOND_PICK_BAN',
  'BLIND_PICK', 'C_DRAW', 'LINEUP',
  'GAME_A_READY', 'GAME_A_PLAYING', 'GAME_A_RESULT',
  'GAME_B_READY', 'GAME_B_PLAYING', 'GAME_B_RESULT',
  'GAME_C_READY', 'GAME_C_PLAYING', 'GAME_C_RESULT', 'FINISHED',
];

const GAME_PHASE = /^GAME_([A-O])_(READY|PLAYING|RESULT)$/;

// A reconnect or a skipped server phase must show the latest snapshot immediately.
export function stageChange(previous, next, live = false) {
  if (!live || !previous || !next || previous.room_code !== next.room_code) return null;
  const from = STAGE_ORDER.indexOf(previous.status);
  const to = STAGE_ORDER.indexOf(next.status);
  if (next.rules?.version && next.draft?.workflow) {
    const key = next.status.match(GAME_PHASE)?.[1] || null;
    if (previous.status===next.status) return null;
    if (next.status==='FINISHED') return {from:previous.status,to:next.status,kind:'match-finished',game:key};
    if (key && previous.match?.current_game_key===key && next.status.endsWith('_RESULT')) return {from:previous.status,to:next.status,kind:'game-result',game:key};
    if (key && previous.status===`GAME_${key}_READY` && next.status.endsWith('_PLAYING')) return {from:previous.status,to:next.status,kind:'game-start',game:key};
    return null;
  }
  if (from < 0 || to !== from + 1) return null;
  const game = next.status.match(GAME_PHASE)?.[1] || null;
  let kind = 'advance';
  if (next.status === 'DRAW') kind = 'first-draw';
  else if (next.status === 'SECOND_PICK_BAN' || next.status === 'BLIND_PICK') kind = 'pick-lock';
  else if (next.status === 'C_DRAW') kind = 'c-draw';
  else if (next.status === 'GAME_A_READY') kind = 'lineup-reveal';
  else if (game && next.status.endsWith('_PLAYING')) kind = 'game-start';
  else if (game && next.status.endsWith('_RESULT')) kind = 'game-result';
  else if (next.status === 'FINISHED') kind = 'match-finished';
  return { from: previous.status, to: next.status, kind, game };
}

export function phaseSeconds(deadline, now) {
  const due = Date.parse(deadline || '');
  return Number.isFinite(due) ? Math.max(0, Math.ceil((due - now) / 1000)) : null;
}

export function ownDeadline(snapshot) {
  const side = snapshot?.me?.seat?.side;
  if (!side) return null;
  if (snapshot.status === 'LINEUP') return snapshot.lineup?.deadlines?.[side] || null;
  if (snapshot.status === 'BLIND_PICK') return snapshot.draft?.deadlines?.[side] || null;
  return null;
}
