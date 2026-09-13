import { battleActorRenderKey } from './battleActor.js';
import { rankBattleResults } from './battleResultRanking.js';

const terminal = (player) => ['completed', 'timed_out', 'disqualified', 'forfeited'].includes(player?.status);
const STORAGE_KEY = '2048tables:battle-finish-dismissed:v1';

export function battleFinishNotices(room) {
  if (!room?.round?.round_id) return {};
  const final = room.round.status === 'completed';
  const results = room.results || [];
  const ranked = rankBattleResults(results, { battleMode: room.mode_key });
  const winners = ranked.filter(player => player.rank === 1).length;
  const remaining = results.filter(player => !terminal(player)).length;
  return Object.fromEntries(ranked.filter(terminal).map(player => {
    const actor = battleActorRenderKey(player);
    const rank = final ? player.rank : null;
    let title = 'finished';
    if (player.status === 'timed_out') title = 'timedOut';
    else if (player.status === 'forfeited' || player.mode_data?.finish_reason === 'forfeit') title = 'exited';
    else if (player.status !== 'completed') title = 'ended';
    else if (final) {
      if (rank == null) title = 'unranked';
      else if (results.length === 1) title = 'solo';
      else if (rank === 1) title = winners > 1 ? 'jointFirst' : 'winner';
      else title = 'placed';
    }
    return [actor, {
      key: JSON.stringify([room.room_id, room.round.round_id, actor, final ? 'final' : 'personal']),
      final, title, rank, remaining,
      winner: final && rank === 1 && results.length > 1,
      goodness: Math.max(0, Math.min(1, Number(player.goodness_of_fit ?? 0))),
    }];
  }));
}

export function createBattleFinishDismissals(storage = () => globalThis.sessionStorage) {
  let dismissed = [];
  try {
    const saved = JSON.parse(storage()?.getItem(STORAGE_KEY) || '[]');
    if (Array.isArray(saved)) dismissed = saved.filter(key => typeof key === 'string').slice(-64);
  } catch { /* Playback must not depend on browser storage availability. */ }
  return {
    has: (key) => dismissed.includes(key),
    dismiss(key) {
      if (!key || dismissed.includes(key)) return;
      dismissed = [...dismissed, key].slice(-64);
      try { storage()?.setItem(STORAGE_KEY, JSON.stringify(dismissed)); } catch { /* Memory-only fallback. */ }
    },
  };
}
