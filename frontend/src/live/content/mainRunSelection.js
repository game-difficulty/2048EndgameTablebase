export function leaderLanes(slots) {
  const running = slots.filter(s => s.run);
  if (!running.length) return [];
  const score = Math.max(...running.map(s => s.run.score));
  return running.filter(s => s.run.score === score).map(s => s.lane);
}

export const MAIN_SWITCH_SCORE_GAP = 1000;
const ended = slot => slot.status === 'ended' || slot.run?.ended_at != null;

export function chooseMainLane(slots, previous = 0, immediate = false) {
  const existing = slots.filter(s => s.run);
  if (!existing.length) return previous;
  const alive = existing.filter(s => !ended(s));
  // While anyone is playing, do not bounce back to a finished high scorer.
  // Once everyone finishes, show the final highest scorer, stably on ties.
  const candidates = alive.length ? alive : existing;
  const leaders = leaderLanes(candidates);
  const best = leaders.includes(previous) ? previous : Math.min(...leaders);
  const current = existing.find(s => s.lane === previous);
  if (immediate || !current || ended(current)) return best;
  const challenger = candidates.find(s => s.lane === best);
  return challenger.run.score - current.run.score >= MAIN_SWITCH_SCORE_GAP ? best : previous;
}

export function selectMain(view, lane) {
  return { ...view, layout: 'focus', mainMode: 'manual', selectedLane: lane };
}

export function followMain(view, slots, immediate = false) {
  return view.layout === 'focus' && view.mainMode === 'auto'
    ? { ...view, selectedLane: chooseMainLane(slots, view.selectedLane, immediate) } : view;
}

export function changeLayout(view, layout, slots) {
  return followMain({ ...view, layout }, slots, layout === 'focus' && view.layout !== 'focus');
}

export function changeMainMode(view, mainMode, slots) {
  return followMain({ ...view, mainMode }, slots, mainMode === 'auto');
}
