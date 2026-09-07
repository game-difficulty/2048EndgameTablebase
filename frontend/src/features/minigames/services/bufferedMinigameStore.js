// Merge only locally changed entries: other tabs may own other games.
export function createBufferedMinigameStore(store, {
  schedule = (callback) => setTimeout(callback, 500),
  cancel = (handle) => clearTimeout(handle),
  onError = () => {},
} = {}) {
  let state = store.read();
  let scheduled = null;
  let difficultyDirty = false;
  const summaries = new Map();
  const snapshots = new Map();

  const track = (pending, before = {}, after = {}) => {
    for (const key of new Set([...Object.keys(before), ...Object.keys(after)])) {
      if (before[key] !== after[key]) pending.set(key, after[key]);
    }
  };
  const merge = (current = {}, pending) => {
    const result = { ...current };
    for (const [key, value] of pending) {
      if (value === undefined) delete result[key];
      else result[key] = value;
    }
    return result;
  };
  const mergeSummaries = (current = {}) => {
    const result = merge(current, summaries);
    for (const [key, value] of summaries) {
      if (!value) continue;
      result[key] = { ...result[key] };
      for (const field of ['bestScore', 'highestTile', 'highestExp', 'trophy']) {
        result[key][field] = Math.max(Number(current[key]?.[field]) || 0, Number(value[field]) || 0);
      }
    }
    return result;
  };
  const flush = () => {
    if (scheduled !== null) cancel(scheduled);
    scheduled = null;
    if (!difficultyDirty && !summaries.size && !snapshots.size) return true;
    try {
      store.update((current) => ({
        ...current,
        difficulty: difficultyDirty ? state.difficulty : current.difficulty,
        summaries: mergeSummaries(current.summaries),
        activeGameSnapshots: merge(current.activeGameSnapshots, snapshots),
      }));
      difficultyDirty = false;
      summaries.clear();
      snapshots.clear();
      return true;
    } catch (error) {
      // Keep the newest in-memory state and retry on the next save/flush.
      onError(error);
      return false;
    }
  };
  return {
    read: () => state,
    update(updater) {
      const next = updater(state);
      difficultyDirty ||= next.difficulty !== state.difficulty;
      track(summaries, state.summaries, next.summaries);
      track(snapshots, state.activeGameSnapshots, next.activeGameSnapshots);
      state = next;
      if (scheduled === null) scheduled = schedule(flush);
      return state;
    },
    discardSnapshot(key) { snapshots.delete(key); },
    flush,
  };
}
