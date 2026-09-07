export function recordSummaries(payload, userId) {
  if (Number(payload?.user_id) !== userId || !Array.isArray(payload?.records)) {
    throw new Error('Invalid personal records response');
  }
  const summaries = {};
  for (const row of payload.records) {
    if (typeof row.game_id !== 'string' || ![0, 1].includes(row.difficulty)) continue;
    const exponent = Math.max(0, Math.min(63, Math.trunc(Number(row.best_tile_exp) || 0)));
    summaries[`${row.game_id}:${row.difficulty}`] = {
      bestScore: Math.max(0, Number(row.best_score) || 0),
      trophy: Math.max(0, Math.min(4, Math.trunc(Number(row.trophy_tier) || 0))),
      highestExp: exponent,
      highestTile: exponent ? 2 ** exponent : 0,
    };
  }
  return summaries;
}

export function createPersonalRecordsSync({ fetchRecords, readCache, writeCache, onChange, now = Date.now }) {
  let userId = null;
  let generation = 0;
  let inFlight = null;
  let rerun = false;
  let lastSuccess = null;
  let current = { userId: null, summaries: {}, loaded: false };
  const publish = (payload) => {
    current = { userId, summaries: recordSummaries(payload, userId), loaded: true };
    onChange(current);
  };
  const setUser = (value) => {
    const selected = Number.isSafeInteger(Number(value)) && Number(value) > 0 ? Number(value) : null;
    if (selected === userId) return;
    userId = selected;
    generation += 1;
    inFlight = null;
    rerun = false;
    lastSuccess = null;
    current = { userId, summaries: {}, loaded: false };
    if (userId) {
      try { current = { userId, summaries: recordSummaries(readCache(userId), userId), loaded: true }; }
      catch { /* An unavailable or old cache is not an account record. */ }
    }
    onChange(current);
  };
  const refresh = ({ force = false } = {}) => {
    if (!userId) return Promise.resolve(false);
    if (inFlight) {
      rerun ||= force;
      return inFlight;
    }
    if (!force && lastSuccess !== null && now() - lastSuccess < 60_000) return Promise.resolve(true);
    const epoch = generation;
    const owner = userId;
    const task = (async () => {
      let success = false;
      do {
        rerun = false;
        try {
          const payload = await fetchRecords();
          if (generation !== epoch) return false;
          publish(payload);
          lastSuccess = now();
          success = true;
          try { writeCache(owner, payload); } catch { /* Current memory remains usable. */ }
        } catch {
          // A failed refresh must not replace a previously confirmed record.
          if (generation !== epoch) return false;
          success = false;
        }
      } while (rerun && generation === epoch);
      return success;
    })();
    inFlight = task;
    void task.finally(() => { if (inFlight === task) inFlight = null; });
    return task;
  };
  return { setUser, refresh, read: () => current };
}
