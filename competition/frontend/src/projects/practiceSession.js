export const PRACTICE_SESSION_TTL = 5 * 60 * 1000;
export const PRACTICE_SESSION_KEY = 'tournament:practice-session-view:v1';
let sharedSession;
export function getPracticeSession(options) {
  return sharedSession ??= createPracticeSession(options);
}

// Display cache only: never store tokens, roles or permission flags here.
export function createPracticeSession({ storage, readSession, bridge, origins = [], now = Date.now }) {
  let memory = null;
  let pending = null;
  const recent = timestamp => Number.isFinite(timestamp) && now() >= timestamp && now() - timestamp < PRACTICE_SESSION_TTL;
  function read() {
    try { if (storage) memory = JSON.parse(storage.getItem(PRACTICE_SESSION_KEY) || 'null'); } catch { /* Storage may be unavailable. */ }
    return memory || {};
  }
  function write(value) {
    memory = value;
    try { storage?.setItem(PRACTICE_SESSION_KEY, JSON.stringify(value)); } catch { /* Keep the in-page cache. */ }
  }
  function remember(user) {
    const display = user ? { id: user.id, display_name: user.display_name } : null;
    write({ ...read(), user: display, checkedAt: now() });
    return display;
  }
  function peek() {
    const cached = read();
    return { user: cached.user || null, fresh: recent(cached.checkedAt) };
  }
  async function check() {
    try { return { user: remember((await readSession()).user), unauthorized: false }; }
    catch (error) {
      if (error.status !== 401) throw error;
      return { user: remember(null), unauthorized: true };
    }
  }
  async function resolve(force) {
    const result = await check();
    if (!result.unauthorized || !origins.length) return result.user;
    if (!force && recent(read().bridgeAttemptAt)) return null;
    // Persist before trying siblings, so reloads cannot restart the whole chain.
    write({ ...read(), bridgeAttemptAt: now() });
    for (const origin of origins) {
      try { await bridge(origin); } catch { /* Try the next sibling if unavailable. */ }
      const next = await check();
      if (!next.unauthorized) return next.user;
    }
    return null;
  }
  function sync({ force = false } = {}) {
    if (pending) return pending;
    const cached = peek();
    if (!force && cached.fresh) return Promise.resolve(cached.user);
    pending = resolve(force).finally(() => { pending = null; });
    return pending;
  }
  return { peek, sync, clear: () => remember(null) };
}
