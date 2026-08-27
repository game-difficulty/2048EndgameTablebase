const FALLBACK_TTL_MS = 12000;
const FALLBACK_RENEW_MS = 4000;

const randomId = () => {
  if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
};

const parseLease = (raw) => {
  try {
    const parsed = JSON.parse(raw || 'null');
    return parsed && typeof parsed.ownerId === 'string' ? parsed : null;
  } catch (_error) {
    return null;
  }
};

const localStorageOrNull = () => {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage || null;
  } catch (_error) {
    return null;
  }
};

export function createRankedRunLock({ onLost } = {}) {
  const ownerId = randomId();
  let generation = 0;
  let heldRunId = null;
  let releaseWebLock = null;
  let fallbackTimer = null;
  let fallbackKey = '';
  let channel = null;
  let storageListener = null;

  const loseFallbackLock = () => {
    if (!heldRunId) return;
    const lostRunId = heldRunId;
    heldRunId = null;
    if (fallbackTimer !== null) window.clearInterval(fallbackTimer);
    fallbackTimer = null;
    channel?.close();
    channel = null;
    if (storageListener) window.removeEventListener('storage', storageListener);
    storageListener = null;
    if (typeof onLost === 'function') onLost(lostRunId);
  };

  const resolveFallbackConflict = (lease) => {
    if (!lease || lease.ownerId === ownerId || Number(lease.expiresAt) <= Date.now()) return;
    if (ownerId.localeCompare(lease.ownerId) > 0) {
      loseFallbackLock();
    }
  };

  const writeFallbackLease = () => {
    if (!heldRunId || !fallbackKey) return;
    try {
      const storage = localStorageOrNull();
      if (!storage) throw new Error('localStorage unavailable');
      const lease = { ownerId, runId: heldRunId, expiresAt: Date.now() + FALLBACK_TTL_MS };
      storage.setItem(fallbackKey, JSON.stringify(lease));
      channel?.postMessage(lease);
    } catch (_error) {
      loseFallbackLock();
    }
  };

  const acquireFallback = (runId) => {
    const storage = localStorageOrNull();
    if (!storage) return false;
    fallbackKey = `2048tables:gamer-ranked-lock:${runId}`;
    const current = parseLease(storage.getItem(fallbackKey));
    if (current && current.ownerId !== ownerId && Number(current.expiresAt) > Date.now()) return false;
    heldRunId = runId;
    try {
      if (typeof BroadcastChannel === 'function') {
        channel = new BroadcastChannel(`2048tables:gamer-ranked-lock:${runId}`);
        channel.onmessage = (event) => resolveFallbackConflict(event.data);
      }
      storageListener = (event) => {
        if (event.key === fallbackKey) resolveFallbackConflict(parseLease(event.newValue));
      };
      window.addEventListener('storage', storageListener);
      writeFallbackLease();
      const stored = parseLease(storage.getItem(fallbackKey));
      if (!stored || stored.ownerId !== ownerId) {
        loseFallbackLock();
        return false;
      }
      fallbackTimer = window.setInterval(writeFallbackLease, FALLBACK_RENEW_MS);
      return true;
    } catch (_error) {
      loseFallbackLock();
      return false;
    }
  };

  const release = () => {
    generation += 1;
    const releasedRunId = heldRunId;
    heldRunId = null;
    if (releaseWebLock) releaseWebLock();
    releaseWebLock = null;
    if (fallbackTimer !== null) window.clearInterval(fallbackTimer);
    fallbackTimer = null;
    channel?.close();
    channel = null;
    if (storageListener) window.removeEventListener('storage', storageListener);
    storageListener = null;
    const storage = localStorageOrNull();
    if (fallbackKey && storage) {
      const stored = parseLease(storage.getItem(fallbackKey));
      if (stored?.ownerId === ownerId) storage.removeItem(fallbackKey);
    }
    fallbackKey = '';
    return releasedRunId;
  };

  const acquireWebLock = (runId, attempt) => new Promise((resolve) => {
    let resolved = false;
    const settle = (value) => {
      if (!resolved) {
        resolved = true;
        resolve(value);
      }
    };
    navigator.locks.request(
      `2048tables:gamer-ranked:${runId}`,
      { mode: 'exclusive', ifAvailable: true },
      async (lock) => {
        if (!lock || attempt !== generation) {
          settle(false);
          return;
        }
        heldRunId = runId;
        settle(true);
        await new Promise((releaseLock) => {
          releaseWebLock = releaseLock;
        });
      },
    ).catch(() => settle(false));
  });

  const acquire = async (rawRunId) => {
    release();
    const runId = String(rawRunId || '');
    if (!runId) return false;
    const attempt = generation;
    if (typeof navigator !== 'undefined' && navigator.locks?.request) {
      const acquired = await acquireWebLock(runId, attempt);
      return acquired && attempt === generation;
    }
    return acquireFallback(runId);
  };

  return {
    acquire,
    release,
    isHeld: (runId) => Boolean(heldRunId && heldRunId === String(runId || '')),
  };
}
