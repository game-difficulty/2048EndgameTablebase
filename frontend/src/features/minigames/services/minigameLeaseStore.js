const keyFor = (runId) => `2048tables:minigame-ranked-lease:${String(runId || '')}`;

const sessionStorageOrNull = () => {
  if (typeof window === 'undefined') return null;
  try {
    return window.sessionStorage || null;
  } catch (_error) {
    return null;
  }
};

export function readMinigameLease(runId) {
  const storage = sessionStorageOrNull();
  if (!storage || !runId) return null;
  try {
    const value = JSON.parse(storage.getItem(keyFor(runId)) || 'null');
    if (!value || typeof value.leaseToken !== 'string') return null;
    return value;
  } catch (_error) {
    return null;
  }
}

export function writeMinigameLease(runId, { leaseToken, expiresAt = '', userId = null }) {
  const storage = sessionStorageOrNull();
  if (!storage || !runId || !leaseToken) return false;
  try {
    storage.setItem(keyFor(runId), JSON.stringify({
      leaseToken: String(leaseToken),
      expiresAt: String(expiresAt || ''),
      userId: userId == null ? null : Number(userId),
    }));
    return true;
  } catch (_error) {
    return false;
  }
}

export function removeMinigameLease(runId) {
  const storage = sessionStorageOrNull();
  if (!storage || !runId) return;
  try {
    storage.removeItem(keyFor(runId));
  } catch (_error) {
    // Session recovery is optional; the server lease remains authoritative.
  }
}
