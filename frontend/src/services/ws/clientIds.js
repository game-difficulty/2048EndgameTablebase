const STORAGE_PREFIX = '2048tables:ws-client-id:v1';

function randomId() {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const bytes = new Uint8Array(12);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (byte) => byte.toString(36).padStart(2, '0')).join('');
  }
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 14)}`;
}

function canUseSessionStorage() {
  if (typeof window === 'undefined' || !window.sessionStorage) {
    return false;
  }
  try {
    const key = `${STORAGE_PREFIX}:probe`;
    window.sessionStorage.setItem(key, '1');
    window.sessionStorage.removeItem(key);
    return true;
  } catch (_error) {
    return false;
  }
}

export function getStableWsClientId(namespace) {
  const safeNamespace = String(namespace || 'client')
    .toLowerCase()
    .replace(/[^a-z0-9_-]/gu, '_')
    .slice(0, 32) || 'client';
  const key = `${STORAGE_PREFIX}:${safeNamespace}`;
  if (!canUseSessionStorage()) {
    return `${safeNamespace}_${randomId()}`;
  }
  const existing = window.sessionStorage.getItem(key);
  if (existing) {
    return existing;
  }
  const next = `${safeNamespace}_${randomId()}`;
  window.sessionStorage.setItem(key, next);
  return next;
}
