const STORAGE_KEY = '2048tables:guest-session-token';
export const GUEST_SESSION_STORAGE_KEY = STORAGE_KEY;
const FALLBACK_COOKIE_NAME = 'tb_guest_session_fallback';

function storage(kind) {
  if (typeof window === 'undefined') return null;
  try {
    return window[kind] || null;
  } catch (_error) {
    return null;
  }
}

function clearCookie() {
  try {
    if (typeof document !== 'undefined') {
      document.cookie = `${FALLBACK_COOKIE_NAME}=; Path=/; Max-Age=0; SameSite=Lax; Secure`;
    }
  } catch (_error) {
    // The HttpOnly guest cookie remains the primary credential.
  }
}

export function clearGuestSession() {
  for (const kind of ['localStorage', 'sessionStorage']) {
    try {
      storage(kind)?.removeItem(STORAGE_KEY);
    } catch (_error) {
      // Ignore unavailable browser storage.
    }
  }
  clearCookie();
}

function parse(raw) {
  try {
    const value = JSON.parse(raw || 'null');
    const expiresAt = Date.parse(value?.expires_at || '');
    if (!value?.token || (Number.isFinite(expiresAt) && expiresAt <= Date.now())) {
      return null;
    }
    return value;
  } catch (_error) {
    return null;
  }
}

function readCookie() {
  if (typeof document === 'undefined') return null;
  const prefix = `${FALLBACK_COOKIE_NAME}=`;
  const item = document.cookie.split(';')
    .map((part) => part.trim())
    .find((part) => part.startsWith(prefix));
  if (!item) return null;
  try {
    return parse(decodeURIComponent(item.slice(prefix.length)));
  } catch (_error) {
    return null;
  }
}

export function readGuestSession() {
  for (const kind of ['localStorage', 'sessionStorage']) {
    try {
      const value = parse(storage(kind)?.getItem(STORAGE_KEY));
      if (value) return value;
    } catch (_error) {
      // Try the next fallback.
    }
  }
  return readCookie();
}

export function getGuestSessionToken() {
  return readGuestSession()?.token || '';
}

export function storeGuestSession(payload) {
  const token = String(payload?.guest_session_token || '').trim();
  if (!token) return;
  const value = {
    token,
    expires_at: payload?.expires_at || payload?.guest?.expires_at || '',
    guest_id: payload?.guest?.guest_id || null,
  };
  const serialized = JSON.stringify(value);
  for (const kind of ['localStorage', 'sessionStorage']) {
    try {
      storage(kind)?.setItem(STORAGE_KEY, serialized);
    } catch (_error) {
      // The HttpOnly guest cookie remains usable.
    }
  }
  try {
    if (typeof document !== 'undefined') {
      const expiresMs = Date.parse(value.expires_at || '');
      const maxAge = Number.isFinite(expiresMs)
        ? Math.max(1, Math.floor((expiresMs - Date.now()) / 1000))
        : 30 * 24 * 60 * 60;
      document.cookie = `${FALLBACK_COOKIE_NAME}=${encodeURIComponent(serialized)}; Path=/; Max-Age=${maxAge}; SameSite=Lax; Secure`;
    }
  } catch (_error) {
    // The HttpOnly guest cookie remains usable.
  }
}
