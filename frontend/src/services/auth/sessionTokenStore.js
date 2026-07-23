const STORAGE_KEY = '2048tables:device-session-token';
const FALLBACK_COOKIE_NAME = 'tb_device_session_fallback';

function storage(kind) {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    return window[kind] || null;
  } catch (_error) {
    return null;
  }
}

function isExpired(expiresAt) {
  if (!expiresAt) {
    return false;
  }
  const expiresMs = Date.parse(expiresAt);
  return Number.isFinite(expiresMs) && expiresMs <= Date.now();
}

export function clearDeviceSession() {
  for (const kind of ['localStorage', 'sessionStorage']) {
    const target = storage(kind);
    if (!target) {
      continue;
    }
    try {
      target.removeItem(STORAGE_KEY);
    } catch (_error) {
      // Ignore storage failures; cookie auth remains the primary session path.
    }
  }
  try {
    if (typeof document !== 'undefined') {
      document.cookie = `${FALLBACK_COOKIE_NAME}=; Path=/; Max-Age=0; SameSite=Lax; Secure`;
    }
  } catch (_error) {
    // Ignore storage failures; cookie auth remains the primary session path.
  }
}

function parseSession(raw) {
  try {
    const session = JSON.parse(raw || 'null');
    if (!session?.token || isExpired(session.expires_at)) {
      clearDeviceSession();
      return null;
    }
    return session;
  } catch (_error) {
    clearDeviceSession();
    return null;
  }
}

function readStorageSession(kind) {
  const target = storage(kind);
  if (!target) {
    return null;
  }
  try {
    return parseSession(target.getItem(STORAGE_KEY));
  } catch (_error) {
    return null;
  }
}

function readCookieSession() {
  if (typeof document === 'undefined') {
    return null;
  }
  const prefix = `${FALLBACK_COOKIE_NAME}=`;
  const rawCookie = document.cookie
    .split(';')
    .map((item) => item.trim())
    .find((item) => item.startsWith(prefix));
  if (!rawCookie) {
    return null;
  }
  try {
    return parseSession(decodeURIComponent(rawCookie.slice(prefix.length)));
  } catch (_error) {
    clearDeviceSession();
    return null;
  }
}

export function readDeviceSession() {
  return (
    readStorageSession('localStorage')
    || readStorageSession('sessionStorage')
    || readCookieSession()
  );
}

export function getDeviceSessionToken() {
  return readDeviceSession()?.token || '';
}

export function storeDeviceSession(payload) {
  const token = String(payload?.device_session_token || '').trim();
  if (!token) {
    return;
  }
  const session = {
    token,
    expires_at: payload?.expires_at || '',
    user_id: payload?.user?.id ?? null,
  };
  const serialized = JSON.stringify(session);
  for (const kind of ['localStorage', 'sessionStorage']) {
    const target = storage(kind);
    if (!target) {
      continue;
    }
    try {
      target.setItem(STORAGE_KEY, serialized);
    } catch (_error) {
      // Ignore storage failures; cookie auth remains the primary session path.
    }
  }
  try {
    if (typeof document !== 'undefined') {
      const expiresMs = Date.parse(session.expires_at || '');
      const maxAge = Number.isFinite(expiresMs)
        ? Math.max(1, Math.floor((expiresMs - Date.now()) / 1000))
        : 14 * 24 * 60 * 60;
      document.cookie = `${FALLBACK_COOKIE_NAME}=${encodeURIComponent(serialized)}; Path=/; Max-Age=${maxAge}; SameSite=Lax; Secure`;
    }
  } catch (_error) {
    // Ignore storage failures; cookie auth remains the primary session path.
  }
}

export function authHeaders(headers = {}) {
  const token = getDeviceSessionToken();
  if (!token) {
    return headers;
  }
  return {
    ...headers,
    Authorization: `Bearer ${token}`,
  };
}
