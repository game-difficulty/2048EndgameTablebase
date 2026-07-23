import { getBackendUrl } from '../runtime/backendUrl';
import { emitAuthRequired } from '../auth/authEvents';
import { authHeaders, clearDeviceSession } from '../auth/sessionTokenStore';

async function requestJson(path, { method = 'GET', body } = {}) {
  const response = await fetch(getBackendUrl(path), {
    method,
    headers: authHeaders(body ? { 'Content-Type': 'application/json' } : {}),
    credentials: 'include',
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      clearDeviceSession();
      emitAuthRequired();
    }
    const detail = typeof payload?.detail === 'string'
      ? payload.detail
      : payload?.detail?.message;
    const error = new Error(detail || `Request failed: ${response.status}`);
    error.status = response.status;
    error.detail = payload?.detail || null;
    throw error;
  }
  return payload;
}

export const adminClient = {
  overview: ({
    q = '',
    page = 1,
    pageSize = 20,
    tier = 'all',
  } = {}) => {
    const params = new URLSearchParams();
    if (q) {
      params.set('q', q);
    }
    params.set('page', String(page));
    params.set('page_size', String(pageSize));
    if (tier && tier !== 'all') {
      params.set('tier', String(tier));
    }
    return requestJson(`/api/admin/overview?${params.toString()}`);
  },
  adjustUserTokens: (userId, payload) => requestJson(`/api/admin/users/${encodeURIComponent(userId)}/tokens`, {
    method: 'POST',
    body: payload,
  }),
};
