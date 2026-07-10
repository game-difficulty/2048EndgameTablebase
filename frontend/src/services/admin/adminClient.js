import { getBackendUrl } from '../runtime/backendUrl';
import { emitAuthRequired } from '../auth/authEvents';

async function requestJson(path) {
  const response = await fetch(getBackendUrl(path), {
    credentials: 'include',
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
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
  overview: ({ q = '', limit = 20 } = {}) => {
    const params = new URLSearchParams();
    if (q) {
      params.set('q', q);
    }
    params.set('limit', String(limit));
    return requestJson(`/api/admin/overview?${params.toString()}`);
  },
};
