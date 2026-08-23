import { emitAuthRequired } from '../../../services/auth/authEvents';
import { authHeaders, clearDeviceSession } from '../../../services/auth/sessionTokenStore';
import { getBackendUrl } from '../../../services/runtime/backendUrl';

export function createRankedRequestId() {
  const cryptoApi = globalThis.crypto;
  if (typeof cryptoApi?.randomUUID === 'function') return cryptoApi.randomUUID();
  if (typeof cryptoApi?.getRandomValues !== 'function') {
    return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
  }
  const bytes = cryptoApi.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

async function requestJson(path, { method = 'GET', body } = {}) {
  const response = await fetch(getBackendUrl(path), {
    method,
    credentials: 'include',
    headers: authHeaders({
      Accept: 'application/json',
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    }),
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      clearDeviceSession();
      emitAuthRequired();
    }
    const error = new Error(typeof payload?.detail === 'string' ? payload.detail : `HTTP ${response.status}`);
    error.status = response.status;
    error.code = typeof payload?.detail === 'string' ? payload.detail : '';
    throw error;
  }
  return payload;
}

export const createRankedRun = (requestId) => requestJson('/api/gamer/runs', {
  method: 'POST',
  body: { request_id: requestId },
});

export const submitRankedRun = (runId, payload) => requestJson(`/api/gamer/runs/${encodeURIComponent(runId)}/submit`, {
  method: 'POST',
  body: payload,
});

export const fetchRankedRun = (runId) => requestJson(`/api/gamer/runs/${encodeURIComponent(runId)}`);
