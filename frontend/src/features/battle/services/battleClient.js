import { emitAuthRequired, emitTokenBalanceUpdated, emitTokenRequired } from '../../../services/auth/authEvents.js';
import {
  authHeaders,
  clearDeviceSession,
  getDeviceSessionToken,
} from '../../../services/auth/sessionTokenStore.js';
import {
  clearGuestSession,
  getGuestSessionToken,
} from '../../../services/auth/guestSessionStore.js';
import { getBackendUrl } from '../../../services/runtime/backendUrl.js';
import { handleProtectedResponseError } from '../../../services/files/browserFiles.js';
import { buildBattleKickPayload } from '../core/battleActor.js';

function handleBattleUnauthorized() {
  if (!getDeviceSessionToken() && getGuestSessionToken()) {
    clearGuestSession();
    window.dispatchEvent(new CustomEvent('guest-session-invalidated'));
    return;
  }
  clearDeviceSession();
  emitAuthRequired();
}

async function request(path, { method = 'GET', body, signal } = {}) {
  const response = await fetch(getBackendUrl(path), {
    method,
    credentials: 'include',
    headers: authHeaders(body ? { 'Content-Type': 'application/json' } : {}),
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      handleBattleUnauthorized();
    }
    if (response.status === 402 && payload?.detail?.code === 'INSUFFICIENT_TOKENS') {
      emitTokenRequired(payload.detail);
    }
    const detail = payload?.detail;
    const message = typeof detail === 'string' ? detail : detail?.message;
    const error = new Error(message || `Battle request failed: ${response.status}`);
    error.status = response.status;
    error.code = detail?.code || payload?.code || '';
    error.detail = detail || payload;
    throw error;
  }
  if (payload?.token_balance) {
    emitTokenBalanceUpdated(payload.token_balance);
  }
  return payload;
}

function filenameFromHeaders(headers, fallback) {
  const disposition = headers.get('content-disposition') || '';
  const utf8 = /filename\*=UTF-8''([^;]+)/iu.exec(disposition);
  if (utf8?.[1]) {
    try {
      return decodeURIComponent(utf8[1]);
    } catch (_error) {
      return utf8[1];
    }
  }
  return /filename="?([^";]+)"?/iu.exec(disposition)?.[1] || fallback;
}

export const battleClient = {
  current: ({ signal } = {}) => request('/api/battle/me', { signal }),
  rooms: ({ signal } = {}) => request('/api/battle/rooms', { signal }),
  get: (roomCode, { signal } = {}) => request(`/api/battle/rooms/${encodeURIComponent(roomCode)}`, { signal }),
  create: (payload) => request('/api/battle/rooms', { method: 'POST', body: payload }),
  join: (roomCode, payload = {}) => request(`/api/battle/rooms/${encodeURIComponent(roomCode)}/join`, {
    method: 'POST',
    body: payload,
  }),
  leave: (roomCode, payload = {}) => request(`/api/battle/rooms/${encodeURIComponent(roomCode)}/leave`, {
    method: 'POST',
    body: payload,
  }),
  ready: (roomCode, ready, requestId) => request(`/api/battle/rooms/${encodeURIComponent(roomCode)}/ready`, {
    method: 'POST',
    body: { ready: Boolean(ready), request_id: requestId },
  }),
  start: (roomCode, requestId) => request(`/api/battle/rooms/${encodeURIComponent(roomCode)}/start`, {
    method: 'POST',
    body: { request_id: requestId },
  }),
  forfeit: (roomCode, roundId, requestId) => request(
    `/api/battle/rooms/${encodeURIComponent(roomCode)}/rounds/${encodeURIComponent(roundId)}/forfeit`,
    {
      method: 'POST',
      body: { request_id: requestId },
    },
  ),
  kick: (roomCode, actor, requestId) => {
    return request(`/api/battle/rooms/${encodeURIComponent(roomCode)}/kick`, {
      method: 'POST',
      body: buildBattleKickPayload(actor, requestId),
    });
  },
  role: (roomCode, role, requestId) => request(`/api/battle/rooms/${encodeURIComponent(roomCode)}/role`, {
    method: 'POST',
    body: { role, request_id: requestId },
  }),
  updateSettings: (roomCode, payload) => request(
    `/api/battle/rooms/${encodeURIComponent(roomCode)}/settings`,
    { method: 'PATCH', body: payload },
  ),
  renewHost: (roomCode, requestId) => request(
    `/api/battle/rooms/${encodeURIComponent(roomCode)}/host/renew`,
    { method: 'POST', body: { request_id: requestId } },
  ),
  route: async (roomCode, roundId, { signal } = {}) => {
    const response = await fetch(getBackendUrl(
      `/api/battle/rooms/${encodeURIComponent(roomCode)}/rounds/${encodeURIComponent(roundId)}/artifact`,
    ), {
      method: 'GET',
      credentials: 'include',
      headers: authHeaders({ Accept: 'application/octet-stream' }),
      signal,
    });
    if (!response.ok) {
      if (response.status === 401) {
        handleBattleUnauthorized();
      }
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload?.detail?.message || payload?.detail || `Route download failed: ${response.status}`);
    }
    return {
      buffer: await response.arrayBuffer(),
      hash: response.headers.get('etag')?.replaceAll('"', '') || '',
      stepCount: Number(response.headers.get('x-battle-route-steps') || 0),
      certaintyStep: Number(response.headers.get('x-battle-certainty-step') || -1),
      terminationReason: response.headers.get('x-battle-termination') || '',
    };
  },
  replay: async (roomCode, roundId, { signal } = {}) => {
    const response = await fetch(getBackendUrl(
      `/api/battle/rooms/${encodeURIComponent(roomCode)}/rounds/${encodeURIComponent(roundId)}/replay`,
    ), {
      method: 'GET',
      credentials: 'include',
      headers: authHeaders({ Accept: 'application/octet-stream' }),
      signal,
    });
    if (!response.ok) {
      if (response.status === 401) {
        handleBattleUnauthorized();
        const error = new Error('Battle replay session expired.');
        error.status = 401;
        throw error;
      }
      await handleProtectedResponseError(response, 'Battle replay download failed');
    }
    return {
      buffer: await response.arrayBuffer(),
      filename: filenameFromHeaders(response.headers, 'battle_replay.rpl'),
      pattern: response.headers.get('x-replay-pattern') || '',
      source: response.headers.get('x-replay-source') || 'Battle',
      useVariant: response.headers.get('x-replay-variant') === '1',
      moveCount: Number(response.headers.get('x-replay-moves') || 0),
    };
  },
};

export function battleRequestId(prefix = 'battle') {
  const random = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}-${random}`;
}
