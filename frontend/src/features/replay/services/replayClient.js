import { emitTokenBalanceUpdated } from '../../../services/auth/authEvents';
import { authHeaders } from '../../../services/auth/sessionTokenStore';
import { handleProtectedResponseError } from '../../../services/files/browserFiles';
import { getBackendUrl } from '../../../services/runtime/backendUrl';

export function createReplayRequestId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

async function requestWithTransportRetry(url, init, handleResponse) {
  let lastError = null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const response = await fetch(url, init);
      return await handleResponse(response);
    } catch (error) {
      if (error?.status) throw error;
      lastError = error;
    }
  }
  const error = new Error(lastError?.message || 'Network request failed.');
  error.code = 'NETWORK_ERROR';
  throw error;
}

function requestInit(payload) {
  return {
    method: 'POST',
    credentials: 'include',
    headers: authHeaders({
      Accept: 'application/json, application/octet-stream',
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify(payload),
  };
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

function balanceFromHeaders(headers) {
  const bonus = Number(headers.get('x-token-bonus'));
  const paid = Number(headers.get('x-token-paid'));
  const total = Number(headers.get('x-token-total'));
  if (![bonus, paid, total].every(Number.isFinite)) return null;
  return { bonus, paid, total };
}

export async function authorizeLocalReplayLoad({ requestId, filename, size }) {
  return requestWithTransportRetry(
    getBackendUrl('/api/replay/load-authorization'),
    requestInit({ request_id: requestId, filename, size }),
    async (response) => {
      if (!response.ok) {
        await handleProtectedResponseError(response, 'Replay authorization failed');
      }
      const payload = await response.json();
      if (payload?.token_balance) emitTokenBalanceUpdated(payload.token_balance);
      return payload;
    },
  );
}

export async function fetchLatestReplay({ requestId }) {
  return requestWithTransportRetry(
    getBackendUrl('/api/replay/latest'),
    requestInit({ request_id: requestId }),
    async (response) => {
      if (!response.ok) {
        await handleProtectedResponseError(response, 'Latest replay request failed');
      }
      const tokenBalance = balanceFromHeaders(response.headers);
      if (tokenBalance) emitTokenBalanceUpdated(tokenBalance);
      return {
        buffer: await response.arrayBuffer(),
        filename: filenameFromHeaders(response.headers, 'tester_latest.rpl'),
        pattern: response.headers.get('x-replay-pattern') || '',
        source: response.headers.get('x-replay-source') || 'Tester session',
        useVariant: response.headers.get('x-replay-variant') === '1',
      };
    },
  );
}
