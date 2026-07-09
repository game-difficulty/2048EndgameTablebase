import { getBackendUrl } from '../runtime/backendUrl';
import { emitAuthRequired, emitTokenBalanceUpdated, emitTokenRequired } from './authEvents';

async function requestJson(path, { method = 'GET', body } = {}) {
  const response = await fetch(getBackendUrl(path), {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    credentials: 'include',
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      emitAuthRequired();
    }
    if (response.status === 402 && payload?.detail?.code === 'INSUFFICIENT_TOKENS') {
      emitTokenRequired(payload.detail);
    }
    const detail = typeof payload?.detail === 'string'
      ? payload.detail
      : payload?.detail?.message;
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  if (payload?.token_balance) {
    emitTokenBalanceUpdated(payload.token_balance);
  }
  return payload;
}

export const authClient = {
  me: () => requestJson('/api/auth/me'),
  sendEmailCode: (payload) => requestJson('/api/auth/send-email-code', { method: 'POST', body: payload }),
  register: (payload) => requestJson('/api/auth/register', { method: 'POST', body: payload }),
  login: (payload) => requestJson('/api/auth/login', { method: 'POST', body: payload }),
  requestPasswordReset: (payload) => requestJson('/api/auth/request-password-reset', { method: 'POST', body: payload }),
  resetPassword: (payload) => requestJson('/api/auth/reset-password', { method: 'POST', body: payload }),
  changePassword: (payload) => requestJson('/api/auth/change-password', { method: 'POST', body: payload }),
  deactivate: (payload) => requestJson('/api/auth/deactivate', { method: 'POST', body: payload }),
  logout: () => requestJson('/api/auth/logout', { method: 'POST' }),
};
