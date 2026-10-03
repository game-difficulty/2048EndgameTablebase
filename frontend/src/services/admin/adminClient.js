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
  decideAssistanceReview: (id, approved, note) => requestJson(
    `/api/admin/assistance-reviews/${id}/decision`, { method: 'POST', body: { approved, note } }),
  profileReviews: ({ page = 1, status = 'all', changeType = 'all', query = '' } = {}) => {
    const params = new URLSearchParams({ page, status, change_type: changeType, q: query });
    return requestJson(`/api/admin/profile-reviews?${params}`);
  },
  reviewProfile: (id, action) => requestJson(`/api/admin/profile-reviews/${id}`, { method: 'POST', body: { action } }),
  approvePendingProfileReviews: () => requestJson('/api/admin/profile-reviews/approve-pending', { method: 'POST' }),
  verseClaims: (userId = null) => requestJson('/api/admin/verse-claims'
    + (userId ? `?user_id=${encodeURIComponent(userId)}` : '')),
  decideVerseClaim: (id, approved, note) => requestJson(
    `/api/admin/verse-claims/` + encodeURIComponent(id) + '/decision',
    { method: 'POST', body: { approved, note } },
  ),
  retryVerseClaim: (id, note) => requestJson('/api/admin/verse-claims/' + encodeURIComponent(id) + '/retry',
    { method: 'POST', body: { note } }),
  revokeVerseClaim: (id, note) => requestJson('/api/admin/verse-claims/' + encodeURIComponent(id) + '/revoke',
    { method: 'POST', body: { note } }),
  archiveApplications: (userId = null) => requestJson('/api/admin/archive-applications'
    + (userId ? `?user_id=${encodeURIComponent(userId)}` : '')),
  approvalTransactions: ({ q = '', kind = 'all', stage = 'all', page = 1, pageSize = 30 } = {}) => {
    const params = new URLSearchParams({ kind, stage, page: String(page), page_size: String(pageSize) });
    if (q) params.set('q', q);
    return requestJson(`/api/admin/approval-transactions?${params.toString()}`);
  },
  decideArchiveApplication: (id, approved, note) => requestJson(
    `/api/admin/archive-applications/${encodeURIComponent(id)}/decision`,
    { method: 'POST', body: { approved, note } }),
  revokeArchiveApplication: (id, note) => requestJson(
    `/api/admin/archive-applications/${encodeURIComponent(id)}/revoke`,
    { method: 'POST', body: { note } }),
  liveStatus: () => requestJson('/api/admin/live'),
  setLiveEnabled: (enabled) => requestJson('/api/admin/live', { method: 'POST', body: { enabled } }),
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
  updateUserStatus: (userId, status) => requestJson(`/api/admin/users/${encodeURIComponent(userId)}/status`, {
    method: 'POST',
    body: { status },
  }),
  resetManagedPassword: (userId, newPassword) => requestJson(`/api/admin/users/${encodeURIComponent(userId)}/managed-password`, {
    method: 'POST',
    body: { new_password: newPassword },
  }),
};
