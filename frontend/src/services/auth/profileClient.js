import { getBackendUrl } from '../runtime/backendUrl';
import { emitAuthRequired } from './authEvents';
import { authHeaders, clearDeviceSession } from './sessionTokenStore';

async function profileRequest(path, options = {}) {
  const response = await fetch(getBackendUrl(path), {
    ...options,
    headers: authHeaders(options.headers || {}),
    credentials: 'include',
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      clearDeviceSession();
      emitAuthRequired();
    }
    const detail = payload?.detail || null;
    const message = typeof detail === 'string' ? detail : detail?.message;
    const error = new Error(message || `Request failed: ${response.status}`);
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return payload;
}

export const profileClient = {
  updateDisplayName(displayName) {
    return profileRequest('/api/profile/display-name', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_name: displayName }),
    });
  },

  updateAvatar(blob) {
    const form = new FormData();
    const extension = blob?.type === 'image/png' ? 'png' : blob?.type === 'image/webp' ? 'webp' : 'jpg';
    form.append('avatar', blob, `avatar.${extension}`);
    return profileRequest('/api/profile/avatar', {
      method: 'PUT',
      body: form,
    });
  },

  removeAvatar() {
    return profileRequest('/api/profile/avatar', { method: 'DELETE' });
  },
};

