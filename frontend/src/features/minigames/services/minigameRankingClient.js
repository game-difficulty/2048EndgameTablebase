import { emitAuthRequired } from '../../../services/auth/authEvents';
import { authHeaders, clearDeviceSession } from '../../../services/auth/sessionTokenStore';
import { getBackendUrl } from '../../../services/runtime/backendUrl';

async function readJson(response, { authenticated = false } = {}) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (authenticated && response.status === 401) {
      clearDeviceSession();
      emitAuthRequired();
    }
    const detail = payload?.detail;
    const error = new Error(typeof detail === 'string' ? detail : `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

export async function fetchMinigameCatalog() {
  const response = await fetch(getBackendUrl('/api/minigame-rankings/catalog'), {
    headers: { Accept: 'application/json' },
  });
  return readJson(response);
}

export async function fetchMinigameLeaderboard(gameId, { difficulty = 1, limit = 100 } = {}) {
  const query = new URLSearchParams({
    difficulty: String(Number(difficulty) ? 1 : 0),
    limit: String(limit),
  });
  const response = await fetch(
    getBackendUrl(`/api/minigame-rankings/games/${encodeURIComponent(gameId)}?${query}`),
    { headers: { Accept: 'application/json' }, cache: 'no-store' }
  );
  return readJson(response);
}

export async function fetchMinigameTrophyLeaderboard({ difficulty = 1, limit = 100 } = {}) {
  const query = new URLSearchParams({
    difficulty: String(Number(difficulty) ? 1 : 0),
    limit: String(limit),
  });
  const response = await fetch(
    getBackendUrl(`/api/minigame-rankings/overall?${query}`),
    { headers: { Accept: 'application/json' }, cache: 'no-store' }
  );
  return readJson(response);
}

export async function submitMinigameScore(payload) {
  const response = await fetch(getBackendUrl('/api/minigame-rankings/scores'), {
    method: 'POST',
    credentials: 'include',
    headers: authHeaders({
      Accept: 'application/json',
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify(payload),
  });
  return readJson(response, { authenticated: true });
}
