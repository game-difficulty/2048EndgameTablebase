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
    const error = new Error(
      typeof detail === 'string' ? detail : String(detail?.message || detail?.code || `HTTP ${response.status}`)
    );
    error.status = response.status;
    error.code = typeof detail === 'string' ? detail : String(detail?.code || '');
    throw error;
  }
  return payload;
}

export function createMinigameRequestId() {
  if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID();
  const bytes = new Uint8Array(16);
  if (typeof globalThis.crypto?.getRandomValues === 'function') globalThis.crypto.getRandomValues(bytes);
  else for (let index = 0; index < bytes.length; index += 1) bytes[index] = Math.floor(Math.random() * 256);
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

async function rankedRequest(path, { method = 'GET', body } = {}) {
  const response = await fetch(getBackendUrl(path), {
    method,
    credentials: 'include',
    headers: authHeaders({
      Accept: 'application/json',
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    }),
    body: body ? JSON.stringify(body) : undefined,
  });
  return readJson(response, { authenticated: true });
}

export const createMinigameRankedRun = ({
  requestId,
  gameId,
  difficulty,
  leaseToken,
  replaceRunId = null,
  replaceLeaseToken = null,
}) => rankedRequest(
  '/api/minigame-rankings/runs',
  {
    method: 'POST',
    body: {
      request_id: requestId,
      game_id: gameId,
      difficulty: Number(difficulty) ? 1 : 0,
      lease_token: leaseToken,
      replace_run_id: replaceRunId,
      replace_lease_token: replaceLeaseToken,
    },
  }
);

export const heartbeatMinigameRankedRun = (runId, leaseToken) => rankedRequest(
  `/api/minigame-rankings/runs/${encodeURIComponent(runId)}/heartbeat`,
  { method: 'POST', body: { lease_token: leaseToken } }
);

export const claimMinigameRankedRun = (runId, leaseToken) => rankedRequest(
  `/api/minigame-rankings/runs/${encodeURIComponent(runId)}/claim`,
  { method: 'POST', body: { lease_token: leaseToken } }
);

export const abandonMinigameRankedRun = (runId, leaseToken) => rankedRequest(
  `/api/minigame-rankings/runs/${encodeURIComponent(runId)}/abandon`,
  { method: 'POST', body: { lease_token: leaseToken } }
);

export const qualifyMinigameRankedRun = (runId, payload) => rankedRequest(
  `/api/minigame-rankings/runs/${encodeURIComponent(runId)}/qualify`,
  { method: 'POST', body: payload }
);

export const submitMinigameRankedRun = (runId, payload) => rankedRequest(
  `/api/minigame-rankings/runs/${encodeURIComponent(runId)}/submit`,
  { method: 'POST', body: payload }
);

export const submitMinigameRankedCheckpoint = (runId, payload) => rankedRequest(
  `/api/minigame-rankings/runs/${encodeURIComponent(runId)}/checkpoints`,
  { method: 'POST', body: payload }
);

export const fetchMinigameRankedCheckpoint = (runId, revision) => rankedRequest(
  `/api/minigame-rankings/runs/${encodeURIComponent(runId)}/checkpoints/${encodeURIComponent(revision)}`
);

export const fetchMinigameRankedRun = (runId) => rankedRequest(
  `/api/minigame-rankings/runs/${encodeURIComponent(runId)}`
);

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
