import { authHeaders } from './auth/sessionTokenStore.js';
import { getBackendUrl } from './runtime/backendUrl';

export function reportAiSetboard(payload) {
  // Best effort and off the move/animation critical path. No continuous AI telemetry.
  void fetch(getBackendUrl('/api/assistance/ai-setboard'), {
    method: 'POST', credentials: 'include', keepalive: true,
    headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify(payload),
  }).catch(() => {});
}
