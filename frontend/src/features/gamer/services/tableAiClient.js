import { authHeaders } from '../../../services/auth/sessionTokenStore.js';
import { emitAuthRequired, emitTokenBalanceUpdated, emitTokenRequired } from '../../../services/auth/authEvents.js';
import { getBackendUrl } from '../../../services/runtime/backendUrl.js';

function routeError(payload, status) {
  const error = new Error('Tablebase request failed');
  error.status = status;
  error.code = payload?.detail?.code || payload?.detail || '';
  if (status === 401) emitAuthRequired();
  if (status === 402) emitTokenRequired(payload?.detail || {});
  return error;
}

export async function readTableRoute(body, { signal, onResult }) {
  let lastError;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const response = await fetch(getBackendUrl('/api/gamer/tablebase/route'), {
        method: 'POST', credentials: 'include', cache: 'no-store', signal,
        headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify(body),
      });
      if (!response.ok) throw routeError(await response.json().catch(() => ({})), response.status);
      const deliver = (line) => {
        if (signal.aborted || !line.trim()) return;
        const item = JSON.parse(line);
        if (item.type === 'error') throw routeError(item, item.status);
        emitTokenBalanceUpdated(item.token_balance);
        onResult(item);
      };
      if (!response.body?.getReader) {
        (await response.text()).split('\n').forEach(deliver);
      } else {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let pending = '';
        try {
          while (true) {
            const { done, value } = await reader.read();
            pending += decoder.decode(value, { stream: !done });
            const lines = pending.split('\n');
            pending = lines.pop();
            lines.forEach(deliver);
            if (done) { if (pending.trim()) deliver(pending); break; }
          }
        } finally { await reader.cancel().catch(() => {}); }
      }
      return;
    } catch (error) {
      lastError = error;
      if (signal.aborted || error.status) throw error;
    }
  }
  throw lastError;
}
