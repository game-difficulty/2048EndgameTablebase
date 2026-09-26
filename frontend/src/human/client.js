import { authHeaders } from '../services/auth/sessionTokenStore.js';
import { eventBytes } from './engine.js';
import { decodeReceipt, uploadBody } from './wire.js';

export async function request(path, { body, method = 'GET', headers = {}, binary = false, keepalive = false, timeoutMs = 8000 } = {}) {
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, { credentials: 'include', cache: 'no-store', method, keepalive,
      headers: authHeaders({ ...(path.startsWith('/api/human/runs/') ? { 'X-Human-Protocol': '2' } : {}),
        ...(body !== undefined ? { 'Content-Type': binary ? 'application/octet-stream' : 'application/json' } : {}), ...headers }),
      body: body === undefined ? undefined : binary ? body : JSON.stringify(body), signal: controller.signal });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({})); const detail = payload.detail;
      const error = new Error(detail?.code || (typeof detail === 'string' ? detail : `HTTP_${response.status}`));
      error.retryAfter = Math.max(1, Number(response.headers.get('Retry-After')) || 2); error.code = error.message; error.status = response.status; error.detail = detail; throw error;
    }
    return response;
  } finally { clearTimeout(timeout); }
}
export async function json(path, options) { return decodeReceipt(await (await request(path, options)).json()); }
export const getStatus = (run, browser) => json(`/api/human/runs/${run.id}/status`, { headers: { 'X-Human-Browser': browser } });
export async function upload(run, events, browser, writer, action, status, keepalive = false) {
  const start = status.seq;
  const packed = await uploadBody(eventBytes(events.slice(start)));
  return json(`/api/human/runs/${run.id}/${action}`, {
    method: 'POST', body: packed.body, binary: true, keepalive,
    headers: { ...packed.headers, 'X-Human-Browser': browser, 'X-Human-Writer': writer, 'X-Human-Epoch': String(status.epoch),
      'X-Human-Start': String(start), 'X-Human-Count': String(run.seq),
      'X-Human-Prefix': start === 0 ? run.initialHash : events[start - 1]?.[2] || 'missing',
      'X-Human-Reason': run.reason || '', 'X-Human-Permit': run.permit || '' },
  });
}
