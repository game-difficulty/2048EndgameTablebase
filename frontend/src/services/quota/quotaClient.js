import { getBackendUrl } from '../runtime/backendUrl';

export async function getQuotaRules() {
  const response = await fetch(getBackendUrl('/api/quota/rules'), {
    credentials: 'include',
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`Quota rules request failed: ${response.status}`);
  }
  return payload;
}
