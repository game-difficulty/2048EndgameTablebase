import { authHeaders } from '../auth/sessionTokenStore.js';
import { getBackendUrl } from '../runtime/backendUrl.js';
import { writeSharedTilePalette } from '../../utils/sharedTilePalette.js';
import { normalizeVthTheme, VTH_TILE_VALUES } from './vthThemeFormat.js';
export { normalizeVthTheme, VTH_TILE_VALUES, VTH_STYLE_KEYS } from './vthThemeFormat.js';

const CACHE_KEY = 'saved-vth-theme-cache-v1';
const verifiedThemes = new Set();
let appearanceRevision = 0;

function cacheRead() { try { return JSON.parse(localStorage.getItem(CACHE_KEY) || '{}'); } catch { return {}; } }
function cacheWrite(value) { try { localStorage.setItem(CACHE_KEY, JSON.stringify(value)); } catch { /* optional cache */ } }
function cacheTheme(item) { const cache = cacheRead(); cache[item.id] = item; cacheWrite(cache); return item; }

async function api(path = '', options = {}) {
  const response = await fetch(getBackendUrl(`/api/profile/themes${path}`), {
    credentials: 'include', cache: 'no-store', ...options,
    headers: authHeaders(options.body ? { 'Content-Type': 'application/json' } : {}),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const error = new Error(typeof payload.detail === 'string' ? payload.detail : `HTTP_${response.status}`);
    error.code = error.message; throw error;
  }
  return response.status === 204 ? null : response.json();
}

export const listSavedThemes = () => api();
export async function getSavedTheme(id, { force = false } = {}) {
  const cached = cacheRead()[id];
  if (cached && !force) return cached;
  return cacheTheme(await api(`/${id}`));
}
export async function createSavedTheme(name, theme) { return cacheTheme(await api('', { method: 'POST', body: JSON.stringify({ name, theme: normalizeVthTheme(theme) }) })); }
export async function updateSavedTheme(id, name, theme) { return cacheTheme(await api(`/${id}`, { method: 'PUT', body: JSON.stringify({ name, theme: normalizeVthTheme(theme) }) })); }
export async function deleteSavedTheme(id) { await api(`/${id}`, { method: 'DELETE' }); const cache = cacheRead(); delete cache[id]; cacheWrite(cache); verifiedThemes.delete(Number(id)); }

export function clearSavedThemeStyles() {
  appearanceRevision += 1;
  if (typeof document === 'undefined') return;
  for (const value of VTH_TILE_VALUES) {
    document.documentElement.style.removeProperty(`--color-shadow-${value}`);
    document.documentElement.style.removeProperty(`--color-outline-${value}`);
  }
}
export function applySavedThemePayload(theme, darkMode) {
  const normalized = normalizeVthTheme(theme);
  const mode = darkMode ? (normalized.dark || normalized.light) : (normalized.light || normalized.dark);
  const palette = [];
  for (const value of VTH_TILE_VALUES) {
    const style = mode[value];
    document.documentElement.style.setProperty(`--color-tile-${value}`, style['--tile-background']);
    document.documentElement.style.setProperty(`--color-text-${value}`, style['--tile-color']);
    document.documentElement.style.setProperty(`--color-shadow-${value}`, style['--tile-shadow-color']);
    document.documentElement.style.setProperty(`--color-outline-${value}`, style['--tile-outline-color']);
    palette.push({ background: style['--tile-background'], color: style['--tile-color'] });
  }
  writeSharedTilePalette(palette); return palette;
}
export async function applyActiveSavedTheme(id, darkMode) {
  if (!Number.isInteger(Number(id)) || Number(id) <= 0) { clearSavedThemeStyles(); return false; }
  const revision = ++appearanceRevision;
  const cached = cacheRead()[id];
  if (cached) applySavedThemePayload(cached.theme, darkMode);
  if (cached && verifiedThemes.has(Number(id))) return true;
  const item = await getSavedTheme(Number(id), { force: true });
  if (revision !== appearanceRevision) return false;
  applySavedThemePayload(item.theme, darkMode);
  verifiedThemes.add(Number(id)); return true;
}
export function downloadVth(name, theme) {
  const blob = new Blob([JSON.stringify(normalizeVthTheme(theme), null, 2)], { type: 'application/json' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
  link.download = `${String(name || 'theme').replace(/[\\/:*?"<>|]/g, '_')}.vth`; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}
