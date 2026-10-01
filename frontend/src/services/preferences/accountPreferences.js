import { ref } from 'vue';
import themes from '../../../../docs_and_configs/themes.json' with { type: 'json' };

import { authHeaders } from '../auth/sessionTokenStore.js';
import { getBackendUrl } from '../runtime/backendUrl.js';
import { createLocalStorageStore } from '../storage/localStorageStore.js';
import { ensureStoredLanguage } from './languagePreference.js';

export const ACCOUNT_GLOBAL_KEYS = Object.freeze([
  'language', 'dark_mode', 'theme', 'use_custom_theme', 'custom_colors',
  'font_size_factor', 'ui_scale', 'do_animation', 'saved_theme_id',
]);
export const ACCOUNT_HUMAN_KEYS = Object.freeze([
  'alwaysConfirmRestart', 'showSpeed', 'showFourPercent',
]);
const GLOBAL = new Set(ACCOUNT_GLOBAL_KEYS);
const HUMAN = new Set(ACCOUNT_HUMAN_KEYS);
const ALLOWED = new Set([...GLOBAL, ...HUMAN]);
const globalStore = createLocalStorageStore({ key: 'user-preferences', version: 1, defaultValue: {} });
const humanStore = createLocalStorageStore({ key: 'human-settings', version: 1, defaultValue: {} });
const ownerStore = createLocalStorageStore({ key: 'account-preferences-owner', version: 1, defaultValue: null });
const guestStore = createLocalStorageStore({ key: 'guest-presentation-preferences', version: 1, defaultValue: null });
const pendingStore = createLocalStorageStore({ key: 'account-preferences-pending', version: 1, defaultValue: {} });

export const preferenceSyncStatus = ref('idle');
export const preferenceSyncError = ref('');
const RETRY_DELAYS = [1000, 3000, 10000];
let retryTimer = null;
let retryCount = 0;
let lastRecoveryAt = 0;
let retryable = false;
let reading = false;
let accountId = null;
let generation = 0;
let loaded = false;
let pending = {};
let timer = null;
let writing = null;
let applying = false;

export function preferenceSyncMessage(language) {
  const zh = String(language).startsWith('zh');
  if (preferenceSyncError.value === 'auth') return zh ? '登录状态已失效，请重新登录后同步账号设置。' : 'Your session has expired. Sign in again to sync account settings.';
  if (preferenceSyncError.value === 'validation') return zh ? '部分账号设置未被服务器接受，请调整设置后重试。' : 'Some account settings were rejected. Adjust them and try again.';
  if (preferenceSyncError.value === 'server') return zh ? '账号设置暂时无法同步，请稍后重试。' : 'Account settings cannot sync right now. Try again later.';
  return zh ? '账号设置尚未同步，连接恢复后将重试。' : 'Account settings have not synced. We will retry when the connection recovers.';
}

function clearRetry() { clearTimeout(retryTimer); retryTimer = null; }
function synced() {
  clearRetry();
  retryCount = 0;
  retryable = false;
  preferenceSyncError.value = '';
}
function scheduleRecovery() {
  if (!accountId || !retryable || retryTimer || retryCount >= RETRY_DELAYS.length) return;
  const serial = generation;
  retryTimer = setTimeout(() => {
    retryTimer = null;
    lastRecoveryAt = Date.now();
    if (serial === generation) void recover();
  }, RETRY_DELAYS[retryCount++]);
}
function failed(error) {
  const status = Number(error.status) || 0;
  retryable = !status || status === 429 || status >= 500;
  preferenceSyncError.value = status === 401 || status === 403 ? 'auth' : status >= 400 && status < 500 && status !== 429 ? 'validation' : status ? 'server' : 'network';
  preferenceSyncStatus.value = 'error';
  scheduleRecovery();
}
function recover() {
  if (!accountId || reading || writing) return;
  if (!loaded) return activateAccountPreferences(accountId, { force: true });
  return flush();
}

function pick(source, keys) {
  return Object.fromEntries(keys.filter(key => Object.hasOwn(source || {}, key)).map(key => [key, source[key]]));
}

function normalized(source) {
  const result = {};
  const palette = globalStore.read().colors || [];
  for (const [key, value] of Object.entries(source || {})) {
    if (!ALLOWED.has(key)) continue;
    if (['dark_mode', 'use_custom_theme', 'do_animation', ...ACCOUNT_HUMAN_KEYS].includes(key)) {
      if (typeof value === 'boolean') result[key] = value;
    } else if (key === 'language') {
      if (value === 'zh' || value === 'en') result[key] = value;
    } else if (key === 'theme') {
      if (typeof value === 'string' && Object.hasOwn(themes, value)) result[key] = value;
    } else if (key === 'font_size_factor' || key === 'ui_scale') {
      const min = key === 'font_size_factor' ? 50 : 90;
      const max = key === 'font_size_factor' ? 150 : 125;
      if (Number.isInteger(value) && value >= min && value <= max && value % 5 === 0) result[key] = value;
    } else if (key === 'saved_theme_id') {
      if (Number.isInteger(value) && value >= 0) result[key] = value;
    } else if (key === 'custom_colors' && Array.isArray(value) && value.length > 0 && value.length <= 36) {
      const colors = Array.from({ length: 36 }, (_, index) => value[index] || palette[index] || '#000000');
      if (colors.every(color => typeof color === 'string' && /^#[\da-f]{6}$/i.test(color))) result[key] = colors;
    }
  }
  return result;
}

function localValues() {
  ensureStoredLanguage(globalStore);
  return normalized({ ...pick(globalStore.read(), ACCOUNT_GLOBAL_KEYS), ...pick(humanStore.read(), ACCOUNT_HUMAN_KEYS) });
}

function writePending(userId, changes) {
  pendingStore.update(current => {
    const next = { ...current };
    if (Object.keys(changes).length) next[userId] = changes;
    else delete next[userId];
    return next;
  });
}

function apply(values) {
  applying = true;
  try {
    const global = globalStore.read();
    const human = humanStore.read();
    for (const key of ACCOUNT_GLOBAL_KEYS) delete global[key];
    delete global.colors; // Derived palette must not leak from the previously active account.
    for (const key of ACCOUNT_HUMAN_KEYS) delete human[key];
    globalStore.write({ ...global, ...pick(values, ACCOUNT_GLOBAL_KEYS),
      ...(Array.isArray(values?.colors) ? { colors: values.colors } : {}) });
    humanStore.write({ ...human, ...pick(values, ACCOUNT_HUMAN_KEYS) });
    window.dispatchEvent(new Event('account-preferences-changed'));
  } finally {
    applying = false;
  }
}

async function request(method, body) {
  const serial = generation;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(getBackendUrl('/api/profile/preferences'), {
      method,
      credentials: 'include',
      cache: 'no-store',
      headers: authHeaders(body ? { 'Content-Type': 'application/json' } : {}),
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      // Only discard a theme reference that the server explicitly identified as invalid.
      if (response.status === 422 && payload.detail === 'invalid_saved_theme' && body?.preferences?.saved_theme_id && serial === generation) {
        body.preferences.saved_theme_id = 0;
        return request(method, body);
      }
      const error = new Error(`Preference sync failed: ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return response.json();
  } finally {
    clearTimeout(timeout);
  }
}

function scheduleFlush() {
  clearTimeout(timer);
  if (accountId && loaded && Object.keys(pending).length) {
    timer = setTimeout(() => { void flush(); }, 400);
  }
}

async function flush() {
  if (writing || !accountId || !loaded || !Object.keys(pending).length) return;
  const id = accountId;
  const serial = generation;
  const changes = pending;
  pending = {};
  preferenceSyncStatus.value = 'saving';
  writing = request('PATCH', { preferences: changes });
  try {
    const result = await writing;
    if (serial === generation && id === accountId) {
      writePending(id, pending);
      apply({ ...result.preferences, ...pending });
      synced();
      preferenceSyncStatus.value = Object.keys(pending).length ? 'saving' : 'saved';
    }
  } catch (error) {
    if (serial === generation && id === accountId) {
      pending = { ...changes, ...pending };
      writePending(id, pending);
      failed(error);
    }
  } finally {
    writing = null;
    if (accountId && loaded && preferenceSyncStatus.value !== 'error') scheduleFlush();
  }
}

export function saveAccountPreferences(changes) {
  if (applying) return;
  const accepted = normalized(changes);
  if (!Object.keys(accepted).length) return;
  if (!accountId) {
    const guest = guestStore.read() || {};
    guestStore.write({ ...guest, ...accepted, colors: globalStore.read().colors });
    return;
  }
  pending = { ...pending, ...accepted };
  writePending(accountId, { ...(pendingStore.read()[accountId] || {}), ...pending });
  if (!loaded) {
    if (!reading) scheduleRecovery();
    return;
  }
  preferenceSyncStatus.value = 'saving';
  scheduleFlush();
}

export async function activateAccountPreferences(userId, { force = false } = {}) {
  const nextId = userId == null ? null : String(userId);
  if (nextId === accountId && !force && (loaded || reading)) {
    if (preferenceSyncStatus.value === 'error') scheduleRecovery();
    return;
  }
  const retainedPending = nextId ? {
    ...(pendingStore.read()[nextId] || {}),
    ...(nextId === accountId ? pending : {}),
  } : {};
  generation += 1;
  const serial = generation;
  clearTimeout(timer);
  clearRetry();
  if (nextId !== accountId) synced();
  reading = false;
  pending = retainedPending;
  loaded = false;
  const oldOwner = ownerStore.read();
  const previous = localValues();
  if (!nextId) {
    accountId = null;
    if (oldOwner != null) apply(guestStore.read() || {});
    ownerStore.write(null);
    preferenceSyncStatus.value = 'idle';
    return;
  }
  if (oldOwner == null && guestStore.read() == null) {
    guestStore.write({ ...previous, colors: globalStore.read().colors });
  }
  accountId = nextId;
  // A previous account's cached presentation must never become the new account's seed.
  const seed = oldOwner == null || String(oldOwner) === nextId ? previous : {};
  if (oldOwner != null && String(oldOwner) !== nextId) apply({});
  ownerStore.write(nextId);
  preferenceSyncStatus.value = 'loading';
  reading = true;
  try {
    let result = await request('GET');
    if (serial !== generation) return;
    const missing = Object.fromEntries(Object.entries(seed).filter(([key]) => !Object.hasOwn(result.preferences, key)));
    if (Object.keys(missing).length) {
      result = await request('PATCH', { preferences: missing, only_if_missing: true });
      if (serial !== generation) return;
    }
    loaded = true;
    apply({ ...result.preferences, ...pending });
    synced();
    preferenceSyncStatus.value = Object.keys(pending).length ? 'saving' : 'saved';
    scheduleFlush();
  } catch (error) {
    if (serial !== generation) return;
    loaded = false;
    failed(error);
  } finally {
    if (serial === generation) reading = false;
  }
}

export function retryAccountPreferences() {
  clearRetry();
  retryCount = 0;
  return recover();
}

export function refreshAccountPreferences() {
  if (reading || writing) return Promise.resolve();
  if (accountId) return activateAccountPreferences(accountId, { force: true });
  return Promise.resolve();
}

function resumeSync() {
  if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return;
  if (retryCount >= RETRY_DELAYS.length && Date.now() - lastRecoveryAt >= 30000) retryCount = 0;
  if (preferenceSyncStatus.value === 'error') scheduleRecovery();
}
window.addEventListener?.('online', resumeSync);
window.addEventListener?.('focus', resumeSync);
if (typeof document !== 'undefined') document.addEventListener('visibilitychange', resumeSync);
