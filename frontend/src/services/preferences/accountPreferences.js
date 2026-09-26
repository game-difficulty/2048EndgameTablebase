import { ref } from 'vue';
import themes from '../../../../docs_and_configs/themes.json' with { type: 'json' };

import { authHeaders } from '../auth/sessionTokenStore.js';
import { getBackendUrl } from '../runtime/backendUrl.js';
import { createLocalStorageStore } from '../storage/localStorageStore.js';

export const ACCOUNT_GLOBAL_KEYS = Object.freeze([
  'language', 'dark_mode', 'theme', 'use_custom_theme', 'custom_colors',
  'font_size_factor', 'ui_scale', 'do_animation',
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
let accountId = null;
let generation = 0;
let loaded = false;
let pending = {};
let timer = null;
let writing = null;
let applying = false;

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
    } else if (key === 'custom_colors' && Array.isArray(value) && value.length > 0 && value.length <= 36) {
      const colors = Array.from({ length: 36 }, (_, index) => value[index] || palette[index] || '#000000');
      if (colors.every(color => typeof color === 'string' && /^#[\da-f]{6}$/i.test(color))) result[key] = colors;
    }
  }
  return result;
}

function localValues() {
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
    if (!response.ok) throw new Error(`Preference sync failed: ${response.status}`);
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
      preferenceSyncStatus.value = 'saved';
    }
  } catch {
    if (serial === generation && id === accountId) {
      pending = { ...changes, ...pending };
      writePending(id, pending);
      preferenceSyncStatus.value = 'error';
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
    preferenceSyncStatus.value = 'error';
    return;
  }
  preferenceSyncStatus.value = 'saving';
  scheduleFlush();
}

export async function activateAccountPreferences(userId, { force = false } = {}) {
  const nextId = userId == null ? null : String(userId);
  if (nextId === accountId && loaded && !force) return;
  const retainedPending = nextId ? {
    ...(pendingStore.read()[nextId] || {}),
    ...(nextId === accountId ? pending : {}),
  } : {};
  generation += 1;
  const serial = generation;
  clearTimeout(timer);
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
    preferenceSyncStatus.value = Object.keys(pending).length ? 'saving' : 'saved';
    scheduleFlush();
  } catch {
    if (serial !== generation) return;
    loaded = false;
    preferenceSyncStatus.value = 'error';
  }
}

export function retryAccountPreferences() {
  if (accountId && !loaded) return activateAccountPreferences(accountId, { force: true });
  if (accountId) {
    preferenceSyncStatus.value = 'saving';
    void flush();
  }
}

export function refreshAccountPreferences() {
  if (accountId) return activateAccountPreferences(accountId, { force: true });
  return Promise.resolve();
}
