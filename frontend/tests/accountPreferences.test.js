import test, { mock } from 'node:test';
import assert from 'node:assert/strict';

const values = new Map();
const listeners = new Map();
globalThis.document = { visibilityState: 'visible', createElement: () => ({}), addEventListener: (name, callback) => listeners.set(name, callback) };
const localStorage = {
  getItem: key => values.get(key) ?? null,
  setItem: (key, value) => values.set(key, value),
  removeItem: key => values.delete(key),
};
globalThis.window = {
  localStorage,
  location: { href: 'https://2048tables.online/' },
  dispatchEvent: () => {},
  addEventListener: (name, callback) => listeners.set(name, callback),
};

let remoteUser = '1';
let failNextPatch = false;
let failNextGet = false;
let rejectTheme = false;
let permanentStatus = 0;
let patchCalls = 0;
const remote = new Map();
globalThis.fetch = async (_url, options) => {
  if (options.method === 'PATCH') patchCalls++;
  if (options.method === 'GET' && failNextGet) {
    failNextGet = false;
    throw new TypeError('Failed to fetch');
  }
  if (options.method === 'PATCH' && permanentStatus) return { ok: false, status: permanentStatus, json: async () => ({ detail: 'invalid_preferences' }) };
  if (options.method === 'PATCH' && failNextPatch) {
    failNextPatch = false;
    return { ok: false, status: 503, json: async () => ({}) };
  }
  const current = remote.get(remoteUser) || { preferences: {}, revision: 0 };
  if (options.method === 'PATCH') {
    const { preferences, only_if_missing } = JSON.parse(options.body);
    if (rejectTheme && preferences.saved_theme_id) return { ok: false, status: 422, json: async () => ({ detail: 'invalid_saved_theme' }) };
    for (const [key, value] of Object.entries(preferences)) {
      if (!only_if_missing || !Object.hasOwn(current.preferences, key)) current.preferences[key] = value;
    }
    current.revision++;
    remote.set(remoteUser, current);
  }
  return { ok: true, json: async () => structuredClone(current) };
};

mock.module('../src/services/auth/sessionTokenStore.js', { namedExports: {
  authHeaders: headers => headers,
} });

const { createLocalStorageStore } = await import('../src/services/storage/localStorageStore.js');
const {
  activateAccountPreferences, preferenceSyncStatus, preferenceSyncError, preferenceSyncMessage, refreshAccountPreferences, retryAccountPreferences,
  saveAccountPreferences,
} = await import('../src/services/preferences/accountPreferences.js');
const globalStore = createLocalStorageStore({ key: 'user-preferences', version: 1, defaultValue: {} });
const humanStore = createLocalStorageStore({ key: 'human-settings', version: 1, defaultValue: {} });

test('migrates explicit local settings once, reloads account values, and isolates account changes', async () => {
  globalStore.write({ language: 'zh', dark_mode: true, colors: Array(36).fill('#abcdef') });
  humanStore.write({ alwaysConfirmRestart: true, swipeSensitivity: 150 });
  await activateAccountPreferences(1);
  assert.deepEqual(remote.get('1').preferences, {
    language: 'zh', dark_mode: true, alwaysConfirmRestart: true,
  });

  globalStore.update(current => ({ ...current, theme: 'Chrome' }));
  saveAccountPreferences({ theme: 'Chrome' });
  await new Promise(resolve => setTimeout(resolve, 450));
  assert.equal(remote.get('1').preferences.theme, 'Chrome');

  remote.get('1').preferences.theme = 'Classic';
  await refreshAccountPreferences();
  assert.equal(globalStore.read().theme, 'Classic');

  remoteUser = '2';
  await activateAccountPreferences(2);
  assert.equal(globalStore.read().theme, undefined);
  assert.equal(globalStore.read().language, undefined);
  assert.equal(globalStore.read().colors, undefined);
  assert.equal(humanStore.read().alwaysConfirmRestart, undefined);
  assert.equal(humanStore.read().swipeSensitivity, 150);
  assert.deepEqual(remote.get('2'), undefined);

  await activateAccountPreferences(null);
  assert.equal(globalStore.read().language, 'zh');
  assert.equal(globalStore.read().colors[0], '#abcdef');
  assert.equal(humanStore.read().alwaysConfirmRestart, true);
});

test('failed account write stays pending and can be retried', async () => {
  remoteUser = '1';
  await activateAccountPreferences(1);
  failNextPatch = true;
  globalStore.update(current => ({ ...current, dark_mode: false }));
  saveAccountPreferences({ dark_mode: false });
  await new Promise(resolve => setTimeout(resolve, 450));
  assert.equal(preferenceSyncStatus.value, 'error');
  assert.equal(remote.get('1').preferences.dark_mode, true);
  retryAccountPreferences();
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(remote.get('1').preferences.dark_mode, false);
  assert.equal(preferenceSyncStatus.value, 'saved');
});

test('temporary write failure automatically recovers without another settings change', async () => {
  failNextPatch = true;
  saveAccountPreferences({ dark_mode: true });
  await new Promise(resolve => setTimeout(resolve, 450));
  assert.equal(preferenceSyncError.value, 'server');
  await new Promise(resolve => setTimeout(resolve, 1050));
  assert.equal(preferenceSyncStatus.value, 'saved');
  assert.equal(remote.get('1').preferences.dark_mode, true);
});

test('temporary initial read failure automatically recovers', async () => {
  failNextGet = true;
  await refreshAccountPreferences();
  assert.equal(preferenceSyncError.value, 'network');
  await new Promise(resolve => setTimeout(resolve, 1050));
  assert.equal(preferenceSyncStatus.value, 'saved');
});

test('invalid saved theme does not block other pending settings', async () => {
  rejectTheme = true;
  saveAccountPreferences({ saved_theme_id: 999, dark_mode: false });
  await new Promise(resolve => setTimeout(resolve, 450));
  assert.equal(preferenceSyncStatus.value, 'saved');
  assert.equal(remote.get('1').preferences.saved_theme_id, 0);
  assert.equal(remote.get('1').preferences.dark_mode, false);
  assert.equal(globalStore.read().saved_theme_id, 0);
  rejectTheme = false;
});

test('validation failures remain pending and are not mistaken for network errors', async () => {
  permanentStatus = 422;
  saveAccountPreferences({ dark_mode: true });
  await new Promise(resolve => setTimeout(resolve, 450));
  const calls = patchCalls;
  assert.equal(preferenceSyncError.value, 'validation');
  assert.match(preferenceSyncMessage('zh'), /未被服务器接受/);
  await new Promise(resolve => setTimeout(resolve, 1050));
  assert.equal(patchCalls, calls);
  permanentStatus = 0;
  await retryAccountPreferences();
  assert.equal(preferenceSyncStatus.value, 'saved');
});

test('expired sessions stop automatic retries and request sign-in', async () => {
  permanentStatus = 401;
  saveAccountPreferences({ dark_mode: false });
  await new Promise(resolve => setTimeout(resolve, 450));
  assert.equal(preferenceSyncError.value, 'auth');
  assert.match(preferenceSyncMessage('zh'), /重新登录/);
  permanentStatus = 0;
  await retryAccountPreferences();
  assert.equal(preferenceSyncStatus.value, 'saved');
});

test('switching accounts cancels the old account retry', async () => {
  failNextPatch = true;
  saveAccountPreferences({ dark_mode: true });
  await new Promise(resolve => setTimeout(resolve, 450));
  remoteUser = '2';
  await activateAccountPreferences(2);
  const calls = patchCalls;
  await new Promise(resolve => setTimeout(resolve, 1050));
  assert.equal(patchCalls, calls);
  assert.equal(preferenceSyncStatus.value, 'saved');
  await activateAccountPreferences(null);
});

test('stale local theme seed is cleared without blocking initial account loading', async () => {
  remoteUser = '3';
  globalStore.write({ language: 'zh', saved_theme_id: 999, dark_mode: false });
  rejectTheme = true;
  await activateAccountPreferences(3);
  assert.equal(preferenceSyncStatus.value, 'saved');
  assert.equal(remote.get('3').preferences.saved_theme_id, 0);
  assert.equal(remote.get('3').preferences.language, 'zh');
  assert.equal(remote.get('3').preferences.dark_mode, false);
  rejectTheme = false;
  await activateAccountPreferences(null);
});

test('retries are bounded and returning to the foreground starts a fresh bounded recovery', async t => {
  remoteUser = '1';
  await activateAccountPreferences(1);
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
  const settle = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };
  const calls = patchCalls;
  permanentStatus = 503;
  saveAccountPreferences({ dark_mode: true });
  for (const delay of [400, 1000, 3000, 10000, 40000]) {
    t.mock.timers.tick(delay);
    await settle();
  }
  assert.equal(patchCalls, calls + 4);
  assert.equal(preferenceSyncStatus.value, 'error');
  permanentStatus = 0;
  document.visibilityState = 'hidden';
  listeners.get('visibilitychange')();
  t.mock.timers.tick(2000);
  await settle();
  assert.equal(patchCalls, calls + 4);
  document.visibilityState = 'visible';
  listeners.get('visibilitychange')();
  t.mock.timers.tick(1000);
  await settle();
  assert.equal(preferenceSyncStatus.value, 'saved');
  assert.equal(patchCalls, calls + 5);
  await activateAccountPreferences(null);
});
