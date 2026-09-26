import test, { mock } from 'node:test';
import assert from 'node:assert/strict';

const values = new Map();
const localStorage = {
  getItem: key => values.get(key) ?? null,
  setItem: (key, value) => values.set(key, value),
  removeItem: key => values.delete(key),
};
globalThis.window = {
  localStorage,
  location: { href: 'https://2048tables.online/' },
  dispatchEvent: () => {},
};

let remoteUser = '1';
let failNextPatch = false;
const remote = new Map();
globalThis.fetch = async (_url, options) => {
  if (options.method === 'PATCH' && failNextPatch) {
    failNextPatch = false;
    return { ok: false, status: 503 };
  }
  const current = remote.get(remoteUser) || { preferences: {}, revision: 0 };
  if (options.method === 'PATCH') {
    const { preferences, only_if_missing } = JSON.parse(options.body);
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
  activateAccountPreferences, preferenceSyncStatus, refreshAccountPreferences, retryAccountPreferences,
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
