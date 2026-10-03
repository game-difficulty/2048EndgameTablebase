import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Exercise the actual store with browser, storage and transport boundaries stubbed.
const source = fs.readFileSync(new URL('../src/app/useAppSettings.js', import.meta.url), 'utf8')
  .replace(/^import .*;\r?\n/gm, '')
  .replace('export function useAppSettingsStore', 'function useAppSettingsStore');

function harness(initial = {}) {
  let stored = structuredClone(initial);
  let transport;
  const listeners = {};
  const attrs = {};
  let renderedDark;
  const context = {
    ref: value => ({ value }), computed: fn => ({ get value() { return fn(); } }),
    i18n: { global: { locale: { value: 'en' } } },
    document: { documentElement: {
      getAttribute: key => attrs[key], setAttribute: (key, value) => { attrs[key] = value; },
      removeAttribute: key => { delete attrs[key]; }, style: { setProperty() {} },
    } },
    window: { addEventListener: (key, fn) => { listeners[key] = fn; }, removeEventListener() {} },
    createLocalStorageStore: () => ({ key: 'preferences', read: () => structuredClone(stored),
      update: fn => { stored = fn(stored); } }),
    createWsClient: options => { transport = options; return { connect() {}, send() { return true; } }; },
    normalizeBCFamilyModulus: value => value,
    applyTileColors() {}, resolveTileColors: value => value, writeSharedTilePalette() {},
    applyActiveSavedTheme() { return Promise.resolve(true); }, clearSavedThemeStyles() {},
    ACCOUNT_GLOBAL_KEYS: ['language', 'dark_mode', 'theme', 'use_custom_theme', 'custom_colors',
      'font_size_factor', 'ui_scale', 'do_animation', 'saved_theme_id'],
    saveAccountPreferences() {},
    rememberRenderedAppearance: value => { renderedDark = value; },
  };
  vm.createContext(context);
  vm.runInContext(`${source};globalThis.store=useAppSettingsStore();`, context);
  context.store.start();
  return {
    setStored: value => { stored = value; },
    stored: () => structuredClone(stored),
    renderedDark: () => renderedDark,
    emit: (name, event = {}) => listeners[name](event),
    receive: () => transport.onMessage({ type: 'SETTINGS_DATA', payload: {
      config: { language: 'zh', dark_mode: true, theme: 'Chrome' },
      theme_map: { Chrome: ['#abcdef'] },
    } }),
    state: () => JSON.parse(JSON.stringify({ language: context.store.config.value.language,
      dark: context.store.config.value.dark_mode, theme: context.store.config.value.theme })),
  };
}

test('empty and partial preferences do not flash on repeated focus and settings responses', () => {
  for (const stored of [{}, { language: 'en' }, { language: 'en', dark_mode: false, theme: 'Classic' }]) {
    const h = harness(stored);
    h.receive();
    const expected = h.state();
    for (let i = 0; i < 3; i++) {
      h.emit('focus', { type: 'focus' });
      assert.deepEqual(h.state(), expected);
      h.receive();
      assert.deepEqual(h.state(), expected);
    }
  }
});

test('account replacement and cross-tab removal reset missing fields to server defaults', () => {
  for (const event of ['account-preferences-changed', 'storage']) {
    const h = harness({ language: 'en', dark_mode: false, theme: 'Classic' });
    h.receive();
    h.setStored({});
    h.emit(event, { key: 'preferences' });
    assert.deepEqual(h.state(), { language: 'zh', dark: true, theme: 'Chrome' });
    h.emit('focus');
    h.receive();
    assert.deepEqual(h.state(), { language: 'zh', dark: true, theme: 'Chrome' });
  }
});

test('focus still applies explicit preference updates made by another page', () => {
  const h = harness();
  h.receive();
  h.setStored({ language: 'en', dark_mode: false });
  h.emit('focus');
  assert.deepEqual(h.state(), { language: 'en', dark: false, theme: 'Chrome' });
});

test('server defaults are remembered for startup without becoming account preferences', () => {
  const h = harness({ language: 'en' });
  h.receive();
  assert.equal(h.renderedDark(), true);
  assert.deepEqual(h.stored(), { language: 'en' });
  h.setStored({ language: 'en', dark_mode: false });
  h.emit('account-preferences-changed');
  assert.equal(h.renderedDark(), false);
});
