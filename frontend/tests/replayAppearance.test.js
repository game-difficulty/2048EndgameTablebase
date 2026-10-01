import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import themes from '../../docs_and_configs/themes.json' with { type: 'json' };
import { resolveTileColors } from '../src/utils/tileColors.js';

const source = readFileSync(new URL('../public/verse-replay/appearance.js', import.meta.url), 'utf8');
const settle = async () => { for (let i = 0; i < 40; i++) await Promise.resolve(); };
function viewer({ preferences = {}, cookie = '', systemDark = false, account = null, saved = null, cache = {}, denied = false } = {}) {
  const attributes = {};
  const listeners = {};
  const storage = new Map([['2048tables:user-preferences', JSON.stringify({ version: 1, value: preferences })], ['saved-vth-theme-cache-v1', JSON.stringify(cache)]]);
  let writes = 0;
  const requests = [];
  const style = { textContent: '' };
  const document = { documentElement: { setAttribute: (key, value) => { attributes[key] = value; }, getAttribute: key => attributes[key] },
    cookie, head: { appendChild() {} }, createElement: () => style, visibilityState: 'visible', addEventListener: (key, fn) => { listeners[key] = fn; } };
  const localStorage = { getItem: key => { if (denied) throw new Error('storage disabled'); return storage.get(key) || null; }, setItem() { writes++; } };
  const media = { matches: systemDark, addEventListener: (key, fn) => { listeners['media-' + key] = fn; } };
  const context = { document, window: { localStorage, sessionStorage: localStorage, matchMedia: () => media,
    ReplayThemeCatalog: Object.fromEntries(Object.entries(themes).map(([name, colors]) => [name, resolveTileColors(colors)])), addEventListener: (key, fn) => { listeners[key] = fn; } },
    setTimeout, clearTimeout, AbortController,
    fetch: async (url, options) => {
      requests.push({ url, options });
      const payload = url.includes('/themes/') ? saved : account && { preferences: account };
      return { ok: !!payload, json: async () => payload };
    } };
  vm.runInNewContext(source, context);
  return { attributes, style, listeners, requests, media, storage, writes: () => writes };
}
test('cached dark mode and preset palette apply synchronously, without writes', async () => {
  const view = viewer({ preferences: { dark_mode: true, theme: 'Chrome' } });
  assert.equal(view.attributes['data-theme'], 'dark');
  assert.ok(view.style.textContent.includes(resolveTileColors(themes.Chrome)[0].background));
  assert.ok(view.style.textContent.includes('.node-badge.value-2'));
  await settle();
  assert.equal(view.writes(), 0);
  assert.ok(view.requests.every(request => !request.options.method));
});
test('custom palette uses exactly the main-site text colors and rejects placeholders', () => {
  const colors = ['#abcdef', '#eeffff', '#010101', '#ffffee', '#020202'];
  const view = viewer({ preferences: { use_custom_theme: true, custom_colors: colors } });
  resolveTileColors(colors).forEach(tile => assert.ok(view.style.textContent.includes('background:' + tile.background + ';color:' + tile.color)));
  assert.equal(viewer({ preferences: { colors: Array(36).fill('#000000') } }).style.textContent, '');
});
test('shared palette cookie is a cross-subdomain fallback', () => {
  const view = viewer({ cookie: '2048tables-tile-palette=' + encodeURIComponent(JSON.stringify([{ background: '#123456', color: '#ffffff' }])) });
  assert.match(view.style.textContent, /background:#123456;color:#ffffff/);
});
function theme() {
  const entries = Object.fromEntries(Array.from({ length: 16 }, (_, i) => [2 ** (i + 1), { '--tile-background': '#123456', '--tile-color': '#abcdef', '--tile-shadow-color': '#aabbcc22', '--tile-outline-color': '#ddffaa55' }]));
  return { dark: entries, light: Object.fromEntries(Object.entries(entries).map(([key, value]) => [key, { ...value, '--tile-background': '#fedcba' }])) };
}
test('saved theme inherits mode, foreground, shadow and outline', async () => {
  const view = viewer({ account: { dark_mode: true, saved_theme_id: 7 }, saved: { theme: theme() } });
  await settle();
  assert.equal(view.attributes['data-theme'], 'dark');
  assert.match(view.style.textContent, /background:#123456;color:#abcdef;box-shadow:0 0 10px #aabbcc22,inset 0 0 0 1px #ddffaa55/);
  view.listeners.focus();
  assert.equal(view.attributes['data-theme'], 'dark');
  assert.match(view.style.textContent, /#123456/);
});
test('system preference is only used when no explicit mode exists', () => {
  const view = viewer({ systemDark: true, denied: true });
  assert.equal(view.attributes['data-theme'], 'dark');
  view.media.matches = false;
  view.listeners['media-change']();
  assert.equal(view.attributes['data-theme'], 'light');
  const explicit = viewer({ preferences: { dark_mode: false }, systemDark: true });
  explicit.listeners['media-change']();
  assert.equal(explicit.attributes['data-theme'], 'light');
});
test('same-origin storage updates both tile classes, and cached themes apply before requests', () => {
  const view = viewer({ preferences: { saved_theme_id: 7, dark_mode: false }, cache: { 7: { theme: theme() } } });
  assert.match(view.style.textContent, /#fedcba/);
  view.storage.set('2048tables:user-preferences', JSON.stringify({ version: 1, value: { dark_mode: true, theme: 'Default' } }));
  view.listeners.storage({ key: '2048tables:user-preferences' });
  assert.equal(view.attributes['data-theme'], 'dark');
  assert.ok(view.style.textContent.includes(resolveTileColors(themes.Default)[0].background));
});
