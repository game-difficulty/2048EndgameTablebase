import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { rememberRenderedAppearance } from '../src/services/preferences/renderedAppearance.js';

const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const startupScript = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0]?.[1];
assert.ok(startupScript);

function boot(value, search = '', storageError = false, languages = ['en-US'], rendered = null, script = startupScript) {
  const attributes = { lang: 'en' };
  const document = { documentElement: {
    get lang() { return attributes.lang; },
    set lang(value) { attributes.lang = value; },
    setAttribute(key, value) { attributes[key] = value; },
    removeAttribute(key) { delete attributes[key]; },
  } };
  let storedValue = value == null ? null : JSON.stringify({ version: 1, value });
  const window = { location: { href: `https://2048tables.online/${search}` }, navigator: { languages },
    localStorage: { getItem(key) {
      if (storageError) throw new Error('Storage blocked');
      if (key === '2048tables:rendered-appearance') return rendered;
      return storedValue;
    }, setItem(_key, next) {
      if (storageError) throw new Error('Storage blocked');
      storedValue = next;
    } },
  };
  runInNewContext(script, { window, document, URL, JSON, Date });
  return { attributes, backend: window.__APP_BACKEND_ORIGIN__, storedValue };
}

test('saved appearance is applied before Vue renders', () => {
  assert.deepEqual(boot({ dark_mode: true, language: 'zh' }).attributes,
    { lang: 'zh-CN', 'data-theme': 'dark' });
  assert.deepEqual(boot({ dark_mode: false, language: 'en' }).attributes,
    { lang: 'en' });
});

test('main and tables restore the rendered server default before modules load', () => {
  const tables = readFileSync(new URL('../tables/index.html', import.meta.url), 'utf8');
  const tablesScript = [...tables.matchAll(/<script>([\s\S]*?)<\/script>/g)][0]?.[1];
  assert.ok(tablesScript);
  for (const script of [startupScript, tablesScript]) {
    for (const dark of [true, false]) {
      const cached = JSON.stringify({ version: 1, dark_mode: dark });
      const result = boot({ language: 'zh' }, '', false, ['en-US'], cached, script);
      assert.equal(result.attributes['data-theme'] === 'dark', dark);
      assert.equal(JSON.parse(result.storedValue).value.dark_mode, undefined);
      const explicit = boot({ language: 'zh', dark_mode: !dark }, '', false, ['en-US'], cached, script);
      assert.equal(explicit.attributes['data-theme'] === 'dark', !dark);
    }
    const malformed = boot(null, '', false, ['en-US'], 'invalid', script);
    assert.equal(malformed.attributes['data-theme'], undefined);
    assert.equal(boot(null, '', true, ['en-US'], null, script).attributes['data-theme'], undefined);
  }
});

test('rendered appearance uses an isolated cache and ignores storage failures', () => {
  const original = globalThis.window;
  const values = new Map();
  let writes = 0;
  try {
    globalThis.window = { localStorage: {
      getItem: key => values.get(key),
      setItem: (key, value) => { writes += 1; values.set(key, value); },
    } };
    rememberRenderedAppearance(true);
    rememberRenderedAppearance(true);
    assert.equal(writes, 1);
    assert.deepEqual([...values.keys()], ['2048tables:rendered-appearance']);
    assert.equal(JSON.parse(values.get('2048tables:rendered-appearance')).dark_mode, true);
    Object.defineProperty(globalThis.window, 'localStorage', { get() { throw new Error('blocked'); } });
    assert.doesNotThrow(() => rememberRenderedAppearance(false));
  } finally { globalThis.window = original; }
});

test('explicit startup theme wins without changing saved language', () => {
  assert.deepEqual(boot({ dark_mode: true, language: 'zh' }, '?startup_theme=light').attributes,
    { lang: 'zh-CN' });
  assert.deepEqual(boot({ dark_mode: false, language: 'zh' }, '?startup_theme=dark').attributes,
    { lang: 'zh-CN', 'data-theme': 'dark' });
});

test('missing or blocked storage keeps defaults and backend origin', () => {
  assert.deepEqual(boot(null).attributes, { lang: 'en' });
  assert.deepEqual(boot(null, '', true, ['zh-CN']).attributes, { lang: 'zh-CN' });
  assert.equal(boot(null, '?backend_port=8766').backend, 'https://2048tables.online:8766');
});

test('new browsers detect and persist the first supported browser language', () => {
  const chinese = boot(null, '', false, ['ja-JP', 'zh-TW', 'en-US']);
  assert.equal(chinese.attributes.lang, 'zh-CN');
  assert.equal(JSON.parse(chinese.storedValue).value.language, 'zh');
  assert.equal(boot(null, '', false, ['fr-FR']).attributes.lang, 'en');
});

test('Vue i18n starts in the language selected by the HTML boot script', () => {
  const source = readFileSync(new URL('../src/app/i18n.js', import.meta.url), 'utf8')
    .replace(/^import .*;\r?\n/gm, '')
    .replace('export default i18n;', 'globalThis.i18n = i18n;');
  for (const [lang, expected] of [['zh-CN', 'zh'], ['en', 'en']]) {
    const context = { document: { documentElement: { lang } },
      createI18n: config => config, en: {}, zh: {} };
    runInNewContext(source, context);
    assert.equal(context.i18n.locale, expected);
  }
});
