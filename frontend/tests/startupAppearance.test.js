import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const startupScript = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0]?.[1];
assert.ok(startupScript);

function boot(value, search = '', storageError = false, languages = ['en-US']) {
  const attributes = { lang: 'en' };
  const document = { documentElement: {
    get lang() { return attributes.lang; },
    set lang(value) { attributes.lang = value; },
    setAttribute(key, value) { attributes[key] = value; },
    removeAttribute(key) { delete attributes[key]; },
  } };
  let storedValue = value == null ? null : JSON.stringify({ version: 1, value });
  const window = { location: { href: `https://2048tables.online/${search}` }, navigator: { languages },
    localStorage: { getItem() {
      if (storageError) throw new Error('Storage blocked');
      return storedValue;
    }, setItem(_key, next) {
      if (storageError) throw new Error('Storage blocked');
      storedValue = next;
    } },
  };
  runInNewContext(startupScript, { window, document, URL, JSON, Date });
  return { attributes, backend: window.__APP_BACKEND_ORIGIN__, storedValue };
}

test('saved appearance is applied before Vue renders', () => {
  assert.deepEqual(boot({ dark_mode: true, language: 'zh' }).attributes,
    { lang: 'zh-CN', 'data-theme': 'dark' });
  assert.deepEqual(boot({ dark_mode: false, language: 'en' }).attributes,
    { lang: 'en' });
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
