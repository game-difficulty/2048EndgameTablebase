import test from 'node:test';
import assert from 'node:assert/strict';

import { detectPreferredLanguage, ensureStoredLanguage } from '../src/services/preferences/languagePreference.js';

test('detects the first supported browser language in preference order', () => {
  assert.equal(detectPreferredLanguage({ languages: ['zh-CN', 'en-US'] }), 'zh');
  assert.equal(detectPreferredLanguage({ languages: ['en-GB', 'zh-CN'] }), 'en');
  assert.equal(detectPreferredLanguage({ languages: ['ja-JP', 'zh-TW', 'en-US'] }), 'zh');
  assert.equal(detectPreferredLanguage({ languages: ['ja-JP'] }), 'en');
  assert.equal(detectPreferredLanguage({ language: 'zh-HK' }), 'zh');
});

test('stores a detected language once and preserves an explicit preference', () => {
  let value = { theme: 'Default' };
  const store = {
    read: () => structuredClone(value),
    write: next => { value = structuredClone(next); },
  };
  assert.equal(ensureStoredLanguage(store, { languages: ['zh-CN'] }), 'zh');
  assert.deepEqual(value, { theme: 'Default', language: 'zh' });
  assert.equal(ensureStoredLanguage(store, { languages: ['en-US'] }), 'zh');
  assert.equal(value.language, 'zh');
});
