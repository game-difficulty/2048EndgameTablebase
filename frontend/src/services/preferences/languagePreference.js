const SUPPORTED_LANGUAGES = new Set(['zh', 'en']);

export function detectPreferredLanguage(browser = globalThis.navigator) {
  const candidates = Array.isArray(browser?.languages) && browser.languages.length
    ? browser.languages
    : [browser?.language];
  for (const candidate of candidates) {
    const language = String(candidate || '').trim().toLowerCase();
    if (language.startsWith('zh')) return 'zh';
    if (language.startsWith('en')) return 'en';
  }
  return 'en';
}

export function ensureStoredLanguage(store, browser = globalThis.navigator) {
  const current = store.read() || {};
  if (SUPPORTED_LANGUAGES.has(current.language)) return current.language;
  const language = detectPreferredLanguage(browser);
  store.write({ ...current, language });
  return language;
}
