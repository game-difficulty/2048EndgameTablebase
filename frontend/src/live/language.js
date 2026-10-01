const key = '2048-live-language';
export function liveLanguage() {
  try { const saved = localStorage.getItem(key); if (saved === 'zh' || saved === 'en') return saved; } catch {}
  return navigator.language.startsWith('zh') ? 'zh' : 'en';
}
export function saveLiveLanguage(value) {
  try { localStorage.setItem(key, value); } catch {}
}
