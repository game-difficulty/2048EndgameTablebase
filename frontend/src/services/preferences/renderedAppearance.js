const STORAGE_KEY = '2048tables:rendered-appearance';

// Cache presentation only: a server default must not become an account preference.
export function rememberRenderedAppearance(darkMode) {
  try {
    const storage = window.localStorage;
    const next = JSON.stringify({ version: 1, dark_mode: !!darkMode });
    if (storage.getItem(STORAGE_KEY) !== next) storage.setItem(STORAGE_KEY, next);
  } catch (_) {
    // Storage may be unavailable; rendering must still succeed.
  }
}
