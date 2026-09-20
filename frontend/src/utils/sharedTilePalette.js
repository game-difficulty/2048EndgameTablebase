const COOKIE_NAME = '2048tables-tile-palette';

export function writeSharedTilePalette(colors) {
  if (!Array.isArray(colors) || colors.length === 0 || typeof document === 'undefined') return;
  try {
    const value = encodeURIComponent(JSON.stringify(colors.slice(0, 36)));
    const domain = location.hostname.endsWith('2048tables.online') ? '; domain=.2048tables.online' : '';
    document.cookie = `${COOKIE_NAME}=${value}; path=/; max-age=31536000; samesite=lax${domain}`;
  } catch (_error) {
    // A storage restriction must not affect the main UI.
  }
}

export function readSharedTilePalette() {
  if (typeof document === 'undefined') return null;
  const prefix = `${COOKIE_NAME}=`;
  const value = document.cookie.split('; ').find(item => item.startsWith(prefix))?.slice(prefix.length);
  if (!value) return null;
  try {
    const colors = JSON.parse(decodeURIComponent(value));
    return Array.isArray(colors) && colors.length > 0 ? colors : null;
  } catch (_error) {
    return null;
  }
}
