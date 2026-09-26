const COOKIE_NAME = '2048tables-tile-palette';

// Main-site settings historically initialize all 36 entries to black before
// the theme catalog arrives. This is an uninitialized palette, not a theme.
export function isPlaceholderTilePalette(colors) {
  return Array.isArray(colors) && colors.length > 0 && colors.every(item => {
    const background = typeof item === 'string' ? item : item?.background;
    return typeof background === 'string' && /^(#000|#000000|black|rgb\(0,0,0\))$/i.test(background.replace(/\s/g, ''));
  });
}

export function writeSharedTilePalette(colors) {
  if (!Array.isArray(colors) || colors.length === 0 || isPlaceholderTilePalette(colors) || typeof document === 'undefined') return;
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
  try {
    const prefix = `${COOKIE_NAME}=`;
    const value = document.cookie.split('; ').find(item => item.startsWith(prefix))?.slice(prefix.length);
    if (!value) return null;
    const colors = JSON.parse(decodeURIComponent(value));
    return Array.isArray(colors) && colors.length > 0 && !isPlaceholderTilePalette(colors) ? colors : null;
  } catch (_error) {
    return null;
  }
}
