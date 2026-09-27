export const VTH_TILE_VALUES = Object.freeze(Array.from({ length: 16 }, (_, index) => 2 ** (index + 1)));
export const VTH_STYLE_KEYS = Object.freeze(['--tile-color', '--tile-background', '--tile-shadow-color', '--tile-outline-color']);
const COLOR = /^#[\da-f]{6}(?:[\da-f]{2}|\s*\/\s*(?:0(?:\.\d+)?|1(?:\.0+)?|\d{1,3}%))?$/i;

export function normalizeVthTheme(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('invalid_theme_file');
  const modes = Object.keys(raw);
  if (!modes.length || modes.some(mode => !['light', 'dark'].includes(mode))) throw new Error('invalid_theme_file');
  const result = {};
  for (const mode of modes) {
    const entries = raw[mode];
    if (!entries || typeof entries !== 'object' || Array.isArray(entries)) throw new Error('invalid_theme_file');
    result[mode] = {};
    for (const tile of [...VTH_TILE_VALUES.map(String), 'Super']) {
      if (!(tile in entries)) continue;
      const style = entries[tile];
      if (!style || typeof style !== 'object' || VTH_STYLE_KEYS.some(key => !COLOR.test(String(style[key] || '').trim()))) throw new Error('invalid_theme_color');
      result[mode][tile] = Object.fromEntries(VTH_STYLE_KEYS.map(key => [key, String(style[key]).trim()]));
    }
    if (VTH_TILE_VALUES.some(tile => !result[mode][tile])) throw new Error('theme_tiles_missing');
  }
  return result;
}
