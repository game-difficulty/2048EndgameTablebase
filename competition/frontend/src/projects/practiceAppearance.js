import themes from '../../../../docs_and_configs/themes.json' with { type: 'json' };

const LIGHT_TEXT = '#776e65';
const DARK_TEXT = '#f9f6f2';
const TILE_VALUES = Array.from({ length: 16 }, (_, index) => 2 ** (index + 1));

function luminance(color) {
  const hex = String(color || '').replace('#', '');
  if (!/^[\da-f]{6}$/i.test(hex)) return 0;
  const [red, green, blue] = [0, 2, 4].map(offset => Number.parseInt(hex.slice(offset, offset + 2), 16));
  return red * 0.299 + green * 0.587 + blue * 0.114;
}

function paletteStyles(colors) {
  const initial = colors.map(color => luminance(color) < luminance('#eedab3'));
  const darkText = initial.map((flag, index) => flag || (
    index > 0 && index < initial.length - 1 && initial[index - 1] && initial[index + 1]
  ));
  return Object.fromEntries(colors.map((backgroundColor, index) => [TILE_VALUES[index], {
    backgroundColor,
    color: darkText[index] ? DARK_TEXT : LIGHT_TEXT,
  }]));
}

export function resolvePracticeAppearance(preferences, mode = 'light') {
  if (!preferences) return { tileStyles: {}, fontScale: 1 };
  const factor = Number(preferences.font_size_factor);
  const fontScale = Number.isFinite(factor) && factor >= 50 && factor <= 150 ? factor / 100 : 1;
  const saved = preferences.saved_theme?.[mode] || preferences.saved_theme?.[mode === 'dark' ? 'light' : 'dark'];
  if (saved) {
    return {
      fontScale,
      tileStyles: Object.fromEntries(TILE_VALUES.flatMap(value => {
        const tile = saved[value];
        return tile ? [[value, {
          backgroundColor: tile['--tile-background'],
          color: tile['--tile-color'],
          boxShadow: `0 0 10px ${tile['--tile-shadow-color']}, inset 0 0 0 1px ${tile['--tile-outline-color']}`,
        }]] : [];
      })),
    };
  }
  const custom = preferences.use_custom_theme && Array.isArray(preferences.custom_colors)
    ? preferences.custom_colors : null;
  const colors = custom?.length ? custom : themes[preferences.theme] || themes.Default;
  return { fontScale, tileStyles: paletteStyles(colors.slice(0, TILE_VALUES.length)) };
}

export function tileLabelSize(value, columns, fontScale = 1) {
  // Main BaseBoard: 48/40/32/24px at 600px and four columns.
  const digits = String(value).length;
  const reference = value >= 2 && value <= 64 ? 48 : digits > 4 ? 24 : digits > 3 ? 32 : 40;
  const percent = reference * 100 / 600 * 4 / Math.max(1, Number(columns) || 4) * fontScale;
  return `${percent.toFixed(4)}cqw`;
}
