import { isPlaceholderTilePalette } from '../utils/sharedTilePalette.js';
// Shared by the live board, milestones and replay-history badges.
const backgrounds = {
  2: '#eee4da', 4: '#ede0c8', 8: '#f2b179', 16: '#f59563',
  32: '#f67c5f', 64: '#f65e3b', 128: '#edcf72', 256: '#edcc61',
  512: '#edc850', 1024: '#edc53f', 2048: '#edc22e',
  4096: '#8056b3', 8192: '#6652a3', 16384: '#4c5498',
  32768: '#287c8e', 65536: '#247a60', 131072: '#a74b72',
};

const extendedColors = ['#536cad', '#287c8e', '#247a60', '#9a6630', '#a74b72', '#8056b3'];
const validColor = value => {
  if (typeof value !== 'string' || !value.trim()) return false;
  // Theme-independent, opaque colors only; unresolved CSS variables are not a palette.
  if (/^#[0-9a-f]{3}$|^#[0-9a-f]{6}$/i.test(value.trim())) return true;
  return /^(rgb|hsl)\(/i.test(value.trim()) && typeof CSS !== 'undefined' && typeof CSS.supports === 'function' && CSS.supports('color', value);
};
let userPalette = null;

export function setLiveTilePalette(colors) {
  userPalette = Array.isArray(colors) && colors.length > 0 && !isPlaceholderTilePalette(colors) ? colors.slice(0, 36).map(item =>
    typeof item === 'string' ? { background: item, color: null } : item
  ) : null;
}

export function liveTileColors(value) {
  const index = Math.log2(value) - 1;
  const custom = userPalette?.[index];
  const fallback = backgrounds[value] || extendedColors[Math.max(0, Number.isInteger(index) ? index - 17 : 0) % extendedColors.length];
  const background = validColor(custom?.background) ? custom.background : fallback;
  return {
    background,
    color: validColor(custom?.background) && validColor(custom?.color) ? custom.color : (value <= 4 ? '#776e65' : '#f9f6f2'),
  };
}

// Empty cells belong to the Live surface rather than the streamer's Play theme.
// Keeping this as CSS variables makes the same rule follow Live light/dark colors.
export function liveEmptyTileColors() {
  return {
    background: 'var(--color-empty)',
    color: 'var(--text-secondary)',
  };
}

export function liveBoardPalette() {
  return Object.fromEntries(Array.from({ length: 31 }, (_, index) => {
    const value = 2 ** (index + 1);
    const colors = liveTileColors(value);
    return [[`--color-tile-${value}`, colors.background], [`--color-text-${value}`, colors.color]];
  }).flat());
}
