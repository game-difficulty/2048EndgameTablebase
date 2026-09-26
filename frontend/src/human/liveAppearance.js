const TILE_VALUES = Array.from({ length: 27 }, (_, index) => 2 ** (index + 1));
const HEX = /^#[0-9a-f]{6}$/i;

const color = (styles, name, fallback) => {
  const value = styles.getPropertyValue(name).trim();
  return HEX.test(value) ? value.toLowerCase() : fallback;
};

export function captureLiveAppearance(root = document.documentElement) {
  const styles = getComputedStyle(root);
  const tiles = {};
  for (const value of TILE_VALUES) {
    tiles[value] = {
      background: color(styles, `--color-tile-${value}`, '#000000'),
      color: color(styles, `--color-text-${value}`, '#f9f6f2'),
    };
  }
  return {
    version: 1,
    empty: {
      background: color(styles, '--empty-tile', '#716e69'),
      color: color(styles, '--muted', '#b5b1ac'),
    },
    tiles,
  };
}

export function liveAppearanceTileStyle(appearance, value) {
  const entry = value ? appearance?.tiles?.[value] : appearance?.empty;
  return entry?.background && entry?.color
    ? { background: entry.background, color: entry.color }
    : null;
}

