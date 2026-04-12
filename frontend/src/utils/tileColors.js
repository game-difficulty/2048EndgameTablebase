const LIGHT_TEXT_COLOR = '#776e65';
const DARK_TEXT_COLOR = '#f9f6f2';
const REFERENCE_TILE_COLOR = '#eedab3';

const hexToRgb = (hexColor) => {
  const normalized = String(hexColor || '').replace('#', '');
  if (normalized.length !== 6) return [0, 0, 0];
  return [0, 2, 4].map(offset => parseInt(normalized.slice(offset, offset + 2), 16));
};

const getLuminance = (hexColor) => {
  const [r, g, b] = hexToRgb(hexColor);
  return 0.299 * r + 0.587 * g + 0.114 * b;
};

const isDarkerThan = (colorHex, referenceHex = REFERENCE_TILE_COLOR) =>
  getLuminance(referenceHex) > getLuminance(colorHex);

const fillMidFalses = (flags) => {
  if (flags.length < 3) return [...flags];
  const result = [...flags];
  for (let index = 1; index < flags.length - 1; index += 1) {
    if (!flags[index] && flags[index - 1] && flags[index + 1]) {
      result[index] = true;
    }
  }
  return result;
};

export const applyTileColors = (colors) => {
  if (!Array.isArray(colors) || colors.length === 0) return;

  const fontFlags = fillMidFalses(colors.map(color => isDarkerThan(color)));
  colors.forEach((color, index) => {
    const tileValue = 2 ** (index + 1);
    document.documentElement.style.setProperty(`--color-tile-${tileValue}`, color);
    document.documentElement.style.setProperty(
      `--color-text-${tileValue}`,
      fontFlags[index] ? DARK_TEXT_COLOR : LIGHT_TEXT_COLOR
    );
  });
};
