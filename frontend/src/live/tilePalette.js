// Shared by the live board, milestones and replay-history badges.
const backgrounds = {
  2: '#eee4da', 4: '#ede0c8', 8: '#f2b179', 16: '#f59563',
  32: '#f67c5f', 64: '#f65e3b', 128: '#edcf72', 256: '#edcc61',
  512: '#edc850', 1024: '#edc53f', 2048: '#edc22e',
  4096: '#9100cf', 8192: '#590080', 16384: '#36004d',
};

let userPalette = null;

export function setLiveTilePalette(colors) {
  userPalette = Array.isArray(colors) && colors.length > 0 ? colors.slice(0, 36).map(item =>
    typeof item === 'string' ? { background: item, color: null } : item
  ) : null;
}

export function liveTileColors(value) {
  const index = Math.log2(value) - 1;
  const custom = userPalette?.[index];
  const background = custom?.background || backgrounds[value] || '#000000';
  return {
    background,
    color: custom?.color || (value <= 4 ? '#776e65' : '#f9f6f2'),
  };
}

export function liveBoardPalette() {
  return Object.fromEntries(Array.from({ length: 31 }, (_, index) => {
    const value = 2 ** (index + 1);
    const colors = liveTileColors(value);
    return [[`--color-tile-${value}`, colors.background], [`--color-text-${value}`, colors.color]];
  }).flat());
}
