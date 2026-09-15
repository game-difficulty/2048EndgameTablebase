// Shared by the live board, milestones and replay-history badges.
const backgrounds = {
  2: '#eee4da', 4: '#ede0c8', 8: '#f2b179', 16: '#f59563',
  32: '#f67c5f', 64: '#f65e3b', 128: '#edcf72', 256: '#edcc61',
  512: '#edc850', 1024: '#edc53f', 2048: '#edc22e',
  4096: '#9100cf', 8192: '#590080', 16384: '#36004d',
};

export function liveTileColors(value) {
  return {
    background: backgrounds[value] || '#000000',
    color: value <= 4 ? '#776e65' : '#f9f6f2',
  };
}

export function liveBoardPalette() {
  return Object.fromEntries(Array.from({ length: 31 }, (_, index) => {
    const value = 2 ** (index + 1);
    const colors = liveTileColors(value);
    return [[`--color-tile-${value}`, colors.background], [`--color-text-${value}`, colors.color]];
  }).flat());
}
