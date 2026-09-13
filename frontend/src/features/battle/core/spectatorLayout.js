export function spectatorLayout(count) {
  const size = Math.max(1, Math.min(8, Number(count) || 1));
  const columns = size <= 4 ? size : size <= 6 ? 3 : 4;
  const span = 24 / columns;
  const remainder = size % columns;
  return {
    maxWidth: `${columns * (columns <= 2 ? 480 : columns === 3 ? 400 : 320) + (columns - 1) * 12}px`,
    items: Array.from({ length: size }, (_, index) => ({
      gridColumn: remainder && index === size - remainder
        ? `${1 + (columns - remainder) * span / 2} / span ${span}` : `span ${span}`,
    })),
  };
}
