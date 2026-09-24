// Shared by the main and Play boards, including their practice and replay views.
export const getTileLabelStyle = (tile) => {
  const len = String(tile.value).length;
  const smallTileScale = tile.value >= 2 && tile.value <= 64 ? 1.2 : 1;
  let fontSize = 'var(--tile-label-small, 2.5rem)';

  if (len > 4) {
    fontSize = 'var(--tile-label-large, 1.5rem)';
  } else if (len > 3) {
    fontSize = 'var(--tile-label-medium, 2rem)';
  }

  return {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: `calc(${fontSize} * var(--tile-font-scale, 1) * ${smallTileScale})`,
    lineHeight: 1,
  };
};
