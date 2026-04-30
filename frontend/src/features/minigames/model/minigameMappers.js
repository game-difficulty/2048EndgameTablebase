import { getMinigameAssetUrl } from '../../../services/runtime/backendUrl';

const FALLBACK_TILE = 131072;

export const formatTileNumber = (tileValue) => {
  if (!tileValue) return '';
  return tileValue >= 1024 ? `${Math.floor(tileValue / 1024)}k` : String(tileValue);
};

export const formatBoardTileNumber = (tileValue) => {
  if (!tileValue) return '';
  return String(tileValue);
};

export const exponentToTileValue = (exponent) => {
  if (!exponent || exponent <= 0) return 0;
  return 2 ** exponent;
};

export const getTrophyMeta = (level) => {
  const normalized = Number(level || 0);
  if (normalized >= 4) return { label: 'Grand', tone: 'grand' };
  if (normalized === 3) return { label: 'Gold', tone: 'gold' };
  if (normalized === 2) return { label: 'Silver', tone: 'silver' };
  if (normalized === 1) return { label: 'Bronze', tone: 'bronze' };
  return { label: 'None', tone: 'none' };
};

export const getTrophyVisualMeta = (level) => {
  const normalized = Number(level || 0);
  if (normalized <= 0) return null;
  if (normalized >= 4) {
    return {
      backgroundSrc: getMinigameAssetUrl('trophybg.png'),
      trophySrc: getMinigameAssetUrl('grand.png'),
      trophyAlt: 'Grand trophy',
    };
  }
  if (normalized >= 3) {
    return {
      backgroundSrc: getMinigameAssetUrl('trophybg.png'),
      trophySrc: getMinigameAssetUrl('gold.png'),
      trophyAlt: 'Gold trophy',
    };
  }
  if (normalized === 2) {
    return {
      backgroundSrc: getMinigameAssetUrl('trophybg.png'),
      trophySrc: getMinigameAssetUrl('silver.png'),
      trophyAlt: 'Silver trophy',
    };
  }
  return {
    backgroundSrc: getMinigameAssetUrl('trophybg.png'),
    trophySrc: getMinigameAssetUrl('bronze.png'),
    trophyAlt: 'Bronze trophy',
  };
};

export const buildBoardCells = (board, shape, view = {}) => {
  const rows = Number(shape?.rows || 4);
  const cols = Number(shape?.cols || 4);
  const hiddenMask = Array.isArray(view.hiddenMask) ? view.hiddenMask : [];
  const blockedMask = Array.isArray(view.blockedMask) ? view.blockedMask : [];
  const smallLabels = Array.isArray(view.smallLabels) ? view.smallLabels : [];
  const tileOverlays = view.tileOverlays || {};
  const tileTextOverride = view.tileTextOverride || {};
  const tileStyleVariant = view.tileStyleVariant || {};
  const coverSprites = view.coverSprites || {};

  return Array.from({ length: rows * cols }, (_, index) => {
    const rawValue = Number(board?.[index] || 0);
    const tileValue = exponentToTileValue(rawValue);
    const blocked = Boolean(blockedMask[index]);
    const hidden = Boolean(hiddenMask[index]);
    const overrideKey = String(index);
    const displayOverride = tileTextOverride[overrideKey] ?? tileTextOverride[index];
    const overlay = tileOverlays[overrideKey] ?? tileOverlays[index] ?? null;
    let variant = tileStyleVariant[overrideKey] ?? tileStyleVariant[index] ?? null;
    const coverSpriteEntry = coverSprites[overrideKey] ?? coverSprites[index] ?? [];
    const coverSpriteList = Array.isArray(coverSpriteEntry)
      ? coverSpriteEntry.map(String)
      : coverSpriteEntry
        ? [String(coverSpriteEntry)]
        : [];

    if (hidden && rawValue > 0 && !coverSpriteList.includes('tilebg.png')) {
      coverSpriteList.unshift('tilebg.png');
      if (!variant) {
        variant = { kind: 'mystery-hidden' };
      }
    } else if (!variant && coverSpriteList.includes('tilebg.png')) {
      variant = { kind: 'tilebg' };
    } else if (!variant && coverSpriteList.includes('portal.png')) {
      variant = { kind: 'portal' };
    } else if (!variant && (coverSpriteList.includes('bomb.png') || coverSpriteList.includes('giftbox.png'))) {
      variant = { kind: 'special-object' };
    }

    let displayText = '';
    if (displayOverride != null) {
      displayText = String(displayOverride);
    } else if (!blocked && rawValue > 0) {
      displayText = hidden ? '?' : formatBoardTileNumber(tileValue);
    }

    return {
      index,
      row: Math.floor(index / cols),
      col: index % cols,
      rawValue,
      tileValue,
      blocked,
      hidden,
      smallLabel: smallLabels[index] || '',
      overlay,
      variant,
      coverSprites: coverSpriteList,
      displayText,
      isHole: blocked && coverSpriteList.length === 0,
    };
  });
};

export const getBoardCellStyle = (cell) => {
  const variant = cell.variant && typeof cell.variant === 'object' ? cell.variant : null;

  if (cell.isHole) {
    return {
      background: 'transparent',
      color: 'transparent',
      borderColor: 'transparent',
      boxShadow: 'none',
    };
  }

  if (variant?.kind === 'mystery-hidden' || variant?.kind === 'tilebg') {
    return {
      background: 'transparent',
      color: '#fff',
      borderColor: 'transparent',
      boxShadow: 'none',
    };
  }

  if (variant?.kind === 'portal' || variant?.kind === 'special-object') {
    return {
      background: 'var(--color-empty)',
      color: 'transparent',
      borderColor: 'rgba(255,255,255,0.05)',
      boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.06)',
    };
  }

  if (cell.blocked) {
    return {
      background: 'linear-gradient(145deg, rgba(15,23,42,0.45), rgba(15,23,42,0.28))',
      color: 'transparent',
      borderColor: 'rgba(148,163,184,0.18)',
      opacity: '0.75',
      boxShadow: 'none',
    };
  }

  if (variant?.kind === 'frozen' && Number(variant.exponent || 0) > 0) {
    const frozenValue = 2 ** Number(variant.exponent);
    const safeValue = Math.min(frozenValue || 0, FALLBACK_TILE) || 2;
    return {
      background: `color-mix(in srgb, var(--color-tile-${safeValue}) 72%, rgba(226,232,240,0.45))`,
      color: `var(--color-text-${safeValue})`,
      borderColor: 'rgba(191, 219, 254, 0.45)',
      boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.2), 0 0 0 1px rgba(191,219,254,0.1)',
    };
  }

  if (!cell.rawValue || cell.rawValue <= 0) {
    return {
      background: 'var(--color-empty)',
      color: 'var(--text-secondary)',
      borderColor: 'rgba(255,255,255,0.05)',
      boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.06)',
    };
  }

  const safeValue = Math.min(cell.tileValue || 0, FALLBACK_TILE);
  const tileCssValue = safeValue || 2;
  return {
    background: `var(--color-tile-${tileCssValue})`,
    color: `var(--color-text-${tileCssValue})`,
    borderColor: 'rgba(255,255,255,0.08)',
    boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.08)',
  };
};
