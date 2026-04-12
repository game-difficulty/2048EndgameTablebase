const FALLBACK_VARIANT_PATTERNS = ['2x4', '3x3', '3x4'];

export const normalizePatternName = (patternLike) => {
  const raw = String(patternLike || '').trim();
  if (!raw) return '';
  if (/_\d+$/u.test(raw)) {
    return raw.replace(/_\d+$/u, '');
  }
  return raw;
};

export const getCategoryPatterns = (categories, categoryName) => {
  const patterns = categories?.[categoryName];
  return Array.isArray(patterns) ? patterns.map((pattern) => String(pattern)) : [];
};

export const isVariantPattern = (patternLike, categories = {}) => {
  const pattern = normalizePatternName(patternLike);
  if (!pattern) return false;

  const variantPatterns = new Set([
    ...FALLBACK_VARIANT_PATTERNS,
    ...getCategoryPatterns(categories, 'variant'),
  ]);

  return variantPatterns.has(pattern);
};
