const FALLBACK_VARIANT_PATTERNS = ['2x4', '3x3', '3x4', '3x4free9', '3x3free8'];

export const PATTERN_CATEGORY_ORDER = ['free', 'space10', 'space12', 'others', 'variant'];

// Match the categories in docs_and_configs/patterns_config.json.
const TEN_SPACE_PATTERNS = new Set(['L3', 'L3f', 'L3t', '442', '442t', 't']);
const TWELVE_SPACE_PATTERNS = new Set(['LL', 'LL2', '4431', '444', '444s', '4432f', '4442ff']);

export const getPatternCategory = (patternLike) => {
  const pattern = normalizePatternName(patternLike);
  if (isVariantPattern(pattern)) return 'variant';
  if (/^free\d+$/u.test(pattern)) return 'free';
  if (TEN_SPACE_PATTERNS.has(pattern)) return 'space10';
  if (TWELVE_SPACE_PATTERNS.has(pattern)) return 'space12';
  return 'others';
};

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
