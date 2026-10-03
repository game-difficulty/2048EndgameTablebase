import { isVariantPattern, normalizePatternName } from './patternCategories.js';

export function patternFromReplayFilename(value) {
  const name = String(value || '').split(/[\\/]/u).pop() || '';
  return name.match(/^([A-Za-z0-9]+(?:_[A-Za-z][A-Za-z0-9]*)*_(?:sum-)?\d+)(?=[_.]|$)/u)?.[1] || '';
}

// RPL has no variant header. Resolve once for both movement and presentation.
export function resolveReplayVariant(metadata = {}, categories = {}) {
  const pattern = String(metadata.pattern || '').trim()
    || patternFromReplayFilename(metadata.filename)
    || patternFromReplayFilename(metadata.source);
  const basePattern = normalizePatternName(pattern);
  const namedVariant = isVariantPattern(basePattern, categories);
  const knownPattern = namedVariant || Object.values(categories).some(
    names => Array.isArray(names) && names.includes(basePattern),
  ) || /^free\d+$/u.test(basePattern);
  const explicit = typeof metadata.useVariant === 'boolean' ? metadata.useVariant : null;
  // Repair persisted flags from older clients when the formation is known.
  const useVariant = knownPattern ? namedVariant : (explicit ?? false);
  return {
    pattern,
    useVariant,
    conflict: knownPattern && explicit !== null && explicit !== useVariant,
  };
}
