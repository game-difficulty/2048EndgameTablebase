const DEFAULT_MAX_ENTRIES = 256;
const DEFAULT_TTL_MS = 5 * 60 * 1000;

function normalizeKeyPart(value) {
  return String(value ?? '').trim();
}

function normalizeBoardHex(value) {
  return normalizeKeyPart(value).replace(/^0x/i, '').toLowerCase();
}

export function createTablebaseResultKey({ catalogVersion, fullPattern, boardHex } = {}) {
  const version = normalizeKeyPart(catalogVersion);
  const pattern = normalizeKeyPart(fullPattern);
  const board = normalizeBoardHex(boardHex);
  if (!version || !pattern || !board) {
    throw new TypeError('catalogVersion, fullPattern, and boardHex are required');
  }
  return JSON.stringify([version, pattern, board]);
}

function normalizeResultValue(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (value == null || value === '') {
    return null;
  }
  return String(value);
}

export function normalizeTablebaseResult(payload = {}) {
  const source = payload && typeof payload === 'object' ? payload : {};
  const rawResults = source.results && typeof source.results === 'object'
    ? source.results
    : {};
  const results = {};

  for (const [rawDirection, rawValue] of Object.entries(rawResults)) {
    const direction = String(rawDirection || '').trim().toLowerCase();
    if (direction) {
      results[direction] = normalizeResultValue(rawValue);
    }
  }

  const hasNumericResult = Object.values(results).some((value) => typeof value === 'number');
  const explicitFound = source.found ?? source.table_found;
  const found = typeof explicitFound === 'boolean' ? explicitFound : hasNumericResult;

  return {
    found,
    dtype: String(source.dtype || '?'),
    results: found ? results : {},
  };
}

function cloneResult(result) {
  return {
    found: result.found,
    dtype: result.dtype,
    results: { ...result.results },
  };
}

export class TablebaseResultCache {
  constructor({
    maxEntries = DEFAULT_MAX_ENTRIES,
    ttlMs = DEFAULT_TTL_MS,
    now = () => Date.now(),
  } = {}) {
    if (!Number.isInteger(maxEntries) || maxEntries < 1) {
      throw new TypeError('maxEntries must be a positive integer');
    }
    if (!Number.isFinite(ttlMs) || ttlMs <= 0) {
      throw new TypeError('ttlMs must be positive');
    }
    this.maxEntries = maxEntries;
    this.ttlMs = ttlMs;
    this.now = now;
    this.entries = new Map();
    this.activeCatalogVersion = '';
    this.activeFullPattern = '';
  }

  prepareScope(keyParts = {}) {
    const catalogVersion = normalizeKeyPart(keyParts.catalogVersion);
    const fullPattern = normalizeKeyPart(keyParts.fullPattern);
    if (
      (this.activeCatalogVersion && this.activeCatalogVersion !== catalogVersion)
      || (this.activeFullPattern && this.activeFullPattern !== fullPattern)
    ) {
      this.entries.clear();
    }
    this.activeCatalogVersion = catalogVersion;
    this.activeFullPattern = fullPattern;
  }

  get(keyParts) {
    this.prepareScope(keyParts);
    const key = createTablebaseResultKey(keyParts);
    const entry = this.entries.get(key);
    if (!entry) {
      return null;
    }
    if (this.now() - entry.storedAt >= this.ttlMs) {
      this.entries.delete(key);
      return null;
    }

    this.entries.delete(key);
    this.entries.set(key, entry);
    return cloneResult(entry.value);
  }

  set(keyParts, payload) {
    this.prepareScope(keyParts);
    const key = createTablebaseResultKey(keyParts);
    const value = normalizeTablebaseResult(payload);
    this.entries.delete(key);
    this.entries.set(key, { storedAt: this.now(), value });

    while (this.entries.size > this.maxEntries) {
      const oldestKey = this.entries.keys().next().value;
      this.entries.delete(oldestKey);
    }
    return cloneResult(value);
  }

  clear() {
    this.entries.clear();
    this.activeCatalogVersion = '';
    this.activeFullPattern = '';
  }
}

export const tablebaseResultCache = new TablebaseResultCache();

export const getCachedTablebaseResult = (keyParts) => tablebaseResultCache.get(keyParts);
export const setCachedTablebaseResult = (keyParts, payload) => (
  tablebaseResultCache.set(keyParts, payload)
);
export const clearTablebaseResultCache = () => tablebaseResultCache.clear();

let authEventTarget = null;
const clearOnAuthChanged = () => tablebaseResultCache.clear();

export function installTablebaseResultCacheAuthListener(eventTarget = globalThis.window) {
  if (!eventTarget?.addEventListener || authEventTarget === eventTarget) {
    return;
  }
  authEventTarget?.removeEventListener?.('auth-changed', clearOnAuthChanged);
  authEventTarget = eventTarget;
  authEventTarget.addEventListener('auth-changed', clearOnAuthChanged);
}

installTablebaseResultCacheAuthListener();

export const TABLEBASE_RESULT_CACHE_MAX_ENTRIES = DEFAULT_MAX_ENTRIES;
export const TABLEBASE_RESULT_CACHE_TTL_MS = DEFAULT_TTL_MS;
