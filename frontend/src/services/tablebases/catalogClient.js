import { getBackendUrl } from '../runtime/backendUrl.js';
import { tablebaseResultCache } from './tablebaseResultCache.js';

const VARIANT_PATTERNS = new Set(['2x4', '3x3', '3x4', '3x4free9', '3x3free8']);
const CATEGORY_ORDER = ['4x4', 'variant'];
let currentCatalogVersion = '';

function normalizeTable(rawTable = {}) {
  return {
    pattern: String(rawTable.pattern || ''),
    target: String(rawTable.target || ''),
    fullPattern: String(rawTable.full_pattern || rawTable.fullPattern || ''),
    dtype: String(rawTable.dtype || ''),
    spawnRate: Number(rawTable.spawn_rate ?? rawTable.spawnRate ?? 0.1),
    guestAvailable: Boolean(rawTable.guest_available ?? rawTable.guestAvailable),
    ai: rawTable.ai || null,
  };
}

function deriveCatalogVersion(tables) {
  const serialized = JSON.stringify(tables.map((table) => [
    table.fullPattern,
    table.dtype,
    table.spawnRate,
    table.guestAvailable,
  ]));
  let hash = 0x811c9dc5;
  for (let index = 0; index < serialized.length; index += 1) {
    hash ^= serialized.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `derived-${(hash >>> 0).toString(16).padStart(8, '0')}`;
}

function resolveCatalogVersion(payload, tables) {
  const serverVersion = payload?.catalog_version ?? payload?.catalogVersion ?? payload?.version;
  return String(serverVersion || deriveCatalogVersion(tables));
}

function updateCatalogVersion(nextVersion) {
  const normalizedVersion = String(nextVersion || '');
  if (currentCatalogVersion && normalizedVersion !== currentCatalogVersion) {
    tablebaseResultCache.clear();
  }
  currentCatalogVersion = normalizedVersion;
}

export function getCatalogVersion() {
  return currentCatalogVersion;
}

export async function fetchTablebaseCatalog({ signal } = {}) {
  const response = await fetch(getBackendUrl('/api/tablebases'), {
    method: 'GET',
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    throw new Error(`Failed to load tablebase catalog: ${response.status}`);
  }
  const payload = await response.json();
  const rawTables = Array.isArray(payload.tables) ? payload.tables : [];
  const tables = rawTables.map(normalizeTable);
  const catalogVersion = resolveCatalogVersion(payload, tables);
  updateCatalogVersion(catalogVersion);
  Object.defineProperty(tables, 'catalogVersion', {
    configurable: false,
    enumerable: false,
    writable: false,
    value: catalogVersion,
  });
  return tables;
}

export function groupTablebasesByPattern(tables = []) {
  const groups = {};
  for (const table of tables) {
    if (!table.pattern) {
      continue;
    }
    if (!groups[table.pattern]) {
      groups[table.pattern] = [];
    }
    groups[table.pattern].push(table);
  }
  return groups;
}

export function groupTablebasePatternsByCategory(tables = []) {
  const patternsByCategory = {
    '4x4': new Set(),
    variant: new Set(),
  };

  for (const table of tables) {
    const pattern = String(table?.pattern || '');
    if (!pattern) {
      continue;
    }
    const category = VARIANT_PATTERNS.has(pattern) ? 'variant' : '4x4';
    patternsByCategory[category].add(pattern);
  }

  return CATEGORY_ORDER.reduce((categories, category) => {
    const patterns = [...patternsByCategory[category]].sort();
    if (patterns.length) {
      categories[category] = patterns;
    }
    return categories;
  }, {});
}

export function getCatalogTargets(tables = []) {
  return [...new Set(tables.map((table) => table.target).filter(Boolean))]
    .sort((left, right) => Number(left) - Number(right));
}

export function getCatalogTargetsForPattern(tables = [], pattern = '') {
  const normalizedPattern = String(pattern || '');
  return getCatalogTargets(
    tables.filter((table) => table.pattern === normalizedPattern)
  );
}
