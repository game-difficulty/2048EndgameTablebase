import { getBackendUrl } from '../runtime/backendUrl';

const VARIANT_PATTERNS = new Set(['2x4', '3x3', '3x4', '3x4441']);
const CATEGORY_ORDER = ['4x4', 'variant'];

function normalizeTable(rawTable = {}) {
  return {
    pattern: String(rawTable.pattern || ''),
    target: String(rawTable.target || ''),
    fullPattern: String(rawTable.full_pattern || rawTable.fullPattern || ''),
    dtype: String(rawTable.dtype || ''),
    spawnRate: Number(rawTable.spawn_rate ?? rawTable.spawnRate ?? 0.1),
  };
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
  const tables = Array.isArray(payload.tables) ? payload.tables : [];
  return tables.map(normalizeTable);
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
