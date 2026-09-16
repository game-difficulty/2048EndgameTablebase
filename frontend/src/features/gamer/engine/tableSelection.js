export const TABLE_POLICY_VERSION = 1;

export const normalizeTableSelection = selection => Array.isArray(selection)
  ? [...new Set(selection.filter(value => typeof value === 'string' && value.length > 0))].sort()
  : null;

const DEFAULT_EXCLUDED_TABLES = new Set(['free10_128', '444_1024', '444_2048']);

export const tableAllowed = (table, selection) => selection === null
  ? !DEFAULT_EXCLUDED_TABLES.has(table.fullPattern)
  : selection.includes(table.fullPattern);

export const aiCompatibleTable = table => Boolean(table.ai?.compatible && table.ai.policy_version === TABLE_POLICY_VERSION && !table.pattern.includes('_'));
