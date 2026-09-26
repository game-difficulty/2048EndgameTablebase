const STORAGE_KEY = 'human-analysis-last-selection:v1';

export function normalizeAnalysisSelection(items, tables, maxItems = 6) {
  const available = new Set((tables || []).map(item => `${String(item.pattern)}\0${String(item.target)}`));
  const seen = new Set();
  const result = [];
  for (const item of Array.isArray(items) ? items : []) {
    const pattern = String(item?.pattern || '');
    const target = String(item?.target || '');
    const key = `${pattern}\0${target}`;
    if (!pattern || !target || seen.has(key) || !available.has(key)) continue;
    seen.add(key);
    result.push({ pattern, target });
    if (result.length >= maxItems) break;
  }
  return result;
}

export function loadLastAnalysisSelection(variant, tables, storage = globalThis.localStorage) {
  try {
    const saved = JSON.parse(storage?.getItem(STORAGE_KEY) || '{}');
    return normalizeAnalysisSelection(saved?.[variant], tables);
  } catch {
    return [];
  }
}

export function saveLastAnalysisSelection(variant, items, tables, storage = globalThis.localStorage) {
  const normalized = normalizeAnalysisSelection(items, tables);
  if (!variant || !normalized.length || !storage) return normalized;
  let saved = {};
  try { saved = JSON.parse(storage.getItem(STORAGE_KEY) || '{}') || {}; } catch { saved = {}; }
  storage.setItem(STORAGE_KEY, JSON.stringify({ ...saved, [variant]: normalized }));
  return normalized;
}

export { STORAGE_KEY as ANALYSIS_SELECTION_STORAGE_KEY };
