const DEFAULT_NAMESPACE = '2048tables';

function storageAvailable() {
  if (typeof window === 'undefined') return false;
  try {
    if (!window.sessionStorage) return false;
    const key = `${DEFAULT_NAMESPACE}:session-probe`;
    window.sessionStorage.setItem(key, '1');
    window.sessionStorage.removeItem(key);
    return true;
  } catch (_error) {
    return false;
  }
}

const cloneValue = (value) => (
  value == null || typeof value !== 'object' ? value : JSON.parse(JSON.stringify(value))
);

export function createSessionStorageStore({
  namespace = DEFAULT_NAMESPACE,
  key,
  version = 1,
  defaultValue = null,
} = {}) {
  if (!key) throw new Error('createSessionStorageStore requires a key.');
  const storageKey = `${namespace}:${key}`;

  const read = () => {
    if (!storageAvailable()) return cloneValue(defaultValue);
    try {
      const parsed = JSON.parse(window.sessionStorage.getItem(storageKey) || 'null');
      return parsed?.version === version ? cloneValue(parsed.value) : cloneValue(defaultValue);
    } catch (_error) {
      return cloneValue(defaultValue);
    }
  };

  const write = (value) => {
    const next = cloneValue(value);
    if (storageAvailable()) {
      window.sessionStorage.setItem(storageKey, JSON.stringify({
        version,
        value: next,
        updatedAt: new Date().toISOString(),
      }));
    }
    return cloneValue(next);
  };

  const remove = () => {
    if (storageAvailable()) window.sessionStorage.removeItem(storageKey);
  };

  return { key: storageKey, version, read, write, remove };
}
