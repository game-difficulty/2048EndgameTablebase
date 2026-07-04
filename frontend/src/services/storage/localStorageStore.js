const DEFAULT_NAMESPACE = '2048tables';

function canUseLocalStorage() {
  if (typeof window === 'undefined' || !window.localStorage) {
    return false;
  }
  try {
    const key = `${DEFAULT_NAMESPACE}:probe`;
    window.localStorage.setItem(key, '1');
    window.localStorage.removeItem(key);
    return true;
  } catch (_error) {
    return false;
  }
}

function cloneValue(value) {
  if (value == null || typeof value !== 'object') {
    return value;
  }
  return JSON.parse(JSON.stringify(value));
}

function normalizeEnvelope(rawValue, defaultValue, version) {
  if (!rawValue || typeof rawValue !== 'object') {
    return {
      version,
      value: cloneValue(defaultValue),
      updatedAt: null,
    };
  }

  return {
    version: Number(rawValue.version) || 0,
    value: Object.prototype.hasOwnProperty.call(rawValue, 'value')
      ? rawValue.value
      : cloneValue(defaultValue),
    updatedAt: rawValue.updatedAt || null,
  };
}

export function createLocalStorageStore({
  namespace = DEFAULT_NAMESPACE,
  key,
  version = 1,
  defaultValue = null,
  migrate,
} = {}) {
  if (!key) {
    throw new Error('createLocalStorageStore requires a key.');
  }

  const storageKey = `${namespace}:${key}`;

  const readEnvelope = () => {
    if (!canUseLocalStorage()) {
      return normalizeEnvelope(null, defaultValue, version);
    }

    try {
      const rawText = window.localStorage.getItem(storageKey);
      const parsed = rawText ? JSON.parse(rawText) : null;
      let envelope = normalizeEnvelope(parsed, defaultValue, version);
      if (envelope.version !== version && typeof migrate === 'function') {
        envelope = normalizeEnvelope(
          {
            version,
            value: migrate(envelope.value, envelope.version, version),
            updatedAt: new Date().toISOString(),
          },
          defaultValue,
          version
        );
        window.localStorage.setItem(storageKey, JSON.stringify(envelope));
      }
      return envelope;
    } catch (_error) {
      return normalizeEnvelope(null, defaultValue, version);
    }
  };

  const read = () => cloneValue(readEnvelope().value);

  const write = (value) => {
    const envelope = {
      version,
      value: cloneValue(value),
      updatedAt: new Date().toISOString(),
    };
    if (canUseLocalStorage()) {
      window.localStorage.setItem(storageKey, JSON.stringify(envelope));
    }
    return cloneValue(envelope.value);
  };

  const update = (updater) => {
    const current = read();
    const next = typeof updater === 'function' ? updater(current) : updater;
    return write(next);
  };

  const reset = () => write(defaultValue);

  const remove = () => {
    if (canUseLocalStorage()) {
      window.localStorage.removeItem(storageKey);
    }
  };

  return {
    key: storageKey,
    version,
    read,
    write,
    update,
    reset,
    remove,
    readEnvelope,
  };
}
