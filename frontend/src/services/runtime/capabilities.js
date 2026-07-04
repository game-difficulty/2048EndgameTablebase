export function getPywebviewApi() {
  if (typeof window === 'undefined') {
    return null;
  }
  const api = window.pywebview?.api;
  return api && typeof api === 'object' ? api : null;
}

export function isDesktopRuntime() {
  return !!getPywebviewApi();
}

export function canUseDesktopDialogs() {
  const api = getPywebviewApi();
  return !!api && (
    typeof api.show_dialog === 'function' ||
    typeof api.select_folder === 'function'
  );
}

export function isBrowserRuntime() {
  return typeof window !== 'undefined' && !isDesktopRuntime();
}

export function hasBrowserFilePicker() {
  return typeof window !== 'undefined' && typeof window.showOpenFilePicker === 'function';
}

export function hasBrowserSaveFilePicker() {
  return typeof window !== 'undefined' && typeof window.showSaveFilePicker === 'function';
}

export function canDownloadFiles() {
  return typeof document !== 'undefined' && typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function';
}

export function canOpenExternalUrl() {
  const api = getPywebviewApi();
  return !!api && typeof api.open_external_url === 'function';
}

export function hasLocalStorage() {
  if (typeof window === 'undefined' || !window.localStorage) {
    return false;
  }
  try {
    const probeKey = '__2048_tables_storage_probe__';
    window.localStorage.setItem(probeKey, '1');
    window.localStorage.removeItem(probeKey);
    return true;
  } catch (_error) {
    return false;
  }
}

export function getRuntimeCapabilities() {
  const desktop = isDesktopRuntime();
  return {
    mode: desktop ? 'desktop' : 'browser',
    desktop,
    browser: !desktop && typeof window !== 'undefined',
    hasPywebview: desktop,
    canUseDesktopDialogs: canUseDesktopDialogs(),
    localStorage: hasLocalStorage(),
    canUseBrowserFilePicker: hasBrowserFilePicker(),
    canUseFileSystemAccess: hasBrowserFilePicker() || hasBrowserSaveFilePicker(),
    canDownloadFiles: canDownloadFiles(),
    canOpenExternalUrl: canOpenExternalUrl(),
  };
}
