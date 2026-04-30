const resolveBackendUrl = () => {
  if (typeof window === 'undefined' || !window.location) {
    throw new Error('Backend URL resolution requires a browser window.');
  }

  if (typeof window.__APP_BACKEND_ORIGIN__ === 'string' && window.__APP_BACKEND_ORIGIN__) {
    return new URL(window.__APP_BACKEND_ORIGIN__);
  }

  const currentUrl = new URL(window.location.href);
  const backendPort = currentUrl.searchParams.get('backend_port');
  if (backendPort) {
    currentUrl.port = backendPort;
  }
  return currentUrl;
};

export const getBackendOrigin = () => resolveBackendUrl().origin;

export const getBackendUrl = (path = '/') => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return new URL(normalizedPath, `${getBackendOrigin()}/`).toString();
};

export const toBackendUrl = (pathOrUrl = '') => {
  if (!pathOrUrl) return '';
  if (/^[a-zA-Z][a-zA-Z\d+.-]*:/.test(pathOrUrl)) {
    return pathOrUrl;
  }
  return getBackendUrl(pathOrUrl);
};

export const getBackendWebSocketUrl = (clientId) => {
  const backendUrl = resolveBackendUrl();
  const wsProtocol = backendUrl.protocol === 'https:' ? 'wss:' : 'ws:';
  const encodedClientId = encodeURIComponent(String(clientId ?? ''));
  return `${wsProtocol}//${backendUrl.host}/ws/${encodedClientId}`;
};

export const getMinigameAssetBase = () => getBackendUrl('/minigames-assets/');

export const getMinigameAssetUrl = (assetPath = '') => {
  const normalizedAssetPath = String(assetPath).replace(/^\/+/, '');
  return new URL(normalizedAssetPath, getMinigameAssetBase()).toString();
};
