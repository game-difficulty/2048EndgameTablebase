import { tabUrl, TAB_ROUTES } from './siteProfile.js';

const TRUSTED_ORIGINS = new Set(['https://2048tables.online', 'https://tables.2048tables.online', 'https://play.2048tables.online']);
export function trustedSiteOrigin(origin, location) {
  if (TRUSTED_ORIGINS.has(origin)) return true;
  return ['localhost', '127.0.0.1'].includes(location.hostname) && origin === location.origin;
}
export function validHandoff(event, source, origin, token, type) {
  return event.source === source && event.origin === origin && event.data?.token === token && event.data?.type === type;
}

// Keep file/board context out of URLs and deliver it only to the window we opened.
export function openSiteTab(tab, detail, site, win = window) {
  const url = tabUrl(tab, site, win.location);
  if (!detail) { win.open(url.href, '_blank', 'noopener,noreferrer'); return; }
  const token = win.crypto.randomUUID();
  url.searchParams.set('handoff', token);
  url.searchParams.set('from', win.location.origin);
  const popup = win.open(url.href, '_blank');
  if (!popup) throw new Error('Please allow pop-ups / 请允许弹出窗口');
  const payload = JSON.parse(JSON.stringify({ ...detail, analysisFile: undefined }));
  if (detail.analysisFile instanceof File) payload.analysisFile = detail.analysisFile;
  const receive = event => {
    if (validHandoff(event, popup, url.origin, token, 'tables-ready')) {
      popup.postMessage({ type: 'tables-context', token, tab, detail: payload }, url.origin);
    } else if (validHandoff(event, popup, url.origin, token, 'tables-loaded')) cleanup();
  };
  const cleanup = () => { win.removeEventListener('message', receive); win.clearTimeout(timer); };
  const timer = win.setTimeout(cleanup, 60000);
  win.addEventListener('message', receive);
}

export function receiveSiteContext(onContext, win = window) {
  const url = new URL(win.location.href);
  const token = url.searchParams.get('handoff'), origin = url.searchParams.get('from'), source = win.opener;
  if (!token || !source || !trustedSiteOrigin(origin, win.location)) return () => {};
  const receive = event => {
    if (!validHandoff(event, source, origin, token, 'tables-context')) return;
    const { tab, detail } = event.data;
    if (!Object.values(TAB_ROUTES).includes(tab) || !detail || typeof detail !== 'object') return;
    if (detail.analysisFile && (!(detail.analysisFile instanceof File) || detail.analysisFile.size > 5000000)) return;
    onContext(tab, detail);
    source.postMessage({ type: 'tables-loaded', token }, origin);
    cleanup();
    url.searchParams.delete('handoff'); url.searchParams.delete('from');
    win.history.replaceState(win.history.state, '', url);
    win.opener = null;
  };
  const cleanup = () => { win.removeEventListener('message', receive); win.clearTimeout(timer); };
  const timer = win.setTimeout(cleanup, 60000);
  win.addEventListener('message', receive);
  source.postMessage({ type: 'tables-ready', token }, origin);
  return cleanup;
}
