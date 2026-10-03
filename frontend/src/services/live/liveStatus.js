export function createLiveStatusPoller({
  url, onChange, document: doc = document, fetch: request = fetch,
  isOnline = data => data.online === true,
  setTimeout: later = setTimeout, clearTimeout: cancel = clearTimeout,
}) {
  let stopped = false;
  let timer;
  let controller;

  async function refresh() {
    if (stopped || doc.hidden || controller) return;
    const current = new AbortController();
    controller = current;
    const timeout = later(() => current.abort(), 5000);
    try {
      const response = await request(url, { signal: current.signal, cache: 'no-store' });
      if (!response.ok) throw new Error('Live status unavailable');
      const data = await response.json();
      if (!stopped && controller === current && !current.signal.aborted) onChange(isOnline(data));
    } catch {
      if (!stopped && controller === current) onChange(false);
    } finally {
      cancel(timeout);
      if (controller === current) {
        controller = null;
        if (!stopped && !doc.hidden) timer = later(refresh, 30000);
      }
    }
  }

  function visibilityChanged() {
    cancel(timer);
    controller?.abort();
    controller = null;
    onChange(false);
    if (!doc.hidden) refresh();
  }

  doc.addEventListener('visibilitychange', visibilityChanged);
  refresh();
  return () => {
    stopped = true;
    cancel(timer);
    controller?.abort();
    controller = null;
    doc.removeEventListener('visibilitychange', visibilityChanged);
  };
}
