// One bounded recovery request. HTTP can restore the picture without claiming
// that the WebSocket is healthy; the transport has its own synchronization state.
export function createSnapshotRecovery({ url, install, fetcher = (...args) => fetch(...args) }) {
  let pending, controller, stopped = false;
  function refresh() {
    if (stopped) return Promise.resolve();
    if (pending) return pending;
    controller = new AbortController();
    const signal = controller.signal;
    pending = (async () => {
      let timer;
      try {
        const timeout = new Promise((_, reject) => {
          signal.addEventListener('abort', () => reject(Error('snapshot_cancelled')), { once:true });
          timer = setTimeout(() => controller.abort(), 3000);
        });
        const data = await Promise.race([(async () => {
          const response = await fetcher(url, { signal, cache:'no-store', credentials:'same-origin' });
          if (!response.ok) throw Error('snapshot_unavailable');
          return response.json();
        })(), timeout]);
        if (!stopped && !signal.aborted) install({ ...data, type:'snapshot' });
      } catch { /* Retry on the next bounded watchdog/foreground request. */ }
      finally { clearTimeout(timer); pending = null; }
    })();
    return pending;
  }
  return { refresh, stop() { stopped = true; controller?.abort(); } };
}
