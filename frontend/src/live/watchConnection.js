// A socket being OPEN is not proof that snapshots or replies are still arriving.
export function createWatchConnection({ room, url, canConnect, expired, onOpen, onDisconnect,
  onMessage, onSnapshot, onEnded, onActivity, onResync, now = () => Date.now(), random = Math.random }) {
  let socket, watchdog, retry, probe, probeTimer;
  let stopped = false, started = false, failures = 0, epoch = 0;
  let startedAt = 0, openedAt = null, lastReceived = 0, lastPing = 0, synchronized = false;
  let lastResync = -Infinity, installed = null;
  const competition = room.protocol === 'competition-match-v1';
  function resync() {
    if (!onResync || now() - lastResync < 3000) return;
    lastResync = now();
    void Promise.resolve().then(onResync).catch(() => {});
  }
  function observeSnapshot(data) {
    const next = data?.match;
    if (!next) return;
    if (!installed || next.match_public_key !== installed.match_public_key || next.generation > installed.generation ||
      (next.generation === installed.generation && next.content_sequence >= installed.content_sequence)) installed = next;
  }

  function cancelRetry() {
    ++epoch;
    clearTimeout(retry);
    clearTimeout(probeTimer);
    probe?.abort();
    probe = null;
  }
  function detach() {
    const old = socket;
    socket = null; // Detach before close: CLOSING may never produce a close event.
    synchronized = false;
    clearInterval(watchdog);
    cancelRetry();
    onDisconnect?.();
    try { old?.close(); } catch { /* Already unusable. */ }
  }
  async function schedule() {
    if (stopped || !canConnect()) return;
    const attempt = epoch;
    if (room.dynamic) {
      const controller = probe = new AbortController();
      try {
        const timeout = new Promise((_, reject) => {
          controller.signal.addEventListener('abort', () => reject(new Error('room_query_cancelled')), { once: true });
          probeTimer = setTimeout(() => controller.abort(), 3000);
        });
        const response = await Promise.race([
          fetch(room.api_base, { cache: 'no-store', signal: controller.signal }), timeout,
        ]);
        if (stopped || attempt !== epoch) return;
        if (response.status === 404) { stop(); onEnded?.(); return; }
      } catch { /* A failed lookup must never prevent reconnection. */ }
      finally {
        if (probe === controller) { clearTimeout(probeTimer); probe = null; }
      }
    }
    if (stopped || attempt !== epoch || !canConnect()) return;
    retry = setTimeout(connect, Math.min(15000, 1000 * 2 ** Math.min(failures++, 4)) + random() * 300);
  }
  function recover(current) {
    if (stopped || socket !== current) return;
    detach();
    void schedule();
  }
  function check() {
    if (!stopped && started && canConnect() && competition && now() - lastResync >= 15000) resync();
    const current = socket;
    if (stopped || !current) return;
    if (expired()) { detach(); return; }
    const time = now();
    if (current.readyState >= 2
      || (!synchronized && time - (openedAt ?? startedAt) >= 10000)
      || (synchronized && time - lastReceived >= 30000)) {
      recover(current); return;
    }
    if (current.readyState === 1 && time - lastPing >= 10000) {
      lastPing = time;
      try { current.send('ping'); } catch { recover(current); }
    }
  }
  function connect() {
    if (stopped || !canConnect()) return;
    started = true;
    if (socket) { check(); return; }
    cancelRetry();
    synchronized = false;
    startedAt = lastReceived = lastPing = now(); openedAt = null;
    let current;
    try { current = socket = new WebSocket(url); }
    catch { onDisconnect?.(); void schedule(); return; }
    current.binaryType = 'arraybuffer';
    watchdog = setInterval(check, 1000);
    current.onopen = () => {
      if (socket !== current || stopped) return;
      openedAt = lastReceived = lastPing = now();
      if (competition) resync();
      onOpen?.(); // Backoff resets only after a successfully installed snapshot.
    };
    current.onmessage = async event => {
      if (socket !== current || stopped) return;
      if (expired()) { detach(); return; }
      try {
        const data = event.data instanceof ArrayBuffer ? event.data : JSON.parse(event.data);
        if (!(data instanceof ArrayBuffer) && (!data || typeof data.type !== 'string')) throw Error('invalid_message');
        if (data.type === 'snapshot' && (data.room_id !== room.id || data.protocol !== room.protocol)) throw Error('wrong_snapshot');
        lastReceived = now();
        if (competition && data.type === 'match_watermark' && (!installed ||
          data.match_public_key !== installed.match_public_key || data.generation > installed.generation ||
          (data.generation === installed.generation && data.content_sequence > installed.content_sequence))) resync();
        await onMessage(data);
        if (socket !== current || stopped) return;
        if (data.type === 'snapshot' && (!competition || data.match)) {
          observeSnapshot(data); synchronized = true; failures = 0; onSnapshot?.();
        }
        onActivity?.();
      } catch { recover(current); }
    };
    current.onclose = () => recover(current);
    current.onerror = () => recover(current);
  }
  function reconnect() {
    if (stopped || !started) return;
    detach();
    void schedule();
  }
  function stop() { stopped = true; detach(); }
  return { connect, reconnect, check, observeSnapshot, cancelRetry, disconnect: detach, stop };
}
