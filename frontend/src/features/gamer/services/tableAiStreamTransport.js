// Inject the socket lifecycle so reconnect/receipt handling is testable without a browser.
export function createTableAiStreamTransport({ createClient, onBalance = () => {}, onFailure = () => {}, now = () => Date.now() }) {
  let client = null;
  let current = null;
  const sendOpen = () => {
    if (!current) return;
    client.send('GAMER_STREAM_OPEN', { route_id: current.body.request_id, request: current.body,
      received: current.received, resume: current.opened, allow_through: 7 });
    current.opened = true;
    if (current.consumed >= 0) client.send('GAMER_STREAM_CREDIT', {
      route_id: current.body.request_id, consumed: current.consumed, allow_through: current.allowed });
  };
  const ensureClient = () => {
    if (client) return;
    client = createClient({
      onOpen: sendOpen,
      onMessage: (message) => {
        if (!current) return;
        if (message.action === 'AUTH_REQUIRED') {
          current.callbacks.onError(Object.assign(new Error('Authentication required'), { status: 401 }));
          current = null; return;
        }
        if (message.action !== 'GAMER_STREAM_EVENT' || message.data?.route_id !== current.body.request_id) return;
        const item = message.data;
        if (item.type === 'result') {
          if (!Number.isInteger(item.seq) || item.seq > current.received + 1) {
            const failed = current;
            current = null;
            client.send('GAMER_STREAM_CANCEL', { route_id: failed.body.request_id });
            failed.callbacks.onError(new Error('Invalid stream sequence')); return;
          }
          if (item.seq <= current.received) return;
          current.received = item.seq;
          onBalance(item.token_balance);
          current.callbacks.onResult(item);
        } else if (item.type === 'window') {
          const sent = current.credits.get(item.allow_through);
          if (sent !== undefined) current.callbacks.onLatency?.(now() - sent);
          for (const limit of current.credits.keys()) if (limit <= item.allow_through) current.credits.delete(limit);
        } else if (item.type === 'end' || (item.type === 'error' && item.detail === 'STREAM_GONE')) {
          current.callbacks.onEnd(); current = null;
        } else if (item.type === 'error') {
          const error = Object.assign(new Error('Tablebase stream failed'), { status: item.status, code: item.detail?.code || item.detail });
          onFailure(item);
          current.callbacks.onError(error); current = null;
        }
      },
    });
    client.connect();
  };
  return {
    open(body, callbacks) {
      if (current && client.getSocket()?.readyState === 1) client.send('GAMER_STREAM_CANCEL', { route_id: current.body.request_id });
      const task = { body: { ...body, steps: 1 }, callbacks, opened: false, received: -1,
        consumed: -1, allowed: 7, credits: new Map() };
      current = task;
      const existing = Boolean(client);
      ensureClient();
      if (existing && client.getSocket()?.readyState === 1) sendOpen();
      return {
        credit(consumed, allowed) {
          if (current !== task) return;
          task.consumed = Math.max(task.consumed, consumed);
          task.allowed = Math.max(task.allowed, allowed);
          if (client.getSocket()?.readyState === 1) {
            task.credits.set(task.allowed, now());
            while (task.credits.size > 32) task.credits.delete(task.credits.keys().next().value);
            client.send('GAMER_STREAM_CREDIT', { route_id: body.request_id,
              consumed: task.consumed, allow_through: task.allowed });
          }
        },
        cancel() {
          if (current !== task) return;
          if (client.getSocket()?.readyState === 1) client.send('GAMER_STREAM_CANCEL', { route_id: body.request_id });
          current = null;
        },
      };
    },
    close() { current = null; client?.disconnect(); client = null; },
  };
}
