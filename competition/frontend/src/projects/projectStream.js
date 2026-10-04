import { checkpointDelta } from './checkpointDelta.js';
export const PROJECT_STREAM_PROTOCOL = 'project-stream-v2';

// Four outstanding batches hide RTT; acknowledgements are cumulative and durable.
export class ProjectStreamSender {
  constructor({ createSocket, authenticate, phaseToken, onAck = () => {}, onError = () => {},
    onDiagnostic = () => {}, interval = 80 }) {
    Object.assign(this, { createSocket, authenticate, phaseToken, onAck, onError, onDiagnostic, interval });
    this.frames = []; this.latest = null; this.inflight = new Map();
    this.accepted = 0; this.sent = 0; this.ready = false; this.closed = false;
    this.previous = null; this.retry = 0;
    this.open();
  }
  open() {
    if (this.closed) return;
    clearTimeout(this.reconnect); clearInterval(this.heartbeat);
    this.ready = false;
    const socket = this.socket = this.createSocket();
    this.lastReceived = Date.now();
    this.openedAt = Date.now();
    socket.onopen = () => {
      if (this.socket === socket && !this.closed) socket.send(JSON.stringify({ type: 'authenticate', data: this.authenticate() }));
    };
    socket.onmessage = event => {
      if (this.socket !== socket || this.closed) return;
      try {
        const message = JSON.parse(event.data); this.lastReceived = Date.now();
        if (message.type === 'stream.ready') {
          if (message.protocol !== PROJECT_STREAM_PROTOCOL) { this.disconnected(socket, {code:4406,reason:'refresh_required'}); return; }
          this.ready = true; this.retry = 0; this.previous = null; this.inflight.clear();
          this.accepted = this.sent = Number(message.accepted_sequence);
          this.frames = this.frames.filter(f => f.sequence > this.accepted);
          this.onAck(message); this.schedule(0);
        } else if (message.type === 'stream.ack') {
          this.accepted = Math.max(this.accepted, Number(message.accepted_sequence));
          this.frames = this.frames.filter(f => f.sequence > this.accepted);
          for (const seq of this.inflight.keys()) if (seq <= this.accepted) this.inflight.delete(seq);
          this.onAck(message);
          if (message.stopped) this.close();
          else this.schedule(0);
        } else if (message.type === 'stream.error') {
          const error = Object.assign(new Error(message.error?.message || '同步失败'), message.error);
          if (this.onError(error) === false) this.close();
        }
      } catch (error) { this.onError(error); this.disconnected(socket, {code:4001,reason:'invalid_server_message'}); }
    };
    socket.onclose = event => this.disconnected(socket, event);
    socket.onerror = () => {}; // onclose owns recovery: never run competing reconnect loops.
    this.heartbeat = setInterval(() => {
      if (this.socket !== socket || this.closed) return;
      const oldest = this.inflight.values().next().value;
      if ((!this.ready && Date.now() - this.openedAt > 10000)
        || Date.now() - this.lastReceived > 30000 || (oldest && Date.now() - oldest > 8000)) {
        this.disconnected(socket, {code:4001,reason:'ack_timeout'}); return;
      }
      if (socket.readyState === 1) socket.send(JSON.stringify({ type: 'ping' }));
    }, 2000);
  }
  disconnected(socket, event) {
    if (this.socket !== socket || this.closed) return;
    // Detach immediately: close() can remain CLOSING indefinitely on a dead
    // connection. Neither recovery nor input should wait for its close event.
    this.socket = null; this.ready = false;
    clearInterval(this.heartbeat); clearTimeout(this.timer); this.timer = null;
    try { socket.close(); } catch { /* Already closed. */ }
    this.onDiagnostic({ type: 'closed', code: event.code, reason: event.reason, accepted: this.accepted });
    const fatal = [4400, 4401, 4404, 4406, 4409, 1009].includes(event.code);
    const error = Object.assign(new Error(event.code === 4409 ? '该对局已在另一个页面接管，请只保留一个操作页面。' : fatal ? '同步协议或登录状态失效，请刷新页面后重试。' : '同步连接中断，正在恢复'),
      { status: fatal ? 426 : 0, code: fatal ? 'CLIENT_UPDATE_REQUIRED' : 'STREAM_DISCONNECTED' });
    if (this.onError(error) === false || fatal) { this.close(); return; }
    this.reconnect = setTimeout(() => this.open(), Math.min(500 * 2 ** this.retry++, 5000));
  }
  recover() {
    if (this.closed) return;
    if (!this.socket || this.socket.readyState > 1) { this.open(); return; }
    if (Date.now() - this.lastReceived > 8000) this.disconnected(this.socket, {code:4001,reason:'network_resumed'});
  }
  push(packet) {
    if (this.closed || (this.latest && packet.sequence <= this.latest.sequence)) return;
    this.latest = packet;
    this.frames.push({ sequence: packet.sequence, payload: packet.payload });
    if (this.frames.length > 128) this.frames.shift();
    this.schedule(packet.finished ? 0 : this.interval);
  }
  schedule(delay) {
    if (!this.closed && this.timer == null) this.timer = setTimeout(() => { this.timer = null; this.flush(); }, delay);
  }
  flush() {
    if (!this.ready || this.closed || !this.latest || this.latest.sequence <= this.sent || this.inflight.size >= 4) return;
    if (this.socket.bufferedAmount > 256 * 1024) { this.schedule(this.interval); return; }
    const packet = this.latest, previous = this.previous;
    // Ordered WebSocket batches and durable cumulative ACKs preserve the base.
    // Reconnect resets previous; periodically resending all undo/metric history
    // only creates growing bursts during an otherwise healthy connection.
    const full = !previous || packet.finished;
    const checkpoint = full ? packet.checkpoint : checkpointDelta(previous.checkpoint, packet.checkpoint, previous.sequence);
    const frames = this.frames.filter(f => f.sequence > this.sent);
    const wire = { ...packet, elapsed_ms: Math.round(packet.elapsed_ms), checkpoint, frames, phase_token: this.phaseToken() };
    let text = JSON.stringify({ type: 'project.batch', data: wire });
    while (frames.length > 1 && new TextEncoder().encode(text).length > 750000) {
      frames.shift(); text = JSON.stringify({ type: 'project.batch', data: wire });
    }
    try {
      this.socket.send(text);
      this.sent = packet.sequence; this.previous = packet; this.inflight.set(packet.sequence, Date.now());
      this.onDiagnostic({ type: 'sent', sequence: packet.sequence, from: frames[0]?.sequence,
        bytes: new TextEncoder().encode(text).length, outstanding: this.inflight.size });
    } catch { this.disconnected(this.socket, {code:4001,reason:'send_failed'}); }
  }
  close() {
    this.closed = true; clearInterval(this.heartbeat); clearTimeout(this.timer); clearTimeout(this.reconnect);
    this.socket?.close(1000); this.frames = []; this.inflight.clear();
  }
}
