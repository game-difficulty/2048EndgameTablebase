import { createRandomXoshiroState } from '../../../utils/xoshiro128.js';

const nodeKey = (body) => JSON.stringify([body.board_codes, body.rng_state, Boolean(body.random_only)]);

// Purchased results are a bounded LRU; speculative work is one credit-controlled subscription.
export class TableAiCache {
  constructor({ transport, now = () => Date.now() } = {}) {
    this.transport = transport;
    this.now = now;
    this.entries = new Map();
    this.active = null;
    this.generation = 0;
  }
  key(body) { return JSON.stringify([body.catalog_version, body.full_pattern, body.board_codes]); }
  context(body) { return JSON.stringify([body.catalog_version, body.full_pattern, body.difficulty, body.spawn_rate4]); }
  get(body) {
    const key = this.key(body);
    const entry = this.entries.get(key);
    if (!entry) return null;
    if (this.now() - entry.time >= 300000) { this.entries.delete(key); return null; }
    this.entries.delete(key); this.entries.set(key, entry);
    return entry.value;
  }
  cancelPrefetch() {
    this.generation += 1;
    const task = this.active;
    this.active = null;
    if (task) { task.handle.cancel(); task.done = true; task.wake(); }
  }
  clear() { this.cancelPrefetch(); this.entries.clear(); }
  close() { this.clear(); this.transport.close(); }

  start(body, advanceFirst = false) {
    this.cancelPrefetch();
    const generation = this.generation;
    const task = { context: this.context(body), nodes: new Map(), done: false, error: null,
      changed: null, wake: null, consumed: -1, allowed: 7, lastCredit: -1,
      used: new Set(), lastReturn: null, interval: 50, latency: 350, started: this.now() };
    const wake = () => {
      task.wake?.();
      task.changed = new Promise((resolve) => { task.wake = resolve; });
    };
    wake();
    this.active = task;
    task.handle = this.transport.open({ ...body, advance_first: advanceFirst, steps: 1,
      request_id: createRandomXoshiroState().map((word) => word.toString(16).padStart(8, '0')).join('') }, {
      onResult: (item) => {
        if (generation !== this.generation) return;
        this.entries.set(this.key(item), { time: this.now(), value: item });
        while (this.entries.size > 256) this.entries.delete(this.entries.keys().next().value);
        task.nodes.set(nodeKey(item), item.seq);
        while (task.nodes.size > 256) task.nodes.delete(task.nodes.keys().next().value);
        if (item.seq === 0) task.latency = Math.max(50, this.now() - task.started);
        // Cached playback may already have passed this late-arriving node.
        if (task.used.has(nodeKey(item))) this.advanceCredit(task, item.seq);
        wake();
      },
      onLatency: (ms) => { task.latency = Math.max(50, ms, task.latency * .9 + ms * .1); },
      onEnd: () => { task.done = true; wake(); },
      onError: (error) => { task.error = error; task.done = true; wake(); },
    });
    return task;
  }

  consume(body, value, requestedAt) {
    const task = this.active;
    if (!task || task.context !== this.context(body) || task.done) return value;
    if (task.lastReturn !== null) {
      const interval = Math.max(1, requestedAt - task.lastReturn);
      task.interval = Math.min(interval, .8 * task.interval + .2 * interval);
    }
    task.lastReturn = this.now();
    const key = nodeKey(body);
    task.used.add(key);
    while (task.used.size > 256) task.used.delete(task.used.values().next().value);
    const seq = task.nodes.get(key);
    if (seq !== undefined) this.advanceCredit(task, seq);
    return value;
  }

  advanceCredit(task, seq) {
    if (task.done || seq <= task.consumed) return;
    task.consumed = seq;
    const cover = Math.ceil(task.latency / task.interval);
    const window = Math.max(8, Math.min(64, cover + 12));
    // Include the four-move credit cadence in the refill margin.
    if (task.allowed - seq <= Math.max(4, cover + 4) && (task.lastCredit < 0 || seq - task.lastCredit >= 4 || task.allowed - seq <= 2)) {
      task.allowed = Math.max(task.allowed, Math.min(99999, seq + window));
      task.lastCredit = seq;
      task.handle.credit(seq, task.allowed);
    }
  }

  async lookup(body) {
    const requestedAt = this.now();
    const generation = this.generation;
    let value = this.get(body);
    if (value) {
      if (!this.active || this.active.context !== this.context(body)) this.start(body, true);
      return this.consume(body, value, requestedAt);
    }
    const previous = this.active;
    if (previous && previous.context === this.context(body)) {
      while (!previous.done && generation === this.generation) {
        value = this.get(body);
        if (value) return this.consume(body, value, requestedAt);
        await this.waitForChange(previous);
      }
      if (generation !== this.generation) throw new Error('Superseded AI request');
      value = this.get(body);
      if (value) return this.consume(body, value, requestedAt);
      if (previous.error) throw previous.error;
    }
    const task = this.start(body);
    const currentGeneration = this.generation;
    while (!task.done && currentGeneration === this.generation) {
      value = this.get(body);
      if (value) return this.consume(body, value, requestedAt);
      await this.waitForChange(task);
    }
    if (currentGeneration !== this.generation) throw new Error('Superseded AI request');
    value = this.get(body);
    if (value) return this.consume(body, value, requestedAt);
    throw task.error || new Error('No tablebase result');
  }

  async waitForChange(task) {
    let timer;
    try {
      await Promise.race([task.changed, new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('Tablebase stream timeout')), 30000);
      })]);
    } finally { clearTimeout(timer); }
  }
}
