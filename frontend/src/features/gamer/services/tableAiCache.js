import { createRandomXoshiroState, Xoshiro128StarStar } from '../../../utils/xoshiro128.js';
import { needsEvilSpawn } from '../engine/gamerSpawn.js';

// Account-local purchased results and one streaming short route; never persisted.
export class TableAiCache {
  constructor({ transport, now = () => Date.now() } = {}) {
    this.transport = transport;
    this.now = now;
    this.entries = new Map();
    this.generation = 0;
    this.active = null;
    this.routes = new Map();
    this.demand = null;
  }

  key(body) { return JSON.stringify([body.catalog_version, body.full_pattern, body.board_codes]); }
  context(body) { return JSON.stringify([body.catalog_version, body.full_pattern,
    body.difficulty, body.spawn_rate4, body.random_only]); }
  get(body) {
    const key = this.key(body);
    const entry = this.entries.get(key);
    if (!entry) return null;
    if (this.now() - entry.time >= 300000) { this.entries.delete(key); return null; }
    this.entries.delete(key);
    this.entries.set(key, entry);
    return entry.value;
  }

  cancelPrefetch() {
    this.generation += 1;
    this.active?.abort.abort();
    this.active = null;
    this.routes.clear();
    this.demand = null;
  }

  clear() {
    this.cancelPrefetch();
    this.entries.clear();
  }

  start(body) {
    const generation = this.generation;
    const context = this.context(body);
    let route = this.routes.get(context);
    if (!route || !body.advance_first) {
      route = { tail: null, path: [] };
      this.routes.set(context, route);
      while (this.routes.size > 16) this.routes.delete(this.routes.keys().next().value);
    }
    const task = { abort: new AbortController(), context: this.context(body), changed: null,
      wake: null, error: null, done: false };
    const resetWake = () => { task.changed = new Promise((resolve) => { task.wake = resolve; }); };
    resetWake();
    this.active = task;
    task.promise = this.transport({ ...body, request_id: createRandomXoshiroState().map((word) => word.toString(16).padStart(8, '0')).join(''), steps: 4 }, {
      signal: task.abort.signal,
      onResult: (item) => {
        if (generation !== this.generation || task.abort.signal.aborted) return;
        this.entries.set(this.key(item), { time: this.now(), value: item });
        while (this.entries.size > 256) this.entries.delete(this.entries.keys().next().value);
        route.tail = { ...body, board_codes: item.board_codes, rng_state: item.rng_state,
          random_only: item.random_only ?? body.random_only, advance_first: true };
        const node = { key: this.key(item), state: JSON.stringify(item.rng_state) };
        const duplicate = route.path.findIndex((entry) => entry.key === node.key && entry.state === node.state);
        if (duplicate < 0) route.path.push(node);
        if (route.path.length > 12) route.path.shift();
        task.wake(); resetWake();
      },
    }).catch((error) => { task.error = error; }).finally(() => {
      task.done = true; task.wake();
      if (this.active === task) {
        this.active = null;
        if (!task.error && !task.abort.signal.aborted) this.refill();
      }
    });
    return task;
  }

  refill() {
    const body = this.demand;
    if (!body || this.active) return;
    const route = this.routes.get(this.context(body));
    const position = route?.path.findIndex((node) => node.key === this.key(body)
      && node.state === JSON.stringify(body.rng_state)) ?? -1;
    if (position < 0 || route.path.length - position - 1 > 2) return;
    const tail = route.tail;
    const value = this.get(tail);
    const raw = Object.values(value?.results || {})[0];
    if (typeof raw !== 'number' || !Number.isFinite(raw)
      || raw + (String(value.dtype).startsWith('1-') ? 1 : 0) <= 0) return;
    if (!needsEvilSpawn(new Xoshiro128StarStar(tail.rng_state),
      { difficulty: tail.difficulty, randomOnly: tail.random_only })) this.start(tail);
  }

  consume(body, value) {
    this.demand = body;
    this.refill();
    return value;
  }

  async lookup(body) {
    const generation = this.generation;
    let value = this.get(body);
    if (value) {
      const context = this.context(body);
      const route = this.routes.get(context);
      if (!route?.path.some((node) => node.key === this.key(body) && node.state === JSON.stringify(body.rng_state))) {
        this.routes.set(context, { tail: { ...body, advance_first: true },
          path: [{ key: this.key(body), state: JSON.stringify(body.rng_state) }] });
        while (this.routes.size > 16) this.routes.delete(this.routes.keys().next().value);
      }
      return this.consume(body, value);
    }
    this.demand = null;
    const previous = this.active;
    if (previous && previous.context === this.context(body)) {
      while (!previous.done && generation === this.generation) {
        value = this.get(body);
        if (value) return this.consume(body, value);
        await previous.changed;
      }
      if (generation !== this.generation) throw new Error('Superseded AI request');
      value = this.get(body);
      if (value) return this.consume(body, value);
      if (previous.error) throw previous.error;
    } else if (previous) {
      previous.abort.abort();
      await previous.promise;
    }
    if (generation !== this.generation) throw new Error('Superseded AI request');
    const task = this.start(body);
    while (!task.done && generation === this.generation) {
      value = this.get(body);
      if (value) return this.consume(body, value);
      await task.changed;
    }
    value = this.get(body);
    if (generation !== this.generation) throw new Error('Superseded AI request');
    if (value) return this.consume(body, value);
    throw task.error || new Error('No tablebase result');
  }
}
