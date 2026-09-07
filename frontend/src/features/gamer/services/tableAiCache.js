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
    this.tail = null;
    this.path = [];
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
    this.tail = null;
    this.path = [];
  }

  clear() {
    this.cancelPrefetch();
    this.entries.clear();
  }

  start(body) {
    const generation = this.generation;
    const task = { abort: new AbortController(), context: this.context(body), changed: null,
      wake: null, error: null, done: false };
    const resetWake = () => { task.changed = new Promise((resolve) => { task.wake = resolve; }); };
    resetWake();
    this.active = task;
    task.promise = this.transport({ ...body, request_id: createRandomXoshiroState().map((word) => word.toString(16).padStart(8, '0')).join(''), steps: 4 }, {
      signal: task.abort.signal,
      onResult: (item) => {
        if (generation !== this.generation) return;
        this.entries.set(this.key(item), { time: this.now(), value: item });
        while (this.entries.size > 256) this.entries.delete(this.entries.keys().next().value);
        this.tail = { ...body, board_codes: item.board_codes, rng_state: item.rng_state,
          random_only: item.random_only ?? body.random_only, advance_first: true };
        this.path.push({ key: this.key(item), state: JSON.stringify(item.rng_state) });
        if (this.path.length > 12) this.path.shift();
        task.wake(); resetWake();
      },
    }).catch((error) => { task.error = error; }).finally(() => {
      task.done = true; task.wake();
      if (this.active === task) this.active = null;
    });
    return task;
  }

  async lookup(body) {
    const generation = this.generation;
    let value = this.get(body);
    if (value) {
      let position = -1;
      for (let index = this.path.length - 1; index >= 0; index -= 1) {
        const node = this.path[index];
        if (node.key === this.key(body) && node.state === JSON.stringify(body.rng_state)) {
          position = index; break;
        }
      }
      const samePath = position >= 0 && this.tail && this.context(this.tail) === this.context(body);
      if (!this.active && (!samePath || this.path.length - position - 1 <= 2)) {
        const tail = samePath ? this.tail : { ...body, advance_first: true };
        if (!needsEvilSpawn(new Xoshiro128StarStar(tail.rng_state),
          { difficulty: tail.difficulty, randomOnly: tail.random_only })) this.start(tail);
      }
      return value;
    }
    const previous = this.active;
    if (previous && previous.context === this.context(body)) {
      while (!previous.done && generation === this.generation) {
        value = this.get(body);
        if (value) return value;
        await previous.changed;
      }
      value = this.get(body);
      if (value) return value;
      if (previous.error) throw previous.error;
    } else if (previous) {
      previous.abort.abort();
      await previous.promise;
    }
    if (generation !== this.generation) throw new Error('Superseded AI request');
    const task = this.start(body);
    while (!task.done && generation === this.generation) {
      value = this.get(body);
      if (value) return value;
      await task.changed;
    }
    value = this.get(body);
    if (generation !== this.generation) throw new Error('Superseded AI request');
    if (value) return value;
    throw task.error || new Error('No tablebase result');
  }
}
