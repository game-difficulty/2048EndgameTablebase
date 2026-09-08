// Refresh off the move path; apply a new catalog only at the next decision.
export class TableAiCatalog {
  constructor({ load, now = () => Date.now() }) {
    this.load = load;
    this.now = now;
    this.tables = null;
    this.pending = null;
    this.expires = 0;
  }
  get() {
    if (!this.pending && (!this.tables || this.now() >= this.expires)) {
      const controller = new AbortController();
      const task = { controller };
      this.pending = task;
      const timer = setTimeout(() => controller.abort(), 10000);
      task.promise = Promise.resolve().then(() => this.load({ signal: controller.signal }))
        .then((tables) => {
          if (this.pending === task) { this.tables = tables; this.expires = this.now() + 60000; }
          return tables;
        }).catch((error) => {
          if (this.pending === task) this.expires = this.now() + 5000;
          throw error;
        }).finally(() => {
          clearTimeout(timer);
          if (this.pending === task) this.pending = null;
        });
      // A failed background refresh must not discard usable data or reject unhandled.
      task.promise.catch(() => {});
    }
    return this.tables ? Promise.resolve(this.tables) : this.pending.promise;
  }
  invalidate() { this.expires = 0; }
  clear() {
    this.pending?.controller.abort();
    this.pending = null;
    this.tables = null;
    this.expires = 0;
  }
}
