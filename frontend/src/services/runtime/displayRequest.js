// Share concurrent display reads only. No polling and no stale result cache.
const pending = new Map();
export function displayRequest(key, load) {
  if (!pending.has(key)) {
    pending.set(key, Promise.resolve().then(load).finally(() => pending.delete(key)));
  }
  return pending.get(key);
}
