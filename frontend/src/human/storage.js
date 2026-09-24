import { needsReplayUpload } from './archivePolicy.js';

// A move and its current state commit together. Server acknowledgements never delete local moves.
let opening;
function open() {
  if (!opening) opening = new Promise((resolve, reject) => {
    const req = indexedDB.open('human-play-v1', 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      db.createObjectStore('meta'); db.createObjectStore('runs', { keyPath: 'id' });
      db.createObjectStore('events', { keyPath: ['run', 'seq'] });
    };
    req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error);
  });
  return opening;
}
function result(req) { return new Promise((resolve, reject) => { req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error); }); }
function completed(tx) { return new Promise((resolve, reject) => { tx.oncomplete = resolve; tx.onabort = tx.onerror = () => reject(tx.error || new Error('storage_failed')); }); }
export async function browserId() {
  const db = await open(); const tx = db.transaction('meta', 'readwrite'); const done = completed(tx); const store = tx.objectStore('meta');
  let id = await result(store.get('browser'));
  if (!id) { id = crypto.randomUUID(); store.put(id, 'browser'); }
  await done; return id;
}
export async function meta(key, value) {
  const db = await open(); const tx = db.transaction('meta', value === undefined ? 'readonly' : 'readwrite'); const done = completed(tx);
  const store = tx.objectStore('meta'); const item = value === undefined ? await result(store.get(key)) : (store.put(value, key), value);
  await done; return item;
}
export async function readRun(id) {
  if (!id) return null;
  const db = await open(); return result(db.transaction('runs').objectStore('runs').get(id));
}
export async function readEvents(id) {
  const db = await open(); const rows = await result(db.transaction('events').objectStore('events').getAll(IDBKeyRange.bound([id, 0], [id, Infinity])));
  return rows.map(row => row.event);
}
export async function saveRun(run, { event, expectedSeq } = {}) {
  const db = await open(); const tx = db.transaction(['runs', 'events'], 'readwrite'); const done = completed(tx);
  try {
    const store = tx.objectStore('runs'); const existing = await result(store.get(run.id));
    if (expectedSeq != null && existing?.seq !== expectedSeq) throw new Error('local_writer_conflict');
    if (event) tx.objectStore('events').add({ run: run.id, seq: run.seq, event });
    store.put(JSON.parse(JSON.stringify(run))); await done;
  } catch (error) { try { tx.abort(); } catch {} await done.catch(() => {}); throw error; }
}
export async function pendingArchives(userId) {
  const db = await open(); const runs = await result(db.transaction('runs').objectStore('runs').getAll());
  return runs.filter(r => r.userId === userId && needsReplayUpload(r));
}
export async function acquireSlot(name) {
  if (!navigator.locks) throw new Error('browser_lock_unavailable');
  let release;
  const held = new Promise(resolve => { release = resolve; });
  const acquired = new Promise((resolve, reject) => {
    navigator.locks.request(`human:${name}`, { ifAvailable: true }, async lock => {
      resolve(lock ? release : null); if (lock) await held;
    }).catch(reject);
  });
  return acquired;
}
