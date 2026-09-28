import { evilSpawnRuntime } from './evilSpawnRuntime.js';

let worker = null;
let workerUnavailable = false;
let nextRequestId = 1;
const pending = new Map();

function disableWorker(cause) {
  workerUnavailable = true;
  worker?.terminate();
  worker = null;
  const error = cause instanceof Error ? cause : new Error('EvilGen worker stopped unexpectedly.');
  for (const request of pending.values()) request.reject(error);
  pending.clear();
}

function getWorker() {
  if (workerUnavailable || typeof Worker !== 'function') return null;
  if (worker) return worker;
  try {
    worker = new Worker(new URL('./evilSpawn.worker.js', import.meta.url), { type: 'module' });
    worker.addEventListener('message', ({ data }) => {
      const request = pending.get(data?.id);
      if (!request) return;
      pending.delete(data.id);
      if (data.error) request.reject(new Error(data.error));
      else request.resolve(data.result);
    });
    worker.addEventListener('error', event => disableWorker(event.error));
    worker.addEventListener('messageerror', () => disableWorker(new Error('EvilGen worker returned an unreadable result.')));
    return worker;
  } catch (_error) {
    workerUnavailable = true;
    return null;
  }
}

export function evilSpawn(board, depth, tieSeed) {
  const activeWorker = getWorker();
  if (!activeWorker) return evilSpawnRuntime(board, depth, tieSeed);
  const id = nextRequestId;
  nextRequestId += 1;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    activeWorker.postMessage({ id, board: Array.from(board), depth, tieSeed });
  });
}
