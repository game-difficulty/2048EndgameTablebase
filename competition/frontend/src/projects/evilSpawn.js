import { evilSpawnRuntime } from './evilSpawnRuntime.js';

let worker = null;
let workerUnavailable = false;
let nextRequestId = 1;
const pending = new Map();

function disableWorker(cause) {
  worker?.terminate();
  worker = null;
  const error = cause instanceof Error ? cause : new Error('EvilGen worker stopped unexpectedly.');
  for (const request of pending.values()) { clearTimeout(request.timeout); request.reject(error); }
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
      clearTimeout(request.timeout);
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
    // A stalled WASM load/worker must not hold the player's input lock forever.
    // Rejecting restores the pre-move state; the next move creates a new worker.
    const timeout = setTimeout(() => disableWorker(new Error('AI 计算暂时未响应，请重新操作。')), 20000);
    pending.set(id, { resolve, reject, timeout });
    try { activeWorker.postMessage({ id, board: Array.from(board), depth, tieSeed }); }
    catch (error) { disableWorker(error); }
  });
}
