import { analyzeReplay } from './replayAnalysis.js';
import { ReplayFormatError, parseRplArrayBuffer } from './rplParser.js';

let parseSequence = 0;
let sharedWorker = null;
const pendingParses = new Map();

function parseSynchronously(buffer, markerThreshold) {
  const replay = parseRplArrayBuffer(buffer);
  return {
    replay,
    analysis: analyzeReplay(replay, markerThreshold),
    rawBuffer: buffer,
  };
}

export function parseReplayAsync(buffer, markerThreshold = 1) {
  if (typeof Worker === 'undefined') {
    return Promise.resolve(parseSynchronously(buffer, markerThreshold));
  }
  const id = ++parseSequence;
  return new Promise((resolve, reject) => {
    try {
      if (!sharedWorker) {
        sharedWorker = new Worker(new URL('./replayWorker.js', import.meta.url), { type: 'module' });
        sharedWorker.onmessage = (event) => {
          const pending = pendingParses.get(event.data?.id);
          if (!pending) return;
          pendingParses.delete(event.data.id);
          if (event.data.ok) {
            pending.resolve(event.data);
            return;
          }
          const details = event.data?.error || {};
          pending.reject(new ReplayFormatError(details.message, details.code));
        };
        sharedWorker.onerror = (event) => {
          const error = new ReplayFormatError(event.message || 'Replay parser worker failed.');
          for (const pending of pendingParses.values()) pending.reject(error);
          pendingParses.clear();
          sharedWorker?.terminate();
          sharedWorker = null;
        };
      }
      pendingParses.set(id, { resolve, reject });
      sharedWorker.postMessage({ id, buffer, markerThreshold }, [buffer]);
    } catch (_error) {
      pendingParses.delete(id);
      try {
        resolve(parseSynchronously(buffer, markerThreshold));
      } catch (error) {
        reject(error);
      }
    }
  });
}
