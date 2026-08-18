import { analyzeReplay } from './replayAnalysis.js';
import { ReplayFormatError, parseRplArrayBuffer } from './rplParser.js';

let parseSequence = 0;

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
    const worker = new Worker(new URL('./replayWorker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (event) => {
      if (event.data?.id !== id) return;
      worker.terminate();
      if (event.data.ok) {
        resolve(event.data);
        return;
      }
      const details = event.data?.error || {};
      reject(new ReplayFormatError(details.message, details.code));
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new ReplayFormatError(event.message || 'Replay parser worker failed.'));
    };
    worker.postMessage({ id, buffer, markerThreshold }, [buffer]);
  });
}
