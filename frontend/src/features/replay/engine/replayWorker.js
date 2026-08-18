import { analyzeReplay } from './replayAnalysis.js';
import { parseRplArrayBuffer } from './rplParser.js';

self.onmessage = (event) => {
  const { id, buffer, markerThreshold } = event.data || {};
  try {
    const replay = parseRplArrayBuffer(buffer);
    const analysis = analyzeReplay(replay, markerThreshold);
    self.postMessage(
      { id, ok: true, replay, analysis, rawBuffer: buffer },
      [
        buffer,
        replay.boards.buffer,
        replay.changes.buffer,
        replay.rates.buffer,
        analysis.losses.buffer,
        analysis.goodnessOfFit.buffer,
        analysis.combo.buffer,
        analysis.forced.buffer,
        analysis.pointsRank.buffer,
      ],
    );
  } catch (error) {
    self.postMessage({
      id,
      ok: false,
      error: {
        name: error?.name || 'Error',
        message: error?.message || 'Failed to parse replay.',
        code: error?.code || 'INVALID_REPLAY',
      },
    });
  }
};
