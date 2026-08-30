import { saveReplayPosition, saveReplaySource } from './replaySessionStore.js';


export const REPLAY_TRANSFER_EVENT = '2048tables:replay-source-ready';

export function queueReplayTransfer(replay) {
  if (!(replay?.buffer instanceof ArrayBuffer) || replay.buffer.byteLength < 1) {
    return false;
  }
  const metadata = {
    filename: String(replay.filename || 'battle_replay.rpl'),
    source: String(replay.source || replay.filename || 'Battle'),
    pattern: String(replay.pattern || ''),
    useVariant: Boolean(replay.useVariant),
  };
  saveReplaySource(replay.buffer, metadata);
  saveReplayPosition(0);
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(REPLAY_TRANSFER_EVENT, {
      detail: { ...metadata, buffer: replay.buffer },
    }));
  }
  return true;
}
