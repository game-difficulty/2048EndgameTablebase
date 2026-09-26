import { initialState, VARIANTS, MAX_MOVES } from './engine.js';
import { encodeUleb128, crc32, bytesToBase64 } from '../features/gamer/engine/rankedReplayEncoder.js';

// Export recorded play, without identities or the seed for future spawns.
export function exportCurrentReplay(run, events) {
  if (!run || events.length !== run.seq || events.length > MAX_MOVES) throw new Error('invalid_local_replay');
  const [rows, cols] = VARIANTS[run.variant];
  const board = initialState(run.id, run.variant, run.seed).board;
  const bytes = [0x52, 0x50, 0x4c, 0x31, (rows << 4) | cols, 0, 2];
  board.forEach((tile, i) => { if (tile) bytes.push(i | (tile === 4 ? 16 : 0)); });
  for (const [code, delta] of events) {
    if (!Number.isInteger(code) || code < 0 || code >= 128 || !Number.isInteger(delta) || delta < 0 || delta > 0xffffffff) throw new Error('invalid_local_replay');
    // 0xffffffff is the RPL1 sentinel for an unknown duration. Older local
    // saves may contain it from the former clamp; keep them exportable as the
    // largest exact duration rather than silently decoding them as unknown.
    bytes.push(code, ...encodeUleb128(Math.min(delta, 0xfffffffe)));
  }
  bytes.push(132); // End of this snapshot; the original game may still be active.
  const checksum = crc32(bytes);
  bytes.push(checksum & 255, (checksum >>> 8) & 255, (checksum >>> 16) & 255, checksum >>> 24);
  const binary = Uint8Array.from(bytes);
  return { binary, text: 'REPLAY_v1RPL_B64_' + bytesToBase64(binary),
    filename: `2048-${run.variant}-${run.score}-${run.seq}.vrs`, variant: run.variant, moves: run.seq, score: run.score };
}

export function openReplayViewer(replay, language) {
  const token = crypto.randomUUID();
  const url = new URL('/verse-replay/', location.href);
  url.searchParams.set('lang', language); url.hash = `human=${token}`;
  return new Promise((resolve, reject) => {
    let popup, timeout;
    const clear = () => { window.removeEventListener('message', receive); clearTimeout(timeout); };
    const receive = event => {
      if (event.origin !== url.origin || event.source !== popup || event.data?.token !== token) return;
      if (event.data.type === 'human-replay-ready') {
        popup.postMessage({ type: 'human-replay-data', token, text: replay.text, filename: replay.filename }, url.origin);
      } else if (event.data.type === 'human-replay-loaded') { clear(); resolve(); }
      else if (event.data.type === 'human-replay-error') { clear(); reject(new Error('replay_transfer_failed')); }
    };
    window.addEventListener('message', receive);
    // Open before any await, while the click's popup permission is still active.
    popup = window.open(url.href, '_blank');
    if (!popup) { clear(); reject(new Error('popup_blocked')); return; }
    timeout = setTimeout(() => { clear(); reject(new Error('replay_transfer_failed')); }, 15000);
  });
}
