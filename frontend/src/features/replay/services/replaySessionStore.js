const SOURCE_KEY = '2048tables:replay-source:v1';
const POSITION_KEY = '2048tables:replay-position:v1';

function sessionStore() {
  try {
    return typeof window !== 'undefined' ? window.sessionStorage : null;
  } catch (_error) {
    return null;
  }
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function base64ToArrayBuffer(value) {
  const binary = atob(String(value || ''));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes.buffer;
}

export function saveReplaySource(buffer, metadata = {}) {
  const store = sessionStore();
  if (!store) return false;
  try {
    store.setItem(SOURCE_KEY, JSON.stringify({
      version: 1,
      bytes: arrayBufferToBase64(buffer),
      filename: String(metadata.filename || ''),
      pattern: String(metadata.pattern || ''),
      source: String(metadata.source || metadata.filename || ''),
      useVariant: !!metadata.useVariant,
    }));
    return true;
  } catch (_error) {
    return false;
  }
}

export function saveReplayPosition(step) {
  const store = sessionStore();
  if (!store) return false;
  try {
    store.setItem(POSITION_KEY, String(Math.max(0, Number(step) || 0)));
    return true;
  } catch (_error) {
    return false;
  }
}

export function restoreReplaySession() {
  const store = sessionStore();
  if (!store) return null;
  try {
    const source = JSON.parse(store.getItem(SOURCE_KEY) || 'null');
    if (source?.version !== 1 || !source.bytes) return null;
    return {
      buffer: base64ToArrayBuffer(source.bytes),
      filename: String(source.filename || ''),
      pattern: String(source.pattern || ''),
      source: String(source.source || source.filename || ''),
      useVariant: !!source.useVariant,
      step: Math.max(0, Number(store.getItem(POSITION_KEY)) || 0),
    };
  } catch (_error) {
    return null;
  }
}
