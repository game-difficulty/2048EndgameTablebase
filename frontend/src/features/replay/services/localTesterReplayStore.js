const STORAGE_KEY = '2048tables:tester-latest-replay:v1';

const storageOrNull = () => {
  try {
    return typeof window !== 'undefined' ? window.sessionStorage : null;
  } catch (_error) {
    return null;
  }
};

const bufferToBase64 = (buffer) => {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return window.btoa(binary);
};

const base64ToBuffer = (encoded) => {
  const binary = window.atob(String(encoded || ''));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes.buffer;
};

export function saveLocalTesterReplay({ buffer, filename, pattern, useVariant }) {
  const storage = storageOrNull();
  if (!storage || !(buffer instanceof ArrayBuffer) || buffer.byteLength < 1) return false;
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify({
      version: 1,
      data: bufferToBase64(buffer),
      filename: String(filename || 'tester_latest.rpl'),
      pattern: String(pattern || ''),
      useVariant: Boolean(useVariant),
      savedAt: Date.now(),
    }));
    return true;
  } catch (_error) {
    return false;
  }
}

export function restoreLocalTesterReplay() {
  const storage = storageOrNull();
  if (!storage) return null;
  try {
    const record = JSON.parse(storage.getItem(STORAGE_KEY) || 'null');
    if (record?.version !== 1 || typeof record.data !== 'string') return null;
    const buffer = base64ToBuffer(record.data);
    if (buffer.byteLength < 1) return null;
    return {
      buffer,
      filename: String(record.filename || 'tester_latest.rpl'),
      pattern: String(record.pattern || ''),
      source: String(record.filename || 'Tester session'),
      useVariant: Boolean(record.useVariant),
    };
  } catch (_error) {
    return null;
  }
}
