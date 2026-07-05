import { getBackendUrl } from '../runtime/backendUrl';
import { emitAuthRequired, emitTokenBalanceUpdated, emitTokenRequired } from '../auth/authEvents';

async function handleProtectedResponseError(response, fallbackPrefix) {
  if (!response) {
    throw new Error(`${fallbackPrefix}: unknown`);
  }
  if (response?.status === 401) {
    emitAuthRequired();
  }
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch (_error) {
    payload = null;
  }
  if (response?.status === 402 && payload?.detail?.code === 'INSUFFICIENT_TOKENS') {
    emitTokenRequired(payload.detail);
  }
  const detail = typeof payload?.detail === 'string'
    ? payload.detail
    : payload?.detail?.message || text;
  throw new Error(`${fallbackPrefix}: ${response?.status || 'unknown'} ${detail || ''}`.trim());
}

const EMPTY_FILES = Object.freeze([]);

function createInput({ accept = '', multiple = false } = {}) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = accept;
  input.multiple = multiple;
  input.style.position = 'fixed';
  input.style.left = '-9999px';
  input.style.top = '-9999px';
  return input;
}

export function pickBrowserFiles({ accept = '', multiple = false } = {}) {
  if (typeof document === 'undefined') {
    return Promise.resolve(EMPTY_FILES);
  }

  return new Promise((resolve) => {
    const input = createInput({ accept, multiple });
    const cleanup = () => {
      input.removeEventListener('change', handleChange);
      document.body.removeChild(input);
    };
    const handleChange = () => {
      const files = input.files ? Array.from(input.files) : EMPTY_FILES;
      cleanup();
      resolve(files);
    };
    input.addEventListener('change', handleChange);
    document.body.appendChild(input);
    input.click();
  });
}

export async function tryPickBrowserFiles(options = {}) {
  const files = await pickBrowserFiles(options);
  return {
    handled: true,
    canceled: files.length === 0,
    files,
  };
}

export async function pickSingleBrowserFile(options = {}) {
  const files = await pickBrowserFiles({ ...options, multiple: false });
  return files[0] || null;
}

export function readFileAsText(file, encoding = 'utf-8') {
  if (!file) {
    return Promise.resolve('');
  }
  return file.text ? file.text() : new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file, encoding);
  });
}

export function readFileAsArrayBuffer(file) {
  if (!file) {
    return Promise.resolve(new ArrayBuffer(0));
  }
  return file.arrayBuffer ? file.arrayBuffer() : new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

export async function readFileAsJson(file) {
  const text = await readFileAsText(file);
  return JSON.parse(text);
}

export function downloadBlob(blob, filename = 'download.bin') {
  if (typeof document === 'undefined') {
    return false;
  }
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = 'noopener';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
  return true;
}

export function downloadText(text, filename, mimeType = 'text/plain;charset=utf-8') {
  return downloadBlob(new Blob([text], { type: mimeType }), filename);
}

export function downloadJson(data, filename) {
  const text = JSON.stringify(data, null, 2);
  return downloadText(text, filename, 'application/json;charset=utf-8');
}

export async function downloadResponse(response, filename = '') {
  if (!response?.ok) {
    await handleProtectedResponseError(response, 'Download failed');
  }
  const blob = await response.blob();
  const header = response.headers?.get('content-disposition') || '';
  const headerMatch = /filename="?([^";]+)"?/i.exec(header);
  const resolvedFilename = filename || headerMatch?.[1] || 'download.bin';
  return downloadBlob(blob, resolvedFilename);
}

export async function uploadBrowserFiles(files, { kind = 'generic', fields = {} } = {}) {
  const fileList = Array.isArray(files) ? files : [files].filter(Boolean);
  if (!fileList.length) {
    return { uploads: [] };
  }
  const formData = new FormData();
  formData.append('kind', kind);
  for (const [key, value] of Object.entries(fields || {})) {
    if (value !== undefined && value !== null) {
      formData.append(key, String(value));
    }
  }
  for (const file of fileList) {
    formData.append('files', file, file.name || 'upload.bin');
  }
  const response = await fetch(getBackendUrl('/api/uploads'), {
    method: 'POST',
    body: formData,
    credentials: 'include',
  });
  if (!response.ok) {
    await handleProtectedResponseError(response, 'Upload failed');
  }
  const payload = await response.json();
  if (payload?.token_balance) {
    emitTokenBalanceUpdated(payload.token_balance);
  }
  return payload;
}

export async function postMultipart(url, { files = [], fields = {} } = {}) {
  const formData = new FormData();
  for (const [key, value] of Object.entries(fields || {})) {
    if (value !== undefined && value !== null) {
      formData.append(key, String(value));
    }
  }
  for (const file of files) {
    formData.append('files', file, file.name || 'upload.bin');
  }
  const response = await fetch(getBackendUrl(url), {
    method: 'POST',
    body: formData,
    credentials: 'include',
  });
  if (!response.ok) {
    await handleProtectedResponseError(response, 'Request failed');
  }
  const payload = await response.json();
  if (payload?.token_balance) {
    emitTokenBalanceUpdated(payload.token_balance);
  }
  return payload;
}
