export function emitAuthRequired() {
  window.dispatchEvent(new CustomEvent('auth-required'));
}

export function emitTokenRequired(detail = {}) {
  window.dispatchEvent(new CustomEvent('token-required', { detail }));
}

export function emitAuthChanged() {
  window.dispatchEvent(new CustomEvent('auth-changed'));
}
