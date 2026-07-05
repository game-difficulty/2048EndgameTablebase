export function emitAuthRequired() {
  window.dispatchEvent(new CustomEvent('auth-required'));
}

export function emitTokenRequired(detail = {}) {
  window.dispatchEvent(new CustomEvent('token-required', { detail }));
}

export function emitTokenBalanceUpdated(tokenBalance = null) {
  if (tokenBalance) {
    window.dispatchEvent(new CustomEvent('token-balance-updated', { detail: tokenBalance }));
  }
}

export function emitAuthChanged() {
  window.dispatchEvent(new CustomEvent('auth-changed'));
}
