import { getBackendWebSocketUrl } from '../runtime/backendUrl';
import { emitAuthRequired, emitTokenBalanceUpdated, emitTokenRequired } from '../auth/authEvents';
import { clearDeviceSession, getDeviceSessionToken } from '../auth/sessionTokenStore';
import { clearGuestSession, getGuestSessionToken } from '../auth/guestSessionStore';

export function createWsClient({
  clientId,
  createUrl,
  reconnectDelay = 1000,
  onOpen,
  onMessage,
  onClose,
  onError,
} = {}) {
  let socket = null;
  let reconnectTimer = null;
  let shouldReconnect = true;
  let authTokenSent = '';
  const pendingPayloads = [];

  const clearReconnectTimer = () => {
    if (reconnectTimer !== null) {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  };

  const resolveUrl = () => {
    if (typeof createUrl === 'function') {
      return createUrl(clientId);
    }

    return getBackendWebSocketUrl(clientId);
  };

  const connect = () => {
    clearReconnectTimer();
    shouldReconnect = true;
    socket = new WebSocket(resolveUrl());

    socket.onopen = () => {
      const token = getDeviceSessionToken();
      if (token && socket?.readyState === WebSocket.OPEN) {
        authTokenSent = token;
        socket.send(JSON.stringify({
          action: 'AUTH_SESSION',
          data: { token },
        }));
      } else {
        const guestToken = getGuestSessionToken();
        if (guestToken && socket?.readyState === WebSocket.OPEN) {
          authTokenSent = guestToken;
          socket.send(JSON.stringify({
            action: 'AUTH_GUEST_SESSION',
            data: { token: guestToken },
          }));
        }
      }
      while (pendingPayloads.length > 0 && socket?.readyState === WebSocket.OPEN) {
        socket.send(pendingPayloads.shift());
      }
      onOpen?.(socket);
    };

    socket.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch (error) {
        onError?.(error, event);
        return;
      }
      if (message?.action === 'AUTH_REQUIRED' || message?.data?.code === 'AUTH_REQUIRED') {
        if (authTokenSent && getDeviceSessionToken() === authTokenSent) {
          clearDeviceSession();
        }
        emitAuthRequired();
      }
      if (message?.action === 'GUEST_SESSION_REQUIRED' || message?.data?.code === 'GUEST_SESSION_REQUIRED') {
        if (authTokenSent && getGuestSessionToken() === authTokenSent) {
          clearGuestSession();
          window.dispatchEvent(new CustomEvent('guest-session-invalidated'));
        }
      }
      if (message?.action === 'TOKEN_REQUIRED' || message?.data?.code === 'INSUFFICIENT_TOKENS') {
        emitTokenRequired(message?.data || {});
      }
      if (message?.data?.token_balance) {
        emitTokenBalanceUpdated(message.data.token_balance);
      }
      onMessage?.(message, event, socket);
    };

    socket.onclose = (event) => {
      socket = null;
      authTokenSent = '';
      onClose?.(event);
      if (shouldReconnect) {
        reconnectTimer = window.setTimeout(connect, reconnectDelay);
      }
    };

    socket.onerror = (event) => {
      onError?.(event);
    };
  };

  const disconnect = () => {
    shouldReconnect = false;
    clearReconnectTimer();
    window.removeEventListener('auth-changed', handleAuthChanged);
    pendingPayloads.length = 0;
    if (socket) {
      socket.onclose = null;
      socket.close();
      socket = null;
    }
  };

  const sendRaw = (payload) => {
    const serialized = typeof payload === 'string' ? payload : JSON.stringify(payload);
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      pendingPayloads.push(serialized);
      return true;
    }
    socket.send(serialized);
    return true;
  };

  const send = (action, data = undefined) => {
    const payload = data === undefined ? { action } : { action, data };
    return sendRaw(payload);
  };

  const getSocket = () => socket;

  const handleAuthChanged = () => {
    if (!shouldReconnect || !socket) {
      return;
    }
    socket.close();
  };

  window.addEventListener('auth-changed', handleAuthChanged);

  return {
    connect,
    disconnect,
    send,
    sendRaw,
    getSocket,
  };
}
