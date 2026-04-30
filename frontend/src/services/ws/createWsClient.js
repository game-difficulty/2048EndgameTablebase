import { getBackendWebSocketUrl } from '../runtime/backendUrl';

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
    socket = new WebSocket(resolveUrl());

    socket.onopen = () => {
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
      onMessage?.(message, event, socket);
    };

    socket.onclose = (event) => {
      socket = null;
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

  return {
    connect,
    disconnect,
    send,
    sendRaw,
    getSocket,
  };
}
