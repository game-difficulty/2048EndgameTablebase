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

    if (typeof window !== 'undefined' && window.location) {
      const { protocol, hostname, port } = window.location;
      const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
      const isFrontendDevServer = port === '5173';
      const targetPort = isFrontendDevServer ? '8000' : (port || '8000');
      return `${wsProtocol}//${hostname}:${targetPort}/ws/${clientId}`;
    }

    return `ws://127.0.0.1:8000/ws/${clientId}`;
  };

  const connect = () => {
    clearReconnectTimer();
    socket = new WebSocket(resolveUrl());

    socket.onopen = () => {
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
    if (socket) {
      socket.onclose = null;
      socket.close();
      socket = null;
    }
  };

  const sendRaw = (payload) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }
    socket.send(JSON.stringify(payload));
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
