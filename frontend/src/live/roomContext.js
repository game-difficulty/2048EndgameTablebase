import { inject, provide } from 'vue';
const roomKey = Symbol('live-room');

export async function requestJson(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(path, {
      signal: controller.signal, credentials: 'same-origin', cache: 'no-store',
      ...(body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw Object.assign(Error('request_failed'), { status: response.status, detail: payload.detail });
    }
    return await response.json();
  } finally { clearTimeout(timeout); }
}

export function provideRoom(room) {
  const context = Object.freeze({ room, api: (path, body) => requestJson(
    path.startsWith('/api/') ? path : room.api_base + path, body), url: path => room.api_base + path });
  provide(roomKey, context);
  return context;
}
export function useRoom() {
  const room = inject(roomKey);
  if (!room) throw new Error('A room provider is required');
  return room;
}
