import assert from 'node:assert/strict';
import { after, before, beforeEach, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createSSRApp, h, ref } from 'vue';
import { renderToString } from 'vue/server-renderer';
import { createServer } from 'vite';

let vite;
let useBattleRoomSession;
let api;
const originalWindow = globalThis.window;
const originalSocket = globalThis.WebSocket;

class FakeSocket {
  static OPEN = 1;
  static last;
  readyState = 1;
  constructor() { FakeSocket.last = this; }
  send() {}
  close() { this.readyState = 3; }
  message(data) { this.onmessage?.({ data: JSON.stringify({ action: 'BATTLE_ROOM_STATE', data }) }); }
}
const settle = () => new Promise(resolve => setImmediate(resolve));
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const snapshot = () => ({
  room_id: 'room-1', room_code: 'ABCDEF', status: 'waiting', revision: 1,
  viewer: { actor_key: 'u:2', user_id: 2, role: 'player' }, members: [], results: [],
});
const closed = code => ({ closed: true, room: null, room_id: 'room-1', code });

before(async () => {
  // Vite resolves the same extensionless browser imports as the application.
  vite = await createServer({
    root: fileURLToPath(new URL('..', import.meta.url)), configFile: false,
    server: { middlewareMode: true, hmr: false },
    optimizeDeps: { noDiscovery: true },
  });
  ({ useBattleRoomSession } = await vite.ssrLoadModule('/src/features/battle/core/useBattleRoomSession.js'));
  ({ battleClient: api } = await vite.ssrLoadModule('/src/features/battle/services/battleClient.js'));
});
after(async () => {
  await vite?.close();
  if (originalWindow === undefined) delete globalThis.window;
  else globalThis.window = originalWindow;
  globalThis.WebSocket = originalSocket;
});
beforeEach(() => {
  globalThis.window = Object.assign(new EventTarget(), {
    location: { href: 'http://localhost/' }, clearTimeout,
  });
  globalThis.WebSocket = FakeSocket;
  api.join = async () => ({ room: snapshot() });
  api.leave = async () => ({ left: true });
  api.rooms = async () => ({ rooms: [] });
  api.current = async () => ({ room: snapshot() });
});
async function joined() {
  let session;
  await renderToString(createSSRApp({ setup() {
    session = useBattleRoomSession(ref(false), ref({ id: 2, kind: 'user' }));
    return () => h('div');
  } }));
  await session.join('ABCDEF');
  return session;
}

for (const code of ['ROOM_MEMBERSHIP_REQUIRED', 'ROOM_CLOSED']) {
  test(`voluntary leave ignores ${code} arriving before HTTP success`, async () => {
    const session = await joined();
    const response = deferred();
    let calls = 0;
    api.leave = () => { calls += 1; return response.promise; };
    const leaving = session.leave();
    const duplicate = session.leave();
    assert.equal(leaving, duplicate);
    await settle();
    FakeSocket.last.message(closed(code));
    await settle();
    assert.equal(session.error.value, '');
    assert.equal(session.room.value, null);
    response.resolve({ left: true });
    assert.equal(await leaving, true);
    assert.equal(calls, 1);
    assert.equal(session.error.value, '');
  });
}
test('late room notifications and a stale current request cannot restore a departed room', async () => {
  const session = await joined();
  const current = deferred();
  api.current = () => current.promise;
  const refreshing = session.refreshCurrent();
  const socket = FakeSocket.last;
  assert.equal(await session.leave(), true);
  socket.message(closed('ROOM_MEMBERSHIP_REQUIRED'));
  socket.message({ room: snapshot() });
  current.resolve({ room: snapshot() });
  await refreshing;
  await settle();
  assert.equal(session.room.value, null);
  assert.equal(session.error.value, '');
});
test('lobby refresh failure does not turn a successful leave into a failure', async () => {
  const session = await joined();
  api.rooms = async () => { throw new Error('offline'); };
  assert.equal(await session.leave(), true);
  assert.equal(session.error.value, '');
  assert.equal(session.room.value, null);
});
test('a failed leave retains the room and reports the real error', async () => {
  const session = await joined();
  api.leave = async () => { throw Object.assign(new Error('failed'), { code: 'LEAVE_FAILED' }); };
  assert.equal(await session.leave(), false);
  assert.equal(session.error.value, 'LEAVE_FAILED');
  assert.equal(session.room.value.room_id, 'room-1');
  api.leave = async () => ({ left: true });
  assert.equal(await session.leave(), true);
  assert.equal(session.error.value, '');
});
for (const code of ['KICKED_FROM_ROOM', 'ROOM_CLOSED', 'ROOM_MEMBERSHIP_REQUIRED']) {
  test(`involuntary ${code} is still reported`, async () => {
    const session = await joined();
    FakeSocket.last.message(closed(code));
    await settle();
    assert.equal(session.error.value, code);
    assert.equal(session.room.value, null);
  });
}
