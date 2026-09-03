import assert from 'node:assert/strict';
import test from 'node:test';

import {
  battleActorKey,
  battleActorKind,
  buildBattleKickPayload,
  isBattleGuest,
  normalizeBattleActor,
  sameBattleActor,
} from '../src/features/battle/core/battleActor.js';
import { createBattleChatState, viewerCanChat } from '../src/features/battle/core/chatState.js';

test('battle actors keep registered users and guests in separate stable namespaces', () => {
  const user = normalizeBattleActor({ id: 7, display_name: 'User Seven' });
  const guest = normalizeBattleActor({
    kind: 'guest',
    guest_id: '7',
    display_name: '游客-A7K2',
  });

  assert.equal(user.actor_key, 'u:7');
  assert.equal(guest.actor_key, 'g:7');
  assert.equal(battleActorKind(user), 'user');
  assert.equal(isBattleGuest(guest), true);
  assert.equal(sameBattleActor(user, guest), false);
  assert.equal(sameBattleActor(guest, { actor_key: 'g:7' }), true);
});

test('guest chat requires both room opt-in and an allowed room role', () => {
  const guestPlayer = { actor_key: 'g:abc', actor_kind: 'guest', role: 'player' };
  assert.equal(viewerCanChat({ chat_roles: ['player'] }, guestPlayer), false);
  assert.equal(viewerCanChat({ allow_guest_chat: true, chat_roles: ['player'] }, guestPlayer), true);
  assert.equal(viewerCanChat({ allow_guest_chat: true, chat_roles: ['spectator'] }, guestPlayer), false);
  assert.equal(viewerCanChat({ chat_roles: ['player'] }, { actor_key: 'u:1', role: 'player' }), true);
});

test('guest chat rejections are scoped by actor key rather than nullable user id', () => {
  const state = createBattleChatState();
  const viewer = { actor_key: 'g:mine', actor_kind: 'guest' };

  state.handleWsMessage({
    action: 'BATTLE_CHAT_REJECTED',
    data: { actor_key: 'g:other', code: 'CHAT_RATE_LIMITED' },
  }, viewer);
  assert.equal(state.notice.value, null);

  state.handleWsMessage({
    action: 'BATTLE_CHAT_REJECTED',
    data: { actor_key: 'g:mine', code: 'CHAT_GUEST_NOT_ALLOWED' },
  }, viewer);
  assert.equal(state.notice.value.code, 'CHAT_GUEST_NOT_ALLOWED');
  state.dispose();
});

test('host kick requests address guests by actor key and preserve legacy user ids', () => {
  assert.deepEqual(
    buildBattleKickPayload({ actor_key: 'g:guest-1', actor_kind: 'guest' }, 'kick-1'),
    { actor_key: 'g:guest-1', request_id: 'kick-1' },
  );
  assert.deepEqual(
    buildBattleKickPayload({ user_id: 42 }, 'kick-2'),
    { actor_key: 'u:42', user_id: 42, request_id: 'kick-2' },
  );
  assert.equal(battleActorKey({ guest_id: 'guest-2' }), 'g:guest-2');
});
