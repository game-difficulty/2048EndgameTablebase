import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BATTLE_CHAT_MAX_MESSAGES,
  chatCodePointLength,
  createBattleChatState,
  truncateChatContent,
  validateChatContent,
  viewerCanChat,
} from '../src/features/battle/core/chatState.js';

test('chat length uses Unicode code points and validates outgoing text', () => {
  assert.equal(chatCodePointLength('你好😀'), 3);
  assert.equal(truncateChatContent('😀'.repeat(21)), '😀'.repeat(20));
  assert.deepEqual(validateChatContent('  hello  '), {
    ok: true,
    code: '',
    content: 'hello',
  });
  assert.equal(validateChatContent(' ').code, 'CHAT_EMPTY');
  assert.equal(validateChatContent('x'.repeat(21)).code, 'CHAT_TOO_LONG');
  assert.equal(validateChatContent('hello\nworld').code, 'CHAT_INVALID_CONTENT');
});

test('room chat roles distinguish the host from players and spectators', () => {
  const room = { chat_roles: ['host', 'spectator'] };
  assert.equal(viewerCanChat(room, { is_host: true, role: 'player' }), true);
  assert.equal(viewerCanChat(room, { is_host: false, role: 'player' }), false);
  assert.equal(viewerCanChat(room, { is_host: false, role: 'spectator' }), true);
  assert.equal(viewerCanChat({ chat_roles: [] }, { is_host: true, role: 'player' }), false);
  assert.equal(viewerCanChat({}, { is_host: false, role: 'player' }), true);
});

test('history and deltas deduplicate by message id and retain only the latest 50', () => {
  const state = createBattleChatState();
  const history = Array.from({ length: 52 }, (_, index) => ({
    message_id: index + 1,
    user_id: 1,
    display_name: 'User',
    content: `message ${index + 1}`,
    created_at: '2026-08-30T12:00:00Z',
  }));
  history.push({ ...history[51], content: 'deduplicated' });

  state.handleWsMessage({ action: 'BATTLE_CHAT_HISTORY', data: { messages: history } });
  assert.equal(state.messages.value.length, BATTLE_CHAT_MAX_MESSAGES);
  assert.equal(state.messages.value[0].message_id, 3);
  assert.equal(state.messages.value.at(-1).content, 'deduplicated');

  state.handleWsMessage({ action: 'BATTLE_CHAT_MESSAGE', data: { message: history[51] } });
  assert.equal(state.messages.value.length, BATTLE_CHAT_MAX_MESSAGES);
  state.handleWsMessage({
    action: 'BATTLE_CHAT_MESSAGE',
    data: { message: { ...history[51], message_id: 53, content: 'new' } },
  });
  assert.equal(state.messages.value[0].message_id, 4);
  assert.equal(state.messages.value.at(-1).message_id, 53);
  state.dispose();
});

test('late history does not overwrite a newer reconnect delta', () => {
  const state = createBattleChatState();
  const room = { room_id: 'room-1', room_code: 'ABC123' };
  const makeMessage = (messageId, content) => ({
    message_id: messageId,
    room_id: 'room-1',
    user_id: 1,
    display_name: 'User',
    content,
    created_at: '2026-08-30T12:00:00Z',
  });

  state.handleWsMessage({
    action: 'BATTLE_CHAT_MESSAGE',
    data: { message: makeMessage(3, 'new delta') },
  }, 1, room);
  state.handleWsMessage({
    action: 'BATTLE_CHAT_HISTORY',
    data: {
      room_id: 'room-1',
      messages: [makeMessage(1, 'old'), makeMessage(2, 'recent')],
    },
  }, 1, room);

  assert.deepEqual(
    state.messages.value.map((entry) => entry.content),
    ['old', 'recent', 'new delta'],
  );
  state.dispose();
});

test('rate-limit and rejection notices are only applied to the addressed viewer', () => {
  let clock = 1_000;
  const state = createBattleChatState({ now: () => clock });

  state.handleWsMessage({
    action: 'BATTLE_CHAT_RATE_LIMITED',
    data: { user_id: 2, retry_after_seconds: 12 },
  }, 1);
  assert.equal(state.notice.value, null);

  state.handleWsMessage({
    action: 'BATTLE_CHAT_RATE_LIMITED',
    data: { user_id: 1, retry_after_seconds: 12 },
  }, 1);
  assert.equal(state.notice.value.code, 'CHAT_RATE_LIMITED');
  assert.equal(state.cooldownSeconds.value, 12);

  clock += 12_000;
  state.tickCooldown();
  assert.equal(state.cooldownSeconds.value, 0);
  assert.equal(state.notice.value, null);

  state.handleWsMessage({
    action: 'BATTLE_CHAT_REJECTED',
    data: { user_id: 1, code: 'CHAT_TOO_LONG' },
  }, 1);
  assert.equal(state.notice.value.code, 'CHAT_TOO_LONG');
  state.clear();
  assert.deepEqual(state.messages.value, []);
  assert.equal(state.notice.value, null);
});

test('chat events from a previous room are consumed without changing current room state', () => {
  const state = createBattleChatState();
  const handled = state.handleWsMessage({
    action: 'BATTLE_CHAT_MESSAGE',
    data: {
      room_code: 'OLD123',
      message: {
        message_id: 1,
        user_id: 1,
        content: 'stale',
        created_at: '2026-08-30T12:00:00Z',
      },
    },
  }, 1, { room_id: 'room-new', room_code: 'NEW123' });
  assert.equal(handled, true);
  assert.deepEqual(state.messages.value, []);
  state.dispose();
});
