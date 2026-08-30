import { ref } from 'vue';

export const BATTLE_CHAT_MAX_MESSAGES = 50;
export const BATTLE_CHAT_MAX_CODE_POINTS = 20;

const CONTROL_CHARACTER_PATTERN = /[\u0000-\u001f\u007f]/u;

export function chatCodePointLength(value) {
  return Array.from(String(value ?? '')).length;
}

export function truncateChatContent(value, limit = BATTLE_CHAT_MAX_CODE_POINTS) {
  return Array.from(String(value ?? '')).slice(0, Math.max(0, Number(limit) || 0)).join('');
}

export function validateChatContent(value) {
  const content = String(value ?? '').trim();
  if (!content) return { ok: false, code: 'CHAT_EMPTY', content: '' };
  if (CONTROL_CHARACTER_PATTERN.test(content)) {
    return { ok: false, code: 'CHAT_INVALID_CONTENT', content };
  }
  if (chatCodePointLength(content) > BATTLE_CHAT_MAX_CODE_POINTS) {
    return { ok: false, code: 'CHAT_TOO_LONG', content };
  }
  return { ok: true, code: '', content };
}

export function viewerCanChat(room, viewer) {
  if (!room || !viewer) return false;
  const allowedRoles = Array.isArray(room.chat_roles)
    ? room.chat_roles
    : ['host', 'player', 'spectator'];
  const role = viewer.is_host ? 'host' : viewer.role;
  return allowedRoles.includes(role);
}

function normalizeMessage(candidate) {
  const messageId = candidate?.message_id;
  if (messageId === undefined || messageId === null || messageId === '') return null;
  const content = String(candidate?.content ?? '');
  if (!content) return null;
  return {
    ...candidate,
    message_id: messageId,
    user_id: candidate?.user_id ?? null,
    display_name: String(
      candidate?.display_name
      || candidate?.username
      || candidate?.user_name
      || '',
    ),
    content,
    created_at: String(candidate?.created_at || ''),
  };
}

function sameUser(left, right) {
  if (left === undefined || left === null || right === undefined || right === null) return true;
  return String(left) === String(right);
}

function messageBelongsToRoom(data, roomIdentity) {
  if (!roomIdentity) return true;
  const eventRoomId = data?.room_id ?? data?.message?.room_id;
  const eventRoomCode = data?.room_code ?? data?.message?.room_code;
  if (eventRoomId !== undefined && eventRoomId !== null) {
    return String(eventRoomId) === String(roomIdentity.room_id || '');
  }
  if (eventRoomCode !== undefined && eventRoomCode !== null) {
    return String(eventRoomCode).toUpperCase() === String(roomIdentity.room_code || '').toUpperCase();
  }
  return true;
}

export function createBattleChatState({ now = () => Date.now() } = {}) {
  const messages = ref([]);
  const notice = ref(null);
  const cooldownSeconds = ref(0);
  let cooldownDeadline = 0;
  let cooldownTimer = null;

  const stopCooldownTimer = () => {
    if (cooldownTimer !== null) {
      globalThis.clearInterval(cooldownTimer);
      cooldownTimer = null;
    }
  };

  const tickCooldown = () => {
    if (!cooldownDeadline) {
      cooldownSeconds.value = 0;
      stopCooldownTimer();
      return;
    }
    cooldownSeconds.value = Math.max(0, Math.ceil((cooldownDeadline - now()) / 1000));
    if (cooldownSeconds.value === 0) {
      cooldownDeadline = 0;
      stopCooldownTimer();
      if (notice.value?.code === 'CHAT_RATE_LIMITED') notice.value = null;
    }
  };

  const setNotice = (code, data = {}) => {
    notice.value = { code: String(code || 'CHAT_REJECTED'), ...data };
  };
  const clearNotice = () => {
    if (notice.value?.code !== 'CHAT_RATE_LIMITED') notice.value = null;
  };

  const replaceMessages = (candidates) => {
    const byId = new Map();
    for (const candidate of Array.isArray(candidates) ? candidates : []) {
      const normalized = normalizeMessage(candidate);
      if (normalized) byId.set(String(normalized.message_id), normalized);
    }
    messages.value = Array.from(byId.values()).slice(-BATTLE_CHAT_MAX_MESSAGES);
  };

  const mergeHistory = (candidates) => {
    const byId = new Map();
    for (const candidate of [
      ...(Array.isArray(candidates) ? candidates : []),
      ...messages.value,
    ]) {
      const normalized = normalizeMessage(candidate);
      if (normalized) byId.set(String(normalized.message_id), normalized);
    }
    messages.value = Array.from(byId.values())
      .sort((left, right) => Number(left.message_id) - Number(right.message_id))
      .slice(-BATTLE_CHAT_MAX_MESSAGES);
  };

  const appendMessage = (candidate) => {
    const normalized = normalizeMessage(candidate);
    if (!normalized) return false;
    const messageId = String(normalized.message_id);
    if (messages.value.some((message) => String(message.message_id) === messageId)) return false;
    messages.value = [...messages.value, normalized].slice(-BATTLE_CHAT_MAX_MESSAGES);
    return true;
  };

  const setRateLimit = (data = {}) => {
    const retryAfter = Math.max(1, Math.ceil(Number(
      data.retry_after_seconds ?? data.retry_after ?? 1,
    ) || 1));
    cooldownDeadline = now() + retryAfter * 1000;
    setNotice('CHAT_RATE_LIMITED', { retry_after_seconds: retryAfter });
    tickCooldown();
    if (cooldownTimer === null) cooldownTimer = globalThis.setInterval(tickCooldown, 250);
  };

  const reject = (data = {}) => {
    setNotice(data.code || 'CHAT_REJECTED', data);
  };

  const handleWsMessage = (message, viewerUserId = null, roomIdentity = null) => {
    const action = String(message?.action || '');
    const data = message?.data || {};
    if (action.startsWith('BATTLE_CHAT_') && !messageBelongsToRoom(data, roomIdentity)) {
      return true;
    }
    if (action === 'BATTLE_CHAT_HISTORY') {
      mergeHistory(data.messages);
      return true;
    }
    if (action === 'BATTLE_CHAT_MESSAGE') {
      appendMessage(data.message || data);
      return true;
    }
    if (action === 'BATTLE_CHAT_RATE_LIMITED') {
      if (sameUser(data.user_id, viewerUserId)) setRateLimit(data);
      return true;
    }
    if (action === 'BATTLE_CHAT_REJECTED') {
      if (sameUser(data.user_id, viewerUserId)) reject(data);
      return true;
    }
    return false;
  };

  const clear = () => {
    messages.value = [];
    notice.value = null;
    cooldownDeadline = 0;
    cooldownSeconds.value = 0;
    stopCooldownTimer();
  };

  return {
    messages,
    notice,
    cooldownSeconds,
    replaceMessages,
    appendMessage,
    setRateLimit,
    reject,
    clearNotice,
    handleWsMessage,
    tickCooldown,
    clear,
    dispose: clear,
  };
}
