import {
  computed,
  onMounted,
  onUnmounted,
  ref,
  watch,
} from 'vue';

import { emitAuthRequired } from '../../../services/auth/authEvents.js';
import { createWsClient } from '../../../services/ws/createWsClient.js';
import { battleClient, battleRequestId } from '../services/battleClient.js';
import { createBattleChatState, validateChatContent, viewerCanChat } from './chatState.js';

export function useBattleRoomSession(activeRef, authUserRef) {
  const rooms = ref([]);
  const room = ref(null);
  const loading = ref(false);
  const error = ref('');
  const wsStatus = ref('disconnected');
  const resultDismissedRound = ref('');
  const chat = createBattleChatState();
  const modeAdapters = new Map();
  let heartbeatTimer = null;
  let roomListTimer = null;
  let roomsRefreshPromise = null;

  const errorKey = (requestError, fallback) => (
    requestError?.code || requestError?.message || fallback
  );
  const viewer = computed(() => room.value?.viewer || null);
  const ownResult = computed(() => (
    room.value?.results?.find(
      (item) => Number(item.user_id) === Number(room.value?.viewer?.user_id),
    ) || null
  ));
  const players = computed(() => (
    room.value?.members?.filter((member) => member.role === 'player') || []
  ));
  const spectators = computed(() => (
    room.value?.members?.filter((member) => member.role === 'spectator') || []
  ));
  const matchActive = computed(() => room.value?.status === 'running');
  const spectatorMode = computed(() => viewer.value?.role === 'spectator');
  const ownFinished = computed(() => (
    spectatorMode.value
    || ['completed', 'timed_out', 'disqualified'].includes(ownResult.value?.status)
  ));
  const showResults = computed(() => (
    room.value?.round?.status === 'completed'
    && resultDismissedRound.value !== room.value?.round?.round_id
  ));
  const chatCanSpeak = computed(() => viewerCanChat(room.value, viewer.value));

  const modeKeyFor = (candidate = room.value) => (
    String(candidate?.mode_key || 'goodness').trim().toLowerCase()
  );
  const modeAdapterFor = (candidate = room.value) => (
    modeAdapters.get(modeKeyFor(candidate)) || modeAdapters.get('goodness') || null
  );
  const registerModeAdapter = (adapter) => {
    const key = String(adapter?.key || '').trim().toLowerCase();
    if (!key) throw new TypeError('Battle mode adapter requires a key.');
    if (modeAdapters.has(key)) throw new Error(`Battle mode adapter already registered: ${key}`);
    modeAdapters.set(key, adapter);
    return adapter;
  };

  const applyRoom = async (nextRoom) => {
    if (
      nextRoom
      && room.value?.room_id === nextRoom.room_id
      && Number(nextRoom.revision || 0) < Number(room.value.revision || 0)
    ) return;
    if (nextRoom && room.value?.room_id === nextRoom.room_id) {
      const onlineByUser = new Map(
        (room.value.members || []).map((member) => [Number(member.user_id), member.online]),
      );
      nextRoom.members = (nextRoom.members || []).map((member) => ({
        ...member,
        online: member.online ?? onlineByUser.get(Number(member.user_id)) ?? false,
      }));
      nextRoom.results = (nextRoom.results || []).map((result) => ({
        ...result,
        online: result.online ?? onlineByUser.get(Number(result.user_id)) ?? false,
      }));
    }
    const previousRoom = room.value;
    if (String(previousRoom?.room_id || '') !== String(nextRoom?.room_id || '')) {
      chat.clear();
    }
    const previousAdapter = modeAdapterFor(previousRoom);
    const nextAdapter = modeAdapterFor(nextRoom);
    if (previousRoom && previousAdapter && previousAdapter !== nextAdapter) {
      await previousAdapter.onRoomApplied?.(null, previousRoom);
    }
    room.value = nextRoom || null;
    try {
      await nextAdapter?.onRoomApplied?.(room.value, previousRoom);
    } catch (modeError) {
      error.value = errorKey(modeError, 'battle_mode_load_failed');
    }
  };

  const client = createWsClient({
    clientId: `battle_${globalThis.crypto?.randomUUID?.() || Date.now()}`,
    onOpen: () => {
      wsStatus.value = 'connected';
      client.send(
        'BATTLE_SUBSCRIBE',
        room.value?.room_code ? { room_code: room.value.room_code } : {},
      );
    },
    onClose: () => { wsStatus.value = 'disconnected'; },
    onMessage: async (message) => {
      if (message?.action === 'BATTLE_ROOM_STATE') {
        if (!message?.data?.room && message?.data?.code) {
          error.value = String(message.data.code);
        }
        await applyRoom(message?.data?.room || null);
        return;
      }
      if (chat.handleWsMessage(message, room.value?.viewer?.user_id, room.value)) return;
      await modeAdapterFor()?.handleMessage?.(message);
    },
  });

  const ensureSocket = () => {
    if (!client.getSocket()) client.connect();
  };

  const sendModeAction = (modeAction, payload, { requestId = battleRequestId('action') } = {}) => {
    client.send('BATTLE_ACTION', {
      request_id: requestId,
      room_code: room.value?.room_code,
      round_id: room.value?.round?.round_id,
      mode_action: modeAction,
      payload,
    });
    return requestId;
  };

  const sendChatMessage = (rawContent) => {
    const validation = validateChatContent(rawContent);
    if (!validation.ok) {
      chat.reject({ code: validation.code });
      return false;
    }
    if (!room.value || !viewer.value) {
      chat.reject({ code: 'CHAT_NOT_IN_ROOM' });
      return false;
    }
    if (!chatCanSpeak.value) {
      chat.reject({ code: 'CHAT_ROLE_NOT_ALLOWED' });
      return false;
    }
    if (chat.cooldownSeconds.value > 0) return false;
    if (wsStatus.value !== 'connected') {
      chat.reject({ code: 'CHAT_DISCONNECTED' });
      return false;
    }
    chat.clearNotice();
    client.send('BATTLE_CHAT_SEND', {
      room_code: room.value.room_code,
      request_id: battleRequestId('chat'),
      content: validation.content,
    });
    return true;
  };

  const refreshRooms = async ({ silent = false } = {}) => {
    if (!authUserRef.value) return rooms.value;
    if (!roomsRefreshPromise) {
      roomsRefreshPromise = battleClient.rooms()
        .then((payload) => {
          rooms.value = payload.rooms || [];
          return rooms.value;
        })
        .finally(() => { roomsRefreshPromise = null; });
    }
    return silent ? roomsRefreshPromise.catch(() => rooms.value) : roomsRefreshPromise;
  };

  const shouldAutoRefreshRooms = () => (
    Boolean(activeRef.value)
    && Boolean(authUserRef.value)
    && !room.value
    && (typeof document === 'undefined' || document.visibilityState === 'visible')
  );
  const autoRefreshRooms = () => {
    if (shouldAutoRefreshRooms()) void refreshRooms({ silent: true });
  };
  const handleVisibilityChange = () => {
    if (document.visibilityState === 'visible') autoRefreshRooms();
  };

  const refreshCurrent = async () => {
    if (!authUserRef.value) {
      await applyRoom(null);
      return;
    }
    const payload = await battleClient.current();
    await applyRoom(payload.room || null);
    if (payload.room) ensureSocket();
  };

  const bootstrap = async () => {
    if (!authUserRef.value || loading.value) return;
    loading.value = true;
    error.value = '';
    try {
      await Promise.all([
        modeAdapterFor()?.bootstrap?.(),
        refreshRooms(),
        refreshCurrent(),
      ]);
    } catch (bootstrapError) {
      error.value = errorKey(bootstrapError, 'battle_load_failed');
    } finally {
      loading.value = false;
    }
  };

  const createRoom = async (payload) => {
    loading.value = true;
    error.value = '';
    try {
      const response = await battleClient.create({
        ...payload,
        mode_key: payload.mode_key || modeAdapterFor()?.key || 'goodness',
        request_id: battleRequestId('create'),
      });
      await applyRoom(response.room);
      ensureSocket();
      return response;
    } catch (requestError) {
      error.value = errorKey(requestError, 'battle_create_failed');
      return null;
    } finally {
      loading.value = false;
    }
  };

  const join = async (roomCode, role = undefined) => {
    loading.value = true;
    error.value = '';
    try {
      const response = await battleClient.join(roomCode, {
        role: role === 'auto' ? undefined : role,
        request_id: battleRequestId('join'),
      });
      await applyRoom(response.room);
      ensureSocket();
    } catch (requestError) {
      error.value = errorKey(requestError, 'battle_join_failed');
    } finally {
      loading.value = false;
    }
  };

  const leave = async () => {
    if (!room.value) return;
    try {
      const code = room.value.room_code;
      await battleClient.leave(code, { request_id: battleRequestId('leave') });
      await applyRoom(null);
      client.disconnect();
      await refreshRooms();
    } catch (requestError) {
      error.value = errorKey(requestError, 'battle_leave_failed');
    }
  };

  const toggleReady = async () => {
    if (!room.value || !viewer.value) return;
    try {
      const member = room.value.members.find(
        (item) => Number(item.user_id) === Number(viewer.value.user_id),
      );
      const response = await battleClient.ready(
        room.value.room_code,
        !member?.ready,
        battleRequestId('ready'),
      );
      await applyRoom(response.room);
    } catch (requestError) {
      error.value = errorKey(requestError, 'battle_ready_failed');
    }
  };

  const start = async () => {
    if (!room.value) return;
    try {
      const response = await battleClient.start(
        room.value.room_code,
        battleRequestId('start'),
      );
      await applyRoom(response.room);
    } catch (requestError) {
      error.value = errorKey(requestError, 'battle_start_failed');
    }
  };

  const kickMember = async (userId) => {
    try {
      const response = await battleClient.kick(
        room.value.room_code,
        userId,
        battleRequestId('kick'),
      );
      await applyRoom(response.room);
    } catch (requestError) {
      error.value = errorKey(requestError, 'battle_kick_failed');
    }
  };

  const setRole = async (role) => {
    try {
      const response = await battleClient.role(
        room.value.room_code,
        role,
        battleRequestId('role'),
      );
      await applyRoom(response.room);
    } catch (requestError) {
      error.value = errorKey(requestError, 'battle_role_failed');
    }
  };

  const dismissResults = () => {
    resultDismissedRound.value = String(room.value?.round?.round_id || '');
  };

  watch(activeRef, (active) => {
    if (!active) return;
    if (!authUserRef.value) {
      emitAuthRequired();
      return;
    }
    void bootstrap();
  });
  watch(authUserRef, (user) => {
    if (!user) {
      client.disconnect();
      void applyRoom(null);
      rooms.value = [];
      return;
    }
    if (activeRef.value) void bootstrap();
  });
  watch(chatCanSpeak, (canSpeak) => {
    if (canSpeak && chat.notice.value?.code === 'CHAT_ROLE_NOT_ALLOWED') {
      chat.clearNotice();
    }
  });

  onMounted(() => {
    document.addEventListener('visibilitychange', handleVisibilityChange);
    heartbeatTimer = window.setInterval(() => {
      if (room.value && wsStatus.value === 'connected') {
        client.send('BATTLE_HEARTBEAT', { room_code: room.value.room_code });
      }
    }, 15_000);
    roomListTimer = window.setInterval(autoRefreshRooms, 10_000);
    if (activeRef.value && authUserRef.value) void bootstrap();
  });

  onUnmounted(() => {
    document.removeEventListener('visibilitychange', handleVisibilityChange);
    if (heartbeatTimer != null) window.clearInterval(heartbeatTimer);
    if (roomListTimer != null) window.clearInterval(roomListTimer);
    for (const adapter of modeAdapters.values()) adapter.dispose?.();
    chat.dispose();
    client.disconnect();
  });

  return {
    rooms,
    room,
    loading,
    error,
    wsStatus,
    viewer,
    players,
    spectators,
    ownResult,
    matchActive,
    spectatorMode,
    ownFinished,
    showResults,
    activeModeKey: computed(() => modeKeyFor()),
    registerModeAdapter,
    setModeAdapter: registerModeAdapter,
    applyRoom,
    sendModeAction,
    chatMessages: chat.messages,
    chatNotice: chat.notice,
    chatCooldownSeconds: chat.cooldownSeconds,
    chatCanSpeak,
    sendChatMessage,
    refreshRooms,
    refreshCurrent,
    createRoom,
    join,
    leave,
    toggleReady,
    start,
    kickMember,
    setRole,
    dismissResults,
  };
}
