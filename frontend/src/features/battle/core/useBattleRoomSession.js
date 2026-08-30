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
import { createBattleRoomViewState } from './battleRoomViewState.js';
import { createBattleChatState, validateChatContent, viewerCanChat } from './chatState.js';

export function useBattleRoomSession(activeRef, authUserRef) {
  const rooms = ref([]);
  const room = ref(null);
  const loading = ref(false);
  const error = ref('');
  const wsStatus = ref('disconnected');
  const completedRoundView = ref('');
  const resultVisibleRound = ref('');
  const forfeitPending = ref(false);
  const chat = createBattleChatState();
  const roomViewState = createBattleRoomViewState();
  const modeAdapters = new Map();
  let heartbeatTimer = null;
  let roomListTimer = null;
  let roomsRefreshPromise = null;
  let roomStateEpoch = 0;

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
  const matchActive = computed(() => (
    room.value?.status === 'running'
    || (
      room.value?.round?.status === 'completed'
      && completedRoundView.value === room.value?.round?.round_id
    )
  ));
  const spectatorMode = computed(() => viewer.value?.role === 'spectator');
  const ownFinished = computed(() => (
    spectatorMode.value
    || ['completed', 'timed_out', 'disqualified'].includes(ownResult.value?.status)
  ));
  const showResults = computed(() => (
    Boolean(room.value?.round?.round_id)
    && resultVisibleRound.value === room.value?.round?.round_id
  ));
  const resultMode = computed(() => (
    room.value?.round?.status === 'completed' ? 'final' : 'live'
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
    const viewState = roomViewState.apply(previousRoom, nextRoom);
    completedRoundView.value = viewState.heldRoundId;
    resultVisibleRound.value = viewState.resultRoundId;
    if (String(previousRoom?.room_id || '') !== String(nextRoom?.room_id || '')) {
      chat.clear();
    }
    const previousAdapter = modeAdapterFor(previousRoom);
    const nextAdapter = modeAdapterFor(nextRoom);
    if (previousRoom && previousAdapter && previousAdapter !== nextAdapter) {
      await previousAdapter.onRoomApplied?.(null, previousRoom);
    }
    room.value = nextRoom || null;
    roomStateEpoch += 1;
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
        const data = message?.data || {};
        if (data.room) {
          await applyRoom(data.room);
          return;
        }
        if (!data.closed) return;
        const closedRoomId = String(data.room_id || '');
        if (closedRoomId && room.value && closedRoomId !== String(room.value.room_id || '')) {
          return;
        }
        if (data.code) error.value = String(data.code);
        await applyRoom(null);
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
    const requestEpoch = roomStateEpoch;
    const payload = await battleClient.current();
    if (!payload.room && requestEpoch !== roomStateEpoch) return;
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

  const forfeit = async () => {
    if (
      forfeitPending.value
      || !room.value?.room_code
      || !room.value?.round?.round_id
      || !['playing', 'disconnected'].includes(String(ownResult.value?.status || ''))
    ) return false;
    forfeitPending.value = true;
    error.value = '';
    try {
      const response = await battleClient.forfeit(
        room.value.room_code,
        room.value.round.round_id,
        battleRequestId('forfeit'),
      );
      await applyRoom(response.room);
      return true;
    } catch (requestError) {
      error.value = errorKey(requestError, 'battle_forfeit_failed');
      return false;
    } finally {
      forfeitPending.value = false;
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

  const syncRoomViewState = (viewState) => {
    completedRoundView.value = viewState.heldRoundId;
    resultVisibleRound.value = viewState.resultRoundId;
  };
  const returnToLobby = () => syncRoomViewState(roomViewState.returnToLobby(room.value));
  const openResults = () => syncRoomViewState(roomViewState.openResults(room.value));
  const dismissResults = () => syncRoomViewState(roomViewState.closeResults());

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
    resultMode,
    forfeitPending,
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
    forfeit,
    kickMember,
    setRole,
    returnToLobby,
    openResults,
    dismissResults,
  };
}
