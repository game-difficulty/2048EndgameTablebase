<template>
  <div class="page-root battle-page">
    <div class="battle-page-shell">
      <header class="battle-page-titlebar">
        <div>
          <span class="ui-caption font-black uppercase text-text-secondary">2048 Endgame Tablebase</span>
          <h1>{{ $t('battle.title') }}</h1>
        </div>
        <div class="battle-title-actions">
          <div v-if="isGuestActor" class="battle-guest-identity">
            <span aria-hidden="true">G</span>
            <strong>{{ battleActor.display_name }}</strong>
            <small>{{ $t('battle.guest.marker') }}</small>
          </div>
          <div v-if="room" class="battle-title-room-state">
            <span>{{ room.room_code }}</span>
            <strong>{{ $t(`battle.status.${room.status}`) }}</strong>
          </div>
          <button
            type="button"
            class="battle-rules-button"
            :title="$t('battle.rules.open')"
            :aria-label="$t('battle.rules.open')"
            @click="rulesOpen = true"
          >?</button>
        </div>
      </header>

      <BattleRulesDialog
        v-if="rulesOpen"
        :modes="modeDefinitions"
        @close="rulesOpen = false"
      />

      <div v-if="error" class="battle-error-banner" role="alert">
        <span>{{ localizedError }}</span>
        <button type="button" aria-label="Close" @click="error = ''">×</button>
      </div>

      <component
        :is="modeDefinition.HallView"
        v-if="!room"
        :rooms="rooms"
        :loading="loading"
        :creating="loading"
        :token-balance="Number(authUser?.token_balance?.total || 0)"
        :can-create-room="isRegisteredActor"
        :mode-key="modeDefinition.key"
        :mode-options="modeOptions"
        :build-create-payload="modeDefinition.buildCreatePayload"
        v-bind="modeHallProps"
        @refresh="refreshRooms"
        @join="join"
        @create="requestCreateRoom"
        @login-required="requestLogin"
        @mode-change="selectMode"
      />

      <component
        :is="modeDefinition.MatchView"
        v-else-if="matchActive"
        :room="room"
        :current-actor-key="currentActorKey"
        :current-user-id="Number(room.viewer?.user_id || authUser?.id || 0)"
        :ws-status="wsStatus"
        :dis32k="dis32k"
        :replay-available="replayAvailable"
        :replay-review-available="replayReviewAvailable"
        :replay-busy="replayBusy"
        v-bind="modeMatchProps"
        v-on="modeMatchListeners"
        @open-trainer="openTrainer"
        @show-results="openResults"
        @forfeit="requestForfeit"
        @leave-room="requestLeave"
        @return-lobby="returnToLobby"
        @save-replay="saveOwnReplay"
        @open-replay="openOwnReplay"
      />

      <BattleLobby
        v-else
        :room="room"
        :members="room.members || []"
        :current-actor-key="currentActorKey"
        :current-user-id="Number(room.viewer?.user_id || authUser?.id || 0)"
        :now="now"
        :settings-pending="settingsPending"
        :role-pending="rolePending"
        :host-renew-pending="hostRenewPending"
        @ready="toggleReady"
        @start="start"
        @kick="kickMember"
        @leave="requestLeave"
        @role="setRole"
        @save-settings="updateRoomSettings"
        @renew-host="renewHosting"
      />

      <BattleRoomChat
        v-if="room"
        :messages="chatMessages"
        :notice="chatNotice"
        :cooldown-seconds="chatCooldownSeconds"
        :connected="wsStatus === 'connected'"
        :can-speak="chatCanSpeak"
        :disabled-code="chatDisabledCode"
        :compact="matchActive"
        @send="sendChatMessage"
      />

      <component
        :is="modeDefinition.ResultView"
        v-if="room && showResults"
        :room="room"
        :mode="resultMode"
        :replay-available="replayAvailable"
        :replay-review-available="replayReviewAvailable"
        :replay-busy="replayBusy"
        @close="dismissResults"
        @return-room="returnToLobby"
        @save-replay="saveOwnReplay"
        @open-replay="openOwnReplay"
      />

      <BattleExitDialog
        v-if="forfeitDialogOpen"
        :pending="forfeitPending"
        :error="error ? localizedError : ''"
        @cancel="forfeitDialogOpen = false"
        @confirm="confirmForfeit"
      />
      <BattleExitDialog
        v-if="closeRoomDialogId"
        intent="closeRoom"
        :pending="closeRoomPending"
        :error="error ? localizedError : ''"
        @cancel="closeRoomDialogId = ''"
        @confirm="confirmCloseRoom"
      />
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, toRef, unref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { TAB_IDS } from '../../../app/tabRegistry.js';
import { useAppSettingsStore } from '../../../app/useAppSettings.js';
import { useAuthState } from '../../../services/auth/authState.js';
import { emitAuthRequired } from '../../../services/auth/authEvents.js';
import { downloadBlob } from '../../../services/files/browserFiles.js';
import {
  authorizeLocalReplayLoad,
  createReplayRequestId,
} from '../../replay/services/replayClient.js';
import { queueReplayTransfer } from '../../replay/services/replayTransferStore.js';
import { clearTrainerPracticeContext } from '../../trainer/services/trainerPracticeJump.js';
import BattleLobby from '../components/BattleLobby.vue';
import BattleExitDialog from '../components/BattleExitDialog.vue';
import BattleRoomChat from '../components/BattleRoomChat.vue';
import BattleRulesDialog from '../components/BattleRulesDialog.vue';
import { useBattleSession } from '../composables/useBattleSession.js';
import {
  isBattleGuest,
  normalizeBattleActor,
} from '../core/battleActor.js';
import { shouldLeaveBattleRoomOnTabClose } from '../core/battleRoomViewState.js';
import { isPermanentBattleRoom } from '../core/battleRoomSettings.js';
import { battleClient } from '../services/battleClient.js';

const props = defineProps({
  active: { type: Boolean, default: true },
  hotkeysEnabled: { type: Boolean, default: true },
  authUser: { type: Object, default: null },
  currentActor: { type: Object, default: null },
  ensureGuestSession: { type: Function, default: null },
});
const emit = defineEmits(['navigate-tab']);

const now = ref(Date.now());
const rulesOpen = ref(false);
const forfeitDialogOpen = ref(false);
const closeRoomDialogId = ref('');
const closeRoomPending = ref(false);
const replayBusy = ref(false);
const ensuredGuestActor = ref(null);
const { t, te } = useI18n();
const { config: appConfig } = useAppSettingsStore();
const sharedAuth = useAuthState();
let clockTimer = null;
let inviteAttempted = false;

const battleActor = computed(() => normalizeBattleActor(
  props.authUser
  || props.currentActor
  || unref(sharedAuth.currentActor)
  || ensuredGuestActor.value,
));
const isGuestActor = computed(() => isBattleGuest(battleActor.value));
const isRegisteredActor = computed(() => battleActor.value?.kind === 'user');
const ensureGuestViaEvent = async () => {
  if (typeof window === 'undefined') return null;
  const detail = { promise: null };
  window.dispatchEvent(new CustomEvent('battle-guest-session-required', { detail }));
  return detail.promise ? detail.promise : null;
};
const ensureBattleGuestSession = async () => {
  if (battleActor.value) return battleActor.value;
  const ensure = props.ensureGuestSession || sharedAuth.ensureGuestSession;
  const response = typeof ensure === 'function'
    ? await ensure()
    : await ensureGuestViaEvent();
  const actor = normalizeBattleActor(
    response?.actor || response?.guest || response || unref(sharedAuth.currentActor),
  );
  if (actor) ensuredGuestActor.value = actor;
  return actor;
};

const {
  rooms,
  room,
  loading,
  error,
  wsStatus,
  matchActive,
  showResults,
  resultMode,
  forfeitPending,
  settingsPending,
  rolePending,
  hostRenewPending,
  chatMessages,
  chatNotice,
  chatCooldownSeconds,
  chatCanSpeak,
  chatDisabledCode,
  currentActorKey,
  ownResult,
  modeDefinitions,
  modeDefinition,
  modeSession,
  selectMode,
  refreshRooms,
  createRoom,
  join,
  leave,
  toggleReady,
  start,
  forfeit,
  kickMember,
  setRole,
  updateRoomSettings,
  renewHosting,
  returnToLobby,
  openResults,
  dismissResults,
  sendChatMessage,
} = useBattleSession(
  toRef(props, 'active'),
  toRef(props, 'authUser'),
  computed(() => props.hotkeysEnabled && !forfeitDialogOpen.value && !closeRoomDialogId.value),
  battleActor,
  ensureBattleGuestSession,
);

const modeHallProps = computed(() => unref(modeSession.value?.hallProps) || {});
const dis32k = computed(() => Boolean(appConfig.value.dis_32k));
const modeOptions = computed(() => modeDefinitions.map((definition) => ({
  value: definition.key,
  label: t(definition.labelKey),
  shortLabel: t(definition.shortLabelKey || definition.labelKey),
})));
const modeMatchProps = computed(() => unref(modeSession.value?.matchProps) || {});
const modeMatchListeners = computed(() => modeSession.value?.matchListeners || {});
const localizedError = computed(() => {
  const key = `battle.errors.${String(error.value || '')}`;
  return te(key) ? t(key) : t('battle.errors.UNKNOWN');
});
const replayAvailable = computed(() => (
  Number(ownResult.value?.replay_move_count || 0) > 0
  && ownResult.value?.status !== 'playing'
));
const replayReviewAvailable = computed(() => replayAvailable.value && isRegisteredActor.value);
const requestLogin = () => emitAuthRequired();
const requestCreateRoom = (payload) => {
  if (!isRegisteredActor.value) {
    requestLogin();
    return;
  }
  void createRoom(payload);
};
const openTrainer = () => {
  const detail = modeSession.value?.createPracticeJump?.();
  if (!detail?.hex) return;
  emit('navigate-tab', 'TrainerView', detail);
};
const confirmForfeit = async () => {
  if (await forfeit()) forfeitDialogOpen.value = false;
};
const requestForfeit = () => {
  error.value = '';
  forfeitDialogOpen.value = true;
};
const requestLeave = () => {
  if (room.value?.viewer?.is_host && !isPermanentBattleRoom(room.value)) {
    error.value = '';
    closeRoomDialogId.value = room.value.room_id;
    return;
  }
  void leave();
};
const confirmCloseRoom = async () => {
  if (closeRoomPending.value || !closeRoomDialogId.value || closeRoomDialogId.value !== room.value?.room_id) return;
  closeRoomPending.value = true;
  try {
    if (await leave()) closeRoomDialogId.value = '';
  } finally {
    closeRoomPending.value = false;
  }
};
const fetchOwnReplay = async () => {
  const roomCode = String(room.value?.room_code || '');
  const roundId = String(room.value?.round?.round_id || '');
  if (!roomCode || !roundId || !replayAvailable.value) return null;
  return battleClient.replay(roomCode, roundId);
};
const saveOwnReplay = async () => {
  if (replayBusy.value) return;
  replayBusy.value = true;
  try {
    const replay = await fetchOwnReplay();
    if (!replay) return;
    downloadBlob(
      new Blob([replay.buffer], { type: 'application/octet-stream' }),
      replay.filename,
    );
  } catch (replayError) {
    error.value = replayError?.code || 'BATTLE_REPLAY_LOAD_FAILED';
  } finally {
    replayBusy.value = false;
  }
};
const openOwnReplay = async () => {
  if (replayBusy.value || !isRegisteredActor.value) return;
  replayBusy.value = true;
  try {
    const replay = await fetchOwnReplay();
    if (!replay) return;
    await authorizeLocalReplayLoad({
      requestId: createReplayRequestId(),
      filename: replay.filename,
      size: replay.buffer.byteLength,
    });
    queueReplayTransfer(replay);
    emit('navigate-tab', TAB_IDS.REPLAY);
  } catch (replayError) {
    error.value = replayError?.code || 'BATTLE_REPLAY_LOAD_FAILED';
  } finally {
    replayBusy.value = false;
  }
};

const beforeTabClose = async () => {
  if (!shouldLeaveBattleRoomOnTabClose(room.value)) return true;
  return leave({ refreshRoomList: false });
};

defineExpose({ beforeTabClose });

watch([() => props.active, loading, room], ([active, busy, currentRoom]) => {
  if (!active || busy || currentRoom || inviteAttempted) return;
  const inviteCode = new URLSearchParams(window.location.search).get('room');
  if (!inviteCode) return;
  inviteAttempted = true;
  join(inviteCode, 'auto');
});
watch(() => room.value?.round?.round_id, () => { forfeitDialogOpen.value = false; });
watch(() => room.value?.room_id, () => { closeRoomDialogId.value = ''; });
watch(matchActive, (active) => {
  if (!active) clearTrainerPracticeContext('battle');
});

onMounted(() => { clockTimer = window.setInterval(() => { now.value = Date.now(); }, 1000); });
onUnmounted(() => {
  if (clockTimer != null) window.clearInterval(clockTimer);
  clearTrainerPracticeContext('battle');
});
</script>

<style scoped>
.battle-page { padding: 18px 26px 24px; }
.battle-page-shell { position: relative; width: min(100%, 1220px); min-height: 690px; margin: 0 auto; }
.battle-page-titlebar { min-height: 72px; display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 13px; }
.battle-page-titlebar h1 { margin: 3px 0 0; color: var(--text-main); font: 900 31px/1.05 Cambria, serif; letter-spacing: 0; }
.battle-title-actions { display: flex; align-items: center; gap: 10px; }
.battle-guest-identity { min-width: 0; display: grid; grid-template-columns: 24px auto; grid-template-rows: auto auto; column-gap: 7px; align-items: center; padding: 5px 9px; border: 1px solid var(--border-main); border-radius: 7px; background: var(--bg-card); }
.battle-guest-identity > span { grid-row: 1 / 3; width: 24px; height: 24px; display: grid; place-items: center; border: 1px solid var(--accent); border-radius: 50%; color: var(--accent); font: 900 10px/1 var(--font-mono, monospace); }
.battle-guest-identity strong { max-width: 132px; overflow: hidden; color: var(--text-main); font-size: 11px; font-weight: 900; text-overflow: ellipsis; white-space: nowrap; }
.battle-guest-identity small { color: var(--text-secondary); font-size: 9px; font-weight: 800; }
.battle-title-room-state { display: flex; align-items: center; gap: 9px; }
.battle-title-room-state span, .battle-title-room-state strong { border: 1px solid var(--border-main); border-radius: 999px; padding: 6px 10px; color: var(--text-secondary); font: 900 11px/1 var(--font-mono, monospace); }
.battle-title-room-state strong { color: var(--accent); font-family: inherit; }
.battle-rules-button { width: 36px; height: 36px; display: grid; flex: 0 0 auto; place-items: center; padding: 0; border: 2px solid var(--text-secondary); border-radius: 50%; background: transparent; color: var(--text-secondary); font: 900 18px/1 Georgia, serif; }
.battle-rules-button:hover, .battle-rules-button:focus-visible { border-color: var(--accent); color: var(--accent); outline: none; }
.battle-error-banner { min-height: 40px; display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; padding: 8px 12px; border: 1px solid color-mix(in srgb, #dc4c4c 52%, var(--border-main)); border-radius: 7px; background: color-mix(in srgb, #dc4c4c 8%, var(--bg-card)); color: #c84848; font-size: var(--font-ui-sm); font-weight: 800; }
.battle-error-banner button { width: 26px; height: 26px; border: 0; background: transparent; color: inherit; font-size: 19px; }
</style>
