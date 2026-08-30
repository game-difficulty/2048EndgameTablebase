<template>
  <div class="page-root battle-page">
    <div class="battle-page-shell">
      <header class="battle-page-titlebar">
        <div>
          <span class="ui-caption font-black uppercase text-text-secondary">2048 Endgame Tablebase</span>
          <h1>{{ $t('battle.title') }}</h1>
        </div>
        <div v-if="room" class="battle-title-room-state">
          <span>{{ room.room_code }}</span>
          <strong>{{ $t(`battle.status.${room.status}`) }}</strong>
        </div>
      </header>

      <div v-if="error" class="battle-error-banner" role="alert">
        <span>{{ localizedError }}</span>
        <button type="button" aria-label="Close" @click="error = ''">×</button>
      </div>

      <section v-if="!authUser" class="battle-login-required">
        <div class="battle-login-mark" aria-hidden="true">VS</div>
        <h2>{{ $t('battle.auth.title') }}</h2>
        <p>{{ $t('battle.auth.body') }}</p>
        <button type="button" @click="requestLogin">{{ $t('auth.actions.login') }}</button>
      </section>

      <component
        :is="modeDefinition.HallView"
        v-else-if="!room"
        :rooms="rooms"
        :loading="loading"
        :creating="loading"
        :token-balance="Number(authUser.token_balance?.total || 0)"
        :mode-key="modeDefinition.key"
        :build-create-payload="modeDefinition.buildCreatePayload"
        v-bind="modeHallProps"
        @refresh="refreshRooms"
        @join="join"
        @create="createRoom"
      />

      <component
        :is="modeDefinition.MatchView"
        v-else-if="matchActive"
        :room="room"
        :current-user-id="Number(room.viewer?.user_id || authUser.id)"
        :ws-status="wsStatus"
        v-bind="modeMatchProps"
        v-on="modeMatchListeners"
      />

      <BattleLobby
        v-else
        :room="room"
        :members="room.members || []"
        :current-user-id="Number(room.viewer?.user_id || authUser.id)"
        :now="now"
        @ready="toggleReady"
        @start="start"
        @kick="kickMember"
        @leave="leave"
        @role="setRole"
      />

      <BattleRoomChat
        v-if="room"
        :messages="chatMessages"
        :notice="chatNotice"
        :cooldown-seconds="chatCooldownSeconds"
        :connected="wsStatus === 'connected'"
        :can-speak="chatCanSpeak"
        :compact="matchActive"
        @send="sendChatMessage"
      />

      <component
        :is="modeDefinition.ResultView"
        v-if="room && showResults"
        :room="room"
        @close="dismissResults"
      />
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, toRef, unref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { emitAuthRequired } from '../../../services/auth/authEvents.js';
import BattleLobby from '../components/BattleLobby.vue';
import BattleRoomChat from '../components/BattleRoomChat.vue';
import { useBattleSession } from '../composables/useBattleSession.js';

const props = defineProps({
  active: { type: Boolean, default: true },
  authUser: { type: Object, default: null },
});

const now = ref(Date.now());
const { t, te } = useI18n();
let clockTimer = null;
let inviteAttempted = false;

const {
  rooms,
  room,
  loading,
  error,
  wsStatus,
  matchActive,
  showResults,
  chatMessages,
  chatNotice,
  chatCooldownSeconds,
  chatCanSpeak,
  modeDefinition,
  modeSession,
  refreshRooms,
  createRoom,
  join,
  leave,
  toggleReady,
  start,
  kickMember,
  setRole,
  dismissResults,
  sendChatMessage,
} = useBattleSession(toRef(props, 'active'), toRef(props, 'authUser'));

const modeHallProps = computed(() => unref(modeSession.value?.hallProps) || {});
const modeMatchProps = computed(() => unref(modeSession.value?.matchProps) || {});
const modeMatchListeners = computed(() => modeSession.value?.matchListeners || {});
const localizedError = computed(() => {
  const key = `battle.errors.${String(error.value || '')}`;
  return te(key) ? t(key) : String(error.value || '');
});
const requestLogin = () => emitAuthRequired();

watch([() => props.active, () => props.authUser, loading, room], ([active, user, busy, currentRoom]) => {
  if (!active || !user || busy || currentRoom || inviteAttempted) return;
  const inviteCode = new URLSearchParams(window.location.search).get('room');
  if (!inviteCode) return;
  inviteAttempted = true;
  join(inviteCode, 'auto');
});

onMounted(() => { clockTimer = window.setInterval(() => { now.value = Date.now(); }, 1000); });
onUnmounted(() => { if (clockTimer != null) window.clearInterval(clockTimer); });
</script>

<style scoped>
.battle-page { padding: 18px 26px 24px; }
.battle-page-shell { position: relative; width: min(100%, 1220px); min-height: 690px; margin: 0 auto; }
.battle-page-titlebar { min-height: 72px; display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 13px; }
.battle-page-titlebar h1 { margin: 3px 0 0; color: var(--text-main); font: 900 31px/1.05 Cambria, serif; letter-spacing: 0; }
.battle-title-room-state { display: flex; align-items: center; gap: 9px; }
.battle-title-room-state span, .battle-title-room-state strong { border: 1px solid var(--border-main); border-radius: 999px; padding: 6px 10px; color: var(--text-secondary); font: 900 11px/1 var(--font-mono, monospace); }
.battle-title-room-state strong { color: var(--accent); font-family: inherit; }
.battle-error-banner { min-height: 40px; display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; padding: 8px 12px; border: 1px solid color-mix(in srgb, #dc4c4c 52%, var(--border-main)); border-radius: 7px; background: color-mix(in srgb, #dc4c4c 8%, var(--bg-card)); color: #c84848; font-size: var(--font-ui-sm); font-weight: 800; }
.battle-error-banner button { width: 26px; height: 26px; border: 0; background: transparent; color: inherit; font-size: 19px; }
.battle-login-required { min-height: 560px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); text-align: center; }
.battle-login-mark { width: 64px; height: 64px; display: grid; place-items: center; border: 2px solid var(--accent); border-radius: 50%; color: var(--accent); font: 900 18px/1 var(--font-mono, monospace); }
.battle-login-required h2 { margin: 17px 0 6px; color: var(--text-main); font-size: 24px; font-weight: 900; }
.battle-login-required p { max-width: 440px; margin: 0 0 18px; color: var(--text-secondary); font-size: var(--font-ui-sm); }
.battle-login-required button { min-width: 150px; min-height: 42px; border: 1px solid var(--btn-bg); border-radius: 7px; background: var(--btn-bg); color: white; font-weight: 900; }
</style>
