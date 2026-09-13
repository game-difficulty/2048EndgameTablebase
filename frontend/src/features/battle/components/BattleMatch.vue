<template>
  <div class="battle-match">
    <header class="battle-match-header">
      <div class="battle-match-identity">
        <span class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.match.kicker') }}</span>
        <div class="battle-match-title-line">
          <h2>{{ room.full_pattern }}</h2>
          <span class="battle-room-code">{{ room.room_code }}</span>
          <span :class="['battle-live-badge', wsStatus === 'connected' ? 'online' : 'offline']">{{ $t(`status.${wsStatus}`) }}</span>
        </div>
      </div>
      <div v-if="roundCompleted" class="battle-round-finished" role="status">{{ $t('battle.finish.roundEnded') }}</div>
      <div v-else-if="watchingView" class="battle-round-finished" role="status">{{ $t('battle.match.watching') }}</div>
      <div v-else
        :class="[
          'battle-match-clock',
          { urgent: countdownState.urgent, critical: countdownState.critical },
        ]"
        role="timer"
        :aria-label="`${$t('battle.match.stepTime')} ${countdown}`"
      >
        <span>{{ $t('battle.match.stepTime') }}</span>
        <strong :key="countdownState.seconds">{{ countdown }}</strong>
      </div>
      <div class="battle-match-actions">
        <button type="button" @click="$emit('open-trainer')">{{ $t('battle.match.openTrainer') }}</button>
        <button type="button" @click="$emit('show-results')">
          {{ $t(roundCompleted ? 'battle.finish.fullRanking' : 'battle.match.liveRanking') }}
        </button>
        <button
          v-if="replayAvailable"
          type="button"
          :disabled="replayBusy"
          @click="$emit('save-replay')"
        >{{ $t('battle.replay.save') }}</button>
        <button
          v-if="replayReviewAvailable"
          type="button"
          :disabled="replayBusy"
          @click="$emit('open-replay')"
        >{{ $t('battle.replay.review') }}</button>
        <button
          v-if="roundCompleted"
          type="button"
          class="primary"
          @click="$emit('return-lobby')"
        >{{ $t('battle.result.backToRoom') }}</button>
        <button
          v-else-if="canForfeit"
          type="button"
          class="danger"
          @click="$emit('forfeit')"
        >{{ $t('battle.match.exitBattle') }}</button>
        <button
          v-if="canWatch && !watchingView"
          type="button"
          @click="watchOthers"
        >{{ $t('battle.finish.watch') }}</button>
        <button
          v-if="spectator || ownFinished"
          type="button"
          class="danger"
          @click="$emit('leave-room')"
        >{{ $t(leaveLabelKey) }}</button>
      </div>
    </header>

    <div v-if="watchingView" class="battle-spectator-stage" :style="{ maxWidth: spectatorGrid.maxWidth }">
      <article v-for="(player, index) in playerRows" :key="battleActorRenderKey(player)" class="battle-spectator-board" :style="spectatorGrid.items[index]">
        <div class="battle-mini-head">
          <div class="battle-player-identity">
            <img v-if="player.avatar_url && !isBattleGuest(player)" :src="player.avatar_url" alt="" />
            <span v-else>{{ initials(player.display_name) }}</span>
            <strong>{{ player.display_name }} <small v-if="isBattleGuest(player)" class="battle-player-guest-marker">{{ $t('battle.guest.marker') }}</small></strong>
          </div>
          <div class="battle-mini-stats">
            <strong>{{ percent(player.goodness_of_fit) }}</strong>
            <span>{{ player.route_index }}/{{ totalSteps }}</span>
          </div>
        </div>
        <div v-if="opponentBoards[battleActorRenderKey(player)]" class="battle-visible-board">
          <BattleObservedBoard :frame="opponentBoards[battleActorRenderKey(player)]" :dis32k="dis32k" :is-variant="isVariant" :overlay="opponentOverlayFor(player)" :finish-notice="finishNotices[battleActorRenderKey(player)]" />
        </div>
      </article>
    </div>

    <div v-else class="battle-match-grid">
      <section class="battle-own-stage">
        <div class="battle-own-metrics">
          <div><span>{{ $t('battle.match.goodness') }}</span><strong>{{ percent(ownResult?.goodness_of_fit) }}</strong></div>
          <div><span>{{ $t('battle.match.progress') }}</span><strong>{{ ownResult?.route_index || 0 }}/{{ totalSteps }}</strong></div>
          <div><span>{{ $t('battle.match.status') }}</span><strong>{{ $t(playerStatusKey(ownResult)) }}</strong></div>
        </div>
        <div :class="['battle-board-shell', { urgent: countdownState.urgent }]">
          <BaseBoard :frame="boardFrame" :dis32k="dis32k" :is-variant="isVariant" @swipe="$emit('move', $event)">
            <template #overlay>
              <Transition name="battle-correction">
                <BattleCorrectionOverlay
                  v-if="wrongOverlay"
                  :overlay="wrongOverlay"
                  interactive
                  @continue="$emit('continue-correction')"
                />
              </Transition>
              <BattleFinishNotice v-if="visibleFinishNotice" :notice="visibleFinishNotice" :leave-label="$t(leaveLabelKey)" @dismiss="dismissFinishNotice" @watch="watchOthers" @leave-room="$emit('leave-room')" @show-results="showFinishResults" @return-lobby="returnFromFinish" />
            </template>
          </BaseBoard>
        </div>
        <p v-if="!roundCompleted && !ownFinishNotice" class="battle-input-hint">{{ $t('battle.match.inputHint') }}</p>
      </section>

      <aside class="battle-opponents-panel">
        <div class="battle-panel-heading">
          <div><span class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.match.live') }}</span><h3>{{ $t('battle.match.opponents') }}</h3></div>
          <strong>{{ playerRows.length }}</strong>
        </div>
        <div class="battle-opponent-list">
          <article v-for="player in playerRows" :key="battleActorRenderKey(player)" :class="['battle-opponent-row', isCurrentActor(player) ? 'self' : '']">
            <div class="battle-opponent-topline">
              <div class="battle-player-identity">
                <img v-if="player.avatar_url && !isBattleGuest(player)" :src="player.avatar_url" alt="" />
                <span v-else>{{ initials(player.display_name) }}</span>
                <div><strong>{{ player.display_name }} <small v-if="isBattleGuest(player)" class="battle-player-guest-marker">{{ $t('battle.guest.marker') }}</small></strong><small :class="player.online ? 'online' : 'offline'">{{ $t(onlineStatusKey(player)) }}</small></div>
              </div>
              <strong class="battle-opponent-gof">{{ percent(player.goodness_of_fit) }}</strong>
            </div>
            <div class="battle-progress-track"><i :style="{ width: progressPercent(player.route_index) }"></i></div>
            <div class="battle-opponent-footer"><span>{{ player.route_index }}/{{ totalSteps }}</span><span>{{ $t(playerStatusKey(player)) }}</span></div>
            <div v-if="canSeeBoard(player) && opponentBoards[battleActorRenderKey(player)]" class="battle-revealed-board">
              <BattleObservedBoard :frame="opponentBoards[battleActorRenderKey(player)]" :dis32k="dis32k" :is-variant="isVariant" :overlay="opponentOverlayFor(player)" :finish-notice="finishNotices[battleActorRenderKey(player)]" />
            </div>
          </article>
        </div>
      </aside>
    </div>

  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue';

import BaseBoard from '../../../components/BaseBoard.vue';
import BattleCorrectionOverlay from './BattleCorrectionOverlay.vue';
import BattleObservedBoard from './BattleObservedBoard.vue';
import BattleFinishNotice from './BattleFinishNotice.vue';
import { battleFinishNotices, createBattleFinishDismissals } from '../core/battleFinishNotice.js';
import { spectatorLayout } from '../core/spectatorLayout.js';
import {
  battleActorRenderKey,
  isBattleGuest,
  sameBattleActor,
} from '../core/battleActor.js';
import { battleCountdownState } from '../core/battleCountdown.js';
import { isPermanentBattleRoom } from '../core/battleRoomSettings.js';

const props = defineProps({
  room: { type: Object, required: true },
  currentUserId: { type: Number, default: 0 },
  currentActorKey: { type: String, default: '' },
  boardFrame: { type: Object, required: true },
  opponentBoards: { type: Object, default: () => ({}) },
  opponentOverlays: { type: Object, default: () => ({}) },
  wrongOverlay: { type: Object, default: null },
  resolving: { type: Boolean, default: false },
  spectator: { type: Boolean, default: false },
  ownFinished: { type: Boolean, default: false },
  wsStatus: { type: String, default: 'disconnected' },
  dis32k: { type: Boolean, default: false },
  isVariant: { type: Boolean, default: false },
  replayAvailable: { type: Boolean, default: false },
  replayReviewAvailable: { type: Boolean, default: false },
  replayBusy: { type: Boolean, default: false },
});

const emit = defineEmits([
  'move',
  'continue-correction',
  'open-trainer',
  'show-results',
  'forfeit',
  'leave-room',
  'return-lobby',
  'save-replay',
  'open-replay',
]);
const now = ref(Date.now());
let timer = null;
const totalSteps = computed(() => Number(props.room.route?.step_count || 0));
const roundCompleted = computed(() => props.room.round?.status === 'completed');
const currentIdentity = computed(() => (
  props.room.viewer
  || (props.currentActorKey ? { actor_key: props.currentActorKey } : { user_id: props.currentUserId })
));
const isCurrentActor = (candidate) => sameBattleActor(candidate, currentIdentity.value);
const ownResult = computed(() => props.room.results?.find(isCurrentActor) || null);
const watchingRoundKey = ref('');
const viewKey = computed(() => JSON.stringify([props.room.room_id, props.room.round?.round_id, battleActorRenderKey(currentIdentity.value)]));
const canWatch = computed(() => !roundCompleted.value && props.ownFinished && !props.spectator);
// Watching is a local view choice, not a room role or host change.
const watchingView = computed(() => props.spectator || (canWatch.value && watchingRoundKey.value === viewKey.value));
const leaveLabelKey = computed(() => props.room.viewer?.is_host && !isPermanentBattleRoom(props.room)
  ? 'battle.actions.closeRoom' : 'battle.actions.leave');
const finishNotices = computed(() => battleFinishNotices(props.room));
const ownFinishNotice = computed(() => finishNotices.value[battleActorRenderKey(currentIdentity.value)] || null);
const finishDismissals = createBattleFinishDismissals();
const dismissedFinishKey = ref('');
const visibleFinishNotice = computed(() => {
  const notice = ownFinishNotice.value;
  return !watchingView.value && !props.wrongOverlay && notice
    && dismissedFinishKey.value !== notice.key && !finishDismissals.has(notice.key) ? notice : null;
});
const dismissFinishNotice = () => {
  const key = ownFinishNotice.value?.key;
  finishDismissals.dismiss(key);
  dismissedFinishKey.value = key || '';
};
const showFinishResults = () => { dismissFinishNotice(); emit('show-results'); };
const returnFromFinish = () => { dismissFinishNotice(); emit('return-lobby'); };
const watchOthers = () => {
  if (!canWatch.value) return;
  dismissFinishNotice();
  watchingRoundKey.value = viewKey.value;
};
const canForfeit = computed(() => !props.spectator && ['playing', 'disconnected'].includes(ownResult.value?.status));
const playerRows = computed(() => (props.room.results || []).map((result) => {
  const member = props.room.members?.find((item) => sameBattleActor(item, result)) || {};
  return { ...member, ...result };
}));
const spectatorGrid = computed(() => spectatorLayout(playerRows.value.length));
const countdownState = computed(() => battleCountdownState({
  deadline: ownResult.value?.timeout_at,
  now: now.value,
  status: ownResult.value?.status,
  correcting: Boolean(props.wrongOverlay),
  resolving: props.resolving,
  pausedSeconds: props.room.step_timeout_seconds || 90,
}));
const countdown = computed(() => (
  countdownState.value.seconds == null ? '--' : `${countdownState.value.seconds}s`
));
const percent = (value) => `${(Math.max(0, Math.min(1, Number(value ?? 1))) * 100).toFixed(2)}%`;
const progressPercent = (index) => `${Math.min(100, totalSteps.value ? Number(index || 0) / totalSteps.value * 100 : 0)}%`;
const initials = (value) => String(value || '?').trim().slice(0, 2).toUpperCase();
const onlineStatusKey = (player) => (player?.online ? 'battle.status.online' : 'battle.status.offline');
const playerStatusKey = (player) => (
  player?.status === 'disqualified' && player?.mode_data?.finish_reason === 'forfeit'
    ? 'battle.playerStatus.forfeited'
    : `battle.playerStatus.${player?.status || 'playing'}`
);
const canSeeBoard = (player) => (
  !isCurrentActor(player)
  && (props.spectator || props.ownFinished)
);
const opponentOverlayFor = (player) => {
  return props.opponentOverlays[battleActorRenderKey(player)] || null;
};

onMounted(() => { timer = window.setInterval(() => { now.value = Date.now(); }, 250); });
onUnmounted(() => { if (timer != null) window.clearInterval(timer); });
</script>

<style scoped>
.battle-match { display: flex; flex-direction: column; gap: 14px; }
.battle-match-header { min-height: 60px; display: grid; grid-template-columns: minmax(0,1fr) auto auto; align-items: center; gap: 14px; padding: 9px 12px 9px 16px; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); box-shadow: 0 12px 30px rgba(0,0,0,.06); }
.battle-match-identity { min-width: 0; }
.battle-round-finished { color: var(--accent); font-size: var(--font-ui-sm); font-weight: 900; }
.battle-match-title-line { display: flex; align-items: center; gap: 10px; margin-top: 3px; }
.battle-match-title-line h2 { min-width: 0; margin: 0; overflow: hidden; color: var(--text-main); font-size: 21px; font-weight: 900; text-overflow: ellipsis; white-space: nowrap; }
.battle-room-code, .battle-live-badge { border: 1px solid var(--border-main); border-radius: 999px; padding: 4px 8px; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 900; }
.battle-live-badge.online { color: #278354; border-color: color-mix(in srgb, #37a667 45%, var(--border-main)); }
.battle-live-badge.offline { color: #c24d4d; }
.battle-match-clock { min-width: 92px; padding: 6px 9px; border: 1px solid transparent; border-radius: 7px; text-align: right; transition: border-color .18s ease, background-color .18s ease, box-shadow .18s ease; }
.battle-match-clock span { display: block; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 800; }
.battle-match-clock strong { display: block; min-width: 3ch; color: var(--text-main); font: 900 20px/1.2 var(--font-mono, monospace); transform-origin: right center; }
.battle-match-clock.urgent { border-color: color-mix(in srgb, #d94f56 68%, var(--border-main)); background: color-mix(in srgb, #d94f56 10%, var(--bg-card)); box-shadow: 0 0 0 2px color-mix(in srgb, #d94f56 10%, transparent); }
.battle-match-clock.urgent span, .battle-match-clock.urgent strong { color: #d94f56; }
.battle-match-clock.urgent strong { font-size: 23px; animation: battle-clock-step .18s ease-out; }
.battle-match-clock.critical { animation: battle-clock-critical 1s ease-in-out infinite; }
.battle-match-actions { display: flex; align-items: center; gap: 7px; }
.battle-match-actions button { min-width: 92px; min-height: 38px; padding: 0 11px; border: 1px solid var(--border-main); border-radius: 7px; background: var(--bg-card); color: var(--text-main); font-size: var(--font-ui-xs); font-weight: 900; white-space: nowrap; }
.battle-match-actions button:hover:not(:disabled), .battle-match-actions button:focus-visible { border-color: var(--accent); color: var(--accent); outline: none; }
.battle-match-actions button.primary { border-color: var(--btn-bg); background: var(--btn-bg); color: white; }
.battle-match-actions button.danger { border-color: color-mix(in srgb, #d94f56 66%, var(--border-main)); color: #d94f56; }
.battle-match-actions button:disabled { cursor: default; opacity: .55; }
.battle-match-grid { display: grid; grid-template-columns: 442px minmax(0, 1fr); gap: 16px; align-items: stretch; }
.battle-own-stage, .battle-opponents-panel { border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); box-shadow: 0 14px 32px rgba(0,0,0,.06); }
.battle-own-stage { padding: 14px; }
.battle-own-metrics { display: grid; grid-template-columns: 1.15fr 1fr 1fr; gap: 7px; margin-bottom: 10px; }
.battle-own-metrics div { min-width: 0; padding: 8px 9px; border: 1px solid var(--border-main); border-radius: 7px; background: color-mix(in srgb, var(--bg-main) 70%, transparent); }
.battle-own-metrics span { display: block; color: var(--text-secondary); font-size: 10px; font-weight: 900; text-transform: uppercase; }
.battle-own-metrics strong { display: block; overflow: hidden; text-overflow: ellipsis; color: var(--text-main); font-size: 13px; font-weight: 900; white-space: nowrap; }
.battle-board-shell { position: relative; width: 100%; }
.battle-board-shell::after { position: absolute; inset: -5px; z-index: 70; border: 2px solid transparent; border-radius: 14px; content: ''; pointer-events: none; }
.battle-board-shell.urgent::after { animation: battle-board-urgent .72s ease-out 1; }
.battle-input-hint { margin: 9px 0 0; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 700; text-align: center; }
.battle-correction-enter-active, .battle-correction-leave-active { transition: opacity .16s ease; }
.battle-correction-enter-from, .battle-correction-leave-to { opacity: 0; }
.battle-opponents-panel { min-width: 0; padding: 15px; }
.battle-panel-heading { display: flex; align-items: center; justify-content: space-between; padding-bottom: 11px; border-bottom: 1px solid var(--border-main); }
.battle-panel-heading h3 { margin: 2px 0 0; color: var(--text-main); font-size: var(--font-ui-base); font-weight: 900; }
.battle-panel-heading > strong { color: var(--accent); font: 900 24px/1 var(--font-mono, monospace); }
.battle-opponent-list { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin-top: 11px; }
.battle-opponent-row { min-width: 0; padding: 11px; border: 1px solid var(--border-main); border-radius: 7px; background: color-mix(in srgb, var(--bg-main) 62%, transparent); }
.battle-opponent-row.self { border-color: color-mix(in srgb, var(--accent) 50%, var(--border-main)); }
.battle-opponent-topline, .battle-opponent-footer { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.battle-player-identity { min-width: 0; display: flex; align-items: center; gap: 8px; }
.battle-player-identity > img, .battle-player-identity > span { width: 32px; height: 32px; flex: 0 0 auto; border-radius: 50%; }
.battle-player-identity > img { object-fit: cover; }
.battle-player-identity > span { display: grid; place-items: center; border: 1px solid var(--border-main); color: var(--accent); font-size: 10px; font-weight: 900; }
.battle-player-identity div { min-width: 0; display: flex; flex-direction: column; }
.battle-player-identity strong { overflow: hidden; text-overflow: ellipsis; color: var(--text-main); font-size: var(--font-ui-xs); white-space: nowrap; }
.battle-player-guest-marker { margin-left: 3px; color: var(--accent); font-size: 8px; font-weight: 900; }
.battle-player-identity small { color: var(--text-secondary); font-size: 9px; }
.battle-player-identity small.online { color: #278354; }
.battle-opponent-gof { color: var(--text-main); font: 900 13px/1 var(--font-mono, monospace); }
.battle-progress-track { height: 5px; margin: 10px 0 6px; overflow: hidden; border-radius: 999px; background: color-mix(in srgb, var(--border-main) 68%, transparent); }
.battle-progress-track i { display: block; height: 100%; border-radius: inherit; background: var(--accent); }
.battle-opponent-footer { color: var(--text-secondary); font-size: 9px; font-weight: 800; }
.battle-revealed-board { position: relative; width: 160px; margin: 10px auto 0; }
.battle-visible-board { position: relative; }
.battle-spectator-stage { display: grid; grid-template-columns: repeat(24, minmax(0, 1fr)); gap: 12px; width: 100%; margin: 0 auto; align-items: start; }
.battle-spectator-board { min-width: 0; padding: 10px; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); }
.battle-mini-head { display: flex; align-items: center; justify-content: space-between; gap: 7px; margin-bottom: 8px; }
.battle-mini-stats { text-align: right; }
.battle-mini-stats strong, .battle-mini-stats span { display: block; color: var(--text-main); font-size: 10px; font-weight: 900; }
.battle-mini-stats span { color: var(--text-secondary); }

@keyframes battle-clock-step {
  0% { transform: scale(.84); opacity: .72; }
  65% { transform: scale(1.1); }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes battle-clock-critical {
  0%, 100% { background: color-mix(in srgb, #d94f56 10%, var(--bg-card)); }
  50% { background: color-mix(in srgb, #d94f56 22%, var(--bg-card)); }
}
@keyframes battle-board-urgent {
  0% { border-color: transparent; box-shadow: 0 0 0 0 transparent; }
  35% { border-color: color-mix(in srgb, #d94f56 80%, white); box-shadow: 0 0 20px color-mix(in srgb, #d94f56 38%, transparent); }
  100% { border-color: transparent; box-shadow: 0 0 0 8px transparent; }
}
@media (prefers-reduced-motion: reduce) {
  .battle-match-clock.urgent strong,
  .battle-match-clock.critical,
  .battle-board-shell.urgent::after { animation: none; }
  .battle-board-shell.urgent::after { border-color: color-mix(in srgb, #d94f56 62%, transparent); }
}
</style>
