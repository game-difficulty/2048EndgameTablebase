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
      <div class="battle-match-clock">
        <span>{{ $t('battle.match.stepTime') }}</span>
        <strong>{{ countdown }}</strong>
      </div>
      <div class="battle-match-actions">
        <button type="button" @click="$emit('open-trainer')">{{ $t('battle.match.openTrainer') }}</button>
        <button type="button" @click="$emit('show-results')">
          {{ $t(roundCompleted ? 'battle.match.viewResults' : 'battle.match.liveRanking') }}
        </button>
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
        <button v-else-if="ownForfeited" type="button" disabled>{{ $t('battle.match.exited') }}</button>
      </div>
    </header>

    <div v-if="spectator" class="battle-spectator-stage">
      <article v-for="player in playerRows" :key="player.user_id" class="battle-spectator-board">
        <div class="battle-mini-head">
          <div class="battle-player-identity">
            <img v-if="player.avatar_url" :src="player.avatar_url" alt="" />
            <span v-else>{{ initials(player.display_name) }}</span>
            <strong>{{ player.display_name }}</strong>
          </div>
          <div class="battle-mini-stats">
            <strong>{{ percent(player.goodness_of_fit) }}</strong>
            <span>{{ player.route_index }}/{{ totalSteps }}</span>
          </div>
        </div>
        <BaseBoard v-if="opponentBoards[player.user_id]" :frame="opponentBoards[player.user_id]" :dis32k="dis32k" :is-variant="isVariant" />
      </article>
    </div>

    <div v-else class="battle-match-grid">
      <section class="battle-own-stage">
        <div class="battle-own-metrics">
          <div><span>{{ $t('battle.match.goodness') }}</span><strong>{{ percent(ownResult?.goodness_of_fit) }}</strong></div>
          <div><span>{{ $t('battle.match.progress') }}</span><strong>{{ ownResult?.route_index || 0 }}/{{ totalSteps }}</strong></div>
          <div><span>{{ $t('battle.match.status') }}</span><strong>{{ $t(playerStatusKey(ownResult)) }}</strong></div>
        </div>
        <div class="battle-board-shell">
          <BaseBoard :frame="boardFrame" :dis32k="dis32k" :is-variant="isVariant" @swipe="$emit('move', $event)" />
          <Transition name="battle-correction">
            <div v-if="wrongOverlay" class="battle-wrong-overlay" role="status" aria-live="assertive">
              <span class="battle-wrong-kicker">{{ $t('battle.match.correcting') }}</span>
              <div class="battle-direction-correction">
                <div><small>{{ $t('battle.match.yourMove') }}</small><strong class="wrong">{{ directionLabel(wrongOverlay.selectedDirection) }}</strong></div>
                <span class="battle-correction-arrow">→</span>
                <div><small>{{ $t('battle.match.standardMove') }}</small><strong class="correct">{{ directionLabel(wrongOverlay.standardDirection) }}</strong></div>
              </div>
              <p>{{ $t('battle.match.goodnessDrop', { value: dropPercent(wrongOverlay.drop) }) }}</p>
              <small class="battle-correction-hint">{{ $t('battle.match.continueHint') }}</small>
            </div>
          </Transition>
        </div>
        <p v-if="!roundCompleted" class="battle-input-hint">{{ $t('battle.match.inputHint') }}</p>
      </section>

      <aside class="battle-opponents-panel">
        <div class="battle-panel-heading">
          <div><span class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.match.live') }}</span><h3>{{ $t('battle.match.opponents') }}</h3></div>
          <strong>{{ playerRows.length }}</strong>
        </div>
        <div class="battle-opponent-list">
          <article v-for="player in playerRows" :key="player.user_id" :class="['battle-opponent-row', Number(player.user_id) === Number(currentUserId) ? 'self' : '']">
            <div class="battle-opponent-topline">
              <div class="battle-player-identity">
                <img v-if="player.avatar_url" :src="player.avatar_url" alt="" />
                <span v-else>{{ initials(player.display_name) }}</span>
                <div><strong>{{ player.display_name }}</strong><small :class="player.online ? 'online' : 'offline'">{{ $t(onlineStatusKey(player)) }}</small></div>
              </div>
              <strong class="battle-opponent-gof">{{ percent(player.goodness_of_fit) }}</strong>
            </div>
            <div class="battle-progress-track"><i :style="{ width: progressPercent(player.route_index) }"></i></div>
            <div class="battle-opponent-footer"><span>{{ player.route_index }}/{{ totalSteps }}</span><span>{{ $t(playerStatusKey(player)) }}</span></div>
            <div v-if="canSeeBoard(player) && opponentBoards[player.user_id]" class="battle-revealed-board">
              <BaseBoard compact :frame="opponentBoards[player.user_id]" :dis32k="dis32k" :is-variant="isVariant" />
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

const props = defineProps({
  room: { type: Object, required: true },
  currentUserId: { type: Number, default: 0 },
  boardFrame: { type: Object, required: true },
  opponentBoards: { type: Object, default: () => ({}) },
  wrongOverlay: { type: Object, default: null },
  spectator: { type: Boolean, default: false },
  ownFinished: { type: Boolean, default: false },
  wsStatus: { type: String, default: 'disconnected' },
  dis32k: { type: Boolean, default: false },
  isVariant: { type: Boolean, default: false },
});

defineEmits(['move', 'open-trainer', 'show-results', 'forfeit', 'return-lobby']);
const now = ref(Date.now());
let timer = null;
const totalSteps = computed(() => Number(props.room.route?.step_count || 0));
const roundCompleted = computed(() => props.room.round?.status === 'completed');
const ownResult = computed(() => props.room.results?.find((item) => Number(item.user_id) === Number(props.currentUserId)) || null);
const canForfeit = computed(() => !props.spectator && ownResult.value?.status === 'playing');
const ownForfeited = computed(() => (
  ownResult.value?.status === 'disqualified'
  && ownResult.value?.mode_data?.finish_reason === 'forfeit'
));
const playerRows = computed(() => (props.room.results || []).map((result) => {
  const member = props.room.members?.find((item) => Number(item.user_id) === Number(result.user_id)) || {};
  return { ...member, ...result };
}));
const countdown = computed(() => {
  if (props.wrongOverlay && ownResult.value?.status === 'playing') {
    return `${Number(props.room.step_timeout_seconds || 90)}s`;
  }
  const deadline = ownResult.value?.timeout_at;
  if (!deadline || ownResult.value?.status !== 'playing') return '--';
  return `${Math.max(0, Math.ceil((Date.parse(deadline) - now.value) / 1000))}s`;
});
const percent = (value) => `${(Math.max(0, Math.min(1, Number(value ?? 1))) * 100).toFixed(2)}%`;
const dropPercent = (value) => `${(Math.max(0, Number(value || 0)) * 100).toFixed(2)}%`;
const progressPercent = (index) => `${Math.min(100, totalSteps.value ? Number(index || 0) / totalSteps.value * 100 : 0)}%`;
const initials = (value) => String(value || '?').trim().slice(0, 2).toUpperCase();
const onlineStatusKey = (player) => (player?.online ? 'battle.status.online' : 'battle.status.offline');
const playerStatusKey = (player) => (
  player?.status === 'disqualified' && player?.mode_data?.finish_reason === 'forfeit'
    ? 'battle.playerStatus.forfeited'
    : `battle.playerStatus.${player?.status || 'playing'}`
);
const directionLabel = (direction) => ({ left: '←', right: '→', up: '↑', down: '↓' }[direction] || '?');
const canSeeBoard = (player) => (
  Number(player.user_id) !== Number(props.currentUserId)
  && (props.spectator || props.ownFinished)
);

onMounted(() => { timer = window.setInterval(() => { now.value = Date.now(); }, 250); });
onUnmounted(() => { if (timer != null) window.clearInterval(timer); });
</script>

<style scoped>
.battle-match { display: flex; flex-direction: column; gap: 14px; }
.battle-match-header { min-height: 60px; display: grid; grid-template-columns: minmax(0,1fr) auto auto; align-items: center; gap: 14px; padding: 9px 12px 9px 16px; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); box-shadow: 0 12px 30px rgba(0,0,0,.06); }
.battle-match-identity { min-width: 0; }
.battle-match-title-line { display: flex; align-items: center; gap: 10px; margin-top: 3px; }
.battle-match-title-line h2 { min-width: 0; margin: 0; overflow: hidden; color: var(--text-main); font-size: 21px; font-weight: 900; text-overflow: ellipsis; white-space: nowrap; }
.battle-room-code, .battle-live-badge { border: 1px solid var(--border-main); border-radius: 999px; padding: 4px 8px; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 900; }
.battle-live-badge.online { color: #278354; border-color: color-mix(in srgb, #37a667 45%, var(--border-main)); }
.battle-live-badge.offline { color: #c24d4d; }
.battle-match-clock { min-width: 84px; padding-right: 4px; text-align: right; }
.battle-match-clock span { display: block; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 800; }
.battle-match-clock strong { color: var(--text-main); font: 900 20px/1.2 var(--font-mono, monospace); }
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
.battle-input-hint { margin: 9px 0 0; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 700; text-align: center; }
.battle-wrong-overlay { position: absolute; inset: 0; z-index: 60; display: flex; flex-direction: column; align-items: center; justify-content: center; border-radius: 12px; background: color-mix(in srgb, var(--bg-card) 91%, transparent); backdrop-filter: blur(5px); color: var(--text-main); text-align: center; }
.battle-wrong-kicker { color: var(--text-secondary); font-size: 11px; font-weight: 900; text-transform: uppercase; }
.battle-direction-correction { display: flex; align-items: center; gap: 22px; margin: 17px 0 11px; }
.battle-direction-correction div { display: flex; flex-direction: column; gap: 4px; }
.battle-direction-correction small { color: var(--text-secondary); font-weight: 800; }
.battle-direction-correction strong { font-size: 45px; line-height: 1; }
.battle-direction-correction .wrong { color: #d94f56; }
.battle-direction-correction .correct { color: #2f9c65; }
.battle-correction-arrow { color: var(--text-secondary); font-size: 22px; }
.battle-wrong-overlay p { margin: 0; color: var(--text-main); font-size: var(--font-ui-sm); font-weight: 900; }
.battle-correction-hint { margin-top: 10px; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 800; }
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
.battle-player-identity small { color: var(--text-secondary); font-size: 9px; }
.battle-player-identity small.online { color: #278354; }
.battle-opponent-gof { color: var(--text-main); font: 900 13px/1 var(--font-mono, monospace); }
.battle-progress-track { height: 5px; margin: 10px 0 6px; overflow: hidden; border-radius: 999px; background: color-mix(in srgb, var(--border-main) 68%, transparent); }
.battle-progress-track i { display: block; height: 100%; border-radius: inherit; background: var(--accent); }
.battle-opponent-footer { color: var(--text-secondary); font-size: 9px; font-weight: 800; }
.battle-revealed-board { width: 160px; margin: 10px auto 0; }
.battle-spectator-stage { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 11px; }
.battle-spectator-board { min-width: 0; padding: 10px; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); }
.battle-mini-head { display: flex; align-items: center; justify-content: space-between; gap: 7px; margin-bottom: 8px; }
.battle-mini-stats { text-align: right; }
.battle-mini-stats strong, .battle-mini-stats span { display: block; color: var(--text-main); font-size: 10px; font-weight: 900; }
.battle-mini-stats span { color: var(--text-secondary); }
</style>
