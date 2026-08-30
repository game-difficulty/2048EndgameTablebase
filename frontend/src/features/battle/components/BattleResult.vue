<template>
  <div class="battle-result-backdrop">
    <section class="battle-result-panel" role="dialog" aria-modal="true" aria-labelledby="battle-result-title">
      <header>
        <span class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.result.kicker') }}</span>
        <h2 id="battle-result-title">{{ $t(isDraw ? 'battle.result.draw' : 'battle.result.title') }}</h2>
        <p>{{ room.full_pattern }} · {{ room.route?.step_count || 0 }} {{ $t('battle.result.steps') }}</p>
      </header>
      <div class="battle-result-list">
        <article v-for="(player, index) in rankedResults" :key="player.user_id" :class="['battle-result-row', rankFor(index) === 1 ? 'winner' : '']">
          <span class="battle-result-rank">{{ rankFor(index) }}</span>
          <div class="battle-result-avatar">
            <img v-if="player.avatar_url" :src="player.avatar_url" alt="" />
            <span v-else>{{ initials(player.display_name) }}</span>
          </div>
          <strong>{{ player.display_name }}</strong>
          <span>{{ statusLabel(player) }}</span>
          <b>{{ percent(player.goodness_of_fit) }}</b>
        </article>
      </div>
      <div class="battle-result-actions">
        <button type="button" @click="$emit('close')">{{ $t('battle.result.stayOnMatch') }}</button>
        <button type="button" class="primary" @click="$emit('return-room')">{{ $t('battle.result.backToRoom') }}</button>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps({ room: { type: Object, required: true } });
defineEmits(['close', 'return-room']);
const { t } = useI18n();
const rankedResults = computed(() => [...(props.room.results || [])].sort((left, right) => {
  const leftTimeout = left.status === 'timed_out' ? 1 : 0;
  const rightTimeout = right.status === 'timed_out' ? 1 : 0;
  return leftTimeout - rightTimeout || Number(right.goodness_of_fit) - Number(left.goodness_of_fit);
}));
const isDraw = computed(() => (
  rankedResults.value.length > 1
  && rankedResults.value[0].status !== 'timed_out'
  && rankedResults.value[1].status !== 'timed_out'
  && Number(rankedResults.value[0].goodness_of_fit) === Number(rankedResults.value[1].goodness_of_fit)
));
const rankFor = (index) => {
  if (index <= 0) return 1;
  const current = rankedResults.value[index];
  const previous = rankedResults.value[index - 1];
  return current.status === previous.status
    && Number(current.goodness_of_fit) === Number(previous.goodness_of_fit)
    ? rankFor(index - 1)
    : index + 1;
};
const initials = (value) => String(value || '?').trim().slice(0, 2).toUpperCase();
const percent = (value) => `${(Math.max(0, Math.min(1, Number(value ?? 1))) * 100).toFixed(2)}%`;
const statusLabel = (player) => player.status === 'timed_out' ? t('battle.playerStatus.timed_out') : t('battle.playerStatus.completed');
</script>

<style scoped>
.battle-result-backdrop { position: absolute; inset: 0; z-index: 200; display: grid; place-items: center; padding: 30px; background: rgba(8,14,28,.62); backdrop-filter: blur(6px); }
.battle-result-panel { width: min(720px, 92%); max-height: 82%; overflow: auto; padding: 24px; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); box-shadow: 0 30px 90px rgba(0,0,0,.34); }
.battle-result-panel header { text-align: center; }
.battle-result-panel h2 { margin: 5px 0 2px; color: var(--text-main); font-size: 29px; font-weight: 900; }
.battle-result-panel p { margin: 0; color: var(--text-secondary); font-size: var(--font-ui-sm); }
.battle-result-list { display: flex; flex-direction: column; gap: 7px; margin: 20px 0; }
.battle-result-row { display: grid; grid-template-columns: 35px 40px minmax(0,1fr) 100px 110px; align-items: center; gap: 10px; min-height: 58px; padding: 8px 12px; border: 1px solid var(--border-main); border-radius: 7px; color: var(--text-main); }
.battle-result-row.winner { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 8%, var(--bg-card)); }
.battle-result-rank { color: var(--accent); font: 900 18px/1 var(--font-mono, monospace); }
.battle-result-avatar img, .battle-result-avatar span { width: 38px; height: 38px; border-radius: 50%; }
.battle-result-avatar img { object-fit: cover; }
.battle-result-avatar span { display: grid; place-items: center; border: 1px solid var(--border-main); color: var(--accent); font-size: 10px; font-weight: 900; }
.battle-result-row > span:nth-of-type(2) { color: var(--text-secondary); font-size: var(--font-ui-xs); }
.battle-result-row b { color: var(--accent); font: 900 15px/1 var(--font-mono, monospace); text-align: right; }
.battle-result-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
.battle-result-actions button { min-height: 43px; border: 1px solid var(--border-main); border-radius: 7px; background: var(--bg-card); color: var(--text-main); font-weight: 900; }
.battle-result-actions button.primary { border-color: var(--btn-bg); background: var(--btn-bg); color: white; }
</style>
