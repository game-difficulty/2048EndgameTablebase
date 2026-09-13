<template>
  <div v-if="compact" :class="['battle-finish-strip', { winner: notice.winner }]">
    <strong>{{ title }}{{ notice.final && notice.rank == null ? ' —' : '' }}</strong>
    <span>{{ goodness }}</span>
  </div>
  <div v-else class="battle-finish-overlay" @pointerdown.stop @pointerup.stop @click.self.stop="$emit('dismiss')">
    <section :class="['battle-finish-panel', { winner: notice.winner }]" :aria-label="title" @click.stop>
      <button class="battle-finish-close" type="button" :aria-label="$t('common.close')" :title="$t('common.close')" @click="$emit('dismiss')">×</button>
      <div role="status" aria-live="polite">
        <h3>{{ title }}</h3>
        <div class="battle-finish-score"><span>{{ $t('battle.match.goodness') }}</span><strong>{{ goodness }}</strong></div>
        <p v-if="!notice.final">{{ $t(notice.remaining ? 'battle.finish.waiting' : 'battle.finish.confirming', { count: notice.remaining }) }}</p>
        <p v-else-if="notice.rank == null">{{ $t('battle.finish.rank') }} —</p>
      </div>
      <div class="battle-finish-actions">
        <template v-if="notice.final">
          <button type="button" @click="$emit('show-results')">{{ $t('battle.finish.fullRanking') }}</button>
          <button type="button" class="primary" @click="$emit('return-lobby')">{{ $t('battle.result.backToRoom') }}</button>
        </template>
        <button v-else type="button" @click="$emit('dismiss')">{{ $t('battle.finish.watch') }}</button>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
const props = defineProps({ notice: { type: Object, required: true }, compact: Boolean });
defineEmits(['dismiss', 'show-results', 'return-lobby']);
const { t } = useI18n();
const title = computed(() => t(`battle.finish.${props.compact && props.notice.title === 'finished' ? 'compactFinished' : props.notice.title}`, { rank: props.notice.rank }));
const goodness = computed(() => `${(props.notice.goodness * 100).toFixed(2)}%`);
</script>

<style scoped>
.battle-finish-overlay { position: absolute; inset: 0; z-index: 55; display: grid; place-items: center; padding: 8px; border-radius: 8px; background: rgba(0,0,0,.18); touch-action: manipulation; }
.battle-finish-panel { position: relative; width: 94%; max-width: 350px; max-height: 100%; overflow: auto; padding: 18px 14px 14px; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-main); background: color-mix(in srgb, var(--bg-main) 96%, transparent); color: var(--text-main); text-align: center; box-shadow: 0 8px 28px rgba(0,0,0,.18); }
.battle-finish-panel.winner { border-color: var(--accent); }
.battle-finish-panel h3 { margin: 0 20px 9px; font-size: 22px; line-height: 1.2; font-weight: 900; overflow-wrap: anywhere; }
.battle-finish-score { display: flex; align-items: baseline; justify-content: center; gap: 9px; flex-wrap: wrap; }
.battle-finish-score span, .battle-finish-panel p { color: var(--text-secondary); font-size: 11px; }
.battle-finish-score strong { color: var(--accent); font-size: 26px; line-height: 1.2; font-weight: 900; font-variant-numeric: tabular-nums; }
.battle-finish-panel p { margin: 7px 0 0; }
.battle-finish-close { position: absolute; top: 3px; right: 3px; width: 28px; height: 28px; border: 0; border-radius: 5px; background: transparent; color: var(--text-secondary); font-size: 22px; line-height: 1; }
.battle-finish-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.battle-finish-actions button { flex: 1 1 110px; min-width: 0; min-height: 34px; padding: 6px 8px; border: 1px solid var(--border-main); border-radius: 6px; background: var(--bg-main); color: var(--text-main); font-size: 11px; font-weight: 900; overflow-wrap: anywhere; }
.battle-finish-actions .primary { background: var(--btn-bg); border-color: var(--btn-bg); color: white; }
.battle-finish-panel button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.battle-finish-strip { position: absolute; inset: auto 3px 3px; z-index: 50; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 2px 5px; padding: 4px 6px; border: 1px solid var(--border-main); border-radius: 5px; background: var(--bg-main); background: color-mix(in srgb, var(--bg-main) 96%, transparent); color: var(--text-main); font-size: 10px; line-height: 1.2; pointer-events: none; }
.battle-finish-strip strong { min-width: 0; overflow-wrap: anywhere; }
.battle-finish-strip span { color: var(--accent); font-weight: 900; font-variant-numeric: tabular-nums; }
.battle-finish-strip.winner { border-color: var(--accent); }
</style>
