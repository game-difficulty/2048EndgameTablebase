<template>
  <div
    :class="['battle-wrong-overlay', { compact, interactive }]"
    :role="interactive ? 'button' : 'status'"
    :tabindex="interactive ? 0 : undefined"
    aria-live="assertive"
    @pointerdown.stop
    @click.stop="requestContinue"
    @keydown.enter.prevent="requestContinue"
    @keydown.space.prevent="requestContinue"
  >
    <span class="battle-wrong-kicker">{{ $t('battle.match.correcting') }}</span>
    <div class="battle-direction-correction">
      <div>
        <small>{{ $t('battle.match.yourMove') }}</small>
        <strong class="wrong">{{ directionLabel(overlay.selectedDirection) }}</strong>
      </div>
      <span class="battle-correction-arrow">→</span>
      <div>
        <small>{{ $t('battle.match.standardMove') }}</small>
        <strong class="correct">{{ directionLabel(overlay.standardDirection) }}</strong>
      </div>
    </div>
    <p>{{ $t('battle.match.goodnessDrop', { value: dropPercent(overlay.drop) }) }}</p>
    <small v-if="interactive" class="battle-correction-hint">{{ $t('battle.match.continueHint') }}</small>
  </div>
</template>

<script setup>
const props = defineProps({
  overlay: { type: Object, required: true },
  interactive: { type: Boolean, default: false },
  compact: { type: Boolean, default: false },
});
const emit = defineEmits(['continue']);

const directionLabel = (direction) => ({
  left: '←', right: '→', up: '↑', down: '↓',
}[direction] || '?');
const dropPercent = (value) => `${(Math.max(0, Number(value) || 0) * 100).toFixed(2)}%`;
const requestContinue = () => {
  if (props.interactive) emit('continue');
};
</script>

<style scoped>
.battle-wrong-overlay { position: absolute; inset: 0; z-index: 60; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 0; border-radius: 12px; background: color-mix(in srgb, var(--bg-card) 91%, transparent); backdrop-filter: blur(5px); color: var(--text-main); text-align: center; user-select: none; -webkit-tap-highlight-color: transparent; }
.battle-wrong-overlay.interactive { cursor: pointer; touch-action: manipulation; }
.battle-wrong-overlay:focus-visible { outline: 3px solid color-mix(in srgb, var(--accent) 70%, transparent); outline-offset: -5px; }
.battle-wrong-overlay.interactive:active { background: color-mix(in srgb, var(--bg-card) 84%, var(--accent)); }
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
.battle-wrong-overlay.compact { border-radius: 7px; }
.battle-wrong-overlay.compact .battle-wrong-kicker { font-size: 8px; }
.battle-wrong-overlay.compact .battle-direction-correction { gap: 8px; margin: 5px 0; }
.battle-wrong-overlay.compact .battle-direction-correction small { font-size: 7px; }
.battle-wrong-overlay.compact .battle-direction-correction strong { font-size: 22px; }
.battle-wrong-overlay.compact .battle-correction-arrow { font-size: 12px; }
.battle-wrong-overlay.compact p { font-size: 8px; }
</style>
