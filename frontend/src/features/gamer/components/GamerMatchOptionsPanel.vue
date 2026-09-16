<template>
  <section v-if="options.length" class="gamer-match-options" :aria-label="$t('gamer.matchOptions.title')">
    <div class="gamer-ai-mode" role="group" :aria-label="$t('gamer.matchOptions.aiMode')">
      <button v-for="enabled in [false, true]" :key="String(enabled)" type="button"
        :aria-pressed="tableEnabled === enabled" @click="$emit('table-mode', enabled)">
        {{ $t(enabled ? 'gamer.matchOptions.useTables' : 'gamer.matchOptions.searchOnly') }}
      </button>
    </div>
    <div v-for="option in options" :key="option.key" class="gamer-match-option">
      <span class="gamer-option-label">
        {{ $t(option.labelKey) }}
        <button v-if="option.key === 'rankedParticipation'" type="button" class="ranked-help-button"
          :title="$t('gamer.matchOptions.rankedHelpTitle')" :aria-label="$t('gamer.matchOptions.rankedHelpTitle')"
          :aria-expanded="rankedHelpOpen" :aria-controls="helpId" @click="rankedHelpOpen = !rankedHelpOpen">
          <CircleHelp :size="16" aria-hidden="true" />
        </button>
      </span>
      <button
        type="button"
        class="gamer-match-option-toggle"
        role="switch"
        :aria-checked="option.enabled"
        :disabled="option.disabled"
        @click="$emit('change', option.key, !option.enabled)"
      >
        <span aria-hidden="true" />
      </button>
    </div>
    <p v-if="rankedHelpOpen" :id="helpId" class="ranked-help" role="note">{{ $t('gamer.matchOptions.rankedHelp') }}</p>
  </section>
</template>

<script setup>
import { ref, useId } from 'vue';
import { CircleHelp } from '@lucide/vue';

const rankedHelpOpen = ref(false);
const helpId = useId();
defineProps({
  options: { type: Array, default: () => [] },
  tableEnabled: { type: Boolean, default: false },
});

defineEmits(['change', 'table-mode']);
</script>

<style scoped>
.gamer-option-label { display: inline-flex; align-items: center; gap: 6px; }
.ranked-help-button { display: inline-grid; place-items: center; width: 24px; height: 24px; padding: 0; border: 0; background: transparent; color: var(--text-secondary); cursor: pointer; flex-shrink: 0; }
.ranked-help-button:hover, .ranked-help-button[aria-expanded="true"] { color: var(--accent); }
.ranked-help-button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.ranked-help { margin: 0; padding: 8px 10px; border-left: 2px solid var(--accent); background: var(--bg-input); color: var(--text-secondary); font-size: 13px; line-height: 1.6; overflow-wrap: anywhere; }
.gamer-ai-mode { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px; }
.gamer-ai-mode button { min-height: 2rem; padding: 4px 8px; border-radius: 4px;
  color: var(--text-main); font-size: var(--font-ui-xs); font-weight: 850; }
.gamer-ai-mode button[aria-pressed="true"] { background: var(--accent); color: var(--bg-main); }
.gamer-ai-mode button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.gamer-match-options {
  width: 100%;
  min-height: 5.5rem;
  flex-shrink: 0;
  padding: 0.65rem 0.8rem;
  display: grid;
  gap: 0.45rem;
  border: 1px solid var(--border-main);
  border-radius: 8px;
  background: color-mix(in srgb, var(--bg-card) 94%, transparent);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}

.gamer-match-option {
  min-height: 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  color: var(--text-main);
  font-size: var(--font-ui-xs);
  font-weight: 850;
}

.gamer-match-option-toggle {
  position: relative;
  width: 2.55rem;
  height: 1.35rem;
  flex: 0 0 auto;
  border: 1px solid var(--border-main);
  border-radius: 999px;
  background: color-mix(in srgb, var(--text-secondary) 24%, var(--bg-main));
  transition: background 160ms ease, border-color 160ms ease;
}

.gamer-match-option-toggle > span {
  position: absolute;
  top: 0.16rem;
  left: 0.17rem;
  width: 0.9rem;
  height: 0.9rem;
  border-radius: 999px;
  background: white;
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.24);
  transition: transform 160ms ease;
}

.gamer-match-option-toggle[aria-checked="true"] {
  border-color: color-mix(in srgb, var(--accent) 78%, var(--border-main));
  background: var(--accent);
}

.gamer-match-option-toggle[aria-checked="true"] > span {
  transform: translateX(1.18rem);
}

.gamer-match-option-toggle:focus-visible {
  outline: 2px solid color-mix(in srgb, var(--accent) 48%, transparent);
  outline-offset: 2px;
}

.gamer-match-option-toggle:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
</style>
