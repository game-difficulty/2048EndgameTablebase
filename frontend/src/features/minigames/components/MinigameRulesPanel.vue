<template>
  <section class="minigame-rules-panel w-full rounded-2xl border border-border-main/70 bg-bg-card/90 p-4 shadow-sm">
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <h2 class="ui-text-lg font-black text-text-main">{{ rules.title }}</h2>
        <p class="mt-1 ui-body font-semibold text-text-secondary">{{ rules.summary }}</p>
      </div>
      <span class="pill-badge pill-badge-soft shrink-0">{{ rules.difficultyName }}</span>
    </div>

    <div class="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
      <div
        v-for="trophy in rules.trophies"
        :key="trophy.key"
        class="minigame-rule-trophy rounded-xl border border-border-main/55 bg-bg-main/55 px-2 py-2 text-center"
      >
        <div class="ui-caption font-black uppercase tracking-wide text-text-secondary">{{ trophy.label }}</div>
        <div class="mt-1 ui-body font-black tabular-nums text-text-main">{{ trophy.requirement }}</div>
      </div>
    </div>
    <p class="mt-2 ui-caption font-semibold text-text-secondary">
      {{ rules.trophyMetricLabel }}: {{ rules.trophyMetric }}
    </p>

    <details class="minigame-rule-details mt-3">
      <summary class="cursor-pointer select-none ui-body font-black text-accent-primary">
        {{ rules.detailsLabel }}
      </summary>
      <div class="mt-3 space-y-3 text-text-main">
        <div>
          <h3 class="ui-caption font-black uppercase tracking-wide text-text-secondary">{{ rules.objectiveTitle }}</h3>
          <p class="mt-1 ui-body font-semibold">{{ rules.objective }}</p>
        </div>
        <div v-if="rules.mechanics.length">
          <h3 class="ui-caption font-black uppercase tracking-wide text-text-secondary">{{ rules.specialRulesTitle }}</h3>
          <ul class="mt-1 list-disc space-y-1 pl-5 ui-body font-semibold">
            <li v-for="item in rules.mechanics" :key="item">{{ item }}</li>
          </ul>
        </div>
        <div>
          <h3 class="ui-caption font-black uppercase tracking-wide text-text-secondary">
            {{ rules.difficultyTitle }} · {{ rules.difficultyName }}
          </h3>
          <ul class="mt-1 list-disc space-y-1 pl-5 ui-body font-semibold">
            <li v-for="item in rules.difficultyRules" :key="item">{{ item }}</li>
          </ul>
        </div>
        <p class="rounded-xl bg-bg-main/65 px-3 py-2 ui-caption font-semibold text-text-secondary">
          {{ rules.retained }}
        </p>
      </div>
    </details>
  </section>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

import { getMinigameRuleView } from '../model/minigameRules';

const props = defineProps({
  gameId: { type: String, default: '' },
  difficulty: { type: Number, default: 1 },
});

const { locale } = useI18n();
const rules = computed(() => getMinigameRuleView(props.gameId, props.difficulty, locale.value));
</script>

<style scoped>
.minigame-rules-panel {
  position: relative;
  z-index: 1;
}

.minigame-rule-trophy {
  min-width: 0;
}

.minigame-rule-details summary::marker {
  color: var(--color-accent-primary, currentColor);
}
</style>
