<template>
  <div class="page-root">
    <div class="relative z-[120] mb-4 flex w-full max-w-6xl items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <span class="font-[Cambria,serif] text-3xl font-extrabold tracking-tight text-text-main">{{ currentPatternDisplay }}</span>
        <span :class="connectionBadgeClass">{{ $t(`status.${wsStatus}`) }}</span>
      </div>

      <div class="relative z-[130] top-menu-shell">
        <div ref="patternMenuRoot" class="relative z-[140]">
          <button type="button" class="top-menu-trigger" @click="togglePatternMenu">
            <span>{{ selectedPattern }}</span>
            <span class="ui-kicker opacity-60">{{ patternMenuOpen ? '▲' : '▼' }}</span>
          </button>
          <div v-if="patternMenuOpen" class="absolute right-0 top-full z-[150] mt-2 flex min-w-[360px] overflow-hidden rounded-xl border border-border-main bg-bg-card shadow-xl">
            <div class="max-h-[320px] w-[132px] overflow-y-auto border-r border-border-main/60 bg-bg-main/60 p-1.5">
              <button
                v-for="group in patternGroups"
                :key="group.category"
                type="button"
                @mouseenter="activePatternCategory = group.category"
                @focus="activePatternCategory = group.category"
                @click.stop="activePatternCategory = group.category"
                :class="[
                  'flex w-full items-center rounded-lg px-3 py-2 text-left ui-control font-black uppercase tracking-tighter transition-colors',
                  activePatternCategory === group.category ? 'bg-btn-bg text-white' : 'text-text-main hover:bg-btn-bg/10'
                ]"
              >
                {{ group.category }}
              </button>
            </div>
            <div class="grid max-h-[320px] min-w-[220px] grid-cols-2 content-start gap-1.5 overflow-y-auto p-2">
              <button
                v-for="pattern in activePatternOptions"
                :key="pattern"
                type="button"
                @click.stop="selectPattern(pattern)"
                :class="[
                  'rounded-lg px-3 py-2 text-left ui-control font-black transition-colors',
                  selectedPattern === pattern ? 'surface-prominent text-white' : 'bg-bg-main text-text-main hover:bg-btn-bg/10'
                ]"
              >
                {{ pattern }}
              </button>
            </div>
          </div>
        </div>
        <select v-model="selectedTarget" class="top-menu-select" @change="applyPatternSelection">
          <option v-for="target in availableTargets" :key="target" :value="target" class="bg-bg-card text-text-main">{{ target }}</option>
        </select>
      </div>
    </div>

    <div class="relative z-0 grid w-full min-w-[1020px] max-w-6xl grid-cols-[452px_minmax(0,1fr)] items-start gap-6">
      <section class="flex flex-col">
        <div class="mb-4 flex gap-2">
          <input
            v-model="hexInput"
            type="text"
            spellcheck="false"
            data-tester-text-input="true"
            class="flex-1 rounded-lg border border-border-main bg-bg-main px-3 py-2 font-[Consolas,Monaco,monospace] ui-body font-black tracking-[0.06em] text-text-main outline-none transition-colors placeholder:opacity-50 hover:border-accent/40 focus:border-accent"
            placeholder="0000000000000000"
          />
          <button class="action-btn min-w-[70px]" :disabled="!hexInput.trim()" @click="applyManualBoard">{{ $t('tester.controls.set') }}</button>
          <button class="action-btn min-w-[88px]" :disabled="!selectedPattern || !selectedTarget" @click="resetRandom">{{ $t('tester.controls.random') }}</button>
        </div>

        <div class="mx-auto w-full max-w-[442px]">
          <BaseBoard :board="board" :metadata="metadata" :dis32k="dis32k" :is-variant="isVariant" />
        </div>

        <div class="mt-4 grid grid-cols-3 gap-2">
          <button class="action-btn-small" :disabled="logs.length < 1" @click="saveLog">{{ $t('tester.controls.saveLog') }}</button>
          <button class="action-btn-small" :disabled="recordLength < 1" @click="saveReplay">{{ $t('tester.controls.saveReplay') }}</button>
          <button class="action-btn-small" @click="toggleInsights">{{ showInsights ? $t('tester.controls.focusMode') : $t('tester.controls.showInfo') }}</button>
        </div>
      </section>

      <section class="flex min-w-0 flex-col gap-3">
        <div class="grid grid-cols-3 gap-2">
          <button class="action-btn" @click="openReplayView">{{ $t('tester.controls.goToReplay') }}</button>
          <button class="action-btn" @click="$emit('navigate-tab', 'NotebookView')">{{ $t('tester.controls.goToNotebook') }}</button>
          <button
            class="action-btn tester-btn-accent btn-prominent"
            @click="$emit('open-analysis', { pattern: selectedPattern, target: selectedTarget })"
          >
            {{ $t('tester.controls.openAnalysis') }}
          </button>
        </div>

        <div class="space-y-3 rounded-[20px] bg-bg-card/72 p-3">
          <div class="flex items-center justify-between gap-4">
            <span class="ui-caption font-black uppercase tracking-[0.18em] text-text-secondary">
              {{ $t('tester.previousStepResults') }}
            </span>
            <span class="ui-caption font-black uppercase tracking-[0.18em] text-text-secondary">{{ displayedResultDtype || '?' }}</span>
          </div>
          <div class="grid grid-cols-[198px_minmax(0,1fr)] items-stretch gap-4">
            <div class="result-mini-board">
              <div
                v-for="tile in resultConsoleTiles"
                :key="tile.key"
                class="result-mini-tile"
                :style="getResultMiniTileStyle(tile)"
              >
                {{ tile.label }}
              </div>
            </div>
            <div class="grid gap-3">
              <div
                v-for="item in displayedResults"
                :key="item.dir"
                class="grid min-h-[40px] grid-cols-[0.9rem_minmax(0,1fr)] items-center gap-3 rounded-lg px-2 py-1.5"
                :style="getResultRowStyle(item)"
              >
                <span class="text-center ui-body font-black text-text-secondary">{{ dirLabels[item.dir] }}</span>
                <span class="truncate text-left font-black tabular-nums leading-none" :style="getResultValueStyle(item)">{{ item.display }}</span>
              </div>
            </div>
          </div>
          <div class="rounded-[18px] bg-bg-main/70 px-4 py-3.5">
            <div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span class="feedback-title" :style="feedbackBadgeStyle">{{ feedbackBadgeText }}</span>
              <span v-if="feedbackLossText" class="feedback-combo">{{ feedbackLossText }}</span>
            </div>
            <div class="mt-2.5 flex flex-wrap items-center gap-x-2 gap-y-1 ui-body font-black text-text-main">
              <span class="feedback-label">{{ feedbackPressedLabel }}</span>
              <span class="feedback-move-inline" :style="feedbackPressedMoveStyle">{{ feedbackPressedMove }}</span>
              <span class="feedback-connector">{{ feedbackConnector }}</span>
              <span class="feedback-label">{{ feedbackBestLabel }}</span>
              <span class="feedback-move-inline" :style="feedbackBestMoveStyle">{{ feedbackBestMove }}</span>
            </div>
          </div>
          <div class="rounded-[18px] bg-bg-main/68 px-3.5 py-3">
            <div class="grid grid-cols-3 gap-2.5">
              <div class="rounded-xl border border-border-main/60 bg-bg-card/55 px-3 py-2.5">
                <div class="ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">{{ $t('tester.metrics.gof') }}</div>
                <div class="mt-1 ui-metric font-black tabular-nums text-text-main">{{ insightsActive ? goodnessDisplay : '--' }}</div>
              </div>
              <div class="rounded-xl border border-border-main/60 bg-bg-card/55 px-3 py-2.5">
                <div class="ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">{{ $t('tester.metrics.combo') }}</div>
                <div class="mt-1 ui-metric font-black tabular-nums text-text-main">{{ insightsActive ? `${metrics.combo}x` : '--' }}</div>
              </div>
              <div class="rounded-xl border border-border-main/60 bg-bg-card/55 px-3 py-2.5">
                <div class="ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">{{ $t('tester.metrics.maxCombo') }}</div>
                <div class="mt-1 ui-metric font-black tabular-nums text-text-main">{{ insightsActive ? `${metrics.max_combo}x` : '--' }}</div>
              </div>
            </div>
            <div
              ref="evaluationMixRoot"
              class="relative mt-3"
              @mouseenter="showEvaluationTooltip = true"
              @mouseleave="showEvaluationTooltip = false"
              @mousemove="updateEvaluationTooltip"
            >
              <div class="mb-1.5 ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">{{ $t('tester.metrics.evaluationMix') }}</div>
              <div v-if="evaluationTotal > 0" class="data-bar-track">
                <div
                  v-for="segment in displayedEvaluationSegments"
                  :key="segment.label"
                  class="data-bar-segment"
                  :style="{ width: `${segment.percent}%`, background: segment.color }"
                />
              </div>
              <div v-else class="h-2.5 rounded-full border border-dashed border-border-main/60 bg-bg-card/60" />
              <div
                class="pointer-events-none absolute bottom-[calc(100%+8px)] z-20 w-36 -translate-x-1/2 rounded-xl border border-border-main/70 bg-bg-card/88 p-3 shadow-xl backdrop-blur-sm transition-opacity duration-150"
                :class="showEvaluationTooltip ? 'opacity-100' : 'opacity-0'"
                :style="{ left: `${evaluationTooltipLeft}px` }"
              >
                <div class="mb-2 ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">{{ $t('tester.metrics.evaluationStats') }}</div>
                <div class="space-y-1.5">
                  <div v-for="segment in displayedEvaluationSegments" :key="`${segment.label}-tooltip`" class="flex items-center justify-between gap-3 ui-caption font-black text-text-main">
                    <span class="inline-flex items-center gap-2">
                      <span class="h-2.5 w-2.5 rounded-full" :style="{ background: segment.color }" />
                      {{ segment.shortLabel }}
                    </span>
                    <span class="tabular-nums text-text-secondary">{{ segment.count }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, toRef } from 'vue';

import BaseBoard from '../../../components/BaseBoard.vue';
import { useTesterSession } from '../composables/useTesterSession';

const props = defineProps({
  active: { type: Boolean, default: false },
});

const emit = defineEmits(['navigate-tab', 'open-analysis']);

const evaluationMixRoot = ref(null);
const showEvaluationTooltip = ref(false);
const evaluationTooltipLeft = ref(72);

const updateEvaluationTooltip = (event) => {
  const root = evaluationMixRoot.value;
  if (!(root instanceof HTMLElement)) {
    return;
  }

  const rect = root.getBoundingClientRect();
  const tooltipHalfWidth = 72;
  const padding = 12;
  const pointerX = event.clientX - rect.left;
  evaluationTooltipLeft.value = Math.min(
    rect.width - tooltipHalfWidth - padding,
    Math.max(tooltipHalfWidth + padding, pointerX)
  );
};

const openReplayView = () => {
  window.__pendingReplayLatestLoad = true;
  emit('navigate-tab', 'ReplayReviewView');
};

const {
  wsStatus,
  board,
  metadata,
  dis32k,
  showInsights,
  availableTargets,
  selectedPattern,
  selectedTarget,
  activePatternCategory,
  patternMenuOpen,
  hexInput,
  logs,
  recordLength,
  metrics,
  patternMenuRoot,
  patternGroups,
  activePatternOptions,
  currentPatternDisplay,
  isVariant,
  displayedResultDtype,
  connectionBadgeClass,
  insightsActive,
  evaluationTotal,
  displayedEvaluationSegments,
  dirLabels,
  feedbackBadgeText,
  feedbackBadgeStyle,
  feedbackLossText,
  feedbackPressedLabel,
  feedbackPressedMove,
  feedbackPressedMoveStyle,
  feedbackConnector,
  feedbackBestLabel,
  feedbackBestMove,
  feedbackBestMoveStyle,
  goodnessDisplay,
  displayedResults,
  resultConsoleTiles,
  togglePatternMenu,
  selectPattern,
  applyPatternSelection,
  resetRandom,
  applyManualBoard,
  toggleInsights,
  saveLog,
  saveReplay,
  getResultRowStyle,
  getResultValueStyle,
  getResultMiniTileStyle,
} = useTesterSession(toRef(props, 'active'));
</script>

<style scoped>
.tester-btn-accent {
  color: white;
}

.tester-btn-accent:hover {
  color: white;
}

.feedback-title {
  font-size: var(--font-ui-lg);
  font-weight: 900;
  letter-spacing: 0.01em;
}

.feedback-combo {
  color: var(--text-secondary);
  font-size: calc(0.84rem * var(--ui-scale));
  font-weight: 900;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.feedback-label {
  color: var(--text-secondary);
  font-size: calc(0.8rem * var(--ui-scale));
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.feedback-move-inline {
  color: var(--text-main);
  font-size: var(--font-ui-lg);
  letter-spacing: 0.01em;
}

.feedback-connector {
  color: var(--text-secondary);
  font-size: calc(0.78rem * var(--ui-scale));
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.result-mini-board {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  grid-template-rows: repeat(4, minmax(0, 1fr));
  gap: 0.45rem;
  height: 100%;
  align-self: stretch;
}

.result-mini-tile {
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 0.7rem;
  min-height: 0;
  font-size: calc(0.92rem * var(--ui-scale));
  font-weight: 900;
  letter-spacing: -0.01em;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
}
</style>
