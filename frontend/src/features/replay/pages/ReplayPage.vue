<template>
  <div class="page-root">
    <div class="mb-3 flex w-full min-w-[1060px] max-w-6xl items-start justify-between gap-6">
      <div class="min-w-0 flex-1">
        <div class="truncate text-left font-[Cambria,serif] text-3xl font-extrabold tracking-tight text-text-main" :title="fileDisplay">
          {{ fileDisplay }}
        </div>
      </div>
      <div ref="menuRoot" class="relative z-[120] flex shrink-0 justify-end">
        <button type="button" class="top-menu-shell top-menu-trigger btn-prominent min-w-[148px] justify-center border-transparent px-4 text-white" @click="menuOpen = !menuOpen">
          <span>{{ $t('replay.toolbar.loadReplay') }}</span>
          <span class="ui-kicker opacity-80">{{ menuOpen ? '^' : 'v' }}</span>
        </button>
        <div v-if="menuOpen" class="absolute right-0 top-full z-[130] mt-2 w-52 overflow-hidden rounded-xl border border-border-main/70 bg-bg-card/96 shadow-xl backdrop-blur-md">
          <button class="replay-menu-item" @click="openReplayFile">{{ $t('replay.toolbar.openFile') }}</button>
          <button class="replay-menu-item" @click="loadLatestReplay">{{ $t('replay.toolbar.loadLatest') }}</button>
        </div>
      </div>
    </div>

    <div class="grid w-full min-w-[1060px] max-w-6xl grid-cols-[442px_minmax(0,1fr)] items-start gap-6">
      <section class="flex flex-col">
        <ReplayMarkSlider
          :losses="losses"
          :current-step="currentStep"
          :threshold="sliderThreshold"
          @update-step="handleSliderStep"
          @update-threshold="updateSliderThreshold"
        />

        <div class="mb-4 mt-2 flex gap-2">
          <input
            :value="currentHex"
            type="text"
            readonly
            spellcheck="false"
            class="flex-1 rounded-lg border border-border-main bg-bg-main px-3 py-2 font-[Consolas,Monaco,monospace] ui-body font-black tracking-[0.06em] text-text-main outline-none"
          />
          <button class="action-btn min-w-[150px]" :disabled="!loaded" @click="jumpToPractice">{{ $t('replay.toolbar.jumpToPractice') }}</button>
        </div>

        <div class="pointer-events-none mx-auto w-full max-w-[438px]">
          <BaseBoard :board="board" :metadata="metadata" :dis32k="dis32k" :is-variant="isVariant" />
        </div>
      </section>

      <section class="flex min-w-0 flex-col gap-4">
        <div class="grid grid-cols-3 gap-2">
          <button class="action-btn" :disabled="!loaded" @click="toggleDemo">{{ demoActive ? $t('replay.actions.stop') : $t('replay.actions.autoDemo') }}</button>
          <button class="action-btn" :disabled="!loaded" @click="stepReplay(1)">{{ $t('replay.actions.step') }}</button>
          <button class="action-btn" :disabled="currentStep <= 0" @click="stepReplay(-1)">{{ $t('replay.actions.undo') }}</button>
          <button class="action-btn" :disabled="!hasNextPoint" @click="nextInaccuracy">{{ $t('replay.actions.nextInaccuracy') }}</button>
          <button class="action-btn" :disabled="!loaded" @click="stepReplay(10)">{{ $t('replay.actions.forward10') }}</button>
          <button class="action-btn" :disabled="currentStep <= 0" @click="stepReplay(-10)">{{ $t('replay.actions.back10') }}</button>
        </div>

        <div class="space-y-3 rounded-[20px] bg-bg-card/72 p-3">
          <div class="rounded-[18px] bg-bg-main/58 p-3">
            <div class="mb-2 flex justify-end">
              <span v-if="resultsRefreshing" class="pill-badge pill-badge-soft">
                {{ $t('common.updating') }}
              </span>
            </div>
            <div class="flex flex-col gap-1.5 transition-opacity" :class="resultsRefreshing ? 'opacity-70' : 'opacity-100'">
              <div
                v-for="item in displayedResults"
                :key="item.dir"
                class="grid grid-cols-[1.1rem_minmax(0,1fr)_19ch] items-center gap-3 rounded-lg border border-border-main/20 bg-bg-main/36 px-3 py-2"
              >
                <span class="text-center ui-body font-black text-text-secondary">{{ dirLabels[item.dir] }}</span>
                <div class="data-bar-track">
                  <div class="data-bar-fill" :style="{ width: `${item.pct}%`, background: item.gradient }" />
                </div>
                <span class="truncate text-right font-black tabular-nums leading-none" :style="getResultValueStyle(item)">{{ item.display }}</span>
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
                <div class="ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">{{ $t('replay.metrics.gof') }}</div>
                <div class="mt-1 ui-metric font-black tabular-nums text-text-main">{{ loaded && currentStep < totalSteps ? goodnessDisplay : '--' }}</div>
              </div>
              <div class="rounded-xl border border-border-main/60 bg-bg-card/55 px-3 py-2.5">
                <div class="ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">{{ $t('replay.metrics.combo') }}</div>
                <div class="mt-1 ui-metric font-black tabular-nums text-text-main">{{ loaded && currentStep < totalSteps ? `${Number(combo ?? 0)}x` : '--' }}</div>
              </div>
              <div class="rounded-xl border border-border-main/60 bg-bg-card/55 px-3 py-2.5">
                <div class="ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">{{ $t('replay.metrics.maxCombo') }}</div>
                <div class="mt-1 ui-metric font-black tabular-nums text-text-main">{{ loaded ? `${summaryMaxCombo}x` : '--' }}</div>
              </div>
            </div>
            <div
              ref="evaluationMixRoot"
              class="relative mt-3"
              @mouseenter="showEvaluationTooltip = true"
              @mouseleave="showEvaluationTooltip = false"
              @mousemove="updateEvaluationTooltip"
            >
              <div class="mb-1.5 ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">{{ $t('replay.metrics.evaluationMix') }}</div>
              <div v-if="evaluationTotal > 0" class="data-bar-track">
                <div
                  v-for="segment in evaluationSegments"
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
                <div class="mb-2 ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">{{ $t('replay.metrics.evaluationStats') }}</div>
                <div class="space-y-1.5">
                  <div v-for="segment in evaluationSegments" :key="`${segment.label}-tooltip`" class="flex items-center justify-between gap-3 ui-caption font-black text-text-main">
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
import ReplayMarkSlider from '../components/ReplayMarkSlider.vue';
import { useReplaySession } from '../composables/useReplaySession';

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

const {
  board,
  metadata,
  currentHex,
  loaded,
  currentStep,
  totalSteps,
  losses,
  sliderThreshold,
  dis32k,
  isVariant,
  menuOpen,
  menuRoot,
  demoActive,
  dirLabels,
  fileDisplay,
  goodnessDisplay,
  summaryMaxCombo,
  displayedResults,
  resultsRefreshing,
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
  evaluationTotal,
  evaluationSegments,
  hasNextPoint,
  combo,
  getResultValueStyle,
  toggleDemo,
  stepReplay,
  handleSliderStep,
  updateSliderThreshold,
  nextInaccuracy,
  openReplayFile,
  loadLatestReplay,
  jumpToPractice,
} = useReplaySession(toRef(props, 'active'), emit);
</script>

<style scoped>
.replay-menu-item {
  width: 100%;
  padding: 0.75rem 0.9rem;
  border: none;
  background: transparent;
  color: var(--text-main);
  text-align: left;
  font-size: var(--font-ui-sm);
  font-weight: 900;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  transition: background 0.2s ease;
}

.replay-menu-item:hover {
  background: color-mix(in srgb, var(--accent) 10%, transparent);
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
</style>
