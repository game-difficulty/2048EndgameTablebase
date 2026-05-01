<template>
  <div class="page-root">
    <div class="relative z-[120] mb-4 flex w-full max-w-6xl items-center justify-between gap-4">
      <div class="flex min-w-0 items-center gap-3">
        <span class="truncate font-[Cambria,serif] text-3xl font-extrabold tracking-tight text-text-main">{{ currentPatternDisplay }}</span>
        <span :class="connectionBadgeClass">{{ $t(`status.${wsStatus}`) }}</span>
      </div>

      <div class="relative z-[130] top-menu-shell">
        <div ref="patternMenuRoot" class="relative z-[140]">
          <button
            type="button"
            class="top-menu-trigger"
            @click="togglePatternMenu"
          >
            <span>{{ selectedPattern || $t('notebook.patternMenu.title') }}</span>
            <span class="ui-kicker opacity-60">{{ patternMenuOpen ? '▲' : '▼' }}</span>
          </button>
          <div
            v-if="patternMenuOpen"
            class="absolute right-0 top-full z-[150] mt-2 min-w-[280px] overflow-hidden rounded-xl border border-border-main bg-bg-card shadow-xl"
          >
            <div class="max-h-[320px] overflow-y-auto p-2">
              <button
                v-if="!hasPatterns"
                type="button"
                disabled
                class="flex w-full cursor-not-allowed items-center rounded-lg bg-bg-main/70 px-3 py-2 text-left ui-control font-black uppercase tracking-tighter text-text-secondary opacity-70"
              >
                {{ $t('notebook.patternMenu.empty') }}
              </button>
              <button
                v-for="pattern in availablePatterns"
                :key="pattern"
                type="button"
                @click.stop="selectPattern(pattern)"
                :class="[
                  'flex w-full items-center rounded-lg px-3 py-2 text-left ui-control font-black uppercase tracking-tighter transition-colors',
                  selectedPattern === pattern ? 'surface-prominent text-white' : 'text-text-main hover:bg-btn-bg/10'
                ]"
              >
                {{ pattern }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="grid w-full max-w-6xl grid-cols-[560px_minmax(0,1fr)] items-start gap-6">
      <section class="flex flex-col">
        <div class="mx-auto w-full max-w-[556px]">
          <BaseBoard :board="board" :metadata="metadata" :dis32k="dis32k" :is-variant="isVariant" @swipe="handleBoardSwipe" />
        </div>
      </section>

      <section class="flex min-w-0 flex-col gap-3">
        <div class="console-card">
          <div class="console-card-header">
            <span>{{ $t('notebook.directions.title') }}</span>
          </div>
          <div class="mt-2 grid grid-cols-2 gap-2">
            <button
              v-for="direction in directionButtons"
              :key="direction"
              type="button"
              :disabled="!canAnswer"
              :class="getDirectionButtonClass(direction)"
              @click="answerDirection(direction)"
            >
              {{ $t(`notebook.directions.${direction.toLowerCase()}`) }}
            </button>
          </div>
        </div>

        <div class="console-card">
          <div class="console-card-header">
            <span>{{ $t('notebook.feedback.title') }}</span>
          </div>
          <div class="mt-2 grid grid-cols-4 gap-2">
            <div class="metric-chip">
              <div class="metric-label">{{ $t('notebook.feedback.combo') }}</div>
              <div class="metric-value">{{ combo }}x</div>
            </div>
            <div class="metric-chip">
              <div class="metric-label">{{ $t('notebook.feedback.remaining') }}</div>
              <div class="metric-value">{{ remaining }}</div>
            </div>
            <div class="metric-chip">
              <div class="metric-label">{{ $t('notebook.feedback.correct') }}</div>
              <div class="metric-value">{{ correct }}</div>
            </div>
            <div class="metric-chip">
              <div class="metric-label">{{ $t('notebook.feedback.incorrect') }}</div>
              <div class="metric-value">{{ incorrect }}</div>
            </div>
          </div>
        </div>

        <div class="console-card">
          <div class="console-card-header mb-2">
            <span>{{ $t('notebook.sampling.title') }}</span>
          </div>
          <div class="flex overflow-hidden rounded-lg border border-border-main/50 shadow-sm">
            <button
              v-for="(mode, index) in sampleModes"
              :key="mode"
              type="button"
              @click="setSampleMode(index)"
              :class="[
                'flex-1 py-2 ui-control font-black uppercase tracking-tighter transition-all',
                sampleMode === index ? 'bg-btn-bg text-white shadow-inner' : 'bg-bg-main text-text-secondary hover:bg-btn-bg/10'
              ]"
            >
              {{ $t(`notebook.sampling.options.${mode}`) }}
            </button>
          </div>
        </div>

        <div class="console-card">
          <div class="console-card-header">
            <span>{{ $t('notebook.actions.title') }}</span>
          </div>
          <div class="mt-2 grid grid-cols-3 gap-2">
            <NotebookCountdownButton
              :disabled="!actionEnabled"
              :active="nextCountdownActive"
              :progress="nextCountdownProgress"
              :countdown-label="nextButtonLabel"
              @click="nextProblem"
            >
              {{ $t('notebook.actions.next') }}
            </NotebookCountdownButton>
            <button type="button" class="action-btn" :disabled="!actionEnabled" @click="deleteCurrent">
              {{ $t('notebook.actions.delete') }}
            </button>
            <button type="button" class="action-btn notebook-btn-accent btn-prominent" @click="jumpToTrainer(emit)">
              {{ $t('notebook.actions.practice') }}
            </button>
          </div>
        </div>

        <div class="console-card">
          <div class="console-card-header">
            <span>{{ $t('notebook.threshold.title') }}</span>
          </div>
          <div class="mt-3 flex items-center gap-2">
            <input
              :value="notebookThresholdInput"
              type="number"
              min="0"
              max="1"
              step="0.001"
              class="notebook-threshold-input"
              @input="updateNotebookThresholdInput($event.target.value)"
              @keydown.enter.prevent="saveNotebookThreshold"
            />
            <button
              type="button"
              class="notebook-threshold-save"
              :disabled="!thresholdSaveEnabled"
              @click="saveNotebookThreshold"
            >
              {{ $t('common.apply') }}
            </button>
          </div>
          <div
            v-if="!thresholdInputIsValid"
            class="mt-2 ui-kicker font-black uppercase tracking-[0.16em] text-rose-500"
          >
            {{ $t('notebook.threshold.invalid') }}
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { toRef } from 'vue';

import BaseBoard from '../../../components/BaseBoard.vue';
import NotebookCountdownButton from '../components/NotebookCountdownButton.vue';
import { useNotebookSession } from '../composables/useNotebookSession';

const props = defineProps({
  active: { type: Boolean, default: false },
});

const emit = defineEmits(['navigate-tab']);

const directionButtons = ['Up', 'Down', 'Left', 'Right'];

const handleBoardSwipe = (direction) => {
  const normalized = String(direction || '').toLowerCase();
  if (!normalized) return;
  answerDirection(normalized.charAt(0).toUpperCase() + normalized.slice(1));
};

const {
  wsStatus,
  board,
  metadata,
  dis32k,
  availablePatterns,
  selectedPattern,
  patternMenuOpen,
  patternMenuRoot,
  combo,
  remaining,
  correct,
  incorrect,
  sampleMode,
  notebookThresholdInput,
  thresholdInputIsValid,
  thresholdSaveEnabled,
  canAnswer,
  currentPatternDisplay,
  isVariant,
  connectionBadgeClass,
  sampleModes,
  hasPatterns,
  actionEnabled,
  nextCountdownActive,
  nextCountdownProgress,
  nextButtonLabel,
  togglePatternMenu,
  selectPattern,
  setSampleMode,
  answerDirection,
  nextProblem,
  deleteCurrent,
  jumpToTrainer,
  updateNotebookThresholdInput,
  saveNotebookThreshold,
  getDirectionButtonClass,
} = useNotebookSession(toRef(props, 'active'));
</script>

<style scoped>
.notebook-direction-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  border-radius: 0.75rem;
  border: 1px solid var(--border-main);
  background: var(--bg-card);
  color: var(--text-main);
  padding: 0.65rem 0.9rem;
  font-size: var(--font-ui-sm);
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  transition: all 0.2s ease;
}

.notebook-direction-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.notebook-direction-btn.is-active {
  background: var(--btn-bg);
  border-color: var(--btn-bg);
  color: white;
}

.notebook-direction-btn.is-correct {
  background: color-mix(in srgb, #2e7d32 90%, white 10%);
  border-color: #2e7d32;
  color: white;
  box-shadow: 0 0 0 1px rgba(46, 125, 50, 0.18), 0 8px 18px rgba(46, 125, 50, 0.18);
}

.notebook-direction-btn.is-wrong {
  background: color-mix(in srgb, #c62828 90%, white 10%);
  border-color: #c62828;
  color: white;
  box-shadow: 0 0 0 1px rgba(198, 40, 40, 0.16), 0 8px 18px rgba(198, 40, 40, 0.16);
}

.notebook-direction-btn.is-disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.notebook-direction-btn:disabled {
  cursor: not-allowed;
  opacity: 1;
}

.notebook-btn-accent {
  color: white;
}

.notebook-btn-accent:hover {
  color: white;
}

.notebook-threshold-input {
  flex: 1;
  min-width: 0;
  appearance: textfield;
  -moz-appearance: textfield;
  border-radius: 0.75rem;
  border: 1px solid var(--border-main);
  background: color-mix(in srgb, var(--bg-main) 82%, transparent);
  color: var(--text-main);
  padding: 0.7rem 0.85rem;
  font-size: var(--font-ui-sm);
  font-weight: 900;
  letter-spacing: 0.04em;
  outline: none;
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}

.notebook-threshold-input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent);
}

.notebook-threshold-input::-webkit-inner-spin-button,
.notebook-threshold-input::-webkit-outer-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

.notebook-threshold-save {
  border-radius: 0.75rem;
  border: 1px solid color-mix(in srgb, var(--accent) 40%, var(--border-main));
  background: color-mix(in srgb, var(--accent) 14%, var(--bg-main));
  color: var(--text-main);
  padding: 0.72rem 0.95rem;
  font-size: var(--font-ui-xs);
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  transition: all 0.18s ease;
}

.notebook-threshold-save:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent);
}

.notebook-threshold-save:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

</style>
