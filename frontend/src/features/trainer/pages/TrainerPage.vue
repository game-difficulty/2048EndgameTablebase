<template>
  <div class="page-root">
    <div class="relative z-[120] w-full max-w-6xl flex items-center justify-between mb-4">
      <div class="flex items-center gap-3">
        <span class="text-3xl font-extrabold tracking-tight text-text-main font-[Cambria,serif]">{{ currentPatternDisplay || '\u00a0' }}</span>
        <span :class="['badge-base', wsStatus === 'connected' ? 'badge-connection-connected' : 'badge-connection-disconnected']">
          {{ $t(`status.${wsStatus}`) }}
        </span>
      </div>

      <div class="relative z-[130] top-menu-shell">
        <div ref="patternMenuRoot" class="relative z-[140]">
          <button
            @click="togglePatternMenu"
            type="button"
            class="top-menu-trigger cursor-pointer"
          >
            <span>{{ patternType || $t('trainer.top.selectPattern') }}</span>
            <span class="ui-kicker opacity-60">{{ patternMenuOpen ? '▲' : '▼' }}</span>
          </button>
          <div
            v-if="patternMenuOpen"
            class="absolute right-0 top-full z-[150] mt-2 flex min-w-[360px] overflow-hidden rounded-xl border border-border-main bg-bg-card shadow-xl"
          >
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
                  activePatternCategory === group.category
                    ? 'bg-btn-bg text-white'
                    : 'text-text-main hover:bg-btn-bg/10'
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
                @click.stop="handlePatternSelect(pattern, $event)"
                :class="[
                  'rounded-lg px-3 py-2 text-left ui-control font-black transition-colors',
                  patternType === pattern
                    ? 'surface-prominent text-white'
                    : 'bg-bg-main text-text-main hover:bg-btn-bg/10'
                ]"
              >
                {{ pattern }}
              </button>
            </div>
          </div>
        </div>
        <span class="text-text-secondary font-bold opacity-30 truncate">|</span>
        <select
          v-model="targetValue"
          @change="handleTargetChange"
          class="top-menu-select"
        >
          <option value="" class="bg-bg-card text-text-main">{{ $t('trainer.top.selectTarget') }}</option>
          <option v-for="t in availableTargets" :key="t" :value="t" class="bg-bg-card text-text-main">{{ t }}</option>
        </select>
        <button
          @click="selectFolder"
          class="ml-2 ui-kicker bg-btn-bg hover:bg-btn-hover text-white px-2.5 py-1.5 rounded font-black uppercase tracking-tighter transition-all active:scale-95 shadow-sm"
        >
          {{ $t('trainer.top.path') }}
        </button>
        <button
          @click="applyTablebase"
          class="btn-prominent ui-kicker px-2.5 py-1.5 rounded font-black uppercase tracking-tighter transition-all active:scale-95 shadow-sm"
        >
          {{ $t('trainer.top.load') }}
        </button>
      </div>
    </div>

    <div class="relative z-0 w-full max-w-6xl flex flex-row gap-6 items-start">
      <div class="flex flex-col items-center" style="width: 442px; min-width: 240px;">
        <div class="w-full flex gap-2 mb-3">
          <input
            type="text"
            v-model="hexInput"
            data-trainer-hex-input="true"
            :placeholder="$t('trainer.board.hexPlaceholder')"
            class="flex-1 px-3 py-2 bg-bg-main border border-border-main rounded ui-body font-mono text-text-main outline-none focus:border-accent transition-all placeholder:opacity-50"
          />
          <button @click="setBoard" class="action-btn ui-control px-4 !bg-btn-bg !text-white border-btn-bg hover:bg-btn-hover font-black uppercase shadow-sm">{{ $t('trainer.board.set') }}</button>
        </div>

        <div ref="boardHotkeyTarget" tabindex="-1" class="w-full outline-none focus:outline-none">
          <BaseBoard :board="board" :metadata="metadata" :dis32k="dis32k" :is-variant="isVariant" @cell-click="handleCellClick" />
        </div>

        <div class="w-full mt-4 bg-bg-card border border-border-main rounded-lg p-3 shadow-sm">
          <div class="flex justify-between items-center mb-2">
            <span class="ui-kicker font-black text-text-secondary uppercase tracking-widest leading-none">{{ $t('labels.tilePalette') }}</span>
            <div class="flex items-center gap-3">
              <label class="flex items-center gap-1 ui-kicker text-text-secondary cursor-pointer select-none font-bold uppercase tracking-tighter">
                <input type="checkbox" v-model="dis32k" @change="onDis32kChange" class="cursor-pointer accent-accent" />
                {{ $t('trainer.board.hide32k') }}
              </label>
              <span
                class="ui-kicker font-bold px-2 py-0.5 rounded-full shadow-sm"
                :style="currentPaletteValue === null
                  ? 'background: var(--bg-main); color: var(--text-secondary); border: 1px solid var(--border-main);'
                  : `background: var(--color-tile-${currentPaletteValue}); color: var(--color-text-${currentPaletteValue}); font-weight: bold;` "
              >
                {{ currentPaletteValue === null ? $t('trainer.palette.browse') : currentPaletteValue === 0 ? $t('trainer.palette.erase') : currentPaletteValue }}
              </span>
            </div>
          </div>
          <div class="grid grid-cols-8 gap-1.5">
            <button
              @click="togglePalette(0)"
              :class="['palette-btn', currentPaletteValue === 0 ? 'ring-2 ring-accent shadow-lg scale-110' : '']"
              style="background: var(--bg-main); color: var(--text-secondary); border: 1px solid var(--border-main);"
            >
              {{ $t('trainer.palette.erase') }}
            </button>
            <button
              v-for="val in cellPalette"
              :key="val"
              @click="togglePalette(val)"
              :class="['palette-btn', currentPaletteValue === val ? 'ring-2 ring-accent shadow-lg scale-110' : '']"
              :style="`background: var(--color-tile-${val}); color: var(--color-text-${val}); font-weight: bold;` "
            >
              {{ val >= 1024 ? `${val / 1024}k` : val }}
            </button>
          </div>
        </div>
      </div>

      <div class="flex-1 min-w-[340px] flex flex-col gap-3">
        <div class="console-card">
          <div class="console-card-header border-b border-border-main/20 pb-2 mb-2">
            <span>{{ $t('trainer.results.title') }}</span>
            <div class="flex items-center gap-2">
              <span v-if="replayResultsActive" class="pill-badge pill-badge-accent">
                {{ $t('trainer.results.replay') }}
              </span>
              <span
                class="pill-badge pill-badge-soft transition-opacity"
                :class="resultsUpdatingVisible ? 'opacity-100' : 'opacity-0'"
              >
                {{ $t('common.updating') }}
              </span>
              <label class="flex items-center gap-1 ui-kicker text-text-secondary cursor-pointer select-none font-bold uppercase tracking-tighter">
                <input type="checkbox" v-model="showResults" class="cursor-pointer accent-accent" />
                {{ $t('trainer.results.auto') }}
              </label>
              <button @click="queryResults" class="ui-kicker !bg-btn-bg !text-white border border-btn-bg hover:bg-btn-hover px-2.5 py-1 rounded font-black uppercase tracking-tighter transition-all active:scale-95 shadow-sm">
                {{ $t('trainer.results.refresh') }}
              </button>
            </div>
          </div>
          <template v-if="showResults && !awaitingSpawn">
            <div class="mt-2 flex flex-col gap-1 transition-opacity" :class="resultsUpdatingVisible ? 'opacity-70' : 'opacity-100'">
              <div
                v-for="item in displayedResults"
                :key="item.dir"
                class="grid grid-cols-[1.25rem_23ch_minmax(0,1fr)] items-center gap-3 px-3 py-2 rounded-lg border border-border-main/20 bg-bg-main/30 group hover:bg-bg-main/50 transition-colors"
                :style="getResultRowStyle(item)"
              >
                <span class="text-[16px] font-black w-5 text-center text-text-secondary group-hover:text-text-main">{{ dirLabels[item.dir] }}</span>
                <div class="min-w-0 overflow-hidden">
                  <span
                    class="block w-full whitespace-nowrap text-left font-mono font-black tabular-nums leading-none"
                    :style="getResultValueStyle(item)"
                  >
                    {{ item.display }}
                  </span>
                </div>
                <div class="data-bar-track">
                  <div
                    class="data-bar-fill shadow-[0_0_8px_rgba(0,0,0,0.1)]"
                    :style="{ width: `${item.pct}%`, background: item.gradient }"
                  ></div>
                </div>
              </div>
            </div>
            <div class="mt-1.5 ui-kicker text-text-secondary opacity-50 font-mono">{{ $t('trainer.results.dtype') }}: {{ tableResult.dtype }}</div>
          </template>
          <div v-else class="mt-2 ui-kicker text-text-secondary italic opacity-60">
            {{ awaitingSpawn ? $t('trainer.results.awaitingManualSpawn') : $t('trainer.results.hidden') }}
          </div>
        </div>

        <div class="console-card">
          <div class="console-card-header"><span>{{ $t('trainer.actions.title') }}</span></div>
          <div class="grid grid-cols-4 gap-2 mt-2">
            <button @click="toggleDemo" :class="['action-btn', demoActive ? 'bg-red-500 hover:bg-red-600 text-white' : '']">
              {{ demoActive ? $t('trainer.actions.stop') : $t('trainer.actions.demo') }}
            </button>
            <button @click="trainerStep" class="action-btn">{{ $t('trainer.actions.step') }}</button>
            <button @click="trainerUndo" class="action-btn">{{ $t('trainer.actions.undo') }}</button>
            <button @click="trainerDefault" class="action-btn">{{ $t('trainer.actions.default') }}</button>
          </div>
        </div>

        <div class="console-card">
          <div class="console-card-header mb-2"><span>{{ $t('trainer.spawnMode.title') }}</span></div>
          <div class="flex mt-2 rounded-lg overflow-hidden border border-border-main/50 shadow-sm">
            <button
              v-for="(_, idx) in spawnModes"
              :key="idx"
              @click="setSpawnMode(idx)"
              :class="['flex-1 py-2 ui-control font-black uppercase tracking-tighter transition-all',
                spawnMode === idx
                  ? 'bg-btn-bg text-white shadow-inner'
                  : 'bg-bg-main text-text-secondary hover:bg-btn-bg/10']"
            >
              {{ $t(`trainer.spawnMode.options.${spawnModeLabelKeys[idx]}`) }}
            </button>
          </div>
        </div>

        <div class="console-card">
          <div class="console-card-header"><span>{{ $t('trainer.transform.title') }}</span></div>
          <div class="grid grid-cols-4 gap-2 mt-2">
            <button @click="triggerAction('ROTATE', { type: 'UD' })" class="action-btn">{{ $t('trainer.transform.vFlip') }}</button>
            <button @click="triggerAction('ROTATE', { type: 'LR' })" class="action-btn">{{ $t('trainer.transform.hFlip') }}</button>
            <button @click="triggerAction('ROTATE', { type: 'R90' })" class="action-btn">{{ $t('trainer.transform.r90') }}</button>
            <button @click="triggerAction('ROTATE', { type: 'L90' })" class="action-btn">{{ $t('trainer.transform.l90') }}</button>
          </div>
        </div>

        <div class="console-card">
          <div class="console-card-header mb-2 leading-none">
            <span>{{ $t('trainer.recording.title') }}</span>
            <span class="ui-control font-mono text-text-secondary opacity-60">{{ recordStep }} / {{ recordMax }}</span>
          </div>
          <div class="grid grid-cols-4 gap-2 mt-2">
            <button @click="manageRecord('OPEN')" class="action-btn">{{ $t('trainer.recording.load') }}</button>
            <button
              @click="manageRecord('TOGGLE')"
              :class="recordingState ? '!bg-red-500 !border-red-600 hover:!bg-red-600' : ''"
              class="action-btn"
            >
              {{ recordingState ? $t('trainer.recording.save') : $t('trainer.recording.record') }}
            </button>
            <button @click="manageRecord('PREV')" class="action-btn font-black ui-control">{{ $t('trainer.recording.prev') }}</button>
            <button @click="manageRecord('NEXT')" class="action-btn font-black ui-control">{{ $t('trainer.recording.next') }}</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, toRef } from 'vue';

import BaseBoard from '../../../components/BaseBoard.vue';
import { refocusBoardHotkeyTarget } from '../../../utils/boardHotkeyFocus';
import { useTrainerSession } from '../composables/useTrainerSession';

const props = defineProps({
  active: { type: Boolean, default: true },
});

const spawnModeLabelKeys = ['random', 'best', 'worst', 'manual'];
const boardHotkeyTarget = ref(null);

const {
  currentPatternDisplay,
  isVariant,
  wsStatus,
  togglePatternMenu,
  selectPattern,
  patternType,
  patternMenuOpen,
  patternGroups,
  activePatternCategory,
  activePatternOptions,
  targetValue,
  availableTargets,
  onPatternChange,
  selectFolder,
  applyTablebase,
  hexInput,
  setBoard,
  board,
  metadata,
  dis32k,
  handleCellClick,
  awaitingSpawn,
  currentPaletteValue,
  togglePalette,
  cellPalette,
  replayResultsActive,
  showResults,
  queryResults,
  sortedResults,
  displayedResults,
  resultsRefreshing,
  resultsUpdatingVisible,
  getResultRowStyle,
  dirLabels,
  getResultValueStyle,
  tableResult,
  toggleDemo,
  demoActive,
  trainerStep,
  trainerUndo,
  trainerDefault,
  spawnModes,
  spawnMode,
  setSpawnMode,
  triggerAction,
  recordStep,
  recordMax,
  recordingState,
  manageRecord,
  onDis32kChange,
  patternMenuRoot,
} = useTrainerSession(toRef(props, 'active'));

const focusBoardHotkeys = (event) => {
  refocusBoardHotkeyTarget(boardHotkeyTarget, event?.target);
};

const handlePatternSelect = (pattern, event) => {
  selectPattern(pattern);
  focusBoardHotkeys(event);
};

const handleTargetChange = (event) => {
  onPatternChange();
  focusBoardHotkeys(event);
};
</script>

<style scoped>
.palette-btn {
  aspect-ratio: 1;
  border-radius: 0.48rem;
  font-size: var(--font-ui-2xs);
  font-weight: 900;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  border: 1px solid rgba(0, 0, 0, 0.1);
}

.palette-btn:hover {
  transform: scale(1.15) rotate(2deg);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
  z-index: 10;
}

.palette-btn:active {
  transform: scale(0.9);
}
</style>
