<template>
  <FitToolPage>
  <div class="page-root gamer-page-root pt-6">
    <div class="gamer-workspace">
      <main class="gamer-main w-full max-w-lg flex flex-col items-center">
      <div class="flex justify-between w-full mb-6 items-center gap-3">
        <div class="flex min-w-0 items-center gap-4">
          <h1 class="shrink-0 text-6xl font-bold text-text-main leading-none">2048</h1>
          <div class="flex min-w-0 flex-col gap-1.5">
            <span :class="['badge-base', wsStatus === 'connected' ? 'badge-connection-connected' : 'badge-connection-pending']">
              {{ $t(`status.${wsStatus.toLowerCase().replace('...', '')}`) }}
            </span>
            <button
              v-if="showRankedStatus"
              type="button"
              :class="['gamer-ranked-badge', `status-${rankedStatus}`]"
              :disabled="rankedStatus !== 'submission_failed'"
              :title="rankedReasonKey ? $t(rankedReasonKey) : $t(`gamer.ranked.status.${rankedStatus}`)"
              @click="retryRankedSubmission"
            >
              {{ $t(`gamer.ranked.status.${ranked.errorCode === 'active_run_exists' ? 'occupied' : rankedStatus}`) }}
            </button>
          </div>
        </div>
        <div class="flex shrink-0 space-x-2">
          <div class="bg-board-bg w-[122px] h-[56px] flex flex-col items-center justify-center rounded-md relative shadow-sm transition-all duration-300">
            <span class="text-text-secondary ui-caption font-black uppercase leading-none mb-1 tracking-tight">{{ $t('labels.score') }}</span>
            <span class="font-black text-white leading-none tabular-nums" style="font-size: calc(22px * var(--ui-scale));">{{ score.current }}</span>

            <transition-group name="float-up" tag="div" class="absolute inset-x-0 bottom-4 pointer-events-none flex justify-center">
              <div v-for="anim in scoreAnimations" :key="anim.id" class="absolute ui-metric text-[#3EB489] font-black z-50 float-anim drop-shadow-[0_0_5px_rgba(62,180,137,0.4)]">
                +{{ anim.value }}
              </div>
            </transition-group>
          </div>
          <div class="bg-board-bg w-[122px] h-[56px] flex flex-col items-center justify-center rounded-md shadow-sm transition-all duration-300">
            <span class="text-text-secondary ui-caption font-black uppercase leading-none mb-1 tracking-tight">{{ $t('labels.best') }}</span>
            <span class="font-black text-white leading-none tabular-nums" style="font-size: calc(22px * var(--ui-scale));">{{ score.best }}</span>
          </div>
        </div>
      </div>

      <div class="w-full mb-4">
        <div class="flex w-full space-x-2">
          <button @click="triggerAction('INIT_GAME')" class="flex-1 bg-btn-bg text-white font-bold py-2 px-2 rounded hover:bg-btn-hover ui-body whitespace-nowrap">
            {{ $t('buttons.newGame') }}
          </button>
          <button @click="triggerAction('UNDO')" class="flex-1 bg-btn-bg text-white font-bold py-2 px-2 rounded hover:bg-btn-hover ui-body whitespace-nowrap">
            {{ $t('buttons.undo') }}
          </button>
          <button
            @click="triggerAction('AI_STEP')"
            :disabled="!aiWorkerReady"
            :class="!aiWorkerReady ? 'opacity-55 cursor-not-allowed' : ''"
            class="flex-1 bg-btn-bg text-white font-bold py-2 px-2 rounded hover:bg-btn-hover ui-body whitespace-nowrap"
          >
            {{ $t('buttons.oneStep') }}
          </button>
          <button
            @click="toggleAI"
            :disabled="!aiWorkerReady"
            :class="[aiEnabled ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600', !aiWorkerReady ? 'opacity-55 cursor-not-allowed' : '']"
            class="flex-1 text-white font-bold py-2 px-2 rounded ui-body whitespace-nowrap"
          >
            {{ aiEnabled ? $t('buttons.aiOn') : $t('buttons.aiOff') }}
          </button>
        </div>
      </div>

      <div class="w-full flex space-x-2 mb-4">
        <input type="text" v-model="hexInput" :placeholder="$t('inputs.hexPlaceholder')" class="flex-1 px-3 py-2 bg-bg-main border border-border-main rounded ui-body text-text-main font-mono tracking-widest outline-none focus:border-accent transition-all" @focus="selectTextInputContentsOnFocus" />
        <button @click="setBoard" class="bg-btn-bg text-white font-bold py-1 px-5 rounded hover:bg-btn-hover ui-body shadow-sm transition-all active:scale-95">
          {{ $t('buttons.set') }}
        </button>
        <button
          type="button"
          @click="writeCurrentBoardToHex"
          class="bg-btn-bg text-white font-bold py-1 px-4 rounded hover:bg-btn-hover ui-body shadow-sm transition-all active:scale-95 whitespace-nowrap"
          :title="$t('buttons.loadCurrentBoard')"
        >
          {{ $t('buttons.loadCurrentBoard') }}
        </button>
      </div>

      <div ref="boardHotkeyTarget" tabindex="-1" class="w-full outline-none focus:outline-none">
        <BaseBoard :frame="boardFrame" @swipe="handleBoardSwipe" />
      </div>

      <div class="w-full mt-6 bg-ctrl-bg rounded-md p-4 flex flex-col space-y-4 shadow-sm">
        <div class="w-full flex justify-between gap-4 mb-4">
          <div class="flex-1 flex flex-col justify-center">
            <span class="text-text-main ui-body font-bold opacity-80 mb-1">{{ $t('labels.gameDifficulty') }}</span>
            <input type="range" class="w-full" :min="0" :max="100" :step="1" v-model.number="difficulty" @change="handleUpdateSettings($event)" />
          </div>
          <div class="flex-1 flex flex-col justify-center">
            <span class="text-text-main ui-body font-bold opacity-80 mb-1">{{ $t('labels.aiSpeed') }}</span>
            <input type="range" class="w-full" :min="0" :max="200" :step="1" v-model.number="aiSpeed" @change="handleUpdateSettings($event)" />
          </div>
        </div>
      </div>
      </main>
      <aside class="gamer-sidebar">
        <div class="gamer-sidebar-content">
        <GamerLeaderboardPanel
          :active="active"
          :ranked-status="rankedStatus"
          @navigate-tab="forwardNavigateTab"
        />
        <GamerToolsPanel
          :active="active"
          :options="matchOptions"
          :table-enabled="aiTableEnabled"
          :tables="aiAvailableTables"
          :selection="aiTableSelection"
          :loading="aiTablesLoading"
          :error="aiTablesError"
          :statistics="gameStatistics"
          :get-replay="exportReplay"
          @load-tables="loadAiTables"
          @table-selection="setAiTableSelection"
          @dialog-open="toolDialogOpen = $event"
          @table-mode="setAiTableEnabled"
          @change="handleMatchOptionChange"
        />
        </div>
      </aside>
    </div>
  </div>
  </FitToolPage>
</template>

<script setup>
import { computed, ref, toRef } from 'vue';

import BaseBoard from '../../../components/BaseBoard.vue';
import FitToolPage from '../../../components/FitToolPage.vue';
import { refocusBoardHotkeyTarget } from '../../../utils/boardHotkeyFocus';
import { selectTextInputContentsOnFocus } from '../../../utils/textInputSelection';
import GamerLeaderboardPanel from '../components/GamerLeaderboardPanel.vue';
import GamerToolsPanel from '../components/GamerToolsPanel.vue';
import { useGamerSession } from '../composables/useGamerSession';

const props = defineProps({
  active: { type: Boolean, default: true },
});

const emit = defineEmits(['navigate-tab']);

const boardHotkeyTarget = ref(null);
const toolDialogOpen = ref(false);

const {
  board,
  boardFrame,
  score,
  wsStatus,
  aiEnabled,
  difficulty,
  aiSpeed,
  hexInput,
  scoreAnimations,
  aiWorkerReady,
  rankedParticipationEnabled,
  aiTableEnabled,
  setAiTableEnabled,
  aiTableSelection,
  aiAvailableTables,
  aiTablesLoading,
  aiTablesError,
  setAiTableSelection,
  loadAiTables,
  gameStatistics,
  exportReplay,
  rankedStatus,
  ranked,
  triggerAction,
  toggleAI,
  updateSettings,
  setBoard,
  writeCurrentBoardToHex,
  setRankedParticipationEnabled,
  retryRankedSubmission,
} = useGamerSession(toRef(props, 'active'), toolDialogOpen);

const visibleRankedStatuses = new Set([
  'starting',
  'ranked_unavailable',
  'ranked',
  'ineligible',
  'submitting',
  'pending',
  'validating',
  'submission_failed',
  'submission_limited',
]);
const showRankedStatus = computed(() => visibleRankedStatuses.has(rankedStatus.value));
const rankedReasonKey = computed(() => ({
  spawn_rate_out_of_range: 'gamer.ranked.reasons.spawnRateOutOfRange',
  duplicate_tab: 'gamer.ranked.reasons.duplicateTab',
  active_run_exists: 'gamer.ranked.reasons.activeRunExists',
  run_creation_rate_limit: 'gamer.ranked.reasons.creationRateLimit',
  lease_lost: 'gamer.ranked.reasons.leaseLost',
  lease_required: 'gamer.ranked.reasons.leaseLost',
  lease_mismatch: 'gamer.ranked.reasons.leaseLost',
  user_changed: 'gamer.ranked.reasons.userChanged',
  user_daily_limit: 'gamer.ranked.reasons.userDailyLimit',
  ip_daily_limit: 'gamer.ranked.reasons.ipDailyLimit',
}[ranked.value.errorCode] || ''));

const matchOptions = computed(() => ([
  {
    key: 'rankedParticipation',
    labelKey: 'gamer.matchOptions.rankedParticipation',
    enabled: rankedParticipationEnabled.value,
    disabled: false,
  },
]));

const forwardNavigateTab = (tabId, detail) => {
  emit('navigate-tab', tabId, detail);
};

const handleMatchOptionChange = (key, enabled) => {
  if (key === 'rankedParticipation') setRankedParticipationEnabled(enabled);
};

const handleUpdateSettings = (event) => {
  updateSettings();
  refocusBoardHotkeyTarget(boardHotkeyTarget, event?.target);
};

const handleBoardSwipe = (direction) => {
  triggerAction('USER_MOVE', { dir: direction });
};
</script>

<style scoped>
.gamer-page-root {
  align-items: center;
}

.gamer-workspace {
  width: min(100%, 52rem);
  display: grid;
  grid-template-columns: minmax(0, 32rem) 19rem;
  align-items: stretch;
  gap: 1rem;
}

.gamer-main,
.gamer-sidebar {
  min-width: 0;
}

.gamer-sidebar {
  position: relative;
  min-height: 0;
  align-self: stretch;
}

.gamer-sidebar-content {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.gamer-ranked-badge {
  display: block;
  min-width: 0;
  max-width: min(100%, 11rem);
  overflow: hidden;
  min-height: 1.45rem;
  padding: 0.18rem 0.55rem;
  border: 1px solid var(--border-main);
  border-radius: 0.35rem;
  color: var(--text-secondary);
  background: color-mix(in srgb, var(--bg-card) 88%, transparent);
  font-size: var(--font-ui-xs);
  font-weight: 900;
  line-height: 1.15;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.gamer-ranked-badge.status-ranked,
.gamer-ranked-badge.status-pending,
.gamer-ranked-badge.status-validating,
.gamer-ranked-badge.status-verified {
  color: #16864b;
  border-color: color-mix(in srgb, #22a95f 45%, var(--border-main));
}

.gamer-ranked-badge.status-submission_failed,
.gamer-ranked-badge.status-submission_limited,
.gamer-ranked-badge.status-rejected,
.gamer-ranked-badge.status-ineligible {
  color: #d14a45;
  border-color: color-mix(in srgb, #d14a45 45%, var(--border-main));
}

.gamer-ranked-badge.status-submission_failed {
  cursor: pointer;
}
</style>
