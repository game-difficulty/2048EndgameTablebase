<template>
  <teleport to="body">
    <div
      v-if="open"
      class="analysis-dialog-overlay"
      @click.self="$emit('close')"
    >
      <div role="dialog" aria-modal="true" aria-labelledby="replay-analysis-title" :class="['analysis-dialog-shell', { 'analysis-dialog-shell--history': historyMode }]">
        <div class="analysis-dialog-header flex items-center justify-between border-b border-border-main/60 px-6 py-4">
          <div class="min-w-0">
            <div id="replay-analysis-title" class="analysis-dialog-title">{{ historyMode ? (String(locale).startsWith('zh') ? '分析历史' : 'Analysis history') : $t('analysis.title') }}</div>
          </div>
          <div class="flex min-w-0 items-center gap-3">
            <button
              class="rounded-full border border-border-main bg-bg-main/80 px-3 py-1.5 ui-control font-black text-text-main"
              @click="historyMode = !historyMode"
            >
              {{ historyMode ? $t('analysis.title') : (String(locale).startsWith('zh') ? '分析历史' : 'History') }}
            </button>
            <button v-if="historyMode" type="button" class="analysis-header-refresh" :title="String(locale).startsWith('zh') ? '刷新历史' : 'Refresh history'" :aria-label="String(locale).startsWith('zh') ? '刷新历史' : 'Refresh history'" @click="historyPanelRef?.refresh()">
              <RefreshCw :size="18" aria-hidden="true" />
            </button>
            <span v-else :class="[statusBadgeClass, 'badge-state-compact']" :title="statusBadgeText">{{ statusBadgeText }}</span>
            <button
              class="rounded-full border border-border-main bg-bg-main/80 px-3 py-1.5 ui-control font-black uppercase tracking-wider text-text-main transition-colors hover:border-accent/40 hover:text-accent"
              @click="$emit('close')"
            >
              {{ $t('analysis.close') }}
            </button>
          </div>
        </div>

        <div v-if="historyMode" class="analysis-history-view">
          <AnalysisHistoryPanel ref="historyPanelRef" :language="String(locale)" />
        </div>
        <div v-else class="analysis-dialog-body grid grid-cols-[minmax(340px,0.95fr)_minmax(0,1.05fr)] gap-5 p-6">
          <section class="analysis-input-section">
            <div class="ui-control font-black uppercase tracking-[0.24em] text-text-secondary">{{ $t('analysis.input.title') }}</div>
            <div class="mt-4 space-y-4">
              <div class="grid grid-cols-[minmax(0,2fr)_minmax(0,1fr)] gap-3">
                <div ref="patternMenuRoot" class="relative">
                  <button
                    type="button"
                    class="analysis-input-btn"
                    @click="patternMenuOpen = !patternMenuOpen"
                  >
                    <span class="truncate">{{ selectedPattern || $t('analysis.input.selectPattern') }}</span>
                    <span class="ui-kicker opacity-60">{{ patternMenuOpen ? '^' : 'v' }}</span>
                  </button>
                  <div
                    v-if="patternMenuOpen"
                    class="absolute left-0 top-full z-[240] mt-2 flex min-w-[360px] overflow-hidden rounded-xl border border-border-main bg-bg-card shadow-xl"
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
                          activePatternCategory === group.category ? 'bg-btn-bg text-white' : 'text-text-main hover:bg-btn-bg/10'
                        ]"
                      >
                        {{ $t(`patternCategories.${group.category}`) }}
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

                <GoalPicker
                  v-model="selectedTarget"
                  class="analysis-select-shell w-full"
                  :options="targetOptions"
                  aria-label="Analysis target"
                  trigger-class="analysis-select-trigger w-full min-h-[3.25rem] rounded-[0.9rem] border border-border-main bg-bg-card px-[0.95rem] py-[0.78rem] ui-control font-black uppercase tracking-[0.06em] text-text-main"
                  option-class="ui-control font-black uppercase tracking-[0.06em]"
                  menu-class="z-[240]"
                  @change="markTargetSelection"
                />
              </div>

              <div>
                <div class="mb-2 ui-caption font-black uppercase tracking-[0.22em] text-text-secondary">{{ $t('analysis.input.paths') }}</div>
                <textarea
                  :value="displayPaths"
                  readonly
                  spellcheck="false"
                  class="analysis-textarea"
                  :placeholder="$t('analysis.input.pathsPlaceholder')"
                />
                <div class="mt-2 ui-caption font-black text-text-secondary/90">
                  {{ $t('analysis.notes') }}
                </div>
              </div>

              <div class="grid grid-cols-2 gap-3">
                <button class="analysis-secondary-btn" @click="pickFiles">
                  {{ $t('analysis.input.selectFiles') }}
                </button>
                <button class="analysis-primary-btn btn-prominent" :disabled="!canAnalyze || isRunning" @click="startAnalysis">
                  {{ isRunning ? $t('analysis.progress.running') : $t('analysis.input.analyze') }}
                </button>
              </div>
              <div v-if="analysisError" class="analysis-error" role="alert">
                {{ analysisError }}
              </div>
              <button
                v-if="downloadUrl"
                class="analysis-secondary-btn"
                :disabled="isDownloading"
                @click="downloadResults"
              >
                {{ isDownloading ? $t('analysis.input.downloadingResults') : $t('analysis.input.downloadResults') }}
              </button>
            </div>
          </section>

          <section class="analysis-results-section">
            <div class="flex items-center justify-between">
              <div class="ui-control font-black uppercase tracking-[0.24em] text-text-secondary">{{ $t('analysis.progress.title') }}</div>
              <span :class="[statusBadgeClass, 'badge-state-compact']" :title="statusBadgeText">{{ statusBadgeText }}</span>
            </div>

            <div class="analysis-progress-summary">
              <div class="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-4">
                <div class="min-w-0">
                  <div class="ui-caption font-black uppercase tracking-[0.22em] text-text-secondary">{{ $t('analysis.progress.currentFile') }}</div>
                  <div class="mt-1 truncate ui-body font-black text-text-main" :title="currentFileDisplay">
                    {{ currentFileDisplay }}
                  </div>
                </div>
                <div class="min-w-[96px] shrink-0 whitespace-nowrap text-right">
                  <div class="ui-caption font-black uppercase tracking-[0.22em] text-text-secondary">{{ $t('analysis.progress.completed') }}</div>
                  <div class="mt-1 text-2xl font-black text-text-main">{{ completedCount }} / {{ totalCount }}</div>
                </div>
              </div>
              <div class="mt-4 h-3 overflow-hidden rounded-full bg-border-main/25">
                <div class="h-full rounded-full bg-gradient-to-r from-accent/50 to-accent transition-all duration-300" :style="{ width: `${progressPercent}%` }" />
              </div>
              <div class="mt-3 flex items-center gap-4 ui-caption font-black uppercase tracking-[0.18em] text-text-secondary">
                <span>{{ $t('analysis.progress.done') }} {{ doneCount }}</span>
                <span>{{ $t('analysis.progress.failed') }} {{ failedCount }}</span>
              </div>
            </div>

            <div class="analysis-results-list">
              <div
                v-if="visibleEntries.length"
                class="analysis-list-viewport"
                tabindex="0"
                :aria-label="$t('analysis.progress.title')"
              >
                <div
                  v-for="entry in visibleEntries"
                  :key="entry.key"
                  class="analysis-list-row"
                >
                  <div class="analysis-entry-content">
                    <div class="analysis-entry-title">{{ entry.label }}</div>
                    <div class="analysis-entry-meta">{{ entry.variant?.replace('x', '×') }}{{ entry.variant ? ' · ' : '' }}{{ selectedPattern }}-{{ selectedTarget }}</div>
                    <div v-if="entry.message" class="mt-0.5 truncate ui-caption font-black text-red-500/85" :title="userError(entry.message)">{{ userError(entry.message) }}</div>
                    <AnalysisStageList v-if="entry.artifacts?.length" :artifacts="entry.artifacts" :language="String(locale)" :opening="openingArtifact" @open="artifact => openAnalysisReplay(artifact.artifact_id)" />
                  </div>
                  <div
                    class="badge-state"
                    :class="entry.status === 'done' ? 'badge-state-success' : (entry.status === 'failed' ? 'badge-state-failure' : 'badge-state-running')"
                  >
                    {{ getEntryStatusLabel(entry.status) }}
                  </div>
                </div>
              </div>
              <div v-else class="rounded-xl border border-dashed border-border-main/60 bg-bg-main/50 px-3 py-5 text-center ui-control font-black uppercase tracking-[0.18em] text-text-secondary">
                {{ $t('analysis.progress.empty') }}
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  </teleport>
</template>

<script setup>
import { computed, onUnmounted, ref, watch } from 'vue';
import { RefreshCw } from '@lucide/vue';
import { useI18n } from 'vue-i18n';
import { openAsyncLink } from '../../../services/openAsyncLink.js';
import { userError } from '../../../services/errors/userError.js';

import GoalPicker from '../../../components/GoalPicker.vue';
import AnalysisHistoryPanel from '../../../components/AnalysisHistoryPanel.vue';
import AnalysisStageList from './AnalysisStageList.vue';
import { analysisScoreLabel, replayDisplayName } from '../analysisPresentation.js';
import { downloadResponse, pickBrowserFiles, postMultipart } from '../../../services/files/browserFiles';
import { useAuthState } from '../../../services/auth/authState';
import { emitTokenBalanceUpdated } from '../../../services/auth/authEvents';
import { authHeaders } from '../../../services/auth/sessionTokenStore';
import { getBackendUrl } from '../../../services/runtime/backendUrl';
import {
  fetchTablebaseCatalog,
  getCatalogTargets,
  getCatalogTargetsForPattern,
  groupTablebasePatternsByCategory,
} from '../../../services/tablebases/catalogClient';
import { createWsClient } from '../../../services/ws/createWsClient';

const props = defineProps({
  open: { type: Boolean, default: false },
  context: {
    type: Object,
    default: () => ({}),
  },
});

defineEmits(['close']);

const { t, locale } = useI18n();
const { requireAuth } = useAuthState();

const wsStatus = ref('disconnected');
const categories = ref({});
const targetTiles = ref([]);
const catalogTables = ref([]);
const selectedPattern = ref('');
const selectedTarget = ref('2048');
const userSelectionTouched = ref(false);
const pathsInput = ref('');
const selectedFiles = ref([]);
const downloadUrl = ref('');
const patternMenuOpen = ref(false);
const patternMenuRoot = ref(null);
const activePatternCategory = ref('');
const isRunning = ref(false);
const completedCount = ref(0);
const totalCount = ref(0);
const doneCount = ref(0);
const failedCount = ref(0);
const currentFile = ref('');
const entries = ref([]);
const openingArtifact = ref('');
const analysisError = ref('');
const isDownloading = ref(false);
const activeJobId = ref('');
const historyMode = ref(false);
const historyPanelRef = ref(null);

let client = null;
let pollTimer = null;
let statusRequest = null;
const ACTIVE_ANALYSIS_JOB_KEY = '2048tables:analysis-active-job:v1';
const ANALYSIS_POLL_INTERVAL_MS = 2500;

const patternGroups = computed(() =>
  Object.entries(categories.value || {}).map(([category, items]) => ({
    category,
    items: Array.isArray(items) ? items : [],
  }))
);

const activePatternOptions = computed(() => {
  const group = patternGroups.value.find((item) => item.category === activePatternCategory.value);
  return group?.items || [];
});

const availableTargetsForPattern = computed(() => {
  if (!catalogTables.value.length) {
    return [];
  }
  return getCatalogTargetsForPattern(catalogTables.value, selectedPattern.value);
});

const targetOptions = computed(() =>
  availableTargetsForPattern.value.map((target) => ({
    value: target,
    label: target,
  }))
);

const currentFileDisplay = computed(() => currentFile.value ? replayDisplayName(currentFile.value, String(locale.value).startsWith('zh') ? '当前对局' : 'Current game') : t('analysis.progress.idle'));
const displayPaths = computed(() => pathsInput.value ? pathsInput.value.split('\n').map((name, index) => replayDisplayName(name, `${String(locale.value).startsWith('zh') ? '对局' : 'Game'} ${index + 1}`)).join('\n') : '');
const progressPercent = computed(() => {
  if (totalCount.value <= 0) return 0;
  return Math.max(0, Math.min(100, (completedCount.value / totalCount.value) * 100));
});
const canAnalyze = computed(() => Boolean(selectedPattern.value && selectedTarget.value && selectedFiles.value.length));
const normalizedEntries = computed(() =>
  entries.value.map((entry, index) => ({
    key: `${entry.filename || entry.path}-${entry.status}-${index}`,
    path: entry.filename || entry.path,
    label: analysisScoreLabel(entry.score, locale.value) || replayDisplayName(entry.filename || entry.path, `${String(locale.value).startsWith('zh') ? '对局' : 'Game'} ${index + 1}`),
    variant: entry.variant,
    status: entry.status,
    message: entry.message || '',
    artifacts: entry.artifacts || [],
  }))
);

async function openAnalysisReplay(artifactId) {
  openingArtifact.value = artifactId;
  try {
    await openAsyncLink(async () => {
    const response = await fetch(getBackendUrl(`/api/analysis/replays/${encodeURIComponent(artifactId)}/open-link`), {
      method: 'POST', credentials: 'include', headers: authHeaders({ Accept: 'application/json' }),
    });
    if (!response.ok) throw new Error(String(response.status));
    const data = await response.json();
    return data.url;
    });
  } catch {
    analysisError.value = String(locale.value).startsWith('zh') ? '回放暂时无法打开。' : 'The replay cannot be opened right now.';
  } finally {
    openingArtifact.value = '';
  }
}
const visibleEntries = normalizedEntries;

const statusBadgeText = computed(() => {
  if (isRunning.value) return t('analysis.progress.running');
  if (failedCount.value > 0 && completedCount.value >= totalCount.value && totalCount.value > 0) {
    return t('analysis.progress.completedWithErrors');
  }
  if (completedCount.value > 0 && completedCount.value >= totalCount.value) {
    return t('analysis.progress.completedState');
  }
  return wsStatus.value === 'connected' ? t('status.connected') : t('status.connecting');
});

const statusBadgeClass = computed(() => {
  if (isRunning.value) return 'badge-state badge-state-running';
  if (failedCount.value > 0 && completedCount.value >= totalCount.value && totalCount.value > 0) {
    return 'badge-state badge-state-failure';
  }
  if (wsStatus.value === 'connected') {
    return 'badge-state badge-state-success';
  }
  return 'badge-state badge-state-neutral';
});

const closePatternMenuOnClick = (event) => {
  if (!patternMenuOpen.value || !patternMenuRoot.value) return;
  if (!patternMenuRoot.value.contains(event.target)) {
    patternMenuOpen.value = false;
  }
};

const getEntryStatusLabel = (status) => {
  if (status === 'done') return t('analysis.progress.doneState');
  if (status === 'failed') return t('analysis.progress.failedState');
  return t('analysis.progress.queued');
};

const selectPattern = (pattern) => {
  userSelectionTouched.value = true;
  selectedPattern.value = pattern;
  ensureValidSelection();
  patternMenuOpen.value = false;
};

const markTargetSelection = () => {
  userSelectionTouched.value = true;
};

const applyContext = (context) => {
  if (context?.analysisFile instanceof File) {
    selectedFiles.value = [context.analysisFile];
    pathsInput.value = context.analysisFile.name;
  }
  const nextPattern = String(context?.pattern || '').trim();
  const nextTarget = String(context?.target || '').trim();
  if (nextPattern) {
    selectedPattern.value = nextPattern;
    const matchedGroup = patternGroups.value.find((group) => group.items.includes(nextPattern));
    if (matchedGroup) activePatternCategory.value = matchedGroup.category;
  }
  ensureValidSelection();
  if (nextTarget && availableTargetsForPattern.value.includes(nextTarget)) {
    selectedTarget.value = nextTarget;
  }
};

const preferredTargetFrom = (targets) => (
  targets.includes('512') ? '512' : (targets[0] || '')
);

const ensureValidSelection = () => {
  const groups = patternGroups.value;
  const allPatterns = groups.flatMap((group) => group.items);
  if (allPatterns.length && !allPatterns.includes(selectedPattern.value)) {
    selectedPattern.value = allPatterns[0] || '';
  }
  if (!activePatternCategory.value && groups.length) {
    activePatternCategory.value = groups[0].category;
  }
  const matchedGroup = groups.find((group) => group.items.includes(selectedPattern.value));
  if (matchedGroup) {
    activePatternCategory.value = matchedGroup.category;
  }
  const targets = availableTargetsForPattern.value;
  if (!targets.length) {
    selectedTarget.value = '';
    return;
  }
  if (!targets.includes(selectedTarget.value)) {
    selectedTarget.value = preferredTargetFrom(targets);
  }
};

const loadCatalog = async () => {
  try {
    const tables = await fetchTablebaseCatalog();
    catalogTables.value = tables;
    const nextCategories = groupTablebasePatternsByCategory(tables);
    const patterns = Object.values(nextCategories).flat();
    categories.value = nextCategories;
    targetTiles.value = getCatalogTargets(tables);
    if (patterns.length) {
      ensureValidSelection();
      if (!userSelectionTouched.value) {
        applyContext(props.context);
      }
    } else {
      selectedPattern.value = '';
      selectedTarget.value = '';
      activePatternCategory.value = '';
    }
  } catch (error) {
    console.error(error);
  }
};

const shouldRelaxAnalysisFileAccept = () => {
  if (typeof navigator === 'undefined') return false;
  const userAgent = navigator.userAgent || '';
  const platform = navigator.platform || '';
  return /iPad|iPhone|iPod/u.test(userAgent)
    || (platform === 'MacIntel' && Number(navigator.maxTouchPoints || 0) > 1);
};

const pickFiles = async () => {
  const options = { multiple: true };
  if (!shouldRelaxAnalysisFileAccept()) {
    options.accept = '.txt,.vrs,.rpl';
  }
  const files = await pickBrowserFiles(options);
  if (files.length) {
    analysisError.value = '';
    selectedFiles.value = files;
    pathsInput.value = files.map((file) => file.name).join('\n');
    downloadUrl.value = '';
  }
};

const formatAnalysisError = (error, phase = '') => {
  if (error?.code === 'REMOTE_TABLEBASE_OFFLINE' || error?.code === 'REMOTE_TABLEBASE_TIMEOUT') {
    return t('analysis.errors.tablebaseUnavailable');
  }
  if (error?.code === 'NETWORK_ERROR' || /failed to fetch/i.test(String(error?.message || ''))) {
    if (phase === 'upload') return t('analysis.errors.uploadNetwork');
    if (phase === 'status') return t('analysis.errors.statusNetwork');
    if (phase === 'download') return t('analysis.errors.downloadNetwork');
    return t('analysis.errors.network');
  }
  if (error?.status === 401) {
    return t('analysis.errors.authRequired');
  }
  if (error?.status === 402 || error?.code === 'INSUFFICIENT_TOKENS') {
    return t('analysis.errors.insufficientTokens');
  }
  if (error?.status === 404 || /analysis job not found/i.test(String(error?.message || ''))) {
    return t('analysis.errors.expired');
  }
  if (error?.status === 409) {
    return t('analysis.errors.notReady');
  }
  return userError(error, t('analysis.errors.generic'));
};

const readStoredAnalysisJob = () => {
  if (typeof sessionStorage === 'undefined') return null;
  try {
    const raw = sessionStorage.getItem(ACTIVE_ANALYSIS_JOB_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const jobId = String(parsed?.job_id || '').trim();
    if (!jobId) return null;
    return { ...parsed, job_id: jobId };
  } catch (_error) {
    return null;
  }
};

const storeAnalysisJob = (payload = {}) => {
  const jobId = String(payload?.job_id || activeJobId.value || '').trim();
  if (!jobId || typeof sessionStorage === 'undefined') return;
  const previous = readStoredAnalysisJob() || {};
  const record = {
    ...previous,
    job_id: jobId,
    pattern: payload.pattern || selectedPattern.value || previous.pattern || '',
    target: payload.target || selectedTarget.value || previous.target || '',
    status: payload.status || previous.status || (isRunning.value ? 'running' : ''),
    total: Number(payload.total ?? previous.total ?? totalCount.value ?? 0),
    completed: Number(payload.completed ?? previous.completed ?? completedCount.value ?? 0),
    done: Number(payload.done ?? previous.done ?? doneCount.value ?? 0),
    failed: Number(payload.failed ?? previous.failed ?? failedCount.value ?? 0),
    download_url: payload.download_url || previous.download_url || downloadUrl.value || '',
    updated_at: Date.now(),
    created_at: previous.created_at || Date.now(),
  };
  sessionStorage.setItem(ACTIVE_ANALYSIS_JOB_KEY, JSON.stringify(record));
};

const clearStoredAnalysisJob = (jobId = '') => {
  if (typeof sessionStorage === 'undefined') return;
  const current = readStoredAnalysisJob();
  if (jobId && current?.job_id && current.job_id !== jobId) return;
  sessionStorage.removeItem(ACTIVE_ANALYSIS_JOB_KEY);
};

const applyAnalysisJobPayload = (payload = {}) => {
  const status = String(payload.status || '').toLowerCase();
  const jobId = String(payload.job_id || activeJobId.value || '').trim();
  if (jobId) activeJobId.value = jobId;
  totalCount.value = Number(payload.total ?? totalCount.value ?? 0);
  completedCount.value = Number(payload.completed ?? completedCount.value ?? 0);
  doneCount.value = Number(payload.done ?? doneCount.value ?? 0);
  failedCount.value = Number(payload.failed ?? failedCount.value ?? 0);
  currentFile.value = payload.current_file || (status === 'finished' ? '' : currentFile.value);
  entries.value = Array.isArray(payload.entries) ? payload.entries : entries.value;
  downloadUrl.value = payload.download_url || downloadUrl.value;
  if (payload?.token_balance) emitTokenBalanceUpdated(payload.token_balance);

  if (status === 'finished') {
    isRunning.value = false;
    analysisError.value = '';
    completedCount.value = totalCount.value;
    storeAnalysisJob(payload);
    stopAnalysisPolling();
    return;
  }

  if (status === 'failed') {
    isRunning.value = false;
    analysisError.value = formatAnalysisError({ message: payload.message || '' }, 'status');
    currentFile.value = analysisError.value;
    storeAnalysisJob(payload);
    stopAnalysisPolling();
    return;
  }

  if (status === 'queued' || status === 'running') {
    isRunning.value = true;
    analysisError.value = '';
    storeAnalysisJob(payload);
  }
};

const fetchAnalysisJobStatus = async (jobId) => {
  let response;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15000);
  try {
    response = await fetch(getBackendUrl(`/api/analysis/jobs/${encodeURIComponent(jobId)}`), {
      credentials: 'include',
      headers: authHeaders(),
      signal: controller.signal,
    });
  } catch (cause) {
    const error = new Error('Analysis job status failed: network error');
    error.code = 'NETWORK_ERROR';
    error.cause = cause;
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
  if (!response.ok) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      payload = null;
    }
    const detail = typeof payload?.detail === 'string'
      ? payload.detail
      : payload?.detail?.message || response.statusText;
    const error = new Error(detail || `Analysis job status failed: ${response.status}`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return response.json();
};

const refreshAnalysisJobStatus = async ({ keepPollingOnNetworkError = true } = {}) => {
  const jobId = activeJobId.value || readStoredAnalysisJob()?.job_id || '';
  if (!jobId || statusRequest?.jobId === jobId) return;
  const request = { jobId };
  statusRequest = request;
  try {
    const payload = await fetchAnalysisJobStatus(jobId);
    if (statusRequest !== request || activeJobId.value !== jobId) return;
    applyAnalysisJobPayload(payload);
  } catch (error) {
    if (statusRequest !== request || activeJobId.value !== jobId) return;
    if (error?.status === 404) {
      isRunning.value = false;
      analysisError.value = formatAnalysisError(error, 'status');
      currentFile.value = analysisError.value;
      clearStoredAnalysisJob(jobId);
      activeJobId.value = '';
      stopAnalysisPolling();
      return;
    }
    if (error?.status === 401) {
      isRunning.value = false;
      analysisError.value = formatAnalysisError(error, 'status');
      currentFile.value = analysisError.value;
      stopAnalysisPolling();
      return;
    }
    analysisError.value = formatAnalysisError(error, 'status');
    if (error?.code !== 'NETWORK_ERROR') {
      currentFile.value = analysisError.value;
    }
    if (!keepPollingOnNetworkError) {
      stopAnalysisPolling();
    }
  } finally {
    if (statusRequest === request) statusRequest = null;
  }
};

function stopAnalysisPolling() {
  statusRequest = null;
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

const startAnalysisPolling = () => {
  if (!activeJobId.value || pollTimer !== null) return;
  pollTimer = window.setInterval(() => {
    refreshAnalysisJobStatus();
  }, ANALYSIS_POLL_INTERVAL_MS);
};

const resumeAnalysisPolling = () => {
  const jobId = activeJobId.value || readStoredAnalysisJob()?.job_id || '';
  if (!jobId) return;
  activeJobId.value = jobId;
  // HTTP jobs may run in a different backend from the main-site WebSocket.
  startAnalysisPolling();
};

const restoreStoredAnalysisJob = () => {
  const stored = readStoredAnalysisJob();
  if (!stored?.job_id) return;
  activeJobId.value = stored.job_id;
  if (stored.pattern) selectedPattern.value = stored.pattern;
  if (stored.target) selectedTarget.value = String(stored.target);
  totalCount.value = Number(stored.total || totalCount.value || 0);
  completedCount.value = Number(stored.completed || completedCount.value || 0);
  doneCount.value = Number(stored.done || doneCount.value || 0);
  failedCount.value = Number(stored.failed || failedCount.value || 0);
  downloadUrl.value = stored.download_url || downloadUrl.value;
  isRunning.value = !['finished', 'failed'].includes(String(stored.status || '').toLowerCase());
  refreshAnalysisJobStatus({ keepPollingOnNetworkError: true });
  if (isRunning.value) {
    startAnalysisPolling();
    resumeAnalysisPolling();
  }
};


const startAnalysis = async () => {
  if (!requireAuth()) return;
  if (!canAnalyze.value || !client) return;
  isRunning.value = true;
  analysisError.value = '';
  downloadUrl.value = '';
  completedCount.value = 0;
  totalCount.value = 0;
  doneCount.value = 0;
  failedCount.value = 0;
  currentFile.value = '';
  entries.value = [];
  activeJobId.value = '';
  stopAnalysisPolling();
  clearStoredAnalysisJob();
  try {
    const payload = await postMultipart('/api/analysis/jobs', {
      files: selectedFiles.value,
      fields: {
        pattern: selectedPattern.value,
        target: selectedTarget.value,
      },
    });
    activeJobId.value = String(payload.job_id || '');
    totalCount.value = Number(payload.total || 0);
    completedCount.value = 0;
    storeAnalysisJob({ ...payload, pattern: selectedPattern.value, target: selectedTarget.value, status: 'queued' });
    resumeAnalysisPolling();
    refreshAnalysisJobStatus({ keepPollingOnNetworkError: true });
  } catch (error) {
    isRunning.value = false;
    failedCount.value = 1;
    analysisError.value = formatAnalysisError(error, 'upload');
    currentFile.value = analysisError.value;
    activeJobId.value = '';
    stopAnalysisPolling();
  }
};

const downloadResults = async () => {
  if (!requireAuth()) return;
  if (!downloadUrl.value) return;
  isDownloading.value = true;
  analysisError.value = '';
  try {
    const response = await fetch(getBackendUrl(downloadUrl.value), {
      credentials: 'include',
      headers: authHeaders(),
    });
    await downloadResponse(response);
  } catch (error) {
    analysisError.value = formatAnalysisError(error, 'download');
    if (error?.status === 404) {
      downloadUrl.value = '';
    }
  } finally {
    isDownloading.value = false;
  }
};

const handleMessage = (message) => {
  // Do not let another backend's subscription error stop an HTTP job.
  if (activeJobId.value && ['ANALYSIS_STARTED', 'ANALYSIS_PROGRESS', 'ANALYSIS_FINISHED', 'ANALYSIS_FAILED'].includes(message.type)) return;
  if (message.type === 'ANALYSIS_BOOTSTRAP') {
    ensureValidSelection();
    if (!userSelectionTouched.value) {
      applyContext(props.context);
    }
    return;
  }

  if (message.type === 'ANALYSIS_STARTED') {
    applyAnalysisJobPayload({ ...(message.payload || {}), status: message.payload?.status || 'running' });
    startAnalysisPolling();
    return;
  }

  if (message.type === 'ANALYSIS_FILES_SELECTED') {
    const selected = Array.isArray(message.payload?.paths) ? message.payload.paths : [];
    if (selected.length) {
      mergeSelectedPaths(selected);
    }
    return;
  }

  if (message.type === 'ANALYSIS_PROGRESS') {
    applyAnalysisJobPayload(message.payload || {});
    return;
  }

  if (message.type === 'ANALYSIS_FINISHED') {
    applyAnalysisJobPayload(message.payload || {});
    return;
  }

  if (message.type === 'ANALYSIS_FAILED') {
    applyAnalysisJobPayload({ ...(message.payload || {}), status: 'failed' });
  }
};

const connect = () => {
  if (client) return;
  client = createWsClient({
    clientId: `analysis_${Math.random().toString(36).slice(2, 9)}`,
    onOpen: () => {
      wsStatus.value = 'connected';
      client?.send('ANALYSIS_GET_INIT');
      if (isRunning.value) resumeAnalysisPolling();
    },
    onMessage: handleMessage,
    onClose: () => {
      wsStatus.value = 'disconnected';
    },
  });
  wsStatus.value = 'connecting';
  client.connect();
};

const disconnect = () => {
  client?.disconnect();
  client = null;
  wsStatus.value = 'disconnected';
  stopAnalysisPolling();
};

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      historyMode.value = false;
      userSelectionTouched.value = false;
      document.addEventListener('click', closePatternMenuOnClick);
      loadCatalog();
      connect();
      restoreStoredAnalysisJob();
    } else {
      patternMenuOpen.value = false;
      document.removeEventListener('click', closePatternMenuOnClick);
      disconnect();
    }
  },
  { immediate: true }
);

watch(patternGroups, (groups) => {
  if (!activePatternCategory.value && groups.length) {
    activePatternCategory.value = groups[0].category;
  }
  if (!selectedPattern.value && activePatternOptions.value.length) {
    selectedPattern.value = activePatternOptions.value[0];
  }
  ensureValidSelection();
});

watch(
  () => props.context,
  (nextContext) => {
    applyContext(nextContext);
  },
  { deep: true, immediate: true }
);

watch(activePatternCategory, (category) => {
  const group = patternGroups.value.find((item) => item.category === category);
  if (group && !group.items.includes(selectedPattern.value)) {
    selectedPattern.value = group.items[0] || '';
    ensureValidSelection();
  }
});

onUnmounted(() => {
  document.removeEventListener('click', closePatternMenuOnClick);
  disconnect();
  stopAnalysisPolling();
});

</script>

<style scoped>
.analysis-dialog-overlay { position: fixed; inset: 0; z-index: 220; display: flex; align-items: center; justify-content: center; padding: 16px; background: rgba(0, 0, 0, 0.4); }
.analysis-dialog-shell {
  display: flex;
  width: 100%;
  max-width: 1024px;
  height: 780px;
  max-height: calc(100vh - 32px);
  min-height: 0;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--border-main);
  border-radius: 12px;
  background: var(--bg-card);
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
}

.analysis-dialog-shell--history {
  max-height: calc(100vh - 32px);
}

.analysis-dialog-header {
  flex: 0 0 auto;
  gap: 12px;
  flex-wrap: wrap;
}
.analysis-dialog-title { font-size: 20px; font-weight: 800; color: var(--text-main); }

.analysis-dialog-body {
  grid-template-columns: minmax(280px, 0.9fr) minmax(0, 1.1fr);
  min-height: 0;
  flex: 1 1 auto;
  overflow: hidden;
  overscroll-behavior: contain;
}
.analysis-input-section, .analysis-results-section { min-width: 0; padding: 0; }
.analysis-input-section { overflow-y: auto; border-right: 1px solid var(--border-main); padding-right: 20px; }
.analysis-results-section { display: flex; flex-direction: column; min-height: 0; }
.analysis-progress-summary { flex: 0 0 auto; margin-top: 16px; padding-bottom: 16px; border-bottom: 1px solid var(--border-main); }
.analysis-results-list { display: flex; flex: 1 1 auto; min-height: 0; margin-top: 16px; }
.analysis-entry-content { min-width: 0; flex: 1; }

.analysis-history-view {
  display: flex;
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
  padding: 1.25rem 1.5rem 1.5rem;
}

.analysis-header-refresh {
  display: inline-flex;
  width: 34px;
  height: 34px;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-main);
  border-radius: 8px;
  background: var(--bg-main);
  color: var(--text-main);
}

.analysis-header-refresh:hover {
  color: var(--accent);
}

@media (max-width: 640px) {
  .analysis-history-view { padding: 0.75rem; }
  .analysis-dialog-body { display: block; padding: 16px; overflow-y: auto; }
  .analysis-input-section { overflow: visible; border-right: 0; padding-right: 0; border-bottom: 1px solid var(--border-main); padding-bottom: 16px; }
  .analysis-results-section { margin-top: 20px; }
  .analysis-results-list { flex: 0 0 auto; }
}

.analysis-input-btn,
.analysis-secondary-btn,
.analysis-primary-btn {
  width: 100%;
  min-height: 3.25rem;
  border-radius: 0.9rem;
  border: 1px solid var(--border-main);
  background-color: var(--bg-card);
  color: var(--text-main);
  padding: 0.78rem 0.95rem;
  font-size: var(--font-ui-sm);
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  transition: all 0.2s ease;
}

.analysis-input-btn,
:deep(.analysis-select-trigger) {
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
}

:deep(.analysis-select-trigger) {
  width: 100%;
  min-height: 3.25rem;
  border-radius: 0.9rem;
  border: 1px solid var(--border-main);
  background-color: var(--bg-card);
  color: var(--text-main);
  padding: 0.78rem 0.95rem;
  font-size: var(--font-ui-sm);
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.analysis-input-btn:hover,
.analysis-select-shell:hover :deep(.analysis-select-trigger),
.analysis-secondary-btn:hover,
.analysis-primary-btn:hover {
  border-color: var(--accent);
}

.analysis-textarea {
  min-height: 176px;
  width: 100%;
  resize: vertical;
  border-radius: 1rem;
  border: 1px dashed color-mix(in srgb, var(--border-main) 82%, transparent);
  background: color-mix(in srgb, var(--bg-card) 82%, transparent);
  color: var(--text-main);
  padding: 0.9rem 1rem;
  font-size: var(--font-ui-md);
  font-weight: 800;
  line-height: 1.5;
  outline: none;
}

.analysis-textarea:focus {
  border-color: var(--accent);
}

.analysis-secondary-btn:disabled,
.analysis-primary-btn:disabled {
  cursor: not-allowed;
  opacity: 0.48;
}

.analysis-error {
  border-radius: 0.9rem;
  border: 1px solid color-mix(in srgb, var(--danger, #ef4444) 45%, var(--border-main));
  background: color-mix(in srgb, var(--danger, #ef4444) 10%, var(--bg-card));
  color: color-mix(in srgb, var(--danger, #ef4444) 85%, var(--text-main));
  padding: 0.72rem 0.9rem;
  font-size: var(--font-ui-sm);
  font-weight: 900;
  line-height: 1.45;
}

.analysis-primary-btn {
  color: white;
}

.analysis-list-viewport {
  min-height: 0;
  width: 100%;
  height: 100%;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-right: 6px;
  overscroll-behavior: contain;
  scrollbar-width: thin;
  scrollbar-color: var(--border-main) transparent;
}

.analysis-list-row {
  display: flex;
  flex: 0 0 auto;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
  border-radius: 8px;
  border: 1px solid color-mix(in srgb, var(--border-main) 60%, transparent);
  background: color-mix(in srgb, var(--bg-main) 65%, transparent);
  padding: 0.5rem 0.75rem;
}
.analysis-list-row > :first-child { flex: 1; }
.analysis-list-row > .badge-state { flex: 0 0 auto; }
.analysis-entry-title { margin-bottom: 8px; overflow-wrap: anywhere; font-size: 14px; font-weight: 700; color: var(--text-main); }
.analysis-entry-meta { margin-bottom: 8px; font-size: 12px; color: var(--text-secondary); }
.analysis-input-btn, .analysis-secondary-btn, .analysis-primary-btn, .analysis-textarea,
:deep(.analysis-select-trigger) { border-radius: 8px; letter-spacing: 0; }
.analysis-list-viewport::-webkit-scrollbar { width: 8px; }
.analysis-list-viewport::-webkit-scrollbar-thumb { border: 2px solid var(--bg-card); border-radius: 8px; background: var(--border-main); }
@media (max-width: 640px) {
  .analysis-list-viewport { height: auto; max-height: 460px; }
}
</style>
