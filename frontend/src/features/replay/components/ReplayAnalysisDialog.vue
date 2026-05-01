<template>
  <teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-[220] flex items-center justify-center bg-black/35 px-4 py-8 backdrop-blur-sm"
      @click.self="$emit('close')"
    >
      <div class="w-full max-w-5xl overflow-hidden rounded-[30px] border border-border-main bg-bg-card shadow-[0_24px_80px_rgba(0,0,0,0.28)]">
        <div class="flex items-center justify-between border-b border-border-main/60 px-6 py-4">
          <div>
            <div class="ui-control font-black uppercase tracking-[0.24em] text-text-secondary">{{ $t('analysis.windowTag') }}</div>
            <div class="mt-1 text-2xl font-black text-text-main">{{ $t('analysis.title') }}</div>
          </div>
          <div class="flex items-center gap-3">
            <span :class="statusBadgeClass">{{ statusBadgeText }}</span>
            <button
              class="rounded-full border border-border-main bg-bg-main/80 px-3 py-1.5 ui-control font-black uppercase tracking-wider text-text-main transition-colors hover:border-accent/40 hover:text-accent"
              @click="$emit('close')"
            >
              {{ $t('analysis.close') }}
            </button>
          </div>
        </div>

        <div class="grid gap-5 p-6 lg:grid-cols-[minmax(340px,0.95fr)_minmax(0,1.05fr)]">
          <section class="rounded-[24px] border border-border-main/70 bg-bg-main/65 p-5 shadow-inner">
            <div class="ui-control font-black uppercase tracking-[0.24em] text-text-secondary">{{ $t('analysis.input.title') }}</div>
            <div class="mt-4 space-y-4">
              <div class="grid gap-3 md:grid-cols-[minmax(0,1fr)_132px]">
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

                <select v-model="selectedTarget" class="analysis-select">
                  <option v-for="target in targetTiles" :key="target" :value="target">{{ target }}</option>
                </select>
              </div>

              <div>
                <div class="mb-2 ui-caption font-black uppercase tracking-[0.22em] text-text-secondary">{{ $t('analysis.input.paths') }}</div>
                <textarea
                  v-model="pathsInput"
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
            </div>
          </section>

          <section class="rounded-[24px] border border-border-main/70 bg-bg-main/65 p-5 shadow-inner">
            <div class="flex items-center justify-between">
              <div class="ui-control font-black uppercase tracking-[0.24em] text-text-secondary">{{ $t('analysis.progress.title') }}</div>
              <span :class="statusBadgeClass">{{ statusBadgeText }}</span>
            </div>

            <div class="mt-4 rounded-2xl border border-border-main bg-bg-card/85 p-4">
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

            <div class="mt-4 rounded-2xl border border-border-main bg-bg-card/85 p-4">
              <div
                v-if="visibleEntries.length"
                ref="listViewportRef"
                class="analysis-list-viewport"
                @scroll="handleListScroll"
              >
                <div :style="{ height: `${topSpacerHeight}px` }" />
                <div
                  v-for="entry in visibleEntries"
                  :key="entry.key"
                  class="analysis-list-row"
                >
                  <div class="min-w-0">
                    <div class="truncate ui-body font-black text-text-main" :title="entry.path">{{ entry.path }}</div>
                    <div v-if="entry.message" class="mt-0.5 truncate ui-caption font-black text-red-500/85" :title="entry.message">{{ entry.message }}</div>
                  </div>
                  <div
                    class="badge-state"
                    :class="entry.status === 'done' ? 'badge-state-success' : (entry.status === 'failed' ? 'badge-state-failure' : 'badge-state-running')"
                  >
                    {{ getEntryStatusLabel(entry.status) }}
                  </div>
                </div>
                <div :style="{ height: `${bottomSpacerHeight}px` }" />
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
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { createWsClient } from '../../../services/ws/createWsClient';

const props = defineProps({
  open: { type: Boolean, default: false },
  context: {
    type: Object,
    default: () => ({}),
  },
});

defineEmits(['close']);

const { t } = useI18n();

const wsStatus = ref('disconnected');
const categories = ref({});
const targetTiles = ref(['64', '128', '256', '512', '1024', '2048', '4096', '8192']);
const selectedPattern = ref('');
const selectedTarget = ref('2048');
const pathsInput = ref('');
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
const listViewportRef = ref(null);
const listScrollTop = ref(0);

let client = null;
const LIST_ITEM_HEIGHT = 62;
const LIST_VIEWPORT_HEIGHT = 312;
const LIST_OVERSCAN = 4;

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

const currentFileDisplay = computed(() => currentFile.value || t('analysis.progress.idle'));
const progressPercent = computed(() => {
  if (totalCount.value <= 0) return 0;
  return Math.max(0, Math.min(100, (completedCount.value / totalCount.value) * 100));
});
const canAnalyze = computed(() => Boolean(selectedPattern.value && selectedTarget.value && pathsInput.value.trim()));
const normalizedEntries = computed(() =>
  entries.value.map((entry, index) => ({
    key: `${entry.path}-${entry.status}-${index}`,
    path: entry.path,
    status: entry.status,
    message: entry.message || '',
  }))
);
const visibleCount = computed(() => Math.ceil(LIST_VIEWPORT_HEIGHT / LIST_ITEM_HEIGHT) + LIST_OVERSCAN * 2);
const startIndex = computed(() =>
  Math.max(0, Math.floor(listScrollTop.value / LIST_ITEM_HEIGHT) - LIST_OVERSCAN)
);
const endIndex = computed(() =>
  Math.min(normalizedEntries.value.length, startIndex.value + visibleCount.value)
);
const visibleEntries = computed(() => normalizedEntries.value.slice(startIndex.value, endIndex.value));
const topSpacerHeight = computed(() => startIndex.value * LIST_ITEM_HEIGHT);
const bottomSpacerHeight = computed(() =>
  Math.max(0, (normalizedEntries.value.length - endIndex.value) * LIST_ITEM_HEIGHT)
);

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
  selectedPattern.value = pattern;
  patternMenuOpen.value = false;
};

const applyContext = (context) => {
  const nextPattern = String(context?.pattern || '').trim();
  const nextTarget = String(context?.target || '').trim();
  if (nextPattern) {
    selectedPattern.value = nextPattern;
    const matchedGroup = patternGroups.value.find((group) => group.items.includes(nextPattern));
    if (matchedGroup) activePatternCategory.value = matchedGroup.category;
  }
  if (nextTarget && targetTiles.value.includes(nextTarget)) {
    selectedTarget.value = nextTarget;
  }
};

const mergeSelectedPaths = (paths) => {
  const existing = new Set(
    pathsInput.value
      .split(/\r?\n/u)
      .map((item) => item.trim())
      .filter(Boolean)
  );
  for (const path of paths) {
    const normalized = String(path || '').trim();
    if (normalized) existing.add(normalized);
  }
  pathsInput.value = [...existing].join('\n');
};

const pickFiles = async () => {
  client?.send('ANALYSIS_TRIGGER_SELECT_FILES');
};

const handleListScroll = (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  listScrollTop.value = target.scrollTop;
};

const startAnalysis = () => {
  if (!canAnalyze.value || !client) return;
  const paths = pathsInput.value
    .split(/\r?\n/u)
    .map((item) => item.trim())
    .filter(Boolean);
  isRunning.value = true;
  completedCount.value = 0;
  totalCount.value = 0;
  doneCount.value = 0;
  failedCount.value = 0;
  currentFile.value = '';
  entries.value = [];
  listScrollTop.value = 0;
  if (listViewportRef.value) listViewportRef.value.scrollTop = 0;
  client.send('ANALYSIS_START', {
    pattern: selectedPattern.value,
    target: selectedTarget.value,
    paths,
  });
};

const handleMessage = (message) => {
  if (message.type === 'ANALYSIS_BOOTSTRAP') {
    categories.value = message.payload?.categories || {};
    targetTiles.value = Array.isArray(message.payload?.target_tiles) && message.payload.target_tiles.length
      ? message.payload.target_tiles.map(String)
      : targetTiles.value;
    if (!activePatternCategory.value && patternGroups.value.length) {
      activePatternCategory.value = patternGroups.value[0].category;
    }
    if (!selectedPattern.value && activePatternOptions.value.length) {
      selectedPattern.value = activePatternOptions.value[0];
    }
    applyContext(props.context);
    return;
  }

  if (message.type === 'ANALYSIS_STARTED') {
    isRunning.value = true;
    completedCount.value = 0;
    totalCount.value = Number(message.payload?.total || 0);
    doneCount.value = 0;
    failedCount.value = 0;
    currentFile.value = '';
    entries.value = [];
    listScrollTop.value = 0;
    if (listViewportRef.value) listViewportRef.value.scrollTop = 0;
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
    completedCount.value = Number(message.payload?.completed || 0);
    totalCount.value = Number(message.payload?.total || totalCount.value);
    doneCount.value = Number(message.payload?.done || 0);
    failedCount.value = Number(message.payload?.failed || 0);
    currentFile.value = message.payload?.current_file || '';
    entries.value = Array.isArray(message.payload?.entries) ? message.payload.entries : entries.value;
    return;
  }

  if (message.type === 'ANALYSIS_FINISHED') {
    isRunning.value = false;
    totalCount.value = Number(message.payload?.total || totalCount.value);
    completedCount.value = totalCount.value;
    doneCount.value = Number(message.payload?.done || doneCount.value);
    failedCount.value = Number(message.payload?.failed || failedCount.value);
    entries.value = Array.isArray(message.payload?.entries) ? message.payload.entries : entries.value;
    return;
  }

  if (message.type === 'ANALYSIS_FAILED') {
    isRunning.value = false;
    failedCount.value = Math.max(1, failedCount.value);
    currentFile.value = message.payload?.message || '';
  }
};

const connect = () => {
  if (client) return;
  client = createWsClient({
    clientId: `analysis_${Math.random().toString(36).slice(2, 9)}`,
    onOpen: () => {
      wsStatus.value = 'connected';
      client?.send('ANALYSIS_GET_INIT');
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
};

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      document.addEventListener('click', closePatternMenuOnClick);
      connect();
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
  }
});

onUnmounted(() => {
  document.removeEventListener('click', closePatternMenuOnClick);
  disconnect();
});

onMounted(() => {
  listScrollTop.value = 0;
});
</script>

<style scoped>
.analysis-input-btn,
.analysis-select,
.analysis-secondary-btn,
.analysis-primary-btn {
  width: 100%;
  border-radius: 0.9rem;
  border: 1px solid var(--border-main);
  background: var(--bg-card);
  color: var(--text-main);
  padding: 0.78rem 0.95rem;
  font-size: var(--font-ui-sm);
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  transition: all 0.2s ease;
}

.analysis-input-btn,
.analysis-select {
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
}

.analysis-input-btn:hover,
.analysis-select:hover,
.analysis-secondary-btn:hover,
.analysis-primary-btn:hover {
  border-color: var(--accent);
}

.analysis-select {
  cursor: pointer;
  appearance: none;
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

.analysis-primary-btn {
  color: white;
}

.analysis-list-viewport {
  height: 312px;
  overflow-y: auto;
}

.analysis-list-row {
  display: flex;
  height: 62px;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  border-radius: 0.75rem;
  border: 1px solid color-mix(in srgb, var(--border-main) 60%, transparent);
  background: color-mix(in srgb, var(--bg-main) 65%, transparent);
  padding: 0.5rem 0.75rem;
}
</style>
