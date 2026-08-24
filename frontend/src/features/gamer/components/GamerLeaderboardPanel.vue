<template>
  <section class="gamer-leaderboard-panel" :aria-label="$t('gamer.leaderboard.title')">
    <header class="gamer-leaderboard-head">
      <div>
        <h2>{{ $t('gamer.leaderboard.title') }}</h2>
      </div>
      <button
        type="button"
        class="gamer-leaderboard-icon-button"
        :disabled="loading"
        :title="$t('gamer.leaderboard.refresh')"
        :aria-label="$t('gamer.leaderboard.refresh')"
        @click="loadSelected(true)"
      >
        <span :class="loading ? 'is-spinning' : ''" aria-hidden="true">&#8635;</span>
      </button>
    </header>

    <div class="gamer-leaderboard-filters">
      <div class="gamer-leaderboard-segments" :aria-label="$t('gamer.leaderboard.modeLabel')">
        <button
          v-for="mode in modes"
          :key="mode.value"
          type="button"
          :class="['gamer-leaderboard-segment', selectedMode === mode.value ? 'active' : '']"
          @click="selectedMode = mode.value"
        >
          {{ $t(mode.labelKey) }}
        </button>
      </div>
      <div class="gamer-leaderboard-segments compact" :aria-label="$t('gamer.leaderboard.periodLabel')">
        <button
          v-for="period in periods"
          :key="period.value"
          type="button"
          :class="['gamer-leaderboard-segment', selectedPeriod === period.value ? 'active' : '']"
          @click="selectedPeriod = period.value"
        >
          {{ $t(period.labelKey) }}
        </button>
      </div>
    </div>

    <div class="gamer-leaderboard-body">
      <div v-if="error && !entries.length" class="gamer-leaderboard-message error">
        <span>{{ error }}</span>
        <button type="button" @click="loadSelected(true)">{{ $t('common.retry') }}</button>
      </div>
      <div v-else-if="loading && !entries.length" class="gamer-leaderboard-skeleton" aria-hidden="true">
        <span v-for="index in 10" :key="index" />
      </div>
      <div v-else-if="!entries.length" class="gamer-leaderboard-message">
        {{ $t('gamer.leaderboard.empty') }}
      </div>
      <div v-else class="gamer-leaderboard-list">
        <button
          v-for="entry in entries"
          :key="entry.entry_key"
          type="button"
          :class="['gamer-leaderboard-row', `rank-${entry.rank}`]"
          :disabled="!entry.replay_id"
          :title="entry.replay_id ? $t('gamer.leaderboard.viewReplay') : undefined"
          @click="openReplay(entry.replay_id)"
        >
          <span class="gamer-leaderboard-rank">{{ entry.rank }}</span>
          <span class="gamer-leaderboard-avatar">
            <AccountAvatar
              :user="leaderboardUser(entry)"
              :supporter="entry.is_supporter"
            />
          </span>
          <span class="gamer-leaderboard-player">
            <strong>{{ entry.display_name }}</strong>
            <small>
              {{ $t('leaderboards.maxTile') }} {{ formatInteger(entry.max_tile) }}
              <span v-if="entry.used_ai"> · AI</span>
            </small>
          </span>
          <strong class="gamer-leaderboard-score">{{ formatInteger(entry.score) }}</strong>
        </button>
      </div>
    </div>

    <button type="button" class="gamer-leaderboard-full-link" @click="openFullLeaderboard">
      <span>{{ $t('gamer.leaderboard.viewAll') }}</span>
      <span aria-hidden="true">&gt;</span>
    </button>
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import AccountAvatar from '../../auth/AccountAvatar.vue';
import { fetchLeaderboard } from '../../leaderboards/services/leaderboardClient';

const props = defineProps({
  active: { type: Boolean, default: true },
  rankedStatus: { type: String, default: '' },
});

const emit = defineEmits(['navigate-tab']);
const { locale, t } = useI18n();

const modes = [
  { value: 'general', labelKey: 'gamer.leaderboard.modes.general' },
  { value: 'adversarial', labelKey: 'gamer.leaderboard.modes.adversarial' },
];
const periods = [
  { value: 'all', labelKey: 'gamer.leaderboard.periods.all' },
  { value: 'week', labelKey: 'gamer.leaderboard.periods.week' },
];
const boardKeys = {
  'general:all': 'gamer_high_score',
  'general:week': 'gamer_high_score_weekly',
  'adversarial:all': 'gamer_adversarial',
  'adversarial:week': 'gamer_adversarial_weekly',
};
const selectedMode = ref('general');
const selectedPeriod = ref('all');
const selectedBoardKey = computed(() => boardKeys[`${selectedMode.value}:${selectedPeriod.value}`]);
const boardData = ref(null);
const loading = ref(false);
const error = ref('');
const cache = new Map();
let requestSerial = 0;

const entries = computed(() => boardData.value?.entries || []);
const localeName = computed(() => (
  String(locale.value).toLowerCase().startsWith('zh') ? 'zh-CN' : 'en-US'
));
const formatInteger = (value) => new Intl.NumberFormat(localeName.value, {
  maximumFractionDigits: 0,
}).format(Number(value || 0));
const leaderboardUser = (entry) => ({
  display_name: entry?.display_name || '',
  profile: { avatar_url: entry?.avatar_url || null },
});

const loadSelected = async (force = false) => {
  const key = selectedBoardKey.value;
  const cached = cache.get(key);
  if (cached && !force) boardData.value = cached.payload;
  const serial = ++requestSerial;
  loading.value = true;
  error.value = '';
  try {
    const payload = await fetchLeaderboard(key, { limit: 10 });
    if (serial !== requestSerial) return;
    cache.set(key, { payload, loadedAt: Date.now() });
    boardData.value = payload;
  } catch (loadError) {
    if (serial === requestSerial) {
      error.value = loadError?.message || t('gamer.leaderboard.loadFailed');
    }
  } finally {
    if (serial === requestSerial) loading.value = false;
  }
};

const openReplay = (replayId) => {
  if (!replayId) return;
  window.open(`/verse-replay/?ranked=${encodeURIComponent(replayId)}`, '_blank', 'noopener');
};
const openFullLeaderboard = () => {
  const boardKey = selectedMode.value === 'adversarial'
    ? 'gamer_adversarial'
    : 'gamer_high_score';
  emit('navigate-tab', 'LeaderboardsView', { boardKey });
};

watch(selectedBoardKey, () => {
  boardData.value = cache.get(selectedBoardKey.value)?.payload || null;
  if (props.active) loadSelected();
});
watch(() => props.active, (active) => {
  if (!active) return;
  const cached = cache.get(selectedBoardKey.value);
  if (!cached || Date.now() - cached.loadedAt >= 60_000) loadSelected(true);
});
watch(() => props.rankedStatus, (status, previous) => {
  if (props.active && status === 'verified' && status !== previous) loadSelected(true);
});

onMounted(() => {
  if (props.active) loadSelected();
});
</script>

<style scoped>
.gamer-leaderboard-panel {
  width: 100%;
  min-height: 41rem;
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--border-main);
  border-radius: 8px;
  background: color-mix(in srgb, var(--bg-card) 94%, transparent);
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.1);
}

.gamer-leaderboard-head {
  min-height: 4.5rem;
  padding: 0.85rem 0.9rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--border-main);
}

.gamer-leaderboard-head h2 {
  color: var(--text-main);
  font-size: var(--font-ui-lg);
  font-weight: 950;
}

.gamer-leaderboard-icon-button {
  width: 2.35rem;
  height: 2.35rem;
  display: grid;
  place-items: center;
  border: 1px solid var(--border-main);
  border-radius: 6px;
  color: var(--text-main);
  background: var(--bg-main);
  font-size: 1.2rem;
  font-weight: 900;
}

.gamer-leaderboard-icon-button:hover:not(:disabled) {
  color: var(--accent);
  border-color: var(--accent);
}

.gamer-leaderboard-icon-button:disabled {
  opacity: 0.55;
}

.is-spinning {
  animation: gamer-leaderboard-spin 0.8s linear infinite;
}

.gamer-leaderboard-filters {
  padding: 0.65rem;
  display: grid;
  gap: 0.45rem;
  border-bottom: 1px solid var(--border-main);
}

.gamer-leaderboard-segments {
  min-height: 2.35rem;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  padding: 0.2rem;
  border: 1px solid var(--border-main);
  border-radius: 6px;
  background: color-mix(in srgb, var(--bg-main) 72%, transparent);
}

.gamer-leaderboard-segments.compact {
  min-height: 2.05rem;
}

.gamer-leaderboard-segment {
  min-width: 0;
  padding: 0.38rem 0.45rem;
  border-radius: 4px;
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 900;
  white-space: nowrap;
}

.gamer-leaderboard-segment.active {
  color: white;
  background: var(--accent);
  box-shadow: 0 4px 10px color-mix(in srgb, var(--accent) 20%, transparent);
}

.gamer-leaderboard-body {
  flex: 1 1 auto;
  min-height: 0;
}

.gamer-leaderboard-list {
  display: flex;
  flex-direction: column;
}

.gamer-leaderboard-row {
  min-height: 3.15rem;
  display: grid;
  grid-template-columns: 2rem 2rem minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.5rem;
  padding: 0.42rem 0.7rem;
  border-bottom: 1px solid color-mix(in srgb, var(--border-main) 72%, transparent);
  color: var(--text-main);
  text-align: left;
}

.gamer-leaderboard-row:hover:not(:disabled),
.gamer-leaderboard-row:focus-visible {
  background: color-mix(in srgb, var(--accent) 8%, transparent);
  outline: none;
}

.gamer-leaderboard-row:disabled {
  cursor: default;
}

.gamer-leaderboard-rank {
  width: 1.75rem;
  height: 1.75rem;
  display: grid;
  place-items: center;
  border-radius: 4px;
  color: white;
  background: color-mix(in srgb, var(--text-secondary) 72%, black);
  font-size: var(--font-ui-xs);
  font-weight: 950;
}

.rank-1 .gamer-leaderboard-rank { background: #d6aa45; }
.rank-2 .gamer-leaderboard-rank { background: #8e9aa7; }
.rank-3 .gamer-leaderboard-rank { background: #b77745; }

.gamer-leaderboard-avatar {
  width: 1.85rem;
  height: 1.85rem;
  display: grid;
  place-items: center;
}

.gamer-leaderboard-avatar :deep(.account-avatar-shell) {
  width: 100%;
  height: 100%;
}

.gamer-leaderboard-player {
  min-width: 0;
  display: grid;
  gap: 0.08rem;
}

.gamer-leaderboard-player strong,
.gamer-leaderboard-player small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.gamer-leaderboard-player strong {
  font-size: var(--font-ui-xs);
  font-weight: 900;
}

.gamer-leaderboard-player small {
  color: var(--text-secondary);
  font-size: var(--font-ui-2xs);
  font-weight: 750;
}

.gamer-leaderboard-score {
  color: var(--accent);
  font-size: var(--font-ui-xs);
  font-variant-numeric: tabular-nums;
  font-weight: 950;
  white-space: nowrap;
}

.gamer-leaderboard-message {
  min-height: 20rem;
  padding: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 800;
  text-align: center;
}

.gamer-leaderboard-message.error {
  color: #d14a45;
}

.gamer-leaderboard-message button {
  padding: 0.35rem 0.55rem;
  border: 1px solid currentColor;
  border-radius: 4px;
}

.gamer-leaderboard-skeleton {
  padding: 0.45rem 0.7rem;
  display: grid;
  gap: 0.45rem;
}

.gamer-leaderboard-skeleton span {
  height: 2.7rem;
  border-radius: 4px;
  background: linear-gradient(90deg, var(--bg-main), color-mix(in srgb, var(--bg-card) 65%, white), var(--bg-main));
  background-size: 220% 100%;
  animation: gamer-leaderboard-shimmer 1.3s linear infinite;
}

.gamer-leaderboard-full-link {
  min-height: 2.8rem;
  padding: 0.55rem 0.85rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-top: 1px solid var(--border-main);
  color: var(--text-main);
  background: color-mix(in srgb, var(--bg-main) 72%, transparent);
  font-size: var(--font-ui-xs);
  font-weight: 900;
}

.gamer-leaderboard-full-link:hover,
.gamer-leaderboard-full-link:focus-visible {
  color: var(--accent);
  outline: none;
}

@keyframes gamer-leaderboard-spin {
  to { transform: rotate(360deg); }
}

@keyframes gamer-leaderboard-shimmer {
  to { background-position: -220% 0; }
}
</style>
