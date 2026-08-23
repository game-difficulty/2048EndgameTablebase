<template>
  <div class="page-root overflow-y-auto p-5">
    <div class="leaderboard-shell">
      <header class="leaderboard-header">
        <div>
          <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('leaderboards.kicker') }}</div>
          <h1 class="mt-1 ui-metric font-black text-text-main">{{ $t('leaderboards.title') }}</h1>
          <p class="mt-2 ui-body text-text-secondary">{{ $t('leaderboards.subtitle') }}</p>
        </div>
        <div v-if="boardData" class="leaderboard-updated">
          <span>{{ $t('leaderboards.updated') }}</span>
          <strong>{{ formatDateTime(boardData.generated_at) }}</strong>
        </div>
      </header>

      <nav class="leaderboard-tabs" :aria-label="$t('leaderboards.title')">
        <button
          v-for="board in boards"
          :key="board.key"
          type="button"
          :class="['leaderboard-tab', selectedKey === board.key ? 'active' : '']"
          :disabled="loading"
          @click="selectBoard(board.key)"
        >
          {{ boardTitle(board.key) }}
        </button>
      </nav>

      <div v-if="error" class="leaderboard-error">
        <span>{{ error }}</span>
        <button type="button" class="action-btn-small" @click="loadBoard(selectedKey)">
          {{ $t('common.retry') }}
        </button>
      </div>

      <template v-else-if="loading && !boardData">
        <div class="leaderboard-skeleton podium-skeleton" />
        <div class="leaderboard-skeleton list-skeleton" />
      </template>

      <template v-else-if="boardData">
        <section class="leaderboard-context">
          <div>
            <h2>{{ boardTitle(boardData.key) }}</h2>
            <p>{{ boardDescription(boardData.key) }}</p>
          </div>
          <span v-if="periodLabel" class="leaderboard-period">{{ periodLabel }}</span>
        </section>

        <div v-if="!entries.length" class="leaderboard-empty">
          {{ $t('leaderboards.empty') }}
        </div>

        <template v-else>
          <section class="leaderboard-podium" aria-label="Top three">
            <article
              v-for="entry in podiumEntries"
              :key="entry.entry_key"
              :class="['podium-place', `place-${entry.rank}`]"
            >
              <div class="podium-rank">{{ entry.rank }}</div>
              <div class="leader-avatar large">
                <AccountAvatar
                  :user="leaderboardUser(entry)"
                  :supporter="entry.is_supporter"
                  size="large"
                />
              </div>
              <strong class="podium-name">{{ entry.display_name }}</strong>
              <span v-if="boardData.score_visible" class="podium-score">
                {{ formatScore(entry.score, boardData.unit) }}
              </span>
              <div v-if="isGamerBoard" class="game-entry-meta">
                <span>{{ $t('leaderboards.maxTile') }} {{ formatInteger(entry.max_tile) }}</span>
                <span v-if="entry.used_ai">{{ $t('leaderboards.aiUsed') }}</span>
              </div>
              <button
                v-if="isGamerBoard && entry.replay_id"
                type="button"
                class="replay-link"
                @click="openReplay(entry.replay_id)"
              >
                {{ $t('leaderboards.viewGame') }}
              </button>
            </article>
          </section>

          <section :class="['leaderboard-list', !boardData.score_visible ? 'score-hidden' : '', isGamerBoard ? 'game-board' : '']">
            <div class="leaderboard-list-head">
              <span>{{ $t('leaderboards.rank') }}</span>
              <span>{{ $t('leaderboards.player') }}</span>
              <span v-if="boardData.score_visible">
                {{ isGamerBoard ? $t('leaderboards.gameScore') : $t('leaderboards.tokenUsage') }}
              </span>
            </div>
            <div
              v-for="entry in listEntries"
              :key="entry.entry_key"
              class="leaderboard-row"
            >
              <span class="row-rank">{{ entry.rank }}</span>
              <span class="row-player">
                <span class="leader-avatar">
                  <AccountAvatar
                    :user="leaderboardUser(entry)"
                    :supporter="entry.is_supporter"
                  />
                </span>
                <strong>{{ entry.display_name }}</strong>
              </span>
              <strong v-if="boardData.score_visible" class="row-score">
                <span>{{ formatScore(entry.score, boardData.unit) }}</span>
                <small v-if="isGamerBoard">
                  {{ $t('leaderboards.maxTile') }} {{ formatInteger(entry.max_tile) }}
                  <span v-if="entry.used_ai"> · {{ $t('leaderboards.aiUsed') }}</span>
                </small>
                <button
                  v-if="isGamerBoard && entry.replay_id"
                  type="button"
                  class="replay-link"
                  @click="openReplay(entry.replay_id)"
                >
                  {{ $t('leaderboards.viewGame') }}
                </button>
              </strong>
            </div>
          </section>
        </template>
      </template>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import AccountAvatar from '../../auth/AccountAvatar.vue';
import { fetchLeaderboard, fetchLeaderboardCatalog } from '../services/leaderboardClient';

const props = defineProps({ active: Boolean });
const { locale, t } = useI18n();
const boards = ref([]);
const selectedKey = ref('supporters');
const boardData = ref(null);
const loading = ref(false);
const error = ref('');
let requestSerial = 0;

const entries = computed(() => boardData.value?.entries || []);
const isGamerBoard = computed(() => String(boardData.value?.key || '').startsWith('gamer_'));
const podiumEntries = computed(() => {
  const top = entries.value.filter((entry) => entry.rank <= 3);
  const order = { 2: 0, 1: 1, 3: 2 };
  return [...top].sort((left, right) => (order[left.rank] ?? 9) - (order[right.rank] ?? 9));
});
const listEntries = computed(() => entries.value.filter((entry) => entry.rank > 3));
const periodLabel = computed(() => {
  if (!boardData.value) return '';
  if (boardData.value.key === 'token_last_week') {
    const start = formatDate(boardData.value.period?.start);
    const end = formatDate(boardData.value.period?.end);
    return start && end ? `${start} - ${end}` : '';
  }
  if (boardData.value.cadence === 'live') return t('leaderboards.cadence.live');
  return boardData.value.cadence === 'daily'
    ? t('leaderboards.cadence.daily')
    : t('leaderboards.cadence.weekly');
});

const boardTitle = (key) => t(`leaderboards.boards.${key}.title`);
const boardDescription = (key) => t(`leaderboards.boards.${key}.description`);

const localeName = () => (String(locale.value).toLowerCase().startsWith('zh') ? 'zh-CN' : 'en-US');
const formatDate = (value) => {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return new Intl.DateTimeFormat(localeName(), { month: 'short', day: 'numeric' }).format(date);
};
const formatDateTime = (value) => {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '-';
  return new Intl.DateTimeFormat(localeName(), {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
};
const formatInteger = (value) => new Intl.NumberFormat(localeName(), {
  maximumFractionDigits: 0,
}).format(Number(value || 0));
const formatScore = (value, unit) => {
  const formatted = new Intl.NumberFormat(localeName(), {
    maximumFractionDigits: unit === 'points' ? 0 : 3,
  }).format(Number(value || 0));
  return unit === 'points'
    ? `${formatted} ${t('leaderboards.points')}`
    : `${formatted} Token`;
};
const openReplay = (replayId) => {
  window.open(`/verse-replay/?ranked=${encodeURIComponent(replayId)}`, '_blank', 'noopener');
};
const leaderboardUser = (entry) => ({
  display_name: entry?.display_name || '',
  profile: { avatar_url: entry?.avatar_url || null },
});

const loadBoard = async (key) => {
  if (!key) return;
  const serial = ++requestSerial;
  loading.value = true;
  error.value = '';
  try {
    const payload = await fetchLeaderboard(key);
    if (serial === requestSerial) boardData.value = payload;
  } catch (loadError) {
    if (serial === requestSerial) error.value = loadError?.message || t('leaderboards.loadFailed');
  } finally {
    if (serial === requestSerial) loading.value = false;
  }
};

const selectBoard = (key) => {
  if (key === selectedKey.value && boardData.value) return;
  selectedKey.value = key;
  loadBoard(key);
};

const initialize = async () => {
  if (boards.value.length) {
    if (!boardData.value) await loadBoard(selectedKey.value);
    return;
  }
  loading.value = true;
  error.value = '';
  try {
    const payload = await fetchLeaderboardCatalog();
    boards.value = Array.isArray(payload?.boards) ? payload.boards : [];
    if (!boards.value.some((board) => board.key === selectedKey.value)) {
      selectedKey.value = boards.value[0]?.key || 'supporters';
    }
  } catch (loadError) {
    error.value = loadError?.message || t('leaderboards.loadFailed');
    loading.value = false;
    return;
  }
  loading.value = false;
  await loadBoard(selectedKey.value);
};

onMounted(initialize);
watch(() => props.active, (active) => {
  if (active && !boardData.value && !loading.value) initialize();
});
</script>

<style scoped>
.leaderboard-shell {
  width: min(100%, 72rem);
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.leaderboard-header,
.leaderboard-context,
.leaderboard-list,
.leaderboard-empty,
.leaderboard-error {
  border: 1px solid var(--border-main);
  background: color-mix(in srgb, var(--bg-card) 92%, transparent);
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
}

.leaderboard-header {
  min-height: 7.5rem;
  padding: 1.4rem 1.6rem;
  border-radius: 1rem;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 2rem;
}

.leaderboard-header p {
  max-width: 42rem;
}

.leaderboard-updated {
  display: grid;
  justify-items: end;
  gap: 0.25rem;
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 800;
  white-space: nowrap;
}

.leaderboard-updated strong {
  color: var(--text-main);
}

.leaderboard-tabs {
  display: flex;
  gap: 0.3rem;
  padding: 0.35rem;
  border: 1px solid var(--border-main);
  border-radius: 0.9rem;
  background: color-mix(in srgb, var(--bg-card) 88%, transparent);
  overflow-x: auto;
}

.leaderboard-tab {
  min-width: 10.5rem;
  min-height: 2.8rem;
  padding: 0.65rem 1rem;
  border: 1px solid transparent;
  border-radius: 0.65rem;
  color: var(--text-secondary);
  font-size: var(--font-ui-sm);
  font-weight: 900;
  transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease;
}

.leaderboard-tab:hover:not(:disabled) {
  color: var(--text-main);
  border-color: var(--border-main);
}

.leaderboard-tab.active {
  color: white;
  background: var(--accent);
  border-color: color-mix(in srgb, var(--accent) 80%, black 20%);
}

.leaderboard-context {
  padding: 1rem 1.25rem;
  border-radius: 0.85rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.leaderboard-context h2 {
  color: var(--text-main);
  font-size: 1.2rem;
  font-weight: 900;
}

.leaderboard-context p {
  margin-top: 0.2rem;
  color: var(--text-secondary);
  font-size: var(--font-ui-sm);
  font-weight: 700;
}

.leaderboard-period {
  padding: 0.45rem 0.75rem;
  border: 1px solid var(--border-main);
  border-radius: 999px;
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 900;
  white-space: nowrap;
}

.leaderboard-podium {
  min-height: 14.5rem;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  align-items: end;
  gap: 1rem;
  padding: 0.5rem 5rem 0;
}

.podium-place {
  position: relative;
  min-height: 10.5rem;
  padding: 1.2rem 1rem;
  border: 1px solid var(--border-main);
  border-radius: 1rem 1rem 0.45rem 0.45rem;
  background: var(--bg-card);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.55rem;
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.1);
}

.podium-place.place-1 {
  min-height: 13rem;
  border-color: color-mix(in srgb, #d6aa45 65%, var(--border-main));
  box-shadow: 0 16px 34px color-mix(in srgb, #d6aa45 22%, transparent);
}

.podium-place.place-2 { border-color: color-mix(in srgb, #9aa6b2 55%, var(--border-main)); }
.podium-place.place-3 { border-color: color-mix(in srgb, #b77745 52%, var(--border-main)); }

.podium-rank {
  position: absolute;
  top: 0.7rem;
  left: 0.8rem;
  width: 1.7rem;
  height: 1.7rem;
  border-radius: 50%;
  display: grid;
  place-items: center;
  color: var(--text-main);
  background: color-mix(in srgb, var(--bg-main) 75%, white 25%);
  font-size: var(--font-ui-xs);
  font-weight: 900;
}

.leader-avatar {
  position: relative;
  flex: 0 0 auto;
  width: 2.25rem;
  height: 2.25rem;
  display: grid;
  place-items: center;
}

.leader-avatar.large {
  width: 3.5rem;
  height: 3.5rem;
  font-size: var(--font-ui-md);
}

.leader-avatar :deep(.account-avatar-shell) {
  width: 100%;
  height: 100%;
}

.podium-name {
  max-width: 100%;
  overflow: hidden;
  color: var(--text-main);
  font-size: var(--font-ui-md);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.podium-score {
  color: var(--accent);
  font-size: var(--font-ui-sm);
  font-weight: 900;
}

.game-entry-meta {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 800;
}

.replay-link {
  padding: 0.3rem 0.65rem;
  border: 1px solid var(--border-main);
  border-radius: 0.45rem;
  color: var(--text-main);
  background: var(--bg-card);
  font-size: var(--font-ui-xs);
  font-weight: 900;
}

.replay-link:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.leaderboard-list {
  overflow: hidden;
  border-radius: 1rem;
}

.leaderboard-list-head,
.leaderboard-row {
  display: grid;
  grid-template-columns: 5rem minmax(0, 1fr) 14rem;
  align-items: center;
  gap: 1rem;
  padding: 0 1.25rem;
}

.leaderboard-list.score-hidden .leaderboard-list-head,
.leaderboard-list.score-hidden .leaderboard-row {
  grid-template-columns: 5rem minmax(0, 1fr);
}

.leaderboard-list.game-board .leaderboard-list-head,
.leaderboard-list.game-board .leaderboard-row {
  grid-template-columns: 5rem minmax(0, 1fr) minmax(24rem, 32rem);
}

.leaderboard-list-head {
  min-height: 2.9rem;
  color: var(--text-secondary);
  background: color-mix(in srgb, var(--bg-main) 72%, transparent);
  font-size: var(--font-ui-xs);
  font-weight: 900;
  text-transform: uppercase;
}

.leaderboard-list-head span:last-child { text-align: right; }
.leaderboard-list.score-hidden .leaderboard-list-head span:last-child { text-align: left; }

.leaderboard-row {
  min-height: 4.2rem;
  border-top: 1px solid var(--border-main);
}

.leaderboard-row:nth-child(odd) {
  background: color-mix(in srgb, var(--bg-main) 24%, transparent);
}

.row-rank {
  color: var(--text-secondary);
  font-size: var(--font-ui-md);
  font-weight: 900;
}

.row-player {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 0.8rem;
}

.row-player strong {
  overflow: hidden;
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.row-score {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 0.65rem;
  color: var(--accent);
  font-size: var(--font-ui-sm);
  text-align: right;
}

.row-score small {
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 800;
}

.leaderboard-empty,
.leaderboard-error {
  min-height: 12rem;
  padding: 2rem;
  border-radius: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  color: var(--text-secondary);
  font-size: var(--font-ui-sm);
  font-weight: 800;
}

.leaderboard-error {
  min-height: 5rem;
  color: #ef4444;
}

.leaderboard-skeleton {
  border: 1px solid var(--border-main);
  border-radius: 1rem;
  background: linear-gradient(90deg, var(--bg-card), color-mix(in srgb, var(--bg-card) 75%, white 25%), var(--bg-card));
  background-size: 220% 100%;
  animation: leaderboard-shimmer 1.4s linear infinite;
}

.podium-skeleton { height: 14rem; }
.list-skeleton { height: 18rem; }

@keyframes leaderboard-shimmer {
  to { background-position: -220% 0; }
}
</style>
