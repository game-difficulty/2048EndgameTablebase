<template>
  <section class="minigame-leaderboard-panel" :aria-label="$t('minigames.leaderboard.title')">
    <header class="minigame-leaderboard-head">
      <div class="min-w-0">
        <h2>{{ $t('minigames.leaderboard.title') }}</h2>
        <p>{{ difficultyLabel }}</p>
      </div>
      <button
        type="button"
        class="minigame-leaderboard-icon-button"
        :disabled="loading"
        :title="$t('minigames.leaderboard.refresh')"
        :aria-label="$t('minigames.leaderboard.refresh')"
        @click="loadBoard"
      >
        <span :class="loading ? 'is-spinning' : ''" aria-hidden="true">&#8635;</span>
      </button>
    </header>

    <div class="minigame-leaderboard-body">
      <div v-if="error && !entries.length" class="minigame-leaderboard-message error">
        <span>{{ error }}</span>
        <button type="button" @click="loadBoard">{{ $t('common.retry') }}</button>
      </div>
      <div v-else-if="loading && !entries.length" class="minigame-leaderboard-skeleton" aria-hidden="true">
        <span v-for="index in 10" :key="index" />
      </div>
      <div v-else-if="!entries.length" class="minigame-leaderboard-message">
        {{ $t('minigames.leaderboard.empty') }}
      </div>
      <div v-else class="minigame-leaderboard-list">
        <div
          v-for="entry in entries"
          :key="entry.entry_key"
          :class="['minigame-leaderboard-row', `rank-${entry.rank}`]"
        >
          <span class="minigame-leaderboard-rank">{{ entry.rank }}</span>
          <span class="minigame-leaderboard-avatar">
            <AccountAvatar :user="leaderboardUser(entry)" :supporter="entry.is_supporter" />
          </span>
          <span class="minigame-leaderboard-player">
            <strong>{{ entry.display_name }}</strong>
            <small v-if="entry.highest_tile">
              {{ $t('minigames.leaderboard.highestTile') }} {{ formatInteger(entry.highest_tile) }}
            </small>
          </span>
          <strong class="minigame-leaderboard-score">{{ formatInteger(entry.score) }}</strong>
        </div>
      </div>
    </div>

    <div v-if="!isAuthenticated" class="minigame-leaderboard-auth">
      {{ $t('minigames.leaderboard.loginHint') }}
    </div>
    <button type="button" class="minigame-leaderboard-full-link" @click="openFullLeaderboard">
      <span>{{ $t('minigames.leaderboard.viewAll') }}</span>
      <span aria-hidden="true">&gt;</span>
    </button>
  </section>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import AccountAvatar from '../../auth/AccountAvatar.vue';
import { useAuthState } from '../../../services/auth/authState';
import { fetchMinigameLeaderboard } from '../services/minigameRankingClient';

const props = defineProps({
  active: { type: Boolean, default: true },
  gameId: { type: String, required: true },
  difficulty: { type: Number, default: 1 },
});
const emit = defineEmits(['navigate-tab']);
const { locale, t } = useI18n();
const { isAuthenticated } = useAuthState();
const boardData = ref(null);
const loading = ref(false);
const error = ref('');
let requestSerial = 0;

const entries = computed(() => boardData.value?.entries || []);
const difficultyLabel = computed(() => Number(props.difficulty)
  ? t('minigames.menu.hard')
  : t('minigames.menu.easy'));
const localeName = computed(() => String(locale.value).toLowerCase().startsWith('zh') ? 'zh-CN' : 'en-US');
const formatInteger = (value) => new Intl.NumberFormat(localeName.value, { maximumFractionDigits: 0 }).format(Number(value || 0));
const leaderboardUser = (entry) => ({
  display_name: entry?.display_name || '',
  profile: { avatar_url: entry?.avatar_url || null },
});

const loadBoard = async () => {
  if (!props.gameId) return;
  const serial = ++requestSerial;
  loading.value = true;
  error.value = '';
  try {
    const payload = await fetchMinigameLeaderboard(props.gameId, {
      difficulty: props.difficulty,
      limit: 10,
    });
    if (serial === requestSerial) boardData.value = payload;
  } catch (loadError) {
    if (serial === requestSerial) error.value = loadError?.message || t('minigames.leaderboard.loadFailed');
  } finally {
    if (serial === requestSerial) loading.value = false;
  }
};
const openFullLeaderboard = () => {
  emit('navigate-tab', 'LeaderboardsView', {
    boardKey: 'minigames',
    gameId: props.gameId,
    difficulty: Number(props.difficulty) ? 1 : 0,
  });
};
const handleScoreUpdated = (event) => {
  const detail = event?.detail || {};
  if (detail.gameId === props.gameId && Number(detail.difficulty) === Number(props.difficulty)) {
    loadBoard();
  }
};

watch(() => [props.gameId, props.difficulty], () => {
  boardData.value = null;
  if (props.active) loadBoard();
});
watch(() => props.active, (active) => {
  if (active) loadBoard();
});
onMounted(() => {
  window.addEventListener('minigame-score-updated', handleScoreUpdated);
  if (props.active) loadBoard();
});
onUnmounted(() => window.removeEventListener('minigame-score-updated', handleScoreUpdated));
</script>

<style scoped>
.minigame-leaderboard-panel {
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
.minigame-leaderboard-head {
  min-height: 4.5rem;
  padding: 0.85rem 0.9rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  border-bottom: 1px solid var(--border-main);
}
.minigame-leaderboard-head h2 { color: var(--text-main); font-size: var(--font-ui-lg); font-weight: 950; }
.minigame-leaderboard-head p { margin-top: 0.15rem; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 800; }
.minigame-leaderboard-icon-button {
  width: 2.35rem;
  height: 2.35rem;
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  border: 1px solid var(--border-main);
  border-radius: 6px;
  color: var(--text-main);
  background: var(--bg-main);
  font-size: 1.2rem;
  font-weight: 900;
}
.minigame-leaderboard-icon-button:hover:not(:disabled) { color: var(--accent); border-color: var(--accent); }
.minigame-leaderboard-icon-button:disabled { opacity: 0.55; }
.minigame-leaderboard-body { flex: 1 1 auto; min-height: 0; }
.minigame-leaderboard-list { display: flex; flex-direction: column; }
.minigame-leaderboard-row {
  min-height: 3.15rem;
  display: grid;
  grid-template-columns: 2rem 2rem minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.5rem;
  padding: 0.42rem 0.7rem;
  border-bottom: 1px solid color-mix(in srgb, var(--border-main) 72%, transparent);
  color: var(--text-main);
}
.minigame-leaderboard-rank {
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
.rank-1 .minigame-leaderboard-rank { background: #d6aa45; }
.rank-2 .minigame-leaderboard-rank { background: #8e9aa7; }
.rank-3 .minigame-leaderboard-rank { background: #b77745; }
.minigame-leaderboard-avatar { width: 1.85rem; height: 1.85rem; display: grid; place-items: center; }
.minigame-leaderboard-avatar :deep(.account-avatar-shell) { width: 100%; height: 100%; }
.minigame-leaderboard-player { min-width: 0; display: grid; gap: 0.08rem; }
.minigame-leaderboard-player strong,
.minigame-leaderboard-player small { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.minigame-leaderboard-player strong { font-size: var(--font-ui-xs); font-weight: 900; }
.minigame-leaderboard-player small { color: var(--text-secondary); font-size: var(--font-ui-2xs); font-weight: 750; }
.minigame-leaderboard-score { color: var(--accent); font-size: var(--font-ui-xs); font-variant-numeric: tabular-nums; font-weight: 950; }
.minigame-leaderboard-message {
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
.minigame-leaderboard-message.error { color: #d14a45; }
.minigame-leaderboard-message button { padding: 0.35rem 0.55rem; border: 1px solid currentColor; border-radius: 4px; }
.minigame-leaderboard-skeleton { padding: 0.45rem 0.7rem; display: grid; gap: 0.45rem; }
.minigame-leaderboard-skeleton span { height: 2.7rem; border-radius: 4px; background: var(--bg-main); }
.minigame-leaderboard-auth {
  padding: 0.55rem 0.8rem;
  border-top: 1px solid var(--border-main);
  color: var(--text-secondary);
  font-size: var(--font-ui-2xs);
  font-weight: 800;
  text-align: center;
}
.minigame-leaderboard-full-link {
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
.minigame-leaderboard-full-link:hover { color: var(--accent); }
.is-spinning { animation: minigame-leaderboard-spin 0.8s linear infinite; }
@keyframes minigame-leaderboard-spin { to { transform: rotate(360deg); } }
</style>
