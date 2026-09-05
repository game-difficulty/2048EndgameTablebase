<template>
  <section class="minigame-board-view">
    <div class="minigame-board-toolbar">
      <label class="minigame-board-select">
        <span>{{ $t('minigames.leaderboard.board') }}</span>
        <select v-model="selectedGameId">
          <option value="overall">{{ $t('minigames.leaderboard.overall') }}</option>
          <option v-for="game in games" :key="game.id" :value="game.id">{{ game.title }}</option>
        </select>
      </label>
      <div class="minigame-board-difficulty" :aria-label="$t('minigames.menu.difficulty')">
        <button type="button" :class="{ active: difficulty === 0 }" @click="difficulty = 0">
          {{ $t('minigames.menu.easy') }}
        </button>
        <button type="button" :class="{ active: difficulty === 1 }" @click="difficulty = 1">
          {{ $t('minigames.menu.hard') }}
        </button>
      </div>
      <button
        type="button"
        class="minigame-board-refresh"
        :disabled="loading"
        :title="$t('minigames.leaderboard.refresh')"
        :aria-label="$t('minigames.leaderboard.refresh')"
        @click="loadBoard"
      >
        <span :class="loading ? 'is-spinning' : ''" aria-hidden="true">&#8635;</span>
      </button>
    </div>

    <div class="minigame-board-context">
      <div>
        <h2>{{ selectedTitle }}</h2>
        <p>{{ selectedDescription }}</p>
      </div>
      <span>{{ difficultyLabel }}</span>
    </div>

    <div v-if="error" class="minigame-board-message error">
      <span>{{ error }}</span>
      <button type="button" @click="loadBoard">{{ $t('common.retry') }}</button>
    </div>
    <div v-else-if="loading && !boardData" class="minigame-board-message">
      {{ $t('minigames.leaderboard.loading') }}
    </div>
    <div v-else-if="!entries.length" class="minigame-board-message">
      {{ $t('minigames.leaderboard.empty') }}
    </div>
    <template v-else>
      <div class="minigame-board-podium" aria-label="Top three">
        <article
          v-for="entry in podiumEntries"
          :key="entry.entry_key"
          :class="['minigame-podium-place', `place-${entry.rank}`]"
        >
          <div class="minigame-podium-rank">{{ entry.rank }}</div>
          <AccountAvatar :user="leaderboardUser(entry)" :supporter="entry.is_supporter" size="large" />
          <strong class="minigame-podium-name">{{ entry.display_name }}</strong>
          <strong v-if="isOverall" class="minigame-podium-trophies">
            <span v-for="trophy in trophyStats(entry)" :key="trophy.key" :title="trophy.label">
              <img :src="trophy.src" alt="" />{{ trophy.value }}
            </span>
          </strong>
          <strong v-else class="minigame-podium-score">{{ formatInteger(entry.score) }}</strong>
          <span
            v-if="!isOverall && trophyLevel(entry).src"
            class="minigame-entry-trophy"
            :title="trophyLevel(entry).label"
            :aria-label="trophyLevel(entry).label"
          >
            <img :src="trophyLevel(entry).src" alt="" />
          </span>
        </article>
      </div>

      <div class="minigame-board-list">
        <div class="minigame-board-list-head">
          <span>{{ $t('leaderboards.rank') }}</span>
          <span>{{ $t('leaderboards.player') }}</span>
          <span>{{ isOverall ? $t('minigames.leaderboard.trophies') : $t('leaderboards.gameScore') }}</span>
        </div>
        <div v-for="entry in listEntries" :key="entry.entry_key" class="minigame-board-row">
          <span class="minigame-row-rank">{{ entry.rank }}</span>
          <span class="minigame-row-player">
            <AccountAvatar :user="leaderboardUser(entry)" :supporter="entry.is_supporter" />
            <strong>{{ entry.display_name }}</strong>
          </span>
          <strong v-if="isOverall" class="minigame-row-trophies">
            <span v-for="trophy in trophyStats(entry)" :key="trophy.key" :title="trophy.label">
              <img :src="trophy.src" alt="" />{{ trophy.value }}
            </span>
          </strong>
          <strong v-else class="minigame-row-score">
            <span>{{ formatInteger(entry.score) }}</span>
            <small v-if="trophyLevel(entry).src || entry.highest_tile">
              <span
                v-if="trophyLevel(entry).src"
                class="minigame-entry-trophy"
                :title="trophyLevel(entry).label"
                :aria-label="trophyLevel(entry).label"
              >
                <img :src="trophyLevel(entry).src" alt="" />
              </span>
              <span v-if="entry.highest_tile">
                {{ $t('minigames.leaderboard.highestTile') }} {{ formatInteger(entry.highest_tile) }}
              </span>
            </small>
          </strong>
        </div>
      </div>
    </template>
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import AccountAvatar from '../../auth/AccountAvatar.vue';
import { getMinigameAssetUrl } from '../../../services/runtime/backendUrl';
import {
  fetchMinigameCatalog,
  fetchMinigameLeaderboard,
  fetchMinigameTrophyLeaderboard,
} from '../services/minigameRankingClient';

const props = defineProps({
  active: { type: Boolean, default: true },
  requestedGameId: { type: String, default: '' },
  requestedDifficulty: { type: Number, default: 1 },
  requestSerial: { type: Number, default: 0 },
});
const { locale, t } = useI18n();
const games = ref([]);
const selectedGameId = ref(props.requestedGameId || 'overall');
const difficulty = ref(Number(props.requestedDifficulty) ? 1 : 0);
const boardData = ref(null);
const loading = ref(false);
const error = ref('');
let loadSerial = 0;

const entries = computed(() => boardData.value?.entries || []);
const isOverall = computed(() => selectedGameId.value === 'overall');
const difficultyLabel = computed(() => difficulty.value
  ? t('minigames.menu.hard')
  : t('minigames.menu.easy'));
const selectedTitle = computed(() => isOverall.value
  ? t('minigames.leaderboard.overall')
  : games.value.find((game) => game.id === selectedGameId.value)?.title || selectedGameId.value);
const selectedDescription = computed(() => isOverall.value
  ? t('minigames.leaderboard.overallDescription')
  : t('minigames.leaderboard.gameDescription'));
const podiumEntries = computed(() => {
  const order = { 2: 0, 1: 1, 3: 2 };
  return entries.value.filter((entry) => entry.rank <= 3)
    .sort((left, right) => (order[left.rank] ?? 9) - (order[right.rank] ?? 9));
});
const listEntries = computed(() => entries.value.filter((entry) => entry.rank > 3));
const localeName = computed(() => String(locale.value).toLowerCase().startsWith('zh') ? 'zh-CN' : 'en-US');
const formatInteger = (value) => new Intl.NumberFormat(localeName.value, { maximumFractionDigits: 0 }).format(Number(value || 0));
const leaderboardUser = (entry) => ({
  display_name: entry?.display_name || '',
  profile: { avatar_url: entry?.avatar_url || null },
});
const trophyStats = (entry) => [
  { key: 'grand', label: t('minigames.leaderboard.trophyLevels.grand'), src: getMinigameAssetUrl('grand.png'), value: entry?.trophies?.grand || 0 },
  { key: 'gold', label: t('minigames.leaderboard.trophyLevels.gold'), src: getMinigameAssetUrl('gold.png'), value: entry?.trophies?.gold || 0 },
  { key: 'silver', label: t('minigames.leaderboard.trophyLevels.silver'), src: getMinigameAssetUrl('silver.png'), value: entry?.trophies?.silver || 0 },
  { key: 'bronze', label: t('minigames.leaderboard.trophyLevels.bronze'), src: getMinigameAssetUrl('bronze.png'), value: entry?.trophies?.bronze || 0 },
];
const trophyLevel = (entry) => {
  const tier = Math.max(0, Math.min(4, Number(entry?.trophy_tier) || 0));
  const key = ['none', 'bronze', 'silver', 'gold', 'grand'][tier];
  return {
    key,
    label: t(`minigames.leaderboard.trophyLevels.${key}`),
    src: tier > 0 ? getMinigameAssetUrl(`${key}.png`) : '',
  };
};

const loadBoard = async () => {
  const serial = ++loadSerial;
  loading.value = true;
  error.value = '';
  try {
    const payload = isOverall.value
      ? await fetchMinigameTrophyLeaderboard({ difficulty: difficulty.value })
      : await fetchMinigameLeaderboard(selectedGameId.value, { difficulty: difficulty.value });
    if (serial === loadSerial) boardData.value = payload;
  } catch (loadError) {
    if (serial === loadSerial) error.value = loadError?.message || t('minigames.leaderboard.loadFailed');
  } finally {
    if (serial === loadSerial) loading.value = false;
  }
};
const initialize = async () => {
  try {
    const payload = await fetchMinigameCatalog();
    games.value = Array.isArray(payload?.games) ? payload.games : [];
    if (selectedGameId.value !== 'overall' && !games.value.some((game) => game.id === selectedGameId.value)) {
      selectedGameId.value = 'overall';
    }
    await loadBoard();
  } catch (loadError) {
    error.value = loadError?.message || t('minigames.leaderboard.loadFailed');
  }
};

watch([selectedGameId, difficulty], () => {
  boardData.value = null;
  if (props.active) loadBoard();
});
watch(() => props.active, (active) => { if (active) loadBoard(); });
watch(() => [props.requestedGameId, props.requestedDifficulty, props.requestSerial], ([gameId, requestedDifficulty]) => {
  selectedGameId.value = gameId || 'overall';
  difficulty.value = Number(requestedDifficulty) ? 1 : 0;
});
onMounted(initialize);
</script>

<style scoped>
.minigame-board-view { display: flex; flex-direction: column; gap: 1rem; }
.minigame-board-toolbar,
.minigame-board-context,
.minigame-board-list,
.minigame-board-message {
  border: 1px solid var(--border-main);
  background: color-mix(in srgb, var(--bg-card) 92%, transparent);
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
}
.minigame-board-toolbar {
  min-height: 4rem;
  padding: 0.65rem;
  border-radius: 8px;
  display: grid;
  grid-template-columns: minmax(15rem, 1fr) 15rem 2.6rem;
  align-items: end;
  gap: 0.65rem;
}
.minigame-board-select { min-width: 0; display: grid; gap: 0.25rem; }
.minigame-board-select span { color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 900; }
.minigame-board-select select {
  min-width: 0;
  height: 2.55rem;
  padding: 0 0.75rem;
  border: 1px solid var(--border-main);
  border-radius: 6px;
  color: var(--text-main);
  background: var(--bg-main);
  font-size: var(--font-ui-sm);
  font-weight: 850;
}
.minigame-board-difficulty { height: 2.55rem; display: grid; grid-template-columns: repeat(2, 1fr); padding: 0.2rem; border: 1px solid var(--border-main); border-radius: 6px; background: var(--bg-main); }
.minigame-board-difficulty button { border-radius: 4px; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 900; }
.minigame-board-difficulty button.active { color: white; background: var(--accent); }
.minigame-board-refresh { width: 2.55rem; height: 2.55rem; display: grid; place-items: center; border: 1px solid var(--border-main); border-radius: 6px; color: var(--text-main); background: var(--bg-main); font-size: 1.2rem; }
.minigame-board-context { padding: 1rem 1.25rem; border-radius: 8px; display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.minigame-board-context h2 { color: var(--text-main); font-size: 1.2rem; font-weight: 900; }
.minigame-board-context p { margin-top: 0.2rem; color: var(--text-secondary); font-size: var(--font-ui-sm); font-weight: 700; }
.minigame-board-context > span { padding: 0.4rem 0.65rem; border: 1px solid var(--border-main); border-radius: 6px; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 900; }
.minigame-board-message { min-height: 15rem; padding: 1rem; border-radius: 8px; display: flex; align-items: center; justify-content: center; gap: 0.6rem; color: var(--text-secondary); font-weight: 800; }
.minigame-board-message.error { color: #d14a45; }
.minigame-board-message button { padding: 0.35rem 0.55rem; border: 1px solid currentColor; border-radius: 4px; }
.minigame-board-podium { min-height: 14.5rem; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); align-items: end; gap: 1rem; padding: 0.5rem 5rem 0; }
.minigame-podium-place { position: relative; min-height: 10.5rem; padding: 1.2rem 0.7rem; border: 1px solid var(--border-main); border-radius: 8px 8px 4px 4px; background: var(--bg-card); display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.5rem; box-shadow: 0 14px 28px rgba(15,23,42,.1); }
.minigame-podium-place.place-1 { min-height: 13rem; border-color: color-mix(in srgb, #d6aa45 65%, var(--border-main)); }
.minigame-podium-rank { position: absolute; top: 0.7rem; left: 0.8rem; font-size: var(--font-ui-xs); font-weight: 950; }
.minigame-podium-name { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-main); font-size: var(--font-ui-sm); }
.minigame-podium-score { color: var(--accent); font-size: var(--font-ui-sm); }
.minigame-entry-trophy { min-width: 0; display: inline-flex; align-items: center; gap: 0.28rem; color: var(--text-secondary); font-size: var(--font-ui-2xs); font-weight: 850; }
.minigame-entry-trophy img { width: 1.15rem; height: 1.15rem; flex: 0 0 auto; object-fit: contain; }
.minigame-podium .minigame-entry-trophy img { width: 1.55rem; height: 1.55rem; }
.minigame-podium-trophies,
.minigame-row-trophies { display: flex; align-items: center; justify-content: flex-end; gap: 0.45rem; color: var(--text-main); font-size: var(--font-ui-xs); }
.minigame-podium-trophies span,
.minigame-row-trophies span { display: inline-flex; align-items: center; gap: 0.18rem; }
.minigame-podium-trophies img,
.minigame-row-trophies img { width: 1rem; height: 1rem; object-fit: contain; }
.minigame-board-list { border-radius: 8px; overflow: hidden; }
.minigame-board-list-head,
.minigame-board-row { display: grid; grid-template-columns: 7rem minmax(0, 1fr) minmax(18rem, auto); align-items: center; gap: 1rem; padding: 0.8rem 1.25rem; }
.minigame-board-list-head { color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 900; text-transform: uppercase; }
.minigame-board-list-head span:last-child { text-align: right; }
.minigame-board-row { min-height: 4rem; border-top: 1px solid var(--border-main); color: var(--text-main); }
.minigame-row-rank { font-weight: 950; }
.minigame-row-player { min-width: 0; display: flex; align-items: center; gap: 0.7rem; }
.minigame-row-player strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--font-ui-sm); }
.minigame-row-score { display: grid; justify-items: end; color: var(--accent); font-size: var(--font-ui-sm); }
.minigame-row-score small { display: flex; align-items: center; justify-content: flex-end; gap: 0.65rem; color: var(--text-secondary); font-size: var(--font-ui-2xs); }
.is-spinning { animation: minigame-board-spin .8s linear infinite; }
@keyframes minigame-board-spin { to { transform: rotate(360deg); } }
</style>
