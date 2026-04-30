<template>
  <div class="page-root">
    <div class="w-full max-w-6xl flex flex-col gap-5">
      <div class="flex items-center justify-between gap-4">
        <div>
          <h1 class="font-[Cambria,serif] text-4xl font-extrabold tracking-tight text-text-main">{{ $t('menu.minigames') }}</h1>
          <p class="mt-1 text-text-secondary ui-body">{{ $t('minigames.menu.subtitle') }}</p>
        </div>
        <div class="minigame-trophy-summary flex items-center gap-2 rounded-2xl border border-border-main/60 bg-bg-card/80 px-3 py-2 shadow-sm">
          <div
            v-for="stat in trophyStats"
            :key="stat.key"
            class="minigame-trophy-stat"
            :title="stat.alt"
          >
            <img :src="stat.src" :alt="stat.alt" class="minigame-trophy-stat-icon" />
            <span class="minigame-trophy-stat-value">{{ stat.value }}/{{ totalGames }}</span>
          </div>
        </div>
        <div class="flex items-center gap-3">
          <span :class="connectionBadgeClass">{{ $t(`status.${wsStatus}`) }}</span>
          <div class="top-menu-shell surface-prominent-soft">
            <span class="ui-kicker font-black uppercase tracking-widest text-text-secondary">{{ $t('minigames.menu.difficulty') }}</span>
            <div class="flex overflow-hidden rounded-lg border border-border-main/50 shadow-sm">
              <button
                type="button"
                :class="['px-3 py-2 ui-control font-black uppercase tracking-tighter transition-all', difficulty === 0 ? 'btn-prominent text-white' : 'bg-bg-main text-text-secondary']"
                @click="$emit('set-difficulty', 0)"
              >
                {{ $t('minigames.menu.easy') }}
              </button>
              <button
                type="button"
                :class="['px-3 py-2 ui-control font-black uppercase tracking-tighter transition-all', difficulty === 1 ? 'btn-prominent text-white' : 'bg-bg-main text-text-secondary']"
                @click="$emit('set-difficulty', 1)"
              >
                {{ $t('minigames.menu.hard') }}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div
          v-for="item in allItems"
          :key="item.id"
          class="w-full min-w-0"
          :ref="(el) => setCardRef(item.id, el)"
        >
          <MinigameCard
            :item="item"
            @start="$emit('start-game', $event)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, watch } from 'vue';

import MinigameCard from '../components/MinigameCard.vue';
import { getMinigameAssetUrl } from '../../../services/runtime/backendUrl';

const props = defineProps({
  sections: {
    type: Array,
    default: () => [],
  },
  difficulty: {
    type: Number,
    default: 1,
  },
  wsStatus: {
    type: String,
    default: 'connecting',
  },
  connectionBadgeClass: {
    type: String,
    default: 'badge-base badge-connection-pending',
  },
  focusGameId: {
    type: String,
    default: '',
  },
});

defineEmits(['set-difficulty', 'start-game']);

const cardRefs = new Map();
const allItems = computed(() => props.sections.flatMap((section) => section?.items || []));
const totalGames = computed(() => allItems.value.length);
const trophyStats = computed(() => {
  const items = allItems.value;
  const counts = {
    bronze: 0,
    silver: 0,
    gold: 0,
    grand: 0,
  };
  items.forEach((item) => {
    const trophy = Number(item?.summary?.trophy || 0);
    if (trophy >= 1) counts.bronze += 1;
    if (trophy >= 2) counts.silver += 1;
    if (trophy >= 3) counts.gold += 1;
    if (trophy >= 4) counts.grand += 1;
  });
  return [
    { key: 'bronze', src: getMinigameAssetUrl('bronze.png'), alt: 'Bronze trophies', value: counts.bronze },
    { key: 'silver', src: getMinigameAssetUrl('silver.png'), alt: 'Silver trophies', value: counts.silver },
    { key: 'gold', src: getMinigameAssetUrl('gold.png'), alt: 'Gold trophies', value: counts.gold },
    { key: 'grand', src: getMinigameAssetUrl('grand.png'), alt: 'Grand trophies', value: counts.grand },
  ];
});

const setCardRef = (id, element) => {
  if (!id) return;
  if (!element) {
    cardRefs.delete(id);
    return;
  }
  cardRefs.set(id, element instanceof HTMLElement ? element : element?.$el || null);
};

const scrollToFocusedGame = async (gameId) => {
  if (!gameId) return;
  await nextTick();
  const target = cardRefs.get(gameId);
  if (target instanceof HTMLElement) {
    target.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'smooth' });
  }
};

watch(
  () => [props.focusGameId, props.sections],
  async ([gameId]) => {
    await scrollToFocusedGame(gameId);
  },
  { deep: true, immediate: true }
);
</script>

<style scoped>
.minigame-trophy-summary {
  flex-wrap: wrap;
  justify-content: center;
}

.minigame-trophy-stat {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  min-width: 4.65rem;
  border-radius: 999px;
  background: color-mix(in srgb, var(--bg-main) 70%, transparent);
  padding: 0.35rem 0.55rem;
}

.minigame-trophy-stat-icon {
  width: 1rem;
  height: 1rem;
  object-fit: contain;
  flex: 0 0 auto;
}

.minigame-trophy-stat-value {
  color: var(--text-main);
  font-size: calc(0.82rem * var(--ui-scale));
  font-weight: 900;
  line-height: 1;
  letter-spacing: -0.02em;
}
</style>
