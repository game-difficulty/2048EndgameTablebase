<template>
  <button
    type="button"
    :disabled="!item.implemented"
    :class="[
      'group relative flex w-full flex-col rounded-2xl border p-4 text-left transition-all duration-300',
      item.implemented
        ? 'border-border-main bg-bg-card shadow-sm hover:-translate-y-1 hover:shadow-lg'
        : 'cursor-not-allowed border-border-main/50 bg-bg-card/40 opacity-60'
    ]"
    @click="$emit('start', item.id)"
  >
    <div v-if="trophyVisual" class="minigame-card-trophy-corner" aria-hidden="true">
      <img :src="trophyVisual.backgroundSrc" alt="" class="minigame-card-trophy-bg" />
      <img :src="trophyVisual.trophySrc" :alt="trophyVisual.trophyAlt" class="minigame-card-trophy-icon" />
    </div>
    <div class="mb-3 w-full flex-none overflow-hidden rounded-xl border border-border-main/60 bg-bg-main/70 aspect-square">
      <img :src="item.coverUrl" :alt="item.title" class="h-full w-full object-cover" />
    </div>
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <h3 class="truncate font-black text-text-main ui-text-lg tracking-tight">{{ item.title }}</h3>
        <p class="mt-1 min-h-[3.2rem] text-text-secondary ui-body">{{ item.description }}</p>
      </div>
      <span
        v-if="item.summary?.trophy"
        :class="['pill-badge shrink-0', trophyClass]"
      >
        {{ trophyMeta.label }}
      </span>
    </div>
    <div class="minigame-card-metrics mt-4 grid grid-cols-2 gap-2">
      <div class="metric-chip minigame-card-metric-chip">
        <div class="metric-label minigame-card-metric-label">{{ $t('minigames.menu.bestScore') }}</div>
        <div class="metric-value minigame-card-metric-value">{{ item.summary?.bestScore || 0 }}</div>
      </div>
      <div class="metric-chip minigame-card-metric-chip">
        <div class="metric-label minigame-card-metric-label">{{ $t('minigames.menu.highestTile') }}</div>
        <div class="metric-value minigame-card-metric-value">{{ highestTileDisplay }}</div>
      </div>
    </div>
    <div v-if="!item.implemented" class="mt-3 flex items-center justify-end">
      <span class="ui-caption text-text-secondary opacity-70">
        {{ $t('minigames.menu.pendingMigration') }}
      </span>
    </div>
  </button>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

import { formatTileNumber, getTrophyMeta, getTrophyVisualMeta } from '../model/minigameMappers';

const props = defineProps({
  item: {
    type: Object,
    required: true,
  },
});

defineEmits(['start']);

const { t } = useI18n();

const trophyMeta = computed(() => getTrophyMeta(props.item?.summary?.trophy));
const trophyVisual = computed(() => getTrophyVisualMeta(props.item?.summary?.trophy));
const trophyClass = computed(() => {
  if (trophyMeta.value.tone === 'gold' || trophyMeta.value.tone === 'grand') return 'pill-badge-accent';
  if (trophyMeta.value.tone === 'silver') return 'pill-badge-soft';
  return 'pill-badge';
});
const highestTileDisplay = computed(() => {
  const tile = Number(props.item?.summary?.highestTile || 0);
  return tile ? formatTileNumber(tile) : '--';
});
</script>

<style scoped>
.minigame-card-metric-label {
  min-height: 2.8em;
  display: flex;
  align-items: flex-start;
  margin: 0;
}

.minigame-card-metric-chip {
  display: grid;
  grid-template-rows: minmax(2.8em, auto) auto;
  align-content: start;
}

.minigame-card-metric-value {
  margin-top: 0;
  align-self: start;
}

.minigame-card-trophy-corner {
  position: absolute;
  top: 0.38rem;
  left: 0.38rem;
  width: 4.0rem;
  height: 4.0rem;
  pointer-events: none;
  z-index: 2;
}

.minigame-card-trophy-bg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.minigame-card-trophy-icon {
  position: absolute;
  left: 0.48rem;
  top: 0.3rem;
  width: 1.92rem;
  height: 1.92rem;
  object-fit: contain;
}
</style>
