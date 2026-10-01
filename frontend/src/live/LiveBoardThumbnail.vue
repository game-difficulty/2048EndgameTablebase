<template>
  <div class="live-board-thumbnail" :style="boardStyle" aria-hidden="true">
    <i v-for="(value, index) in values" :key="index" :style="tileStyle(value)">
      <span v-if="value" :style="getTileLabelStyle({ value })">{{ value }}</span>
    </i>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { getTileLabelStyle } from '../components/tileLabelStyle.js';
import { liveAppearanceTileStyle } from '../human/liveAppearance.js';
import { liveEmptyTileColors, liveTileColors } from './tilePalette.js';

const props = defineProps({
  board: { type: Array, default: () => [] },
  variant: { type: String, default: '4x4' },
  appearance: { type: Object, default: null },
});

const dimensions = computed(() => ({
  '4x4': [4, 4],
  '3x4': [3, 4],
  '2x4': [2, 4],
  '3x3': [3, 3],
}[props.variant] || [4, 4]));
const values = computed(() => {
  const count = dimensions.value[0] * dimensions.value[1];
  return Array.from({ length: count }, (_, index) => Number(props.board[index]) || 0);
});
const boardStyle = computed(() => {
  const [rows, cols] = dimensions.value;
  return {
    '--thumbnail-rows': rows,
    '--thumbnail-cols': cols,
    '--thumbnail-width': `${Math.min(90, 50 * cols / rows)}%`,
    '--thumbnail-max-width': `${180 * cols / rows}px`,
    '--tile-label-small': '14px',
    '--tile-label-medium': '11px',
    '--tile-label-large': '8.5px',
  };
});
const tileStyle = value => value
  ? (liveAppearanceTileStyle(props.appearance, value) || liveTileColors(value))
  : liveEmptyTileColors();
</script>

<style scoped>
.live-board-thumbnail {
  z-index: 1;
  box-sizing: border-box;
  display: grid;
  width: var(--thumbnail-width);
  max-width: var(--thumbnail-max-width);
  grid-template-columns: repeat(var(--thumbnail-cols), minmax(0, 1fr));
  gap: 2.5%;
  padding: 2.5%;
  border-radius: 4px;
  background: var(--color-board-bg, #30302e);
  box-shadow: 0 9px 24px #0004;
}
.live-board-thumbnail > i {
  display: flex;
  aspect-ratio: 1;
  min-width: 0;
  align-items: center;
  justify-content: center;
  border-radius: 3px;
  font-style: normal;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
</style>
