<template>
  <div ref="container" class="battle-observed-board" :style="{ '--tile-font-scale': fontScale }">
    <BaseBoard :frame="frame" :dis32k="dis32k" :is-variant="isVariant">
      <template #overlay>
        <BattleCorrectionOverlay v-if="overlay" :overlay="overlay" :compact="width < 280" />
      </template>
    </BaseBoard>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue';
import BaseBoard from '../../../components/BaseBoard.vue';
import { createBoardViewport, createBoardViewportLayout } from '../../../utils/boardViewport.js';
import BattleCorrectionOverlay from './BattleCorrectionOverlay.vue';

const props = defineProps({
  frame: { type: Object, required: true },
  dis32k: Boolean,
  isVariant: Boolean,
  overlay: { type: Object, default: null },
});
const container = ref(null);
const width = ref(160);
const rootFontSize = ref(16);
let observer;
const fontScale = computed(() => {
  const layout = createBoardViewportLayout(createBoardViewport(props.frame.toBoard, props.isVariant));
  const tileWidth = width.value * layout.widthPercent * layout.tileWidthPercent / 10000;
  return Math.min(1.2, tileWidth * .46 / (rootFontSize.value * 2.5));
});
const measure = () => {
  width.value = container.value?.clientWidth || 160;
  rootFontSize.value = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
};
onMounted(() => {
  measure();
  if (typeof ResizeObserver !== 'undefined') {
    observer = new ResizeObserver(measure);
    observer.observe(container.value);
  }
  window.addEventListener('resize', measure);
});
onUnmounted(() => {
  observer?.disconnect();
  window.removeEventListener('resize', measure);
});
</script>

<style scoped>
.battle-observed-board { position: relative; width: 100%; min-width: 0; }
</style>
