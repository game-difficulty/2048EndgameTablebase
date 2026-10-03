import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { threeByThreeTileStyle } from './threeByThreeTileStyle.js';

export function useThreeByThreeTileStyle(element, dimensions, tileSide) {
  return useMeasuredTileStyle(element, dimensions, tileSide, threeByThreeTileStyle);
}

export function useMeasuredTileStyle(element, dimensions, tileSide, resolveStyle) {
  const style = ref({});
  let observer;
  const update = () => {
    const { rows, cols } = dimensions();
    // Measure the untransformed width only; never feed the resized height back into layout.
    style.value = resolveStyle(rows, cols, element.value ? tileSide(element.value) : 0);
  };
  watch(dimensions, update, { flush: 'post' });
  onMounted(() => {
    update();
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(update);
      observer.observe(element.value);
    }
    window.addEventListener('resize', update);
  });
  onBeforeUnmount(() => {
    observer?.disconnect();
    window.removeEventListener('resize', update);
  });
  return style;
}
