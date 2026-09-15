<template>
  <div ref="viewport" :class="['fit-tool-page', { 'fit-tool-page--disabled': disabled }]">
    <div ref="content" class="fit-tool-content" :style="contentStyle"><slot /></div>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue';

const props = defineProps({ disabled: Boolean });
const viewport = ref(null);
const content = ref(null);
const scale = ref(1);
const contentStyle = computed(() => props.disabled ? {} : {
  transform: `scale(${scale.value})`,
  left: `${(1 - scale.value) * 50}%`,
});
let observer;
const measure = () => {
  if (props.disabled || !viewport.value?.clientHeight || !content.value?.offsetHeight) return;
  scale.value = Math.min(1, Math.max(1, viewport.value.clientHeight - 2) / content.value.offsetHeight);
};
onMounted(() => {
  if (typeof ResizeObserver === 'function') {
    observer = new ResizeObserver(measure);
    observer.observe(viewport.value);
    observer.observe(content.value);
  }
  window.addEventListener('resize', measure);
  document.fonts?.ready.then(measure);
  nextTick(measure);
});
watch(() => props.disabled, () => nextTick(measure));
onUnmounted(() => {
  observer?.disconnect();
  window.removeEventListener('resize', measure);
});
</script>

<style scoped>
.fit-tool-page { position: relative; width: 100%; height: 100%; min-height: 0; overflow: hidden; }
.fit-tool-content { position: absolute; top: 0; width: 100%; transform-origin: top left; }
.fit-tool-content :deep(> .page-root) { height: auto; overflow: visible; }
.fit-tool-page--disabled .fit-tool-content { position: relative; height: 100%; }
.fit-tool-page--disabled .fit-tool-content :deep(> .page-root) { height: 100%; overflow: auto; }
</style>
