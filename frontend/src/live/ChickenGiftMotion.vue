<template>
  <span class="chicken-photo-motion">
    <img :src="source" width="240" height="240" alt="" />
    <canvas ref="canvas" width="480" height="480" :class="{ ready }" aria-hidden="true" />
  </span>
</template>
<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue';
import { createChickenRenderer } from './chickenMotion.js';
const props = defineProps({ id: { type:String, required:true } });
const source = computed(() => `/live-gifts/${props.id === 'serious' ? 'serious' : 'chicken-balance'}-reference.jpg`);
const canvas = ref(null), ready = ref(false);
let renderer, frame, image, disposed = false, media, origin = 0, last = -Infinity;
function tick(now) {
  if (disposed || document.hidden || media.matches || !renderer) return;
  if (!origin) origin = now;
  if (now - last >= 1000 / 30) { renderer.draw(now - origin); last = now; }
  frame = requestAnimationFrame(tick);
}
function visibility() {
  cancelAnimationFrame(frame);
  if (document.hidden || media.matches) { ready.value = false; return; }
  if (renderer) { renderer.draw(0); ready.value = true; origin = 0; last = -Infinity; frame = requestAnimationFrame(tick); }
}
function lost(event) {
  event.preventDefault(); cancelAnimationFrame(frame); ready.value = false; renderer = null;
}
onMounted(() => {
  media = matchMedia('(prefers-reduced-motion: reduce)');
  media.addEventListener('change', visibility);
  document.addEventListener('visibilitychange', visibility);
  canvas.value.addEventListener('webglcontextlost', lost);
  image = new Image();
  image.onload = () => {
    if (disposed) return;
    try { renderer = createChickenRenderer(canvas.value, image, props.id); }
    catch { renderer = null; }
    visibility();
  };
  image.src = source.value;
});
onBeforeUnmount(() => {
  disposed = true; cancelAnimationFrame(frame);
  if (image) image.onload = null;
  media?.removeEventListener('change', visibility);
  document.removeEventListener('visibilitychange', visibility);
  canvas.value?.removeEventListener('webglcontextlost', lost);
  renderer?.dispose();
});
</script>
<style scoped>
.chicken-photo-motion { position:relative; display:block; width:100%; height:100%; overflow:hidden; border-radius:4px; }
img,canvas { position:absolute; inset:0; width:100%; height:100%; display:block; }
canvas { opacity:0; } canvas.ready { opacity:1; }
</style>
