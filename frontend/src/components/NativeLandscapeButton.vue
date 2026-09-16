<template>
  <div v-if="visible" class="native-landscape-control" :class="{ inline }">
    <button type="button" :disabled="busy" :title="$t('orientation.rotate')"
      :aria-label="$t('orientation.rotate')" @click="rotate">
      <RotateCw :size="22" aria-hidden="true" />
    </button>
    <p v-if="failed" role="status">{{ $t('orientation.unavailable') }}</p>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue';
import { RotateCw } from '@lucide/vue';
import { requestNativeLandscape } from '../utils/nativeLandscape.js';

defineProps({ inline: Boolean });

const visible = ref(false);
const busy = ref(false);
const failed = ref(false);
let media;
let timer;
const update = () => {
  visible.value = Boolean(media?.matches) && window.innerHeight > window.innerWidth;
  if (!visible.value) failed.value = false;
};
const rotate = async () => {
  if (busy.value) return;
  busy.value = true;
  failed.value = false;
  clearTimeout(timer);
  try { failed.value = !await requestNativeLandscape(document, screen.orientation); }
  finally { busy.value = false; update(); }
  if (failed.value) timer = setTimeout(() => { failed.value = false; }, 6000);
};
onMounted(() => {
  media = window.matchMedia('(any-pointer: coarse)');
  media.addEventListener?.('change', update);
  window.addEventListener('resize', update);
  update();
});
onUnmounted(() => {
  media?.removeEventListener?.('change', update);
  window.removeEventListener('resize', update);
  clearTimeout(timer);
});
</script>

<style scoped>
.native-landscape-control { position: fixed; top: max(8px, env(safe-area-inset-top)); right: max(8px, env(safe-area-inset-right)); z-index: 500; }
.native-landscape-control.inline { position:relative;top:auto;right:auto;flex-shrink:0; }
button { width: 44px; height: 44px; display: grid; place-items: center; border: 1px solid var(--border-main); border-radius: 8px; color: var(--accent); background: var(--bg-card); box-shadow: 0 2px 8px #0002; cursor: pointer; }
button:disabled { opacity: .5; cursor: wait; }
p { position: absolute; right: 0; width: min(260px, calc(100vw - 32px)); padding: 10px; margin-top: 6px; border: 1px solid var(--border-main); border-radius: 6px; font-size: 13px; line-height: 1.5; color: var(--text-main); background: var(--bg-card); }
</style>
