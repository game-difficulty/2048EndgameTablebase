<template>
  <div ref="viewport" class="room-stage-viewport" :style="{ '--room-stage-scale': scale }">
    <div class="room-stage-canvas"><slot /></div>
  </div>
</template>
<script setup>
import { ref, shallowRef, provide, onMounted, onBeforeUnmount } from 'vue';
import { roomSurfaceHost, observeSurfaceBox } from './roomSurfaceSize.js';
// Room contract: content never sets the outer dimensions. Both layouts share this
// 1280x720 coordinate space; resizing only scales the presentation surface.
const viewport = ref(null), scale = ref(1);
const host = shallowRef(null);
provide(roomSurfaceHost, host);
let disconnect;
function fit(width, height) {
  if (width > 0 && height > 0) scale.value = Math.min(width / 1280, height / 720);
}
function refreshLayout() {
  disconnect?.();
  // A surface moved into Document PiP must observe the new owning window.
  // Rebind on return too; the opener may have resized while it was detached.
  host.value = viewport.value.ownerDocument.defaultView;
  disconnect = observeSurfaceBox(viewport.value, fit);
}
onMounted(refreshLayout);
onBeforeUnmount(() => disconnect?.());
defineExpose({ element: () => viewport.value, refreshLayout });
</script>
<style scoped>
.room-stage-viewport { position:relative;width:100%;min-width:0;overflow:hidden;border-radius:12px;background:var(--bg-main); }
.room-stage-viewport::before { content:'';display:block;padding-top:56.25%; }
.room-stage-canvas { position:absolute;inset:0 auto auto 0;width:1280px;height:720px;transform:scale(var(--room-stage-scale));transform-origin:top left;overflow:hidden; }
.room-stage-canvas :deep(.content-stage) { width:100%;height:100%; }
</style>
