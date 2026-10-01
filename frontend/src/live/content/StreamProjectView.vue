<template>
  <div class="stream-project">
    <header><span><small>{{ metric.label }}</small><strong>{{ metric.value }}</strong></span>
      <span><small>{{ payload.time_limit_ms ? (lang === 'zh' ? '倒计时' : 'COUNTDOWN') : (lang === 'zh' ? '用时' : 'ELAPSED') }}</small><strong>{{ elapsed }}</strong></span></header>
    <ObservedProjectBoard :view="view" :tile-styles="palette" @present="shown = $event" @pending="$emit('pending', $event)" />
    <small>{{ payload.move_count || 0 }} {{ lang === 'zh' ? '步' : 'moves' }}</small>
  </div>
</template>
<script setup>
import { computed, shallowRef, ref, watch, onMounted, onBeforeUnmount } from 'vue';
import { shouldRefreshLiveClock } from '../displayClock.js';
import ObservedProjectBoard from '../../../../competition/shared/ObservedProjectBoard.vue';
import { projectPerformanceMetric } from '../../../../competition/shared/projectMetrics.mjs';
import { liveTileColors } from '../tilePalette.js';
const props = defineProps({ view: Object, lang: String, suspended: Boolean });
defineEmits(['pending']);
const shown = shallowRef(null);
const payload = computed(() => (shown.value || props.view)?.payload || {});
const now = ref(performance.now()), receivedAt = ref(performance.now());
let timer;
watch(() => [payload.value.elapsed_ms, props.suspended], () => { now.value = receivedAt.value = performance.now(); });
onMounted(() => { timer = setInterval(() => { if (shouldRefreshLiveClock()) now.value = performance.now(); }, 50); });
onBeforeUnmount(() => clearInterval(timer));
const metric = computed(() => projectPerformanceMetric(shown.value || props.view, props.lang));
const palette = computed(() => Object.fromEntries(Array.from({ length: 30 }, (_, i) => [2 ** (i + 1), liveTileColors(2 ** (i + 1))])));
const elapsed = computed(() => {
  const running = !payload.value.finished && !props.suspended;
  const milliseconds = Number(payload.value.elapsed_ms || 0) + (running ? Math.max(0, now.value - receivedAt.value) : 0);
  const value = Math.max(0, payload.value.time_limit_ms ? payload.value.time_limit_ms - milliseconds : milliseconds);
  return `${String(Math.floor(value / 60000)).padStart(2, '0')}:${String(Math.floor(value / 1000) % 60).padStart(2, '0')}.${String(Math.floor(value / 10) % 100).padStart(2, '0')}`;
});
</script>
<style scoped>
.stream-project{height:100%;display:grid;grid-template-rows:48px minmax(0,1fr) 22px;gap:8px;align-items:center}
header{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;align-items:end;border-bottom:1px solid #334155;padding:0 2px 7px}
header span{display:grid;gap:3px;min-width:0}header span:last-child{text-align:right}small{color:#94a3b8;font-size:11px}header strong{color:#f8fafc;font-size:24px;font-variant-numeric:tabular-nums;line-height:1}
.stream-project>:deep(.tournament-board),.stream-project>:deep(.poly-board){width:min(100%,330px);margin:auto;background:#25334b}
.stream-project>:deep(.cargo-stage){width:min(100%,330px);max-height:100%}
.stream-project :deep(.cargo-number-board){background:#25334b}.stream-project :deep(.board-cell),.stream-project :deep(.poly-cell){background:#3b4960}
.stream-project>small{text-align:center}
</style>
