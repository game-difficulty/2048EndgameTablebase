<template>
  <div class="poly-live-shell">
    <header><span><small>{{ metric.label }}</small><strong>{{ metric.value }}</strong></span><span><small>{{ payload.time_limit_ms ? t('倒计时', 'COUNTDOWN') : t('用时', 'ELAPSED') }}</small><strong>{{ projectTime }}</strong></span></header>
    <PolyominoBoard :snapshot="snapshot" disabled />
    <small>{{ Number(payload.move_count || 0).toLocaleString() }} {{ t('步', 'moves') }}</small>
  </div>
</template>

<script setup>
import { computed, onMounted, onBeforeUnmount, ref, watch } from 'vue';
import { projectPerformanceMetric } from '../../../../competition/shared/projectMetrics.mjs';
import PolyominoBoard from '../../../../competition/frontend/src/projects/PolyominoBoard.vue';

const props = defineProps({ view: { type: Object, required: true }, lang: { type: String, default: 'zh' }, suspended: Boolean });
const payload = computed(() => props.view?.payload || {});
const metric = computed(() => projectPerformanceMetric(props.view, props.lang));
const snapshot = computed(() => ({
  ...payload.value,
  revision: Number(props.view?.sequence || 0),
  transition: payload.value.last_transition || null,
}));
const now = ref(performance.now());
const receivedAt = ref(performance.now());
let interval;
const t = (zh, en) => props.lang === 'zh' ? zh : en;
const projectTime = computed(() => {
  const milliseconds = Math.max(0, Number(payload.value.elapsed_ms || 0)
    + (payload.value.finished || props.suspended ? 0 : now.value - receivedAt.value));
  const limit = Number(payload.value.time_limit_ms || 0);
  const display = limit > 0 ? Math.max(0, limit - milliseconds) : milliseconds;
  const minutes = String(Math.floor(display / 60000)).padStart(2, '0');
  const seconds = String(Math.floor(display / 1000) % 60).padStart(2, '0');
  const centiseconds = String(Math.floor(display / 10) % 100).padStart(2, '0');
  return `${minutes}:${seconds}.${centiseconds}`;
});
watch(() => [payload.value.elapsed_ms, props.suspended], () => { receivedAt.value = performance.now(); now.value = receivedAt.value; }, { immediate: true });
onMounted(() => { interval = setInterval(() => { now.value = performance.now(); }, 16); });
onBeforeUnmount(() => clearInterval(interval));
</script>

<style scoped>
.poly-live-shell { display:grid; grid-template-rows:48px minmax(0,1fr) 22px; align-items:center; gap:8px; height:100%; }
.poly-live-shell header { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); align-items:end; gap:8px; padding:0 2px 7px; border-bottom:1px solid #334155; }
.poly-live-shell header span { display:grid; gap:3px; min-width:0; }
.poly-live-shell header span:last-child { text-align:right; }
.poly-live-shell header small { color:#94a3b8; font-size:10px; letter-spacing:.08em; }
.poly-live-shell header strong { overflow:hidden; color:#f8fafc; font-size:24px; font-variant-numeric:tabular-nums; line-height:1; text-overflow:ellipsis; white-space:nowrap; }
.poly-live-shell > :deep(.poly-board) { width:min(100%,330px); margin:auto; background:#25334b; }
.poly-live-shell > :deep(.poly-board .poly-cell) { background:#3b4960; border-radius:4px; }
.poly-live-shell > :deep(.poly-board .poly-piece) { border-radius:4px; }
.poly-live-shell > small { color:#94a3b8; text-align:center; }
</style>
