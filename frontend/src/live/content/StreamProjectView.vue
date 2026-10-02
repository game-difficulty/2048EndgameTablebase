<template>
  <div class="stream-project">
    <header :style="{'--metric-count':ruleMetrics.length+2}"><span><small>{{ metric.label }}</small><strong>{{ metric.value }}</strong></span>
      <span v-for="item in ruleMetrics" :key="item.key" :class="['rule-metric', {warning:item.warning}]"><small>{{ item.label }}</small><strong>{{ item.value }}</strong></span>
      <span class="time-metric"><small>{{ payload.time_limit_ms ? (lang === 'zh' ? '倒计时' : 'COUNTDOWN') : (lang === 'zh' ? '用时' : 'ELAPSED') }}</small><strong>{{ elapsed }}</strong></span></header>
    <div class="stream-board-area"><ObservedProjectBoard :view="view" :tile-styles="palette" @present="shown = $event" @pending="$emit('pending', $event)" @gap="$emit('gap', $event)" /></div>
    <small>{{ payload.move_count || 0 }} {{ lang === 'zh' ? '步' : 'moves' }}</small>
  </div>
</template>
<script setup>
import { computed, shallowRef, ref, watch, onMounted, onBeforeUnmount } from 'vue';
import { shouldRefreshLiveClock } from '../displayClock.js';
import ObservedProjectBoard from '../../../../competition/shared/ObservedProjectBoard.vue';
import { projectPerformanceMetric, projectRuleMetrics } from '../../../../competition/shared/projectMetrics.mjs';
import { liveTileColors } from '../tilePalette.js';
const props = defineProps({ view: Object, project: Object, lang: String, suspended: Boolean });
defineEmits(['pending', 'gap']);
const shown = shallowRef(null);
const payload = computed(() => (shown.value || props.view)?.payload || {});
const now = ref(performance.now()), receivedAt = ref(performance.now());
let timer;
watch(() => [payload.value.elapsed_ms, props.suspended], () => { now.value = receivedAt.value = performance.now(); });
onMounted(() => { timer = setInterval(() => { if (shouldRefreshLiveClock()) now.value = performance.now(); }, 50); });
onBeforeUnmount(() => clearInterval(timer));
const metric = computed(() => projectPerformanceMetric(shown.value || props.view, props.lang));
const ruleMetrics = computed(() => projectRuleMetrics(shown.value || props.view, props.lang, props.project));
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
header{display:grid;grid-template-columns:repeat(var(--metric-count,2),minmax(0,1fr));gap:8px;align-items:end;height:48px;box-sizing:border-box;border-bottom:1px solid var(--match-line,#334155);padding:0 2px 7px}
header small{line-height:1}
header span{display:grid;gap:3px;min-width:0}header .time-metric{text-align:right}small{color:var(--match-muted,#94a3b8);font-size:11px}header strong{color:var(--match-text,#f8fafc);font-size:24px;font-variant-numeric:tabular-nums;line-height:1}
header .rule-metric{text-align:center}header .rule-metric small{font-size:11px;white-space:nowrap}header .rule-metric strong{font-size:24px;line-height:1}header .rule-metric.warning strong{color:var(--match-accent,#d8bd69)}
.stream-board-area{width:100%;height:100%;min-width:0;min-height:0;container-type:size;display:flex;align-items:center;justify-content:center}
.stream-board-area>:deep(.tournament-board),.stream-board-area>:deep(.poly-board){width:min(100%,330px,100cqh);margin:auto;background:var(--match-tint,#25334b)}
/* Cargo includes two entrance rows and one exit row: fit the entire 4:7 stage without clamping its height. */
.stream-board-area>:deep(.cargo-stage){width:min(100%,330px,calc(100cqh * 4 / 7));flex-shrink:0}
.stream-project :deep(.cargo-number-board){background:var(--match-tint,#25334b)}.stream-project :deep(.board-cell),.stream-project :deep(.poly-cell){background:var(--match-cell,#3b4960)}
.stream-project>small{text-align:center}
</style>
