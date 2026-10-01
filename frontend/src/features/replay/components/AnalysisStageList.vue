<template>
  <div class="analysis-stages">
    <div class="stage-columns" aria-hidden="true">
      <span>{{ text('阶段 / 原局步数', 'Stage / original moves') }}</span>
      <span>{{ text('吻合度', 'Accuracy') }}</span>
      <span></span>
    </div>
    <button v-for="stage in artifacts" :key="stage.artifact_id" type="button" class="stage-row"
      :disabled="!stage.available || opening === stage.artifact_id"
      :aria-label="`${text('阶段', 'Stage')} ${stage.segment_index + 1}, ${range(stage)}, ${text('吻合度', 'accuracy')} ${formatAnalysisFit(stage.goodness_of_fit)}`"
      @click="$emit('open', stage)">
      <span class="stage-range">
        <span class="stage-index">{{ String(stage.segment_index + 1).padStart(2, '0') }}</span>
        <span>{{ range(stage) }}</span>
        <small v-if="!stage.available">{{ text('已过期', 'Expired') }}</small>
      </span>
      <strong class="stage-fit">{{ formatAnalysisFit(stage.goodness_of_fit) }}</strong>
      <LoaderCircle v-if="opening === stage.artifact_id" :size="16" class="stage-loading" aria-hidden="true" />
      <ExternalLink v-else :size="16" aria-hidden="true" />
    </button>
  </div>
</template>

<script setup>
import { ExternalLink, LoaderCircle } from '@lucide/vue';
import { formatAnalysisFit } from '../analysisPresentation.js';

const props = defineProps({ artifacts: { type: Array, default: () => [] }, language: { type: String, default: 'zh' }, opening: { type: String, default: '' } });
defineEmits(['open']);
const text = (zh, en) => props.language.startsWith('en') ? en : zh;
const range = stage => {
  const format = value => new Intl.NumberFormat(props.language.startsWith('en') ? 'en-US' : 'zh-CN').format(value);
  return `${format(Number(stage.source_start_index) + 1)}–${format(Number(stage.source_end_index))}`;
};
</script>

<style scoped>
.analysis-stages { min-width: 0; border-top: 1px solid var(--border-main); }
.stage-columns, .stage-row { display: grid; grid-template-columns: minmax(0, 1fr) 88px 18px; align-items: center; gap: 12px; }
.stage-columns { padding: 8px 10px; color: var(--text-secondary); font-size: 12px; }
.stage-columns > :nth-child(2) { text-align: right; }
.stage-row { width: 100%; min-height: 42px; border: 0; border-top: 1px solid var(--border-main); padding: 9px 10px; background: transparent; color: var(--text-main); text-align: left; font-size: 13px; cursor: pointer; }
.stage-row:hover:not(:disabled), .stage-row:focus-visible { background: var(--bg-card); outline: 2px solid var(--accent); outline-offset: -2px; }
.stage-row:disabled { opacity: 0.55; cursor: default; }
.stage-range { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; min-width: 0; font-variant-numeric: tabular-nums; }
.stage-index { min-width: 22px; color: var(--text-secondary); font-size: 12px; }
.stage-range small { color: var(--text-secondary); font-size: 11px; }
.stage-fit { text-align: right; color: var(--accent); font-variant-numeric: tabular-nums; }
.stage-loading { animation: stage-spin 1s linear infinite; }
@keyframes stage-spin { to { transform: rotate(360deg); } }
@media (max-width: 480px) { .stage-columns, .stage-row { grid-template-columns: minmax(0, 1fr) 68px 16px; gap: 6px; padding-left: 4px; padding-right: 4px; } }
@media (prefers-reduced-motion: reduce) { .stage-loading { animation: none; } }
</style>
