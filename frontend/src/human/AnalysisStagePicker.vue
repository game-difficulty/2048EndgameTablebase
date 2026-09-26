<template>
  <component :is="collapsible ? 'details' : 'section'" class="stage-picker" :open="collapsible ? expanded : undefined">
    <summary v-if="collapsible">{{ label('阶段回放', 'Stage replays') }} <span class="stage-count">{{ artifacts.length }}</span></summary>
    <div class="stage-picker-scroll" tabindex="0" :aria-label="label('选择回放阶段', 'Choose a replay stage')">
      <table>
        <thead><tr><th>{{ label('阶段', 'Stage') }}</th><th>{{ label('原局步号', 'Game moves') }}</th><th>{{ label('吻合度', 'Fit') }}</th></tr></thead>
        <tbody><tr v-for="stage in artifacts" :key="stage.artifact_id" :class="{ selected: selectedId === stage.artifact_id, unavailable: stage.available === false }">
          <td><label><input v-model="selectedId" type="radio" :name="groupId" :value="stage.artifact_id" :disabled="stage.available === false" :aria-label="`${label('阶段', 'Stage')} ${stage.segment_index + 1}`">{{ stage.segment_index + 1 }}</label></td>
          <td>{{ number(stage.source_start_index) }}–{{ number(stage.source_end_index) }}<small v-if="stage.available === false"> · {{ label('已过期', 'Expired') }}</small></td>
          <td>{{ percent(stage.goodness_of_fit) }}</td>
        </tr></tbody>
      </table>
    </div>
    <footer>
      <span v-if="selected">{{ label('已选阶段', 'Selected stage') }} {{ selected.segment_index + 1 }}</span>
      <span v-else>{{ label('暂无可用回放', 'No available replay') }}</span>
      <button type="button" :disabled="!selected || busy" @click="$emit('open', selectedId)">{{ busy ? label('正在打开…', 'Opening…') : label('打开回放 ↗', 'Open replay ↗') }}</button>
    </footer>
  </component>
</template>

<script setup>
import { computed, ref, useId, watch } from 'vue';
import { language } from './i18n.js';
const props = defineProps({ artifacts: { type: Array, default: () => [] }, expanded: Boolean, busy: Boolean, collapsible: { type: Boolean, default: true } });
defineEmits(['open']);
const groupId = useId();
const selectedId = ref('');
const label = (zh, en) => language.value === 'en' ? en : zh;
const number = value => value == null ? '—' : Number(value).toLocaleString(language.value === 'en' ? 'en-US' : 'zh-CN');
const percent = value => value != null && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '—';
const selected = computed(() => props.artifacts.find(stage => stage.artifact_id === selectedId.value && stage.available !== false));
watch(() => props.artifacts, () => {
  if (!selected.value) selectedId.value = props.artifacts.find(stage => stage.available !== false)?.artifact_id || '';
}, { immediate: true });
</script>

<style scoped>
.stage-picker { width: 100%; min-width: 0; border: 1px solid var(--line); border-radius: 6px; background: var(--bg-main); }
summary { cursor: pointer; padding: 11px 14px; font-size: 13px; font-weight: 700; }
.stage-count { display: inline-block; margin-left: 6px; color: var(--muted); font-variant-numeric: tabular-nums; }
.stage-picker-scroll { max-height: 272px; overflow: auto; border-top: 1px solid var(--line); }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; white-space: nowrap; }
th { position: sticky; top: 0; z-index: 1; background: var(--bg-card); color: var(--muted); font-weight: 400; text-align: left; }
th, td { padding: 10px 14px; border-bottom: 1px solid var(--line); }
th:last-child, td:last-child { text-align: right; }
td label { display: flex; align-items: center; gap: 9px; cursor: pointer; }
input[type=radio] { width: 16px; height: 16px; padding: 0; margin: 0; accent-color: var(--accent); }
tr.selected { background: var(--bg-card); color: var(--accent-ink); }
tr.unavailable { color: var(--muted); }
footer { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 14px; }
footer span { color: var(--muted); font-size: 12px; }
footer button { font-size: 13px; }
summary:focus-visible, .stage-picker-scroll:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
@media (max-width: 480px) { th, td { padding: 10px; } }
</style>
