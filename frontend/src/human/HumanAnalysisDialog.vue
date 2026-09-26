<template>
  <div v-show="!posterSummaryId" class="modal-backdrop" @click.self="$emit('close')">
    <section class="modal human-analysis-dialog" role="dialog" aria-modal="true" :aria-label="label('对局分析', 'Game analysis')" @keydown.esc="$emit('close')">
      <button class="modal-close" :aria-label="label('关闭', 'Close')" @click="$emit('close')">×</button>
      <h2>{{ label('对局分析', 'Game analysis') }}</h2>
      <p v-if="loading" role="status">{{ label('正在读取可用定式…', 'Loading available formations…') }}</p>
      <div v-else-if="error && !options" class="error-text" role="alert">{{ error }} <button @click="load">{{ label('重试', 'Retry') }}</button></div>
      <template v-else-if="options">
        <p v-if="error" class="error-text" role="alert">{{ error }}</p>
        <p class="small muted">{{ options.run.variant }} · {{ options.run.score.toLocaleString() }} {{ label('分', 'points') }} · {{ options.run.moves.toLocaleString() }} {{ label('步', 'moves') }} · {{ endedAt }}</p>
        <template v-if="!job">
          <div v-if="summaries.length" class="analysis-summaries">
            <h3>{{ label('已完成的定式分析', 'Completed formation analyses') }}</h3>
            <div v-for="item in summaries" :key="item.id" class="analysis-result-row">
              <span>{{ item.pattern }}-{{ item.target }}<small class="muted">{{ label('残局数', 'Endgames') }} {{ item.aggregate?.stage_count || 0 }}</small></span>
              <button v-if="item.aggregate?.poster_eligible" @click="showPoster(item.id)">{{ label('生成展示图', 'Make result card') }}</button>
              <small v-else class="muted">{{ posterUnavailable(item.aggregate) }}</small>
            </div>
          </div>
          <p v-if="summaryError" class="small error-text">{{ summaryError }} <button @click="loadSummaries(generation)">{{ label('重试', 'Retry') }}</button></p>
          <p>{{ label('为这一局选择最多 6 组定式与目标。每组单独计费。', 'Choose up to 6 formation and target pairs. Each pair is charged separately.') }}</p>
          <div class="analysis-selection-tools">
            <button type="button" :disabled="!lastSelection.length || busy" @click="importLastSelection">
              {{ label('导入上次选择', 'Use last selection') }}<span v-if="lastSelection.length">（{{ lastSelection.length }}）</span>
            </button>
            <span v-if="!lastSelection.length">{{ label('此变体暂无上次选择', 'No saved selection for this variant') }}</span>
          </div>
          <div v-if="!patterns.length" class="notice">{{ label('当前棋盘暂无可用分析表库。', 'No analysis tablebase is available for this board.') }}</div>
          <div v-else class="analysis-picker">
            <label>{{ label('分类', 'Category') }}
              <select v-model="patternCategory"><option v-for="group in patternGroups" :key="group.category" :value="group.category">{{ categoryLabel(group.category) }}</option></select>
            </label>
            <label>{{ label('定式', 'Formation') }}
              <select v-model="pattern"><option v-for="name in activePatterns" :key="name" :value="name">{{ name }}</option></select>
            </label>
            <label>{{ label('目标', 'Target') }}
              <select v-model="target"><option v-for="value in targets" :key="value" :value="value">{{ value }}</option></select>
            </label>
            <button :disabled="!target || selected.length >= 6 || selected.some(item => item.pattern === pattern && item.target === target)" @click="addItem">{{ label('加入', 'Add') }}</button>
          </div>
          <div v-if="selected.length" class="analysis-selected">
            <div v-for="(item, index) in selected" :key="`${item.pattern}-${item.target}`">
              <span>{{ index + 1 }}. {{ item.pattern }} · {{ item.target }}</span>
              <button :aria-label="label('移除', 'Remove')" @click="removeItem(index)">×</button>
            </div>
          </div>
          <div class="modal-actions">
            <button :disabled="!selected.length || busy" @click="getQuote">{{ label('查看费用', 'Check cost') }}</button>
          </div>
          <div v-if="quote" class="analysis-quote">
            <strong>{{ label('合计', 'Total') }} {{ quote.total_cost }} {{ label('代币', 'tokens') }}</strong>
            <span>{{ label('余额', 'Balance') }} {{ formatTokenBalance(quote.token_balance?.total) }}</span>
            <button class="primary" :disabled="busy" @click="submit">{{ label('确认并开始分析', 'Confirm and analyze') }}</button>
          </div>
        </template>
        <template v-else>
          <p role="status">{{ statusLabel }} · {{ job.completed }} / {{ job.total }}</p>
          <progress :value="job.completed" :max="job.total" class="analysis-progress"></progress>
          <p v-if="job.current_file" class="small muted">{{ job.current_file }}</p>
          <div v-for="(entry, index) in job.items || []" :key="index" class="analysis-result-row">
            <span>{{ entry.pattern }} · {{ entry.target }}<small v-if="entry.message" class="error-text">{{ serverErrorText(entry.message, language) }}</small></span>
            <span class="analysis-result-actions"><span>{{ { queued: label('等待中', 'Queued'), running: label('分析中', 'Running'), done: label('完成', 'Done'), failed: label('失败', 'Failed') }[entry.status] }}</span>
              <button v-if="entry.status === 'done' && entry.poster_eligible && entry.summary_id" @click="showPoster(entry.summary_id)">{{ label('生成展示图', 'Make result card') }}</button>
              <small v-else-if="entry.status === 'done'" class="muted">{{ label('暂不可出图', 'No result card') }}</small></span>
          </div>
          <p v-if="job.message" class="error-text">{{ serverErrorText(job.message, language) }}</p>
          <div class="modal-actions">
            <a v-if="job.download_url" class="button-link primary" :href="job.download_url">{{ label('下载分析结果', 'Download results') }}</a>
            <button v-if="job.status === 'finished' || job.status === 'failed'" @click="reset">{{ label('重新选择', 'New analysis') }}</button>
          </div>
        </template>
      </template>
    </section>
  </div>
  <HumanAnalysisPosterDialog v-if="posterSummaryId" :run-id="runId" :entries="posterEntries"
    :initial-summary-id="posterSummaryId" :player="player" @close="posterSummaryId = null" />
</template>

<script setup>
import { serverErrorText } from '../services/errors/serverErrorText.js';
import { computed, onUnmounted, ref, watch } from 'vue';
import { groupTablebasePatternsByCategory } from '../services/tablebases/catalogClient.js';
import { json } from './client.js';
import { language } from './i18n.js';
import { loadLastAnalysisSelection, saveLastAnalysisSelection } from './analysisSelectionHistory.js';
import HumanAnalysisPosterDialog from './HumanAnalysisPosterDialog.vue';

const props = defineProps({ runId: { type: String, required: true }, player: { type: Object, default: null } });
defineEmits(['close']);
const options = ref(null), loading = ref(false), busy = ref(false), error = ref('');
const patternCategory = ref(''), pattern = ref(''), target = ref(''), selected = ref([]), lastSelection = ref([]);
const quote = ref(null), job = ref(null), requestId = ref('');
const summaries = ref([]), summaryError = ref(''), posterSummaryId = ref(null);
const label = (zh, en) => language.value === 'en' ? en : zh;
const formatTokenBalance = value => {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString(language.value === 'en' ? 'en-US' : 'zh-CN', {
    minimumFractionDigits: 1, maximumFractionDigits: 1,
  }) : '—';
};
const patternGroups = computed(() => Object.entries(groupTablebasePatternsByCategory(options.value?.tables || []))
  .map(([category, items]) => ({ category, items })));
const patterns = computed(() => patternGroups.value.flatMap(group => group.items));
const activePatterns = computed(() => patternGroups.value.find(group => group.category === patternCategory.value)?.items || []);
const targets = computed(() => (options.value?.tables || []).filter(item => item.pattern === pattern.value).map(item => item.target).sort((a, b) => Number(a) - Number(b)));
const statusLabel = computed(() => ({ queued: label('等待中', 'Queued'), running: label('分析中', 'Running'), finished: label('已完成', 'Finished'), failed: label('失败', 'Failed') })[job.value?.status] || '');
const endedAt = computed(() => options.value?.run.ended_at ? new Date(options.value.run.ended_at * 1000).toLocaleString(language.value === 'en' ? 'en-US' : 'zh-CN') : '');
const posterEntries = computed(() => {
  const items = summaries.value.filter(item => item.aggregate?.poster_eligible).map(item => ({ id: item.id, pattern: item.pattern, target: item.target }));
  const seen = new Set(items.map(item => Number(item.id)));
  for (const entry of job.value?.items || []) {
    if (entry.poster_eligible && entry.summary_id && !seen.has(Number(entry.summary_id))) {
      items.push({ id: entry.summary_id, pattern: entry.pattern, target: entry.target });
      seen.add(Number(entry.summary_id));
    }
  }
  return items;
});
const categoryLabel = category => ({
  free: label('Free', 'Free'), space10: label('10 格', '10 cells'), space12: label('12 格', '12 cells'),
  others: label('其他', 'Other'), variant: label('Variant', 'Variant'),
})[category] || category;
let timer, generation = 0, jobRequestInFlight = false, terminalSummariesLoadedJob = '';
function posterUnavailable(aggregate) {
  if (!aggregate?.stage_count) return label('无满足条件的残局', 'No qualifying endgame');
  return label('此定式尚不支持展示图', 'This formation is not yet supported');
}
function failure(e) {
  if (e.status === 402) return label('代币不足。', 'Insufficient tokens.');
  if (e.status === 429) return label('分析队列已满，请稍后重试。', 'The analysis queue is full. Please try later.');
  if (e.code === 'analysis_price_changed' || e.code === 'analysis_catalog_changed') return label('表库或费用已变化，请重新查看。', 'The tablebase or cost changed. Please check again.');
  if (e.status === 404) return label('此对局无法分析。', 'This game is unavailable for analysis.');
  return serverErrorText(e, language.value, label('分析请求失败。', 'Analysis request failed.'));
}
async function refreshJob(id, currentGeneration) {
  if (jobRequestInFlight) return;
  jobRequestInFlight = true;
  try {
    const data = await json(`/api/analysis/jobs/${encodeURIComponent(id)}`);
    if (currentGeneration !== generation) return;
    job.value = data;
    if (['finished', 'failed'].includes(data.status)) {
      clearInterval(timer);
      if (terminalSummariesLoadedJob !== id) {
        terminalSummariesLoadedJob = id;
        void loadSummaries(currentGeneration);
      }
    }
  } catch (e) { if (currentGeneration === generation) {
    error.value = failure(e); clearInterval(timer);
    if (e.status === 404) { sessionStorage.removeItem(`human-analysis:${props.runId}`); job.value = null; }
  } } finally { jobRequestInFlight = false; }
}
async function loadSummaries(currentGeneration = generation) {
  summaryError.value = '';
  try {
    const result = await json(`/api/human/runs/${encodeURIComponent(props.runId)}/analysis/summaries`);
    if (currentGeneration === generation) summaries.value = result.items || [];
  } catch {
    if (currentGeneration === generation) summaryError.value = label('已完成的分析结果暂时无法读取。', 'Could not load completed analyses.');
  }
}
function showPoster(id) {
  if (posterEntries.value.some(item => Number(item.id) === Number(id))) posterSummaryId.value = Number(id);
}
function poll(id) {
  clearInterval(timer);
  const currentGeneration = generation;
  refreshJob(id, currentGeneration);
  timer = setInterval(() => refreshJob(id, currentGeneration), 2500);
}
async function load() {
  generation += 1; clearInterval(timer); loading.value = true; error.value = ''; quote.value = null;
  terminalSummariesLoadedJob = '';
  options.value = null; selected.value = []; job.value = null; summaries.value = []; summaryError.value = ''; posterSummaryId.value = null;
  const currentGeneration = generation;
  try {
    const data = await json(`/api/human/runs/${encodeURIComponent(props.runId)}/analysis/options`);
    if (currentGeneration !== generation) return;
    options.value = data;
    void loadSummaries(currentGeneration);
    patternCategory.value = patternGroups.value[0]?.category || '';
    pattern.value = activePatterns.value[0] || '';
    target.value = targets.value[0] || '';
    lastSelection.value = loadLastAnalysisSelection(data.run.variant, data.tables);
    const stored = sessionStorage.getItem(`human-analysis:${props.runId}`);
    if (stored) { job.value = { job_id: stored, status: 'queued', completed: 0, total: 1 }; poll(stored); }
    else {
      try {
        const pending = JSON.parse(sessionStorage.getItem(`human-analysis-pending:${props.runId}`) || 'null');
        if (pending?.requestId && Array.isArray(pending.items) && pending.quote) {
          selected.value = pending.items; quote.value = pending.quote; requestId.value = pending.requestId;
        }
      } catch { sessionStorage.removeItem(`human-analysis-pending:${props.runId}`); }
    }
  } catch (e) { if (currentGeneration === generation) error.value = failure(e); }
  finally { if (currentGeneration === generation) loading.value = false; }
}
watch(() => props.runId, load, { immediate: true });
watch(patternCategory, () => {
  if (!activePatterns.value.includes(pattern.value)) pattern.value = activePatterns.value[0] || '';
});
watch(pattern, () => { if (!targets.value.includes(target.value)) target.value = targets.value[0] || ''; });
function invalidateQuote() { quote.value = null; requestId.value = ''; sessionStorage.removeItem(`human-analysis-pending:${props.runId}`); }
function addItem() { selected.value = [...selected.value, { pattern: pattern.value, target: target.value }]; invalidateQuote(); }
function removeItem(index) { selected.value = selected.value.filter((_, i) => i !== index); invalidateQuote(); }
function setPicker(item) {
  const group = patternGroups.value.find(entry => entry.items.includes(item.pattern));
  if (group) patternCategory.value = group.category;
  pattern.value = item.pattern; target.value = item.target;
}
function importLastSelection() {
  if (!lastSelection.value.length) return;
  selected.value = lastSelection.value.map(item => ({ ...item }));
  setPicker(selected.value[0]); invalidateQuote();
}
async function getQuote() {
  busy.value = true; error.value = '';
  try { quote.value = await json(`/api/human/runs/${encodeURIComponent(props.runId)}/analysis/quote`, { method: 'POST', body: { items: selected.value } }); requestId.value = crypto.randomUUID(); }
  catch (e) { error.value = failure(e); }
  finally { busy.value = false; }
}
async function submit() {
  if (!quote.value) return;
  busy.value = true; error.value = '';
  try {
    sessionStorage.setItem(`human-analysis-pending:${props.runId}`, JSON.stringify({ requestId: requestId.value, items: selected.value, quote: quote.value }));
    const data = await json(`/api/human/runs/${encodeURIComponent(props.runId)}/analysis/jobs`, { method: 'POST', body: {
      items: selected.value, expected_cost_units: quote.value.total_cost_units,
      expected_catalog_version: quote.value.catalog_version, request_id: requestId.value,
    } });
    lastSelection.value = saveLastAnalysisSelection(options.value.run.variant, selected.value, options.value.tables);
    terminalSummariesLoadedJob = '';
    job.value = data; sessionStorage.setItem(`human-analysis:${props.runId}`, data.job_id);
    sessionStorage.removeItem(`human-analysis-pending:${props.runId}`); poll(data.job_id);
  } catch (e) { error.value = failure(e); if (e.status === 409) invalidateQuote(); }
  finally { busy.value = false; }
}
function reset() { sessionStorage.removeItem(`human-analysis:${props.runId}`); job.value = null; selected.value = []; invalidateQuote(); error.value = ''; }
onUnmounted(() => { generation += 1; clearInterval(timer); });
</script>
