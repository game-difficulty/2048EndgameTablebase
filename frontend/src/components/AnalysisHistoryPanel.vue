<template>
  <section class="analysis-history-panel" tabindex="0" :aria-label="text('分析历史', 'Analysis history')">
    <div ref="list" class="analysis-history-list" :aria-busy="loading">
    <p v-if="error" class="analysis-history-message analysis-history-error" role="alert">{{ error }}</p>
    <p v-else-if="loading && !jobs.length" class="analysis-history-message">{{ text('正在加载…', 'Loading…') }}</p>
    <p v-else-if="!jobs.length" class="analysis-history-message">{{ text('暂无分析记录', 'No analysis history yet') }}</p>

    <article v-for="job in jobs" :key="job.job_id" class="analysis-history-job">
      <button type="button" class="analysis-history-job-head" :aria-expanded="Boolean(job.open)" @click="toggle(job)">
        <span class="analysis-history-job-heading">
          <strong>{{ originLabel(job.origin) }}</strong>
          <small>{{ formatTime(job.created_at) }}</small>
        </span>
        <span class="analysis-history-job-summary">
          <span>{{ job.done }}/{{ job.total }}</span>
          <span>{{ statusLabel(job.status) }}</span>
          <ChevronDown :size="17" :class="{ 'analysis-history-chevron-open': job.open }" aria-hidden="true" />
        </span>
      </button>

      <div v-if="job.open" class="analysis-history-items">
        <p v-if="job.detailLoading" class="analysis-history-message">{{ text('正在读取阶段…', 'Loading stages…') }}</p>
        <section v-for="item in job.items || []" :key="item.id" class="analysis-history-item">
          <div class="analysis-history-item-title">
            <strong>{{ itemLabel(item) }}</strong>
            <span>{{ modeLabel(item.variant) }} · {{ item.pattern }}-{{ goalTargetLabel(item.target, language) }}</span>
          </div>
          <AnalysisStageList v-if="item.artifacts?.length" :artifacts="item.artifacts" :language="language" :opening="opening" @open="openReplay" />
          <small v-else class="analysis-history-empty">{{ item.status === 'failed' ? text('分析失败', 'Analysis failed') : text('没有可跳转的回放阶段', 'No replay stage is available') }}</small>
        </section>
      </div>
    </article>

    </div>
    <nav v-if="pageIndex > 0 || nextCursor" class="analysis-history-pagination" :aria-label="text('分析历史分页', 'History pagination')">
      <button type="button" :disabled="loading || pageIndex === 0" @click="loadPage(pageIndex - 1)">{{ text('上一页', 'Previous') }}</button>
      <span aria-live="polite">{{ text(`第 ${pageIndex + 1} 页`, `Page ${pageIndex + 1}`) }}</span>
      <button type="button" :disabled="loading || !nextCursor" @click="loadPage(pageIndex + 1)">{{ text('下一页', 'Next') }}</button>
    </nav>
  </section>
</template>

<script setup>
import { goalTargetLabel } from '../utils/goalTarget.js';
import { computed, nextTick, onMounted, ref } from 'vue';
import { ChevronDown } from '@lucide/vue';
import AnalysisStageList from '../features/replay/components/AnalysisStageList.vue';
import { analysisScoreLabel } from '../features/replay/analysisPresentation.js';
import { openAsyncLink } from '../services/openAsyncLink.js';
import { authHeaders } from '../services/auth/sessionTokenStore.js';
import { getBackendUrl } from '../services/runtime/backendUrl.js';

const props = defineProps({ language: { type: String, default: 'zh' } });
const pages = ref([]);
const pageIndex = ref(0);
const jobs = computed(() => pages.value[pageIndex.value]?.items || []);
const nextCursor = computed(() => pages.value[pageIndex.value]?.next_cursor || '');
const list = ref(null);
const loading = ref(false);
const error = ref('');
const opening = ref('');
const text = (zh, en) => String(props.language).startsWith('en') ? en : zh;
const formatTime = value => new Date(Number(value) * 1000).toLocaleString(String(props.language).startsWith('en') ? 'en-US' : 'zh-CN');
const originLabel = value => value === 'human_archive' ? text('对局站归档', 'Play archive') : text('回放上传', 'Replay upload');
const statusLabel = value => ({ queued: text('等待', 'Queued'), running: text('分析中', 'Running'), finished: text('完成', 'Finished'), partial: text('部分完成', 'Partial'), failed: text('失败', 'Failed') })[value] || value;
const modeLabel = variant => variant ? String(variant).replace('x', '×') : text('未知模式', 'Unknown mode');
const itemLabel = item => analysisScoreLabel(item.score, props.language) || text('局分未记录', 'Score unavailable');

async function api(path, init = {}) {
  const response = await fetch(getBackendUrl(path), { credentials: 'include', ...init, headers: authHeaders({ Accept: 'application/json', ...(init.headers || {}) }) });
  if (!response.ok) throw new Error(`${response.status}`);
  return response.json();
}

async function loadPage(index = 0, reset = false) {
  if (loading.value) return;
  if (index < 0 || (!reset && index > 0 && !pages.value[index - 1]?.next_cursor)) return;
  loading.value = true;
  error.value = '';
  try {
    if (reset || !pages.value[index]) {
      const cursor = index === 0 ? '' : pages.value[index - 1].next_cursor;
      const data = await api(`/api/analysis/history?limit=10${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''}`);
      if (reset) pages.value = [];
      pages.value[index] = data;
    }
    pageIndex.value = index;
    await nextTick();
    if (list.value) list.value.scrollTop = 0;
  } catch {
    error.value = text('分析历史读取失败，请稍后重试。', 'Could not load analysis history.');
  } finally {
    loading.value = false;
  }
}

async function toggle(job) {
  job.open = !job.open;
  if (!job.open || job.items) return;
  job.detailLoading = true;
  try {
    Object.assign(job, await api(`/api/analysis/history/${encodeURIComponent(job.job_id)}`));
  } catch {
    error.value = text('分析详情读取失败。', 'Could not load analysis details.');
  } finally {
    job.detailLoading = false;
  }
}

async function openReplay(artifact) {
  opening.value = artifact.artifact_id;
  try {
    await openAsyncLink(async () => (await api(`/api/analysis/replays/${encodeURIComponent(artifact.artifact_id)}/open-link`, { method: 'POST' })).url);
  } catch {
    error.value = text('回放暂时无法打开。', 'The replay cannot be opened right now.');
  } finally {
    opening.value = '';
  }
}

defineExpose({ refresh: () => loadPage(0, true), loading });
onMounted(() => loadPage());
</script>

<style scoped>
.analysis-history-panel {
  display: flex;
  flex: 1 1 auto;
  min-height: 0;
  height: 100%;
  flex-direction: column;
  gap: 10px;
  overflow: hidden;
  color: var(--text-main);
}
.analysis-history-list {
  display: flex;
  flex: 1 1 auto;
  min-height: 0;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
  overscroll-behavior: contain;
  padding: 2px 8px 2px 2px;
  color: var(--text-main);
  scrollbar-color: var(--border-main) transparent;
  scrollbar-width: thin;
}
.analysis-history-list::-webkit-scrollbar { width: 8px; }
.analysis-history-list::-webkit-scrollbar-thumb { border: 2px solid var(--bg-card); border-radius: 8px; background: var(--border-main); }
.analysis-history-job { flex: 0 0 auto; overflow: hidden; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-main); }
.analysis-history-job-head { display: flex; width: 100%; align-items: center; justify-content: space-between; gap: 12px; border: 0; background: transparent; padding: 12px 14px; color: inherit; text-align: left; cursor: pointer; }
.analysis-history-job-head:hover { background: var(--bg-card); }
.analysis-history-job-heading { display: flex; min-width: 0; flex-direction: column; gap: 3px; }
.analysis-history-job-heading small, .analysis-history-item-title span { color: var(--text-secondary); }
.analysis-history-job-summary { display: flex; flex: 0 0 auto; align-items: center; gap: 10px; font-size: 0.82rem; }
.analysis-history-job-summary svg { transition: transform 0.2s ease; }
.analysis-history-chevron-open { transform: rotate(180deg); }
.analysis-history-items { display: flex; flex-direction: column; gap: 14px; border-top: 1px solid var(--border-main); padding: 12px 14px 14px; }
.analysis-history-items, .analysis-history-item { flex: 0 0 auto; }
.analysis-history-item { min-width: 0; }
.analysis-history-item-title { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 4px 12px; margin-bottom: 8px; }
.analysis-history-item-title strong { font-size: 1rem; }
.analysis-history-item-title span { font-size: 0.82rem; }
.analysis-history-message { margin: 6px 0; color: var(--text-secondary); }
.analysis-history-error { color: #d14c45; }
.analysis-history-empty { color: var(--text-secondary); }
.analysis-history-pagination { display: flex; flex: 0 0 auto; align-items: center; justify-content: center; gap: 14px; border-top: 1px solid var(--border-main); padding: 10px 0 2px; font-size: 0.85rem; }
.analysis-history-pagination button { border: 1px solid var(--border-main); border-radius: 6px; background: var(--bg-main); padding: 8px 16px; color: inherit; cursor: pointer; }
.analysis-history-pagination button:hover:not(:disabled) { background: var(--bg-card); }
.analysis-history-pagination button:disabled { opacity: 0.5; cursor: default; }
@media (max-width: 560px) {
  .analysis-history-job-head { padding: 10px; }
  .analysis-history-items { padding: 10px; }
  .analysis-history-job-summary { gap: 6px; }
}
</style>
