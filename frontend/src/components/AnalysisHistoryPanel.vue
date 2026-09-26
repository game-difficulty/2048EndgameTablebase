<template>
  <section class="analysis-history-panel">
    <div class="analysis-history-toolbar">
      <strong>{{ text('分析历史', 'Analysis history') }}</strong>
      <button type="button" :disabled="loading" @click="load(true)">{{ text('刷新', 'Refresh') }}</button>
    </div>
    <p v-if="error" class="analysis-history-error">{{ error }}</p>
    <p v-else-if="loading && !jobs.length" class="analysis-history-empty">{{ text('正在加载…', 'Loading…') }}</p>
    <p v-else-if="!jobs.length" class="analysis-history-empty">{{ text('暂无分析记录', 'No analysis history yet') }}</p>
    <div v-for="job in jobs" :key="job.job_id" class="analysis-history-job">
      <button type="button" class="analysis-history-job-head" @click="toggle(job)">
        <span><b>{{ originLabel(job.origin) }}</b><small>{{ formatTime(job.created_at) }}</small></span>
        <span>{{ job.done }}/{{ job.total }} · {{ statusLabel(job.status) }}　{{ job.open ? '−' : '+' }}</span>
      </button>
      <div v-if="job.open" class="analysis-history-items">
        <p v-if="job.detailLoading" class="analysis-history-empty">{{ text('正在读取阶段…', 'Loading stages…') }}</p>
        <template v-for="item in job.items || []" :key="item.id">
          <div class="analysis-history-item-title">
            <span>{{ item.pattern }} · {{ item.target }}</span><small>{{ item.source_filename }}</small>
          </div>
          <div v-if="item.artifacts?.length" class="analysis-history-stages">
            <button v-for="artifact in item.artifacts" :key="artifact.artifact_id" type="button"
              :disabled="!artifact.available || opening === artifact.artifact_id" @click="openReplay(artifact)">
              {{ text('阶段', 'Stage') }} {{ artifact.segment_index + 1 }}
              <small>{{ artifact.source_start_index + 1 }}–{{ artifact.source_end_index }}</small>
            </button>
          </div>
          <small v-else class="analysis-history-empty">{{ text('没有可跳转的回放阶段', 'No replay stage is available') }}</small>
        </template>
      </div>
    </div>
    <button v-if="nextCursor" type="button" class="analysis-history-more" :disabled="loading" @click="load(false)">
      {{ text('下一页', 'Next page') }}
    </button>
  </section>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { authHeaders } from '../services/auth/sessionTokenStore.js';
import { getBackendUrl } from '../services/runtime/backendUrl.js';

const props = defineProps({ language: { type: String, default: 'zh' } });
const jobs = ref([]), nextCursor = ref(''), loading = ref(false), error = ref(''), opening = ref('');
const text = (zh, en) => String(props.language).startsWith('en') ? en : zh;
const formatTime = value => new Date(Number(value) * 1000).toLocaleString(String(props.language).startsWith('en') ? 'en-US' : 'zh-CN');
const originLabel = value => value === 'human_archive' ? text('对局站归档', 'Play archive') : text('主站上传', 'Main-site upload');
const statusLabel = value => ({ queued: text('等待', 'Queued'), running: text('分析中', 'Running'), finished: text('完成', 'Finished'), partial: text('部分完成', 'Partial'), failed: text('失败', 'Failed') })[value] || value;
async function api(path, init = {}) {
  const response = await fetch(getBackendUrl(path), { credentials: 'include', ...init, headers: authHeaders({ Accept: 'application/json', ...(init.headers || {}) }) });
  if (!response.ok) throw new Error(`${response.status}`);
  return response.json();
}
async function load(reset) {
  if (loading.value) return;
  loading.value = true; error.value = '';
  try {
    const cursor = reset ? '' : nextCursor.value;
    const data = await api(`/api/analysis/history?limit=10${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''}`);
    jobs.value = reset ? data.items : [...jobs.value, ...data.items];
    nextCursor.value = data.next_cursor || '';
  } catch { error.value = text('分析历史读取失败，请稍后重试。', 'Could not load analysis history.'); }
  finally { loading.value = false; }
}
async function toggle(job) {
  job.open = !job.open;
  if (!job.open || job.items) return;
  job.detailLoading = true;
  try { Object.assign(job, await api(`/api/analysis/history/${encodeURIComponent(job.job_id)}`)); }
  catch { error.value = text('分析详情读取失败。', 'Could not load analysis details.'); }
  finally { job.detailLoading = false; }
}
async function openReplay(artifact) {
  opening.value = artifact.artifact_id;
  try {
    const data = await api(`/api/analysis/replays/${encodeURIComponent(artifact.artifact_id)}/open-link`, { method: 'POST' });
    window.open(data.url, '_blank', 'noopener');
  } catch { error.value = text('回放暂时无法打开。', 'The replay cannot be opened right now.'); }
  finally { opening.value = ''; }
}
onMounted(() => load(true));
</script>

<style scoped>
.analysis-history-panel{display:flex;flex-direction:column;gap:10px;max-height:min(66vh,620px);overflow:auto;padding:2px;color:var(--text-main,#463f3a)}
.analysis-history-toolbar,.analysis-history-job-head,.analysis-history-item-title{display:flex;align-items:center;justify-content:space-between;gap:12px}
button{border:1px solid var(--border-main,#c9beb1);border-radius:9px;background:var(--bg-card,#fffaf2);color:inherit;padding:8px 11px;font:inherit;cursor:pointer}button:disabled{opacity:.45;cursor:default}
.analysis-history-job{overflow:hidden;border:1px solid var(--border-main,#c9beb1);border-radius:13px;background:var(--bg-main,#f6efe5)}
.analysis-history-job-head{width:100%;border:0;border-radius:0;text-align:left}.analysis-history-job-head span:first-child{display:flex;min-width:0;flex-direction:column}.analysis-history-job-head small,.analysis-history-item-title small{color:var(--text-secondary,#7d726a)}
.analysis-history-items{display:flex;flex-direction:column;gap:8px;padding:10px}.analysis-history-item-title{padding-top:4px}.analysis-history-item-title small{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.analysis-history-stages{display:flex;flex-wrap:wrap;gap:7px}.analysis-history-stages button{display:flex;flex-direction:column;min-width:92px}.analysis-history-empty{margin:6px 0;color:var(--text-secondary,#7d726a)}.analysis-history-error{color:#b43a32}.analysis-history-more{align-self:center}
</style>
