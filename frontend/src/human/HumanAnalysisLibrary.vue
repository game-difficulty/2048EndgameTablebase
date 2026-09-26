<template>
  <section class="analysis-library-page">
    <header class="section-top">
      <div><span class="player-kicker">2048 · ANALYSIS</span><h1>{{ t('分析库') }}</h1></div>
      <a class="button-link" href="/#game" @click.prevent="$emit('back')">{{ t('返回棋盘') }}</a>
    </header>
    <form class="panel analysis-library-filters" @submit.prevent="search">
      <label>{{ t('玩家') }}<input v-model.trim="filters.username" maxlength="64" :placeholder="t('用户名')"></label>
      <label>{{ t('模式') }}<select v-model="filters.variant"><option value="">{{ t('全部模式') }}</option><option v-for="item in variants" :key="item" :value="item">{{ item.replace('x',' × ') }}</option></select></label>
      <label>{{ t('定式') }}<input v-model.trim="filters.pattern" maxlength="80" :placeholder="t('全部定式')"></label>
      <label>{{ t('目标') }}<input v-model.trim="filters.target" maxlength="12" :placeholder="t('全部目标')"></label>
      <label>{{ t('来源') }}<select v-model="filters.source"><option value="">{{ t('全部来源') }}</option><option value="native">{{ t('本站') }}</option><option value="verse">2048Verse</option><option value="manual">{{ t('补录') }}</option></select></label>
      <button class="primary" type="submit">{{ t('筛选') }}</button>
    </form>

    <p v-if="error" class="notice danger" role="alert">{{ error }} <button @click="load(true)">{{ t('重试') }}</button></p>
    <div v-if="loading && !items.length" class="panel large-empty">{{ t('正在读取分析…') }}</div>
    <div v-else-if="!items.length" class="panel large-empty">{{ t('暂无可展示分析') }}</div>
    <div v-else class="analysis-library-list" :aria-busy="loading">
      <article v-for="item in items" :key="item.id" class="panel analysis-library-card">
        <button class="analysis-library-player" @click="$emit('player',item.subject.display_name)">
          <img v-if="item.subject.avatar_url" :src="item.subject.avatar_url" alt="">
          <span v-else class="analysis-avatar-fallback">{{ item.subject.display_name.slice(0,1) }}</span>
          <strong>{{ item.subject.display_name }}</strong>
        </button>
        <div class="analysis-library-result">
          <span class="variant-tag">{{ item.variant.replace('x',' × ') }}</span>
          <strong>{{ number(item.score) }}</strong>
          <small>{{ date(item.run_ended_at) }} · {{ sourceLabel(item.source) }}</small>
        </div>
        <div class="analysis-library-metrics">
          <span><small>{{ t('定式') }}</small><strong>{{ item.pattern }} · {{ item.target }}</strong></span>
          <span><small>{{ t('评价') }}</small><strong>{{ item.grade || '—' }}</strong></span>
          <span><small>{{ t('吻合度') }}</small><strong>{{ percent(item.mean_goodness_of_fit) }}</strong></span>
          <span><small>{{ t('残局数') }}</small><strong>{{ item.stage_count || 0 }}</strong></span>
        </div>
        <div class="analysis-library-actions">
          <button @click="toggleDetail(item)">{{ detail?.id === item.id ? t('收起') : t('查看分析') }}</button>
          <button @click="$emit('replay',item)">{{ t('查看原局') }}</button>
          <button :disabled="!viewer" :title="viewer ? '' : t('登录后可以帮助分析')" @click="$emit('analyze',item)">{{ t('帮 TA 分析') }}</button>
        </div>
        <div v-if="detail?.id === item.id" class="analysis-library-detail">
          <p v-if="detailLoading">{{ t('正在读取分析…') }}</p>
          <template v-else>
            <button v-for="artifact in detail.artifacts || []" :key="artifact.artifact_id" :disabled="!artifact.available" @click="openArtifact(artifact.artifact_id)">
              {{ t('回放阶段') }} {{ artifact.segment_index + 1 }} · {{ number(artifact.source_start_index) }}–{{ number(artifact.source_end_index) }} ↗
            </button>
            <span v-if="!(detail.artifacts || []).some(item => item.available)" class="muted">{{ t('阶段回放已过期') }}</span>
          </template>
        </div>
      </article>
    </div>
    <nav v-if="items.length" class="analysis-library-pagination" :aria-label="t('分析库分页')">
      <button :disabled="page === 0 || loading" @click="previous">‹ {{ t('上一页') }}</button>
      <span>{{ t(`第 ${page + 1} 页`) }}</span>
      <button :disabled="!nextCursor || loading" @click="next">{{ t('下一页') }} ›</button>
    </nav>
  </section>
</template>

<script setup>
import { reactive, ref } from 'vue';
import { json } from './client.js';
import { language, t } from './i18n.js';

defineProps({ viewer: Object });
defineEmits(['back','player','replay','analyze']);
const variants = ['4x4','3x4','3x3','2x4'];
const filters = reactive({ username:'', variant:'', pattern:'', target:'', source:'' });
const items = ref([]), loading = ref(false), error = ref(''), nextCursor = ref('');
const cursors = ref(['']);
const page = ref(0);
const detail = ref(null), detailLoading = ref(false);
const number = value => new Intl.NumberFormat(language.value === 'en' ? 'en-US' : 'zh-CN').format(Number(value) || 0);
const date = value => value ? new Date(value * 1000).toLocaleString(language.value === 'en' ? 'en-US' : 'zh-CN') : '—';
const percent = value => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '—';
const sourceLabel = value => value === 'verse' ? '2048Verse' : value === 'manual' ? t('补录') : t('本站');
async function load() {
  loading.value = true; error.value = ''; detail.value = null;
  const params = new URLSearchParams({ limit:'20' });
  for (const [key,value] of Object.entries(filters)) if (value) params.set(key,value);
  if (cursors.value[page.value]) params.set('cursor',cursors.value[page.value]);
  try {
    const data = await json(`/api/analysis/library?${params}`);
    items.value = data.items || []; nextCursor.value = data.next_cursor || '';
  } catch { error.value = t('无法读取分析库，请稍后重试。'); }
  finally { loading.value = false; }
}
function search() { page.value = 0; cursors.value = ['']; load(); }
function next() { if (!nextCursor.value) return; cursors.value[page.value + 1] = nextCursor.value; page.value += 1; load(); }
function previous() { if (!page.value) return; page.value -= 1; load(); }
async function toggleDetail(item) {
  if (detail.value?.id === item.id) { detail.value = null; return; }
  detail.value = { id:item.id, artifacts:[] }; detailLoading.value = true;
  try { detail.value = await json(`/api/analysis/library/${item.id}`); }
  catch { error.value = t('无法读取分析详情，请稍后重试。'); detail.value = null; }
  finally { detailLoading.value = false; }
}
async function openArtifact(id) {
  try {
    const data = await json(`/api/analysis/replays/${encodeURIComponent(id)}/open-link`, { method:'POST' });
    window.open(data.url, '_blank', 'noopener');
  } catch { error.value = t('回放暂时无法打开。'); }
}
load();
</script>
