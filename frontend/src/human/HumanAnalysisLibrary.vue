<template>
  <section class="analysis-library-page">
    <header class="section-top">
      <div><span class="player-kicker">2048 · ANALYSIS</span><h1>{{ t('分析库') }}</h1></div>
      <a class="button-link" href="/#game" @click.prevent="$emit('back')">{{ t('返回棋盘') }}</a>
    </header>
    <form class="panel analysis-library-filters" @submit.prevent="search">
      <label>{{ t('玩家') }}<input v-model.trim="filters.username" maxlength="64" :placeholder="t('用户名')"></label>
      <label>{{ t('模式') }}<select v-model="filters.variant"><option v-for="item in variants" :key="item" :value="item">{{ item.replace('x',' × ') }}</option></select></label>
      <label>{{ t('定式') }}<input v-model.trim="filters.pattern" maxlength="80" :placeholder="t('全部定式')"></label>
      <label>{{ t('目标') }}<input v-model.trim="filters.target" maxlength="12" :placeholder="t('全部目标')"></label>
      <label>{{ t('来源') }}<select v-model="filters.source"><option value="">{{ t('全部来源') }}</option><option value="native">{{ t('本站') }}</option><option value="verse">2048Verse</option><option value="manual">{{ t('补录') }}</option></select></label>
      <button class="primary" type="submit">{{ t('筛选') }}</button>
    </form>

    <p v-if="error" class="notice danger" role="alert">{{ error }} <button @click="load">{{ t('重试') }}</button></p>
    <div v-if="loading && !items.length" class="panel large-empty">{{ t('正在读取分析…') }}</div>
    <div v-else-if="!items.length" class="panel large-empty">{{ t('暂无可展示分析') }}</div>
    <div v-else class="analysis-library-list" :aria-busy="loading">
      <p class="analysis-library-scroll-hint">{{ language === 'en' ? 'Swipe sideways to see all columns' : '左右滑动查看完整列表' }} ↔</p>
      <div class="analysis-library-scroll" tabindex="0" :aria-label="t('分析库')">
        <table class="analysis-library-table">
          <thead><tr>
            <th scope="col">{{ t('玩家') }}</th><th scope="col">{{ t('对局') }}</th>
            <th scope="col">{{ t('定式') }} / {{ t('目标') }}</th><th scope="col">{{ t('评价') }}</th>
            <th scope="col" class="numeric">{{ t('吻合度') }}</th><th scope="col" class="numeric">{{ t('残局数') }}</th>
            <th scope="col"><span class="visually-hidden">{{ t('查看分析') }}</span></th>
          </tr></thead>
          <tbody>
            <template v-for="item in items" :key="item.id">
              <tr :class="{ 'analysis-library-active': detail?.id === item.id }">
                <td><button class="analysis-library-player" :title="item.subject.display_name" @click="$emit('player', item.subject.display_name)">
                  <img v-if="item.subject.avatar_url" :src="item.subject.avatar_url" alt="" loading="lazy" @error="item.subject.avatar_url = null">
                  <span v-else class="analysis-avatar-fallback">{{ item.subject.display_name.slice(0,1) }}</span>
                  <strong>{{ item.subject.display_name }}</strong>
                </button></td>
                <td><div class="analysis-library-result">
                  <button class="analysis-score-link" :title="t('查看原局')" @click="$emit('replay', item.run_id)">{{ number(item.score) }}</button>
                  <small>{{ date(item.run_ended_at) }}</small>
                </div></td>
                <td class="analysis-formation"><strong>{{ item.pattern }}</strong><span>{{ item.target }}</span></td>
                <td><span class="analysis-grade">{{ item.grade || '—' }}</span></td>
                <td class="numeric analysis-fit">{{ percent(item.mean_goodness_of_fit) }}</td>
                <td class="numeric">{{ number(item.stage_count) }}</td>
                <td class="analysis-library-actions"><button :aria-expanded="detail?.id === item.id" @click="toggleDetail(item)">{{ detail?.id === item.id ? t('收起') : t('查看分析') }}</button></td>
              </tr>
              <tr v-if="detail?.id === item.id" class="analysis-library-detail-row"><td colspan="7">
                <p v-if="detailLoading" role="status">{{ t('正在读取分析…') }}</p>
                <AnalysisStagePicker v-else :artifacts="detail.artifacts || []" :busy="opening" expanded @open="openArtifact" />
              </td></tr>
            </template>
          </tbody>
        </table>
      </div>
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
import AnalysisStagePicker from './AnalysisStagePicker.vue';

defineEmits(['back','player','replay']);
const variants = ['4x4','3x4','3x3','2x4'];
const filters = reactive({ username:'', variant:'4x4', pattern:'', target:'', source:'' });
const items = ref([]), loading = ref(false), error = ref(''), nextCursor = ref('');
const cursors = ref(['']);
const page = ref(0);
const detail = ref(null), detailLoading = ref(false), opening = ref(false);
let detailRequest = 0, loadRequest = 0;
const number = value => new Intl.NumberFormat(language.value === 'en' ? 'en-US' : 'zh-CN').format(Number(value) || 0);
const date = value => value ? new Date(value * 1000).toLocaleString(language.value === 'en' ? 'en-US' : 'zh-CN') : '—';
const percent = value => value != null && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '—';
async function load() {
  const request = ++loadRequest;
  ++detailRequest;
  loading.value = true; error.value = ''; detail.value = null;
  const params = new URLSearchParams({ limit:'20' });
  for (const [key,value] of Object.entries(filters)) if (value) params.set(key,value);
  if (cursors.value[page.value]) params.set('cursor',cursors.value[page.value]);
  try {
    const data = await json(`/api/analysis/library?${params}`);
    if (request !== loadRequest) return;
    items.value = data.items || []; nextCursor.value = data.next_cursor || '';
  } catch { if (request === loadRequest) error.value = t('无法读取分析库，请稍后重试。'); }
  finally { if (request === loadRequest) loading.value = false; }
}
function search() { page.value = 0; cursors.value = ['']; load(); }
function next() { if (!nextCursor.value) return; cursors.value[page.value + 1] = nextCursor.value; page.value += 1; load(); }
function previous() { if (!page.value) return; page.value -= 1; load(); }
async function toggleDetail(item) {
  const request = ++detailRequest;
  if (detail.value?.id === item.id) { detail.value = null; return; }
  detail.value = { id:item.id, artifacts:[] }; detailLoading.value = true;
  try {
    const data = await json(`/api/analysis/library/${item.id}`);
    if (request === detailRequest) detail.value = data;
  } catch { if (request === detailRequest) { error.value = t('无法读取分析详情，请稍后重试。'); detail.value = null; } }
  finally { if (request === detailRequest) detailLoading.value = false; }
}
async function openArtifact(id) {
  if (opening.value) return;
  opening.value = true;
  try {
    const data = await json(`/api/analysis/replays/${encodeURIComponent(id)}/open-link`, { method:'POST' });
    window.open(data.url, '_blank', 'noopener');
  } catch { error.value = t('回放暂时无法打开。'); }
  finally { opening.value = false; }
}
load();
</script>
