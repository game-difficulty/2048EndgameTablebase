<template>
  <section class="player-page">
    <header class="player-heading">
      <div><span class="player-kicker">2048 · PLAYER</span><h1>{{ profile?.player.display_name || username }}</h1></div>
      <a class="button-link" href="/#game" @click.prevent="$emit('back')">{{ t('返回棋盘') }}</a>
    </header>
    <nav class="player-tabs" :aria-label="t('个人主页导航')">
      <button v-for="item in tabs" :key="item.id" :class="{ active: tab === item.id }" @click="selectTab(item.id)">{{ t(item.label) }}</button>
    </nav>
    <p v-if="error" class="notice danger" role="alert">{{ t(error) }} <button @click="loadHistory(true)">{{ t('重试') }}</button></p>

    <template v-if="tab === 'profile'">
      <div class="player-bests">
        <button v-for="variant in variants" :key="variant" class="player-best" :class="{ active: bestVariant === variant }" @click="bestVariant = variant">
          <span>{{ variant.replace('x', ' × ') }} BEST</span><strong>{{ number(profile?.bests?.[variant] || 0) }}</strong>
        </button>
      </div>
      <div class="panel best-ten-panel">
        <div class="table-title"><div><span class="player-kicker">{{ bestVariant.replace('x', ' × ') }}</span><h2>BEST 10</h2></div><button :disabled="!bestTen.length" @click="downloadPoster">{{ t('下载分享图') }}</button></div>
        <div v-if="bestLoading" class="large-empty">{{ t('正在读取记录…') }}</div>
        <div v-else-if="!bestTen.length" class="large-empty">{{ t('此模式暂无可展示记录') }}</div>
        <div v-else class="best-poster-stage">
          <canvas ref="posterCanvas" class="best-poster-canvas" width="800" height="1350" role="img" :aria-label="t('Best 10 成绩海报')"></canvas>
          <button v-for="(entry,index) in bestTen" v-show="entry.has_replay" :key="entry.id" type="button" class="best-poster-hit" :style="posterHitStyle(index)" :aria-label="`${t('查看回放')} B${index + 1}，${number(entry.score)} ${t('分')}`" @click="$emit('replay',entry)"></button>
        </div>
      </div>
    </template>

    <div v-else-if="tab === 'history'" class="panel history-table">
      <div class="table-title"><h2>{{ t('历史记录') }}</h2><button @click="loadHistory(true)">{{ t('刷新') }}</button></div>
      <div class="history-filters">
        <label>{{ t('模式') }} <select v-model="filterVariant"><option value="all">{{ t('全部模式') }}</option><option v-for="variant in variants" :key="variant" :value="variant">{{ variant.replace('x',' × ') }}</option></select></label>
        <label>{{ t('排序') }} <select v-model="sort"><option value="newest">{{ t('时间：最新优先') }}</option><option value="oldest">{{ t('时间：最早优先') }}</option><option value="score_desc">{{ t('分数：从高到低') }}</option><option value="score_asc">{{ t('分数：从低到高') }}</option></select></label>
        <label>{{ t('每页') }} <select v-model.number="pageSize"><option :value="10">10</option><option :value="20">20</option><option :value="50">50</option></select> {{ t('局') }}</label>
      </div>
      <div v-if="loading" class="large-empty">{{ t('正在读取记录…') }}</div>
      <div v-else-if="!entries.length" class="large-empty">{{ t('暂无可展示记录') }}</div>
      <div v-for="item in entries" :key="item.id" class="profile-history-row">
        <span class="variant-tag">{{ item.variant.replace('x',' × ') }}</span>
        <div class="profile-history-main">
          <div><strong>{{ number(item.score) }}</strong><small>{{ date(item.ended_at) }} <span v-if="item.source === 'verse'">· 2048Verse</span><span v-else-if="item.source === 'manual'">· {{ t('补录') }}</span></small></div>
          <button type="button" class="history-preview-button" :aria-expanded="preview?.id === item.id" aria-controls="history-board-preview" @click="togglePreview(item,$event)">{{ t('终盘预览') }}</button>
        </div>
        <div class="profile-history-actions">
          <button v-if="item.has_replay" @click="$emit('replay',item)">{{ t('回放 ↗') }}</button>
          <button v-if="item.has_replay" @click="$emit('analyze',item)">{{ t(isOwner ? '分析' : '帮 TA 分析') }}</button>
          <label v-if="isOwner && item.source === 'verse' && !item.has_replay" class="text-button">{{ t('补充回放') }}<input type="file" accept=".vrs,.txt" hidden @change="attachReplay(item,$event)"></label>
          <button v-if="isOwner" type="button" class="history-delete-button" :title="t('删除记录')" :aria-label="`${t('删除记录')}：${number(item.score)} ${t('分')}`" @click="askDelete(item)"><Trash2 :size="16" aria-hidden="true" /></button>
        </div>
      </div>
      <nav v-if="historyTotal" class="history-pagination" :aria-label="t('历史记录分页')" :aria-busy="loading">
        <button type="button" class="history-page-nav" :disabled="currentPage <= 1 || loading" :aria-label="t('上一页')" @click="goToPage(currentPage - 1)">‹</button>
        <div class="history-pagination-center">
          <span class="history-page-summary">{{ t(`第 ${historyStart}–${historyEnd} 条，共 ${historyTotal} 条`) }}</span>
          <div class="history-page-numbers">
            <template v-for="item in paginationItems" :key="item.key">
              <span v-if="item.type === 'ellipsis'" class="history-page-ellipsis">…</span>
              <button v-else type="button" class="history-page-number" :class="{ active: item.page === currentPage }" :disabled="loading || item.page === currentPage" :aria-current="item.page === currentPage ? 'page' : undefined" @click="goToPage(item.page)">{{ item.page }}</button>
            </template>
          </div>
        </div>
        <button type="button" class="history-page-nav" :disabled="currentPage >= historyPageCount || loading" :aria-label="t('下一页')" @click="goToPage(currentPage + 1)">›</button>
      </nav>
    </div>

    <PlayerStatistics v-else-if="tab === 'statistics'" :username="username" />
    <PlayerSettings v-else-if="tab === 'settings' && isOwner" :play-settings="playSettings" @update:play-settings="$emit('update:play-settings',$event)" />
    <Teleport to="body">
      <div v-if="preview" id="history-board-preview" ref="previewElement" class="history-board-preview" :style="previewPosition" role="region" :aria-label="t('最终盘面')">
        <div class="history-board-preview-heading"><strong>{{ t('最终盘面') }}</strong><span>{{ t('盘面和') }} {{ number(previewBoardSum) }}</span><span>{{ preview.variant.replace('x',' × ') }}</span></div>
        <div class="history-preview-board" :style="{ '--preview-cols': previewDims[1], '--preview-rows': previewDims[0] }">
          <span v-for="(value,index) in preview.board" :key="index" class="history-preview-cell" :style="tileStyle(value)">
            <span v-if="value" class="history-preview-label" :style="previewTileLabelStyle(value)">{{ value }}</span>
          </span>
        </div>
      </div>
      <div v-if="deleteCandidate" class="modal-backdrop" @click.self="closeDeleteDialog" @keydown.esc="closeDeleteDialog">
        <section class="modal history-delete-dialog" role="dialog" aria-modal="true" :aria-label="t('删除这局记录？')">
          <button class="modal-close" type="button" :disabled="deletingId" :aria-label="t('关闭')" @click="closeDeleteDialog">×</button>
          <h2>{{ t('删除这局记录？') }}</h2>
          <p v-if="deleteError" class="notice danger" role="alert">{{ t(deleteError) }}</p>
          <p><strong>{{ deleteCandidate.variant.replace('x',' × ') }} · {{ number(deleteCandidate.score) }} {{ t('分') }}</strong><br>{{ date(deleteCandidate.ended_at) }}</p>
          <p>{{ t('删除后，这局将从历史记录、排行榜、PB、B10 和个人统计中移除。服务器仍会保留对局及审核资料。') }}</p>
          <div class="modal-actions"><button type="button" :disabled="deletingId" @click="closeDeleteDialog">{{ t('取消') }}</button><button type="button" class="history-delete-confirm" :disabled="deletingId" @click="deleteHistoryRun">{{ t(deletingId ? '删除中…' : '确认删除') }}</button></div>
        </section>
      </div>
    </Teleport>
  </section>
</template>

<script setup>
import { computed, defineAsyncComponent, nextTick, onActivated, onDeactivated, onMounted, onUnmounted, ref, watch } from 'vue';
import { Trash2 } from '@lucide/vue';
import { json, request } from './client.js';
import { t, language } from './i18n.js';
import { tileStyle } from './appearance.js';
import { getTileLabelStyle } from '../components/tileLabelStyle.js';
import PlayerSettings from './PlayerSettings.vue';
import { drawBestTenPoster, bestTenPreviewSize, POSTER_HEIGHT, POSTER_WIDTH, posterCardBounds } from './bestTenPoster.js';
import { isProfileOwner } from './profileOwnership.js';

const props = defineProps({ username: String, viewer: Object, playSettings: Object });
const PlayerStatistics = defineAsyncComponent(() => import('./PlayerStatistics.vue'));
defineEmits(['back','replay','analyze','update:play-settings']);
const variants = ['4x4','3x4','2x4','3x3'];
const profile = ref(null), entries = ref([]), bestTen = ref([]), bestMeta = ref(null), error = ref('');
const loading = ref(false), bestLoading = ref(false), tab = ref('profile');
const filterVariant = ref('4x4'), sort = ref('newest'), bestVariant = ref('4x4');
const currentPage = ref(1), pageSize = ref(20);
const posterCanvas = ref(null), posterDark = ref(document.documentElement.dataset.theme === 'dark');
const posterThemeVersion = ref(0);
const preview = ref(null), previewElement = ref(null), previewPosition = ref({});
const deleteCandidate = ref(null), deletingId = ref(null);
const deleteError = ref('');
let previewTrigger = null;
let historySerial = 0, bestSerial = 0, posterFrame = 0;
const historyCache = new Map(), bestCache = new Map();
function remember(cache, key, value, limit) {
  cache.delete(key); cache.set(key, value);
  while (cache.size > limit) cache.delete(cache.keys().next().value);
}
const isOwner = computed(() => isProfileOwner(profile.value, props.viewer));
const tabs = computed(() => [{id:'profile',label:'个人主页'},{id:'history',label:'历史记录'},
  {id:'statistics',label:'统计'},...(isOwner.value?[{id:'settings',label:'设置'}]:[])]);
const historyTotal = computed(() => Number(profile.value?.total || 0));
const historyPageCount = computed(() => Math.max(1, Number(profile.value?.page_count || 1)));
const historyStart = computed(() => historyTotal.value ? (currentPage.value - 1) * pageSize.value + 1 : 0);
const historyEnd = computed(() => Math.min(historyTotal.value, historyStart.value + entries.value.length - 1));
const paginationItems = computed(() => {
  const total = historyPageCount.value;
  const page = Math.max(1, Math.min(currentPage.value, total));
  if (total <= 7) return Array.from({length:total}, (_,index) => ({key:`page-${index + 1}`,type:'page',page:index + 1}));
  const pages = new Set([1,total,page - 1,page,page + 1]);
  if (page <= 4) [2,3,4,5].forEach(value => pages.add(value));
  if (page >= total - 3) [total - 4,total - 3,total - 2,total - 1].forEach(value => pages.add(value));
  const sorted = [...pages].filter(value => value >= 1 && value <= total).sort((a,b) => a - b);
  const items = [];
  sorted.forEach((value,index) => {
    const previous = sorted[index - 1];
    if (previous && value - previous > 1) items.push({key:`ellipsis-${previous}-${value}`,type:'ellipsis'});
    items.push({key:`page-${value}`,type:'page',page:value});
  });
  return items;
});
const previewDims = computed(() => ({'4x4':[4,4],'3x4':[3,4],'2x4':[2,4],'3x3':[3,3]})[preview.value?.variant] || [4,4]);
const previewBoardSum = computed(() => (preview.value?.board || []).reduce((sum, value) => sum + (Number(value) || 0), 0));
let themeObserver, posterResizeObserver, resourcesActive = false, posterGeneration = 0;
function previewSize(canvas) {
  return bestTenPreviewSize(canvas.parentElement.getBoundingClientRect().width, window.devicePixelRatio || 1);
}
function resizePosterIfNeeded() {
  const canvas = posterCanvas.value;
  if (!resourcesActive || tab.value !== 'profile' || !canvas) return;
  const size = previewSize(canvas);
  if (canvas.width !== size.width || canvas.height !== size.height) schedulePosterRender();
}
function observePosterSize() {
  posterResizeObserver?.disconnect();
  if (!resourcesActive || !posterCanvas.value) return;
  posterResizeObserver = new ResizeObserver(resizePosterIfNeeded);
  posterResizeObserver.observe(posterCanvas.value.parentElement);
}
function onProfileResize() { positionPreview(); resizePosterIfNeeded(); }
watch(posterCanvas, observePosterSize, { flush: 'post' });
function activateResources() {
  if (resourcesActive) return;
  resourcesActive = true;
  posterDark.value = document.documentElement.dataset.theme === 'dark';
  posterThemeVersion.value += 1;
  themeObserver = new MutationObserver(() => {
    posterDark.value = document.documentElement.dataset.theme === 'dark';
    posterThemeVersion.value += 1;
  });
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'style'] });
  document.addEventListener('pointerdown', outsidePreview);
  document.addEventListener('keydown', previewKeydown);
  document.addEventListener('scroll', closePreview, true);
  window.addEventListener('resize', onProfileResize);
  observePosterSize();
}
function deactivateResources({ releasePoster = false } = {}) {
  posterGeneration += 1;
  if (posterFrame) { cancelAnimationFrame(posterFrame); posterFrame = 0; }
  closePreview();
  if (resourcesActive) {
    resourcesActive = false;
    themeObserver?.disconnect(); themeObserver = null;
    posterResizeObserver?.disconnect(); posterResizeObserver = null;
    document.removeEventListener('pointerdown', outsidePreview);
    document.removeEventListener('keydown', previewKeydown);
    document.removeEventListener('scroll', closePreview, true);
    window.removeEventListener('resize', onProfileResize);
  }
  if (releasePoster && posterCanvas.value) {
    posterCanvas.value.width = 1; posterCanvas.value.height = 1;
  }
}
function currentTilePalette() {
  const style = getComputedStyle(document.documentElement);
  const palette = {};
  for (let power = 1; power <= 36; power++) {
    const value = 2 ** power;
    const background = style.getPropertyValue(`--color-tile-${value}`).trim();
    const color = style.getPropertyValue(`--color-text-${value}`).trim();
    if (background && color) palette[value] = [background, color];
  }
  return palette;
}
function posterHitStyle(index) {
  const box = posterCardBounds(index);
  return { left: `${100 * box.x / POSTER_WIDTH}%`, top: `${100 * box.y / POSTER_HEIGHT}%`,
    width: `${100 * box.width / POSTER_WIDTH}%`, height: `${100 * box.height / POSTER_HEIGHT}%` };
}
const number = value => new Intl.NumberFormat(language.value === 'zh' ? 'zh-CN' : 'en-US').format(value || 0);
const date = seconds => seconds ? new Date(seconds * 1000).toLocaleString(language.value === 'zh' ? 'zh-CN' : 'en-US') : '';
function previewTileLabelStyle(value) {
  const style = getTileLabelStyle({ value });
  const width = (248 - (previewDims.value[1] + 1) * 7) / previewDims.value[1] - 6;
  const cap = Math.floor(width / (String(value).length * 0.62));
  return { ...style, fontSize: `min(${style.fontSize}, ${cap}px)` };
}
function closePreview() { preview.value = null; previewTrigger = null; }
function positionPreview() {
  if (!previewTrigger?.isConnected || !previewElement.value) { closePreview(); return; }
  const anchor = previewTrigger.getBoundingClientRect();
  const panel = previewElement.value.getBoundingClientRect();
  const margin = 8, gap = 8;
  let left = anchor.right + gap;
  if (left + panel.width > window.innerWidth - margin) left = anchor.left - panel.width - gap;
  if (left < margin) left = Math.max(margin, Math.min(anchor.left, window.innerWidth - panel.width - margin));
  const top = Math.max(margin, Math.min(anchor.top, window.innerHeight - panel.height - margin));
  previewPosition.value = { left: `${left}px`, top: `${top}px` };
}
async function togglePreview(item, event) {
  if (preview.value?.id === item.id) { closePreview(); return; }
  previewTrigger = event.currentTarget;
  preview.value = item;
  await nextTick();
  if (preview.value?.id === item.id) positionPreview();
}
function outsidePreview(event) {
  if (preview.value && !previewElement.value?.contains(event.target) && !previewTrigger?.contains(event.target)) closePreview();
}
function previewKeydown(event) { if (event.key === 'Escape') closePreview(); }
onMounted(activateResources);
onActivated(() => { activateResources(); schedulePosterRender(); });
onDeactivated(() => deactivateResources({ releasePoster: true }));
onUnmounted(() => deactivateResources({ releasePoster: true }));
function selectTab(next) {
  closePreview();
  if (next === 'history' && tab.value !== 'history' && filterVariant.value !== bestVariant.value) {
    filterVariant.value = bestVariant.value;
  }
  tab.value = next;
  location.hash = next === 'profile' ? '' : next;
  if (next === 'profile') schedulePosterRender();
}
function applyHistory(result) {
  profile.value = result; entries.value = result.entries; currentPage.value = result.page;
  if (tab.value === 'settings' && !isOwner.value) selectTab('profile');
}
async function loadHistory(force = false) {
  const serial = ++historySerial;
  closePreview();
  const params = new URLSearchParams({variant:filterVariant.value,sort:sort.value,
    page:String(currentPage.value),page_size:String(pageSize.value)});
  const key = `${props.viewer?.id ?? 'guest'}:${props.username}?${params}`;
  if (!force && historyCache.has(key)) { applyHistory(historyCache.get(key)); loading.value = false; return; }
  loading.value = true; error.value = '';
  try {
    const result = await json(`/api/human/users/${encodeURIComponent(props.username)}/history?${params}`);
    if (serial !== historySerial) return;
    remember(historyCache, key, result, 12); applyHistory(result);
  } catch { if (serial === historySerial) error.value = '无法读取玩家记录，请稍后重试。'; }
  finally { if (serial === historySerial) loading.value = false; }
}
function goToPage(page) {
  const next = Math.max(1, Math.min(Number(page) || 1, historyPageCount.value));
  if (next === currentPage.value || loading.value) return;
  currentPage.value = next;
  loadHistory();
}
function applyBestTen(result) { bestMeta.value = result; bestTen.value = result.entries; }
async function loadBestTen(force = false) {
  const serial = ++bestSerial;
  const key = `${props.viewer?.id ?? 'guest'}:${props.username}:${bestVariant.value}`;
  if (!force && bestCache.has(key)) { applyBestTen(bestCache.get(key)); bestLoading.value = false; return; }
  bestLoading.value = true;
  try {
    const result = await json(`/api/human/users/${encodeURIComponent(props.username)}/best10?variant=${bestVariant.value}`);
    if (serial === bestSerial) { remember(bestCache, key, result, 8); applyBestTen(result); }
  } catch { if (serial === bestSerial) { bestMeta.value = null; bestTen.value = []; } }
  finally { if (serial === bestSerial) bestLoading.value = false; }
}
async function renderPoster(canvas = null, outputWidth, outputHeight) {
  const previewRender = !canvas;
  const generation = posterGeneration;
  await nextTick();
  canvas ||= posterCanvas.value;
  const shouldRender = () => !previewRender || (resourcesActive && tab.value === 'profile'
    && generation === posterGeneration && canvas === posterCanvas.value);
  if (!shouldRender()) return;
  if (!canvas || !bestTen.value.length) return;
  if (previewRender) {
    const size = previewSize(canvas);
    outputWidth = size.width; outputHeight = size.height;
  }
  await drawBestTenPoster({ canvas, name: profile.value?.player.display_name,
    userId: profile.value?.player.id, variant: bestVariant.value, entries: bestTen.value,
    pbScore: bestMeta.value?.pb_score, pbRank: bestMeta.value?.pb_rank,
    rating: bestMeta.value?.rating, raRank: bestMeta.value?.ra_rank, dark: posterDark.value,
    language: language.value, tilePalette: currentTilePalette(), outputWidth, outputHeight, shouldRender });
}
function schedulePosterRender() {
  posterGeneration += 1;
  if (!resourcesActive || posterFrame || tab.value !== 'profile') return;
  posterFrame = requestAnimationFrame(() => { posterFrame = 0; void renderPoster(); });
}
watch([bestTen, bestVariant, () => profile.value?.player, posterDark, posterThemeVersion, language], schedulePosterRender, { flush: 'post' });
watch(tab, value => { if (value === 'profile') schedulePosterRender(); }, { flush: 'post' });
async function downloadPoster() {
  if (!bestTen.value.length) return;
  if (posterFrame) { cancelAnimationFrame(posterFrame); posterFrame = 0; }
  const canvas = document.createElement('canvas');
  try {
    await renderPoster(canvas, 1600, 2700);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    if (!blob) { error.value = '无法生成分享图。'; return; }
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a'); link.download = `2048-${bestVariant.value}-best10.png`; link.href = url;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  } finally {
    canvas.width = 1; canvas.height = 1;
  }
}
async function attachReplay(item,event) {
  const file=event.target.files?.[0];event.target.value='';
  if (!file) return;
  if (file.size > 2 * 1024 * 1024) { error.value='回放文件不能超过 2 MB。'; return; }
  try {
    await request(`/api/human/verse-replays/`+encodeURIComponent(item.id),
      {method:'POST',body:await file.arrayBuffer(),binary:true,timeoutMs:30000});
    historyCache.clear(); bestCache.clear();
    await Promise.all([loadHistory(true),loadBestTen(true)]);
  } catch(e) { error.value=e.code==='replay_result_mismatch'?'回放终盘或分数与继承记录不一致。':'回放补充失败，请检查文件后重试。'; }
}
function askDelete(item) { closePreview(); deleteError.value = ''; deleteCandidate.value = item; }
function closeDeleteDialog() { if (!deletingId.value) deleteCandidate.value = null; }
async function deleteHistoryRun() {
  const item = deleteCandidate.value;
  if (!item || deletingId.value) return;
  deletingId.value = item.id; deleteError.value = '';
  try {
    await json(`/api/human/runs/${encodeURIComponent(item.id)}/history`, { method:'DELETE' });
    deleteCandidate.value = null; historyCache.clear(); bestCache.clear();
    await Promise.all([loadHistory(true), loadBestTen(true)]);
  } catch { deleteError.value = '删除记录失败，请稍后重试。'; }
  finally { deletingId.value = null; }
}
watch(() => props.username, () => {closePreview();historyCache.clear();bestCache.clear();profile.value=null;entries.value=[];currentPage.value=1;loadHistory();loadBestTen();}, {immediate:true});
watch(() => props.viewer?.id ?? null, (next, previous) => {
  if (next === previous) return;
  historyCache.clear(); bestCache.clear();
  loadHistory(true); loadBestTen(true);
});
watch([filterVariant,sort,pageSize], () => { currentPage.value=1; loadHistory(); });
watch(bestVariant, loadBestTen);
watch(isOwner, owner => { if (!owner && tab.value === 'settings') selectTab('profile'); });
if (['history','statistics','settings'].includes(location.hash.slice(1))) tab.value=location.hash.slice(1);
</script>
