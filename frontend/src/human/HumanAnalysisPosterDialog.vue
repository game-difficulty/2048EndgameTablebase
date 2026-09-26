<template>
  <Teleport to="body">
    <div class="modal-backdrop analysis-poster-backdrop" @click.self="$emit('close')">
      <section ref="dialogElement" class="modal analysis-poster-dialog" role="dialog" aria-modal="true"
        :aria-label="label('分析展示图', 'Analysis result card')" tabindex="-1"
        @keydown.esc.stop="$emit('close')" @keydown.left.prevent="go(-1)" @keydown.right.prevent="go(1)">
        <button class="modal-close" :aria-label="label('关闭', 'Close')" @click="$emit('close')">×</button>
        <div class="analysis-poster-heading">
          <div><h2>{{ label('分析展示图', 'Analysis result card') }}</h2>
            <p class="small muted">{{ currentEntry?.pattern }}-{{ currentEntry?.target }} · {{ currentIndex + 1 }} / {{ entries.length }}</p></div>
          <div class="analysis-poster-controls">
            <button :disabled="currentIndex <= 0" :aria-label="label('上一张', 'Previous card')" @click="go(-1)">‹</button>
            <button :disabled="currentIndex >= entries.length - 1" :aria-label="label('下一张', 'Next card')" @click="go(1)">›</button>
            <button class="primary" :disabled="!summary || drawing || downloading" @click="download">
              {{ downloading ? label('正在生成…', 'Preparing…') : label('下载 PNG', 'Download PNG') }}
            </button>
          </div>
        </div>
        <p v-if="error" class="notice danger" role="alert">{{ error }} <button @click="loadSummary">{{ label('重试', 'Retry') }}</button></p>
        <p v-if="loading" class="large-empty" role="status">{{ label('正在读取分析摘要…', 'Loading analysis summary…') }}</p>
        <div v-show="!loading && !error" class="analysis-poster-preview">
          <canvas ref="canvas" width="2360" height="1640" role="img" :aria-label="label('本局分析展示图预览', 'Analysis result card preview')"></canvas>
        </div>
        <p class="small muted analysis-poster-note">{{ label('图像在浏览器生成；若分析资料不足，等级位显示“待评级”。',
          'The image is generated in your browser; the grade shows Unrated when analysis data is insufficient.') }}</p>
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue';
import { drawAnalysisPoster } from './analysisPoster.js';
import { json } from './client.js';
import { language } from './i18n.js';

const props = defineProps({
  runId: { type: String, required: true },
  entries: { type: Array, required: true },
  initialSummaryId: { type: Number, required: true },
  player: { type: Object, default: null },
});
defineEmits(['close']);
const label = (zh, en) => language.value === 'en' ? en : zh;
const dialogElement = ref(null), canvas = ref(null);
const currentId = ref(props.initialSummaryId), summary = ref(null);
const loading = ref(false), drawing = ref(false), downloading = ref(false), error = ref('');
const currentIndex = computed(() => Math.max(0, props.entries.findIndex(item => Number(item.id) === Number(currentId.value))));
const currentEntry = computed(() => props.entries[currentIndex.value]);
const cachedSummaries = new Map();
let requestSerial = 0, renderSerial = 0;

const loadImage = src => new Promise(resolve => {
  if (!src) { resolve(null); return; }
  const image = new Image();
  image.crossOrigin = 'anonymous';
  image.onload = () => resolve(image);
  image.onerror = () => resolve(null);
  image.src = src;
});
let qrAssets;
async function getQrAssets() {
  if (!qrAssets) qrAssets = Promise.all([
    loadImage('/human/play-qr-art.webp?v=20260926-3'),
    loadImage('/human/play-qr-backdrop.webp?v=20260926-3'),
  ]);
  const assets = await qrAssets;
  if (assets.some(asset => !asset)) { qrAssets = null; throw new Error('poster_assets_unavailable'); }
  return assets;
}
let avatarPromise;
function tilePalette() {
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
function go(delta) {
  const next = currentIndex.value + delta;
  if (next >= 0 && next < props.entries.length) currentId.value = Number(props.entries[next].id);
}
async function renderPoster() {
  const data = summary.value, target = canvas.value;
  if (!data || !target) return;
  const serial = ++renderSerial;
  drawing.value = true;
  try {
    const [qrImage, qrBackdropImage] = await getQrAssets();
    const subject = data.subject || props.player;
    if (!avatarPromise) avatarPromise = loadImage(subject?.avatar_url || subject?.profile?.avatar_url);
    const avatarImage = await avatarPromise;
    if (serial !== renderSerial || summary.value !== data) return;
    await drawAnalysisPoster({ canvas: target,
      data: { name: subject?.display_name || 'Player', pattern: data.pattern,
        target: data.target, goalTile: data.goal_tile, run: data.run,
        aggregate: data.aggregate, grade: data.grade || null },
      qrImage, qrBackdropImage, avatarImage, tilePalette: tilePalette(), language: language.value });
  } catch (cause) {
    console.error('analysis_poster_render_failed', cause);
    if (serial === renderSerial) error.value = label('展示图生成失败，请重试。', 'Could not generate the result card. Please retry.');
  } finally { if (serial === renderSerial) drawing.value = false; }
}
async function loadSummary() {
  const id = Number(currentId.value);
  const serial = ++requestSerial;
  summary.value = null; error.value = ''; loading.value = true;
  try {
    let data = cachedSummaries.get(id);
    if (!data) {
      data = await json(`/api/human/analysis/summaries/${id}`);
      if (data.run_id !== props.runId || !data.aggregate?.poster_eligible) throw new Error('poster_unavailable');
      cachedSummaries.set(id, data);
    }
    if (serial !== requestSerial) return;
    summary.value = data;
    await nextTick();
    await renderPoster();
  } catch {
    if (serial === requestSerial) error.value = label('无法读取这项分析展示图。', 'This analysis result card is unavailable.');
  } finally { if (serial === requestSerial) loading.value = false; }
}
async function download() {
  if (!summary.value || downloading.value || drawing.value) return;
  downloading.value = true; error.value = '';
  try {
    await renderPoster();
    if (error.value) return;
    const target = canvas.value;
    if (!target) throw new Error('poster_canvas_missing');
    const blob = await new Promise(resolve => target.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('poster_export_failed');
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const name = `${summary.value.run.variant}-${summary.value.pattern}-${summary.value.target}`.replace(/[^a-zA-Z0-9_-]/g, '-');
    link.download = `2048-${name}-${props.runId.slice(-8)}-analysis.png`;
    link.href = url; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  } catch { error.value = label('下载 PNG 失败，请重试。', 'Could not download the PNG. Please retry.'); }
  finally { downloading.value = false; }
}
watch(currentId, loadSummary, { immediate: true });
watch(language, () => { if (summary.value) void renderPoster(); });
onMounted(() => nextTick(() => dialogElement.value?.focus()));
onUnmounted(() => { requestSerial += 1; renderSerial += 1; });
</script>
