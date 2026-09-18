<template>
  <div class="pip-probe">
    <button :disabled="busy" @click="toggle" :aria-pressed="active" :aria-label="lang === 'zh' ? '实验小窗' : 'Experimental PiP'" :title="lang === 'zh' ? '实验小窗' : 'Experimental PiP'">
      <PictureInPicture2 :size="17" />
    </button>
    <span v-if="error" role="status">{{ error }}</span>
    <details v-if="active"><summary>{{ lang === 'zh' ? '小窗诊断' : 'PiP diagnostics' }}</summary><pre>{{ report }}</pre></details>
    <video ref="video" class="pip-video" muted playsinline aria-hidden="true" tabindex="-1" />
  </div>
</template>
<script setup>
import { ref, onBeforeUnmount, watch } from 'vue';
import { PictureInPicture2 } from '@lucide/vue';
import { liveTileColors } from './tilePalette.js';
const props = defineProps({ run: Object, state: String, lang: String });
const emit = defineEmits(['active']);
const video = ref(null), active = ref(false), busy = ref(false), error = ref(''), report = ref('');
let canvas, ctx, stream, timer, startTimer, disposed = false, attempt = 0;
let previousTick = 0, previousSeq = null, previousRun = null;
let stats = {};
let canvasWidth = 320, canvasHeight = 320, boardSize = 320, boardOffsetX = 0, boardOffsetY = 0;

function configureCanvas() {
  const mobile = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
  canvasWidth = mobile ? 480 : 320;
  canvasHeight = mobile ? 270 : 320;
  boardSize = canvasHeight;
  boardOffsetX = (canvasWidth - boardSize) / 2;
  boardOffsetY = 0;
}

function draw() {
  if (!ctx || !stream) return;
  const values = props.run?.board || Array(16).fill(0);
  const unit = boardSize / 480;
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvasWidth, canvasHeight);
  ctx.fillStyle = '#1e293b'; ctx.fillRect(boardOffsetX, boardOffsetY, boardSize, boardSize);
  for (let i = 0; i < 16; i++) {
    const value = values[i], x = boardOffsetX + (12 + (i % 4) * 117) * unit, y = boardOffsetY + (12 + Math.floor(i / 4) * 117) * unit;
    const colors = liveTileColors(value);
    ctx.fillStyle = value ? colors.background : '#182333'; ctx.fillRect(x, y, 105 * unit, 105 * unit);
    if (!value) continue;
    ctx.fillStyle = colors.color;
    ctx.font = `bold ${(value >= 10000 ? 24 : value >= 1000 ? 32 : 40) * unit}px sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(value), x + 52.5 * unit, y + 54 * unit);
  }
  if (props.state !== 'live' || props.run?.ended_at) {
    ctx.fillStyle = 'rgba(0,0,0,.65)'; ctx.fillRect(boardOffsetX, boardOffsetY + 180 * unit, boardSize, 120 * unit);
    ctx.fillStyle = '#fff'; ctx.font = `bold ${24 * unit}px sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(props.run?.ended_at ? 'Game over' : props.state || 'Connecting', canvasWidth / 2, boardSize / 2);
  }
  stream.getVideoTracks()[0]?.requestFrame?.();
  stats.draws++;
  stats.seq = props.run?.seq || 0;
  stats.state = props.state;
  if (previousSeq !== stats.seq || previousRun !== props.run?.run_id) {
    stats.updates++;
    if (document.hidden) stats.hiddenUpdates++;
    stats.lastUpdate = new Date().toISOString();
    previousSeq = stats.seq; previousRun = props.run?.run_id;
  }
}
function sample() {
  const time = performance.now();
  if (previousTick && document.hidden) stats.maxHiddenTickGapMs = Math.max(stats.maxHiddenTickGapMs, Math.round(time - previousTick));
  previousTick = time;
  stats.hidden = document.hidden;
  stats.active = active.value;
  stats.videoFrames = video.value?.getVideoPlaybackQuality?.().totalVideoFrames ?? null;
  stats.videoTime = video.value?.currentTime || 0;
  report.value = JSON.stringify(stats, null, 2);
}
function release() {
  clearInterval(timer);
  clearTimeout(startTimer);
  active.value = false;
  sample();
  stream?.getTracks().forEach(track => track.stop());
  stream = null; canvas = null; ctx = null;
  if (video.value) { video.value.pause(); video.value.srcObject = null; }
  emit('active', false);
}
function entered() { active.value = true; emit('active', true); sample(); }
function left() { release(); }
async function toggle() {
  if (active.value) {
    try { await document.exitPictureInPicture(); }
    catch (e) { error.value = `${e.name}: ${e.message}`; }
    return;
  }
  error.value = '';
  if (!document.pictureInPictureEnabled || !video.value?.requestPictureInPicture || !HTMLCanvasElement.prototype.captureStream) {
    error.value = props.lang === 'zh' ? '此浏览器不支持视频小窗或 Canvas 视频流' : 'Video PiP or canvas capture is unavailable';
    return;
  }
  busy.value = true;
  const id = ++attempt;
  const player = video.value;
  try {
    stats = { userAgent: navigator.userAgent, started: new Date().toISOString(), draws: 0, updates: 0, hiddenUpdates: 0, maxHiddenTickGapMs: 0 };
    previousSeq = previousRun = null; previousTick = 0;
    configureCanvas();
    canvas = document.createElement('canvas'); canvas.width = canvasWidth; canvas.height = canvasHeight;
    ctx = canvas.getContext('2d');
    if (!ctx) throw Error('Canvas 2D unavailable');
    stream = canvas.captureStream(10);
    player.srcObject = stream;
    player.addEventListener('enterpictureinpicture', entered);
    player.addEventListener('leavepictureinpicture', left);
    draw();
    timer = setInterval(() => { draw(); sample(); }, 1000);
    await Promise.race([player.play(), new Promise((_, reject) => { startTimer = setTimeout(() => reject(Error('Video start timed out')), 8000); })]);
    clearTimeout(startTimer);
    if (disposed || id !== attempt) return;
    await player.requestPictureInPicture();
    if (disposed) { if (document.pictureInPictureElement === player) await document.exitPictureInPicture(); return; }
  } catch (e) {
    error.value = `${e.name}: ${e.message}`;
    release();
  } finally { busy.value = false; }
}
watch(() => [props.run?.seq, props.run?.run_id, props.state, props.run?.ended_at], draw);
onBeforeUnmount(() => {
  disposed = true; attempt++;
  if (document.pictureInPictureElement === video.value) document.exitPictureInPicture().catch(() => {});
  video.value?.removeEventListener('enterpictureinpicture', entered);
  video.value?.removeEventListener('leavepictureinpicture', left);
  release();
});
</script>
<style scoped>
.pip-probe { position:relative;flex:0 0 44px;font-size:12px; }
.pip-probe button { display:flex;align-items:center;justify-content:center;width:44px;height:44px;padding:0; }
.pip-probe button[aria-pressed="true"] { color:var(--accent-color, #00cfff); }
.pip-probe details, .pip-probe [role="status"] { position:absolute;right:0;top:100%;width:280px;z-index:50;background:var(--bg-panel, #1e293b);padding:8px;border:1px solid var(--border-main, #475569);border-radius:6px; }
.pip-probe summary { cursor:pointer;margin-top:6px; }
.pip-probe pre { white-space:pre-wrap;overflow-wrap:anywhere;max-height:200px;overflow:auto; }
.pip-video { position:fixed;left:-10px;top:0;width:1px;height:1px;pointer-events:none; }
</style>
