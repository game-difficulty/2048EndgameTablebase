<template>
  <section class="multi-content" aria-label="Three AI livestream">
    <header class="content-title"><div><h1>{{ room.title[lang] || room.title.en }}</h1><p>{{ room.description[lang] || room.description.en }}</p></div>
      <span class="room-best">{{ t('历史最高', 'ALL-TIME BEST') }} <b>{{ bestScore.toLocaleString() }}</b></span>
    </header>
    <div class="view-controls">
      <small v-if="state.batch" class="batch-status">{{ batchCaption }}</small>
      <div class="segmented" role="group" :aria-label="t('直播布局', 'Stream layout')">
        <button :aria-pressed="view.layout === 'focus'" @click="setLayout('focus')">{{ t('主屏＋两小屏', 'Main + two previews') }}</button>
        <button :aria-pressed="view.layout === 'equal'" @click="setLayout('equal')">{{ t('三局等大', 'Three equal boards') }}</button>
      </div>
      <div v-if="view.layout === 'focus'" class="segmented" role="group" :aria-label="t('主屏模式', 'Main screen mode')">
        <button :aria-pressed="view.mainMode === 'auto'" :title="t('其他存活 AI 领先至少 1000 分，或当前 AI 已结束时自动切换', 'Switch when another live AI leads by at least 1000 points, or the current AI finishes')" @click="setMode('auto')">{{ t('总是显示最高分局', 'Always show the leader') }}</button>
        <button :aria-pressed="view.mainMode === 'manual'" @click="setMode('manual')">{{ t('自行选择', 'Choose manually') }}</button>
      </div>
    </div>
    <div class="multi-grid" :class="view.layout" :data-layout="view.layout" :data-main-lane="view.selectedLane">
      <AiRunBoard v-for="slot in state.slots" :key="slot.lane" :slot="slot" :frame="state.frames[slot.lane]"
        :class="placement(slot.lane)" :lang="lang" :now="now" :stream-state="streamState"
        :leader="leaders.includes(slot.lane)" :primary="view.layout === 'focus' && slot.lane === view.selectedLane"
        :compact="view.layout === 'focus' && slot.lane !== view.selectedLane"
        :source="state.sources[slot.run?.source_id] || 'AI'" @select="select" @notice="emit('notice', $event)" />
      <LiveRunHistory ref="history" class="history" :lang="lang" :horizontal="view.layout === 'equal'" @notice="emit('notice', $event)" />
    </div>
  </section>
</template>
<script setup>
import { shouldRefreshLiveClock } from '../displayClock.js';
import { computed, ref, shallowRef, onMounted, onUnmounted } from 'vue';
import { useRoom } from '../roomContext.js';
import { boardPipFrame } from './boardPipFrame.js';
import AiRunBoard from './AiRunBoard.vue';
import LiveRunHistory from './LiveRunHistory.vue';
import { emptyMultiState, receiveMultiBatch, receiveMultiJson, syncMultiFrames } from './multiLiveState.js';
import { leaderLanes, followMain, selectMain, changeLayout, changeMainMode } from './mainRunSelection.js';
import { LiveBoardPlayback } from './liveBoardPlayback.js';
const props = defineProps({ lang: String, streamState: String, pipActive: Boolean });
const emit = defineEmits(['notice']);
const { room } = useRoom();
const t = (zh, en) => props.lang === 'zh' ? zh : en;
const state = shallowRef(emptyMultiState()), history = ref(null), best = ref(0), now = ref(Date.now());
const view = shallowRef({ layout: 'focus', mainMode: 'auto', selectedLane: 0 });
const leaders = computed(() => leaderLanes(state.value.slots));
const bestScore = computed(() => Math.max(best.value, ...state.value.slots.map(s => s.run?.score || 0)));
let serverOffset = 0;
const batchCaption = computed(() => {
  const batch = state.value.batch;
  if (!batch) return '';
  if (batch.phase === 'cooldown' || batch.phase === 'void') return `${t(batch.phase === 'void' ? '本批作废' : '本批结束', batch.phase === 'void' ? 'Batch void' : 'Batch complete')} · ${Math.max(0, Math.ceil((batch.next_at || 0) - now.value / 1000))}s`;
  if (batch.phase === 'settling') return t('正在结算', 'Settling');
  return batch.transition ? t('过渡批 · 下一批开放下注', 'Transition · Betting starts next batch') : t('同批对局', 'Synchronized batch');
});
let pending = state.value, playbackTimer = null, timer;
const playback = new LiveBoardPlayback(pending.frames);
function placement(lane) {
  if (view.value.layout === 'equal') return '';
  if (lane === view.value.selectedLane) return 'main-board';
  return lane === state.value.slots.find(s => s.lane !== view.value.selectedLane).lane ? 'preview-one' : 'preview-two';
}
function render(time = performance.now(), force = false) {
  if (playbackTimer !== null) clearTimeout(playbackTimer);
  playbackTimer = null;
  if (!force && document.hidden) return;
  const nextView = followMain(view.value, pending.slots);
  view.value = nextView;
  if (force) {
    pending = syncMultiFrames(pending);
    playback.sync(pending.frames);
  }
  const frames = playback.paint(time);
  state.value = { ...pending, frames };
  // Drain a packet's remaining steps even when no further network message arrives.
  if (playback.pending) playbackTimer = setTimeout(() => render(), playback.nextDelay(performance.now()));
}
function schedule() {
  if (document.hidden) {
    view.value = followMain(view.value, pending.slots);
    pending = syncMultiFrames(pending);
    playback.sync(pending.frames);
    state.value = pending;
    return;
  } // PiP follows current logical state without queuing move animations.
  if (playbackTimer === null) playbackTimer = setTimeout(() => render(), 0);
}
function receive(data) {
  if (data instanceof ArrayBuffer) {
    const transitions = [];
    pending = receiveMultiBatch(pending, data, transitions);
    playback.enqueue(transitions, performance.now());
  }
  else {
    if (data.server_time) serverOffset = data.server_time * 1000 - Date.now();
    const firstSnapshot = data.type === 'snapshot' && !pending.epoch;
    pending = receiveMultiJson(pending, data);
    if (firstSnapshot) view.value = followMain(view.value, pending.slots, true);
    if (data.type === 'snapshot') playback.sync(pending.frames);
    else if (data.type === 'lane_start') playback.sync(pending.frames, [data.slot.lane]);
    if (data.history !== undefined) history.value?.receive(data);
    if (data.best != null) best.value = data.best;
  }
  schedule();
}
function resume() {
  if (playbackTimer !== null) clearTimeout(playbackTimer);
  pending = syncMultiFrames(pending); render(performance.now(), true);
}
function select(lane) { view.value = selectMain(view.value, lane); render(); }
function setLayout(layout) { view.value = changeLayout(view.value, layout, pending.slots); render(); }
function setMode(mode) { view.value = changeMainMode(view.value, mode, pending.slots); render(); }
function getPipFrame() {
  const selection = followMain(view.value, pending.slots);
  return boardPipFrame({slots:pending.slots,selectedLane:selection.selectedLane,layout:selection.layout,
    state:props.streamState,lang:props.lang,title:room.title[props.lang] || room.title.en,now:Date.now()+serverOffset});
}
defineExpose({ receive, resume, getPipFrame });
onMounted(() => { timer = setInterval(() => { if (shouldRefreshLiveClock(props.pipActive)) now.value = Date.now()+serverOffset; }, 500); });
onUnmounted(() => { clearInterval(timer); if (playbackTimer !== null) clearTimeout(playbackTimer); });
</script>
<style scoped>
.multi-content { width:100%;height:100%;min-width:0;min-height:0;box-sizing:border-box;display:flex;flex-direction:column;padding:12px;overflow:hidden; }
.content-title { display:flex;justify-content:space-between;gap:20px;align-items:center; }.content-title h1 { margin:0;font-size:28px; }.content-title p { margin:8px 0 0;color:var(--text-secondary);font-size:13px; }
.room-best { display:flex;flex-direction:column;text-align:right;font-size:11px;color:var(--text-secondary);gap:5px; }.room-best b { font-size:22px;color:var(--text-main); }
.view-controls { display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:14px 0;flex-shrink:0; }.segmented { display:flex;gap:3px;border:1px solid var(--border-main);padding:4px;border-radius:12px; }
.batch-status { color:var(--text-secondary);font-size:11px; }
.segmented button { font-size:12px;padding:8px 12px; }.segmented button[aria-pressed=true] { background:var(--accent);color:#071524;border-color:var(--accent); }
.multi-grid { display:grid;gap:12px;align-items:stretch;flex:1;min-height:0;overflow:hidden; }
.focus { grid-template-columns:240px minmax(0,1fr) 260px;grid-template-rows:repeat(2,minmax(0,1fr)); }
.focus .main-board { grid-column:2;grid-row:1 / 3; }.focus .preview-one { grid-column:1;grid-row:1; }.focus .preview-two { grid-column:1;grid-row:2; }.focus .history { grid-column:3;grid-row:1 / 3; }
.equal { grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:minmax(0,1fr) 150px; }.equal .history { grid-column:1 / -1; }
.multi-grid :deep(.ai-run-card) { min-height:0;display:flex;flex-direction:column;overflow:hidden; }
.multi-grid :deep(.history) { min-height:0;max-height:100%;align-self:stretch;overflow:auto;box-sizing:border-box; }
</style>
