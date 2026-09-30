<template>
  <section class="live-content" aria-label="2048 live content">
      <div class="live-title">
        <div>
          <h1>{{ room.title[lang] || room.title.en }}</h1>
          <p>
            {{
              room.description[lang] || room.description.en
            }}
          </p>
        </div>
      </div>
      <section class="broadcast-grid" :class="{ 'timing-collapsed': timingCollapsed, 'history-collapsed': historyCollapsed }">
        <div class="board-column">
          <div class="score-strip">
            <div>
              <small>{{ t("本局得分", "SCORE") }}</small
              ><strong>{{ format(run?.score) }}</strong>
            </div>
            <div>
              <small>{{ t("历史最高", "ALL-TIME BEST") }}</small
              ><strong>{{ format(Math.max(best, run?.score || 0)) }}</strong>
            </div>
          </div>
          <div class="source-line">
            <span>{{
              run?.source && run.source !== "AI"
                ? run.source
                : t("AI 搜索", "AI search")
            }}</span
            ><small>#{{ format(run?.seq) }}</small>
          </div>
          <BaseBoard :frame="frame">
            <template #overlay>
              <div v-if="streamState === 'loading' || streamState === 'reconnecting'" class="board-overlay loading" role="status" aria-live="polite" aria-busy="true">
                <LoaderCircle :size="30" class="loading-spinner" />
                <h2>{{ streamState === 'loading' ? t('正在加载直播', 'Loading the stream') : t('正在恢复连接', 'Reconnecting') }}</h2>
                <p>{{ streamState === 'loading' ? t('正在连接直播间，请稍候', 'Connecting to the stream. Please wait.') : t('连接恢复后，画面将继续播放', 'Playback will resume when reconnected.') }}</p>
              </div>
              <div v-else-if="streamState === 'paused'" class="board-overlay" role="status">
                <h2>{{ t('直播已暂停', 'Stream paused') }}</h2>
                <p>{{ t('等待主播恢复直播', 'Waiting for the stream to resume') }}</p>
              </div>
              <div v-else-if="streamState === 'offline'" class="board-overlay">
                <WifiOff :size="30" />
                <h2>{{ t("主播暂时离线", "Stream paused") }}</h2>
                <p>{{ t("等待恢复直播", "Waiting for the broadcaster") }}</p>
              </div>
              <div v-else-if="run?.ended_at" class="board-overlay ended">
                <Trophy :size="32" />
                <h2>{{ t("本局结束", "Game over") }}</h2>
                <strong>{{ format(run.score) }}</strong>
                <p>
                  {{ countdown
                  }}{{ t(" 秒后开始新局", "s until the next game") }}
                </p>
              </div>
            </template>
          </BaseBoard>
          <div class="board-code">
            <input
              :value="hex"
              readonly
              aria-label="Board hexadecimal code"
              @focus="$event.target.select()"
            /><button @click="copyHex" :title="t('复制盘面', 'Copy board')">
              <Copy :size="17" />
            </button>
          </div>
        </div>
        <aside class="timing-column" :class="{ 'side-collapsed': timingCollapsed }">
          <button v-if="timingCollapsed" class="expand-panel" @click="timingCollapsed = false" :title="t('展开用时', 'Expand run time')" :aria-label="t('展开用时', 'Expand run time')" aria-expanded="false" aria-controls="live-timing-content"><PanelLeftOpen :size="18" /></button>
          <div v-show="!timingCollapsed" id="live-timing-content">
          <div class="panel-heading">
            <Clock :size="17" />
            <h2>{{ t("本局用时", "RUN TIME") }}</h2>
            <button class="collapse-panel" @click="timingCollapsed = true" :title="t('收起用时', 'Collapse run time')" :aria-label="t('收起用时', 'Collapse run time')" aria-expanded="true" aria-controls="live-timing-content"><PanelLeftClose :size="18" /></button>
          </div>
          <div class="elapsed">{{ duration(run?.elapsed_ms) }}</div>
          <div class="table-heading">
            <span>{{ t("棋块", "TILE") }}</span
            ><span>{{ t("累计用时", "REACHED AT") }}</span>
          </div>
          <div class="node-row" v-for="tile in milestones" :key="tile">
            <LiveTile :value="tile" /><span>{{
              run?.nodes?.[tile] != null ? duration(run.nodes[tile]) : "—"
            }}</span>
          </div>
          </div>
        </aside>
        <aside class="history-column" :class="{ 'side-collapsed': historyCollapsed }">
          <button v-if="historyCollapsed" class="expand-panel" @click="historyCollapsed = false" :title="t('展开最近对局', 'Expand recent runs')" :aria-label="t('展开最近对局', 'Expand recent runs')" aria-expanded="false" aria-controls="live-history-content"><PanelRightOpen :size="18" /></button>
          <div v-show="!historyCollapsed" id="live-history-content">
          <div class="panel-heading">
            <History :size="17" />
            <h2>{{ t("最近对局", "RECENT RUNS") }}</h2>
            <button @click="refreshHistory()" :title="t('刷新', 'Refresh')">
              <RefreshCw :size="16" />
            </button>
            <button class="collapse-panel" @click="historyCollapsed = true" :title="t('收起最近对局', 'Collapse recent runs')" :aria-label="t('收起最近对局', 'Collapse recent runs')" aria-expanded="true" aria-controls="live-history-content"><PanelRightClose :size="18" /></button>
          </div>
          <div class="table-heading">
            <span>{{ t("得分 / 最大棋块", "SCORE / BEST TILE") }}</span
            ><span>{{ t("结束时间", "FINISHED") }}</span>
          </div>
          <p v-if="!history.length" class="empty">
            {{ t("等待第一局完成", "Waiting for the first completed run") }}
          </p>
          <a
            class="history-row"
            v-for="game in history"
            :key="game.id"
            :href="replayUrl(game.id)"
            target="_blank"
            rel="noopener"
            ><span
              ><strong>{{ format(game.score) }}</strong
              ><small>{{ game.max_tile === 65536 ? '65k' : game.max_tile >= 1024 ? `${game.max_tile / 1024}K` : game.max_tile }}</small></span>
            <time>{{ dateTime(game.ended) }} <ExternalLink :size="12" /></time
          ></a>
          <nav class="history-pagination" :aria-label="t('历史对局分页', 'Run history pages')" :aria-busy="historyLoading">
            <button :disabled="historyPage === 1 || historyLoading" @click="loadHistory(historyPage - 1)" :title="t('上一页', 'Previous page')">‹</button>
            <template v-for="(page, index) in historyButtons" :key="index">
              <span v-if="page === null">…</span>
              <button v-else :aria-current="page === historyPage ? 'page' : undefined" :disabled="historyLoading" @click="loadHistory(page)">{{ page }}</button>
            </template>
            <button :disabled="historyPage === historyPages || historyLoading" @click="loadHistory(historyPage + 1)" :title="t('下一页', 'Next page')">›</button>
          </nav>
          </div>
        </aside>
      </section>
  </section>
</template>
<script setup>
import { shouldRefreshLiveClock } from '../displayClock.js';
import { ref, computed, onMounted, onUnmounted } from 'vue';
import { Clock, History, RefreshCw, PanelLeftOpen, PanelLeftClose, PanelRightOpen, PanelRightClose, Copy, ExternalLink, Trophy, WifiOff, LoaderCircle } from '@lucide/vue';
import BaseBoard from '../../components/BaseBoard.vue';
import { createSnapshotBoardFrame } from '../../components/boardFrame.js';
import { applyLiveStep } from '../liveEngine.js';
import LiveTile from '../LiveTile.vue';
import { boardPipFrame } from './boardPipFrame.js';
import { useRoom } from '../roomContext.js';
const props = defineProps({ lang: String, streamState: String, pipActive: Boolean });
const emit = defineEmits(['notice']);
const { room, api } = useRoom();
const t = (zh, en) => props.lang === 'zh' ? zh : en;
const showNotice = text => emit('notice', text);
const timingCollapsed = ref(false), historyCollapsed = ref(false);
const run = ref(null), frame = ref(createSnapshotBoardFrame('empty', Array(16).fill(0)));
const best = ref(0), history = ref([]), now = ref(Date.now());
const milestones = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
const hex = computed(() =>
  (run.value?.board || Array(16).fill(0))
    .map((v) => (v ? Math.min(15, Math.log2(v)).toString(16) : "0"))
    .join(""),
);
const countdown = computed(() =>
  Math.max(
    0,
    Math.ceil(((run.value?.restart_at || 0) * 1000 - now.value) / 1000),
  ),
);
const format = (n) =>
  Number(n || 0).toLocaleString(props.lang === "zh" ? "zh-CN" : "en-US");
const duration = (ms) => {
  const s = Math.floor((ms || 0) / 1000);
  return `${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};
const dateTime = (at) =>
  new Date(at * 1000).toLocaleString([], {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
const replayUrl = (id) =>
  `https://2048tables.online/verse-replay/?live=${encodeURIComponent(id)}&room=${encodeURIComponent(room.id)}`;
const historyPage = ref(1), historyTotal = ref(0), historyLoading = ref(false);
const historyPages = computed(() => Math.max(1, Math.ceil(historyTotal.value / 10)));
const historyButtons = computed(() => {
  const pages = [...new Set([1, historyPages.value, historyPage.value - 1, historyPage.value, historyPage.value + 1])]
    .filter(page => page > 0 && page <= historyPages.value).sort((a,b) => a-b);
  return pages.flatMap((page,index) => index && page - pages[index-1] > 1 ? [null,page] : [page]);
});
let historyRequest = 0;
function updateHistorySummary(data) {
  historyTotal.value = data.history_total ?? data.history?.length ?? 0;
  if (historyPage.value === 1 && !historyLoading.value) history.value = data.history || [];
}
async function loadHistory(page) {
  const request = ++historyRequest;
  historyLoading.value = true;
  try {
    const data = await api(`/history?page=${page}`);
    if (request !== historyRequest) return;
    history.value = data.history;
    historyPage.value = data.page;
    historyTotal.value = data.total;
  } catch {
    if (request === historyRequest) showNotice(t('历史对局加载失败，请重试。', 'Could not load run history. Please retry.'));
  } finally {
    if (request === historyRequest) historyLoading.value = false;
  }
}
async function refreshHistory() {
  try { receive({ ...await api('/state'), type: 'summary' }); } catch { showNotice(t('暂时无法更新', 'Could not refresh')); }
  await loadHistory(historyPage.value);
}
function receive(data) {
  if (data instanceof ArrayBuffer) {
    const next = applyLiveStep(run.value, data);
    run.value = next.run;
    if (!document.hidden) frame.value = next.frame;
    else if (props.pipActive) resume();
    return;
  }
  if (data.type === 'snapshot') {
    run.value = data.run;
    resume();
  } else if (data.type === 'source' && run.value) run.value.source = data.source;
  if (data.history !== undefined) { updateHistorySummary(data); best.value = data.best; }
}
function resume() {
  frame.value = createSnapshotBoardFrame(`${run.value?.run_id}:${run.value?.seq}:sync`, run.value?.board || Array(16).fill(0));
}
async function copyHex() {
  try {
    await navigator.clipboard.writeText(hex.value);
    showNotice(t("盘面已复制", "Board copied"));
  } catch {
    showNotice(t("请选择盘面编码复制", "Select the board code to copy it"));
  }
}
function getPipFrame() { return boardPipFrame({slots:[{lane:0,run:run.value,status:'running'}],state:props.streamState,lang:props.lang,title:room.title[props.lang] || room.title.en}); }
defineExpose({ receive, resume, getPipFrame });
let tick;
onMounted(() => { tick = setInterval(() => { if (shouldRefreshLiveClock(props.pipActive)) now.value = Date.now(); }, 500); });
onUnmounted(() => { clearInterval(tick); historyRequest++; });
</script>
<style scoped>
.live-title {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: center;
  margin-bottom: 24px;
}
h1 {
  font-size: 28px;
  line-height: 1.2;
  margin: 0 0 8px;
}
h2 {
  font-size: 15px;
  margin: 0;
}
p {
  line-height: 1.65;
}
.live-title p {
  margin: 0;
  color: var(--text-secondary);
}
.broadcast-grid {
  display: grid;
  grid-template-columns: var(--timing-track, 182px) minmax(480px, 1fr) var(--history-track, 262px);
  grid-template-areas: "timing board history";
  gap: 26px;
  align-items: start;
}
.board-column {
  grid-area:board;
  min-width: 0;
  width:var(--live-board-size);
  --tile-label-small:calc(var(--live-board-size) / 12);
  --tile-label-medium:calc(var(--live-board-size) / 15);
  --tile-label-large:calc(var(--live-board-size) / 20);
  justify-self:center;
}
.timing-column { grid-area:timing;max-height:660px;overflow:auto; }
.history-column { grid-area:history; }
.timing-collapsed { --timing-track:38px; }
.history-collapsed { --history-track:38px; }
.live-content .collapse-panel,.live-content .expand-panel { padding:4px;min-width:30px;min-height:30px;flex-shrink:0; }
.panel-heading .collapse-panel { margin-left:auto; }
.history-column .panel-heading .collapse-panel { margin-left:0; }
.timing-column.side-collapsed,.history-column.side-collapsed { padding:0;border:0;overflow:visible; }
.board-column :deep(.board-stage) { max-width:100%; }
.board-column :deep(.board-stage) { height:var(--live-board-size); }
.score-strip {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.score-strip > div {
  border-bottom: 2px solid var(--accent);
  padding-bottom: 10px;
}
.score-strip > div + div {
  border-color: #d6b461;
}
small {
  font-size: 11px;
  color: var(--text-secondary);
}
.score-strip small {
  display: block;
  margin-bottom: 6px;
}
.score-strip strong {
  font-size: 28px;
  font-variant-numeric: tabular-nums;
}
.source-line {
  display: flex;
  justify-content: space-between;
  padding: 13px 0;
  font-weight: 700;
  color: var(--accent);
}
.board-overlay {
  position: absolute;
  inset: 0;
  z-index: 20;
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 20px;
  background: #0a121bd9;
  color: #fff;
  border-radius: 10px;
}
.board-overlay h2 {
  font-size: 24px;
}
.board-overlay p {
  margin: 0;
  color: #ced5df;
}
.board-overlay.ended {
  background: #172028d9;
}
.loading-spinner { color:var(--accent);animation:live-loading-spin 1.1s linear infinite; }
@keyframes live-loading-spin { to { transform:rotate(360deg); } }
@media (prefers-reduced-motion:reduce) { .loading-spinner { animation:none; } }
.ended strong {
  font-size: 34px;
  color: #e9c76d;
}
.board-code {
  display: flex;
  gap: 8px;
  margin-top: 14px;
}
.board-code input {
  width: 100%;
  font-family: monospace;
  font-size: 16px;
}
.timing-column,
.history-column {
  border-left: 1px solid var(--border-main);
  padding-left: 24px;
  min-width: 0;
}
.panel-heading {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 36px;
  margin-bottom: 12px;
}
.timing-column { border-left:0;padding-left:0;border-right:1px solid var(--border-main);padding-right:24px; }
.panel-heading button,
.panel-heading small {
  margin-left: auto;
}
.panel-heading svg {
  color: var(--accent);
}
.elapsed {
  font-size: 30px;
  font-variant-numeric: tabular-nums;
  margin: 18px 0 25px;
}
.table-heading {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 0;
  color: var(--text-secondary);
  font-size: 11px;
  border-bottom: 1px solid var(--border-main);
}
.node-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-main);
  font-variant-numeric: tabular-nums;
}
.history-column {
  max-height: 660px;
  overflow: auto;
}
.history-pagination { display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:5px;padding:12px 0; }
.history-pagination button { min-width:28px;min-height:28px;padding:3px 7px; }
.history-pagination [aria-current=page] { background:var(--accent);color:var(--bg-main); }
.history-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  text-decoration: none;
  padding: 14px 0;
  border-bottom: 1px solid var(--border-main);
  color: var(--text-main);
}
.history-row:hover {
  color: var(--accent);
}
.history-row span small {
  display: block;
  margin-top: 3px;
}
.history-row time {
  font-size: 11px;
  white-space: nowrap;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 4px;
}
.history-row strong {
  font-size: 17px;
  font-variant-numeric: tabular-nums;
}
.empty {
  color: var(--text-secondary);
  padding: 28px 0;
  font-size: 13px;
}

.live-content { --live-board-size:480px; min-width:0; padding:18px; border:1px solid var(--border-main); border-radius:10px; background:var(--bg-card); }
.live-content button { display:inline-flex;align-items:center;justify-content:center;gap:8px;min-width:36px;min-height:36px;border:1px solid var(--border-main);border-radius:6px;padding:6px 10px;background:var(--bg-card);cursor:pointer; }
.live-content button:hover { border-color:var(--accent);color:var(--accent); }
.live-content input { min-width:0;border:1px solid var(--border-main);border-radius:6px;padding:8px 10px;background:var(--bg-input);color:var(--text-main); }
/* Fit the room-owned 1280x720 surface; histories never resize that surface. */
.live-content { height:100%;box-sizing:border-box;display:flex;flex-direction:column;overflow:hidden;--live-board-size:460px; }
.live-title { flex-shrink:0; }
.broadcast-grid { flex:1;min-height:0;overflow:hidden; }
.timing-column,.history-column { max-height:100%;overflow:auto;box-sizing:border-box; }
</style>
