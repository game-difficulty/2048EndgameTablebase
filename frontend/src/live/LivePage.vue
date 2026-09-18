<template>
  <div class="live-page">
    <header class="live-header">
      <a href="https://2048tables.online/" class="brand"
        ><Radio :size="24" /><strong>2048 <span>AI LIVE</span></strong></a
      >
      <nav>
        <span :class="['live-status', streamState === 'live' ? 'on' : '']">{{
          streamState === 'loading' ? t('连接中', 'CONNECTING') : streamState === 'reconnecting' ? t('重连中', 'RECONNECTING') : streamState === 'live' ? t("直播中", "LIVE") : t("暂停", "OFFLINE")
        }}</span
        ><span class="viewer-count"><Users :size="16" /> {{ viewers }}</span>
        <button @click="toggleTheme" :title="t('切换主题', 'Switch theme')">
          <Sun :size="18" />
        </button>
        <button @click="lang = lang === 'zh' ? 'en' : 'zh'" title="Language">
          {{ lang === "zh" ? "EN" : "中" }}
        </button>
        <button v-if="!user" @click="loginOpen = true">
          {{ t("登录", "Sign in") }}</button
        ><LiveAccountMenu v-else :user="user" @saved="user = $event" @refresh="refreshIdentity" @logout="logout" />
        <a v-if="!user" class="register-link" href="https://2048tables.online/?auth=register" target="_blank" rel="noopener">{{ t('注册', 'Register') }}</a>
        <NativeLandscapeButton inline class="live-landscape" />
      </nav>
    </header>
    <main>
      <div class="live-layout">
      <div ref="stageColumn" class="stage-column">
      <div class="live-title">
        <div>
          <h1>{{ t("2048 AI 直播", "2048 AI Live") }}</h1>
          <p>
            {{
              t(
                "AI 实时思考，搜索与残局定式共同决策。",
                "Live decisions powered by search and endgame tables.",
              )
            }}
          </p>
        </div>
      </div>
      <GiftEffects ref="giftEffects" v-model:mode="effectsMode" :lang="lang" :catalog="giftCatalog" />
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
            <button @click="refreshSummary(); loadHistory(historyPage)" :title="t('刷新', 'Refresh')">
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
      <LuckyBags :state="luckyState" :user="user" :connected="connected" :lang="lang" @login="loginOpen = true" @balance="giftPanel?.refreshBalance()">
        <template #default="{ bag, open, caption }">
          <GiftPanel ref="giftPanel" :user="user" :online="online && connected" :lang="lang" @login="loginOpen = true" @catalog="giftCatalog = $event" @red-envelope="redEnvelopes?.compose()">
            <template #leading><button v-if="bag" class="lucky-strip-entry" @click="open(bag)" :title="t('福袋','Lucky bags')"><LuckyBagIcon /><b>{{ t('福袋','Lucky bags') }}</b><small>{{ caption }}</small></button></template>
          </GiftPanel>
        </template>
      </LuckyBags>
      <RedEnvelopes ref="redEnvelopes" :state="redState" :user="user" :connected="connected" :lang="lang" @login="loginOpen = true" @balance="giftPanel?.refreshBalance()" />
      </div>
      <section class="history-stats-strip">
        <div>
          <small>{{ t("历史完成", "ALL-TIME RUNS") }}</small
          ><strong>{{ format(allTime.games) }}</strong>
        </div>
        <div>
          <small>{{ t("达到 32K", "32K RUNS") }}</small
          ><strong>{{ format(allTime.tile32) }}</strong>
        </div>
        <div>
          <small>{{ t("达到 65k", "65k RUNS") }}</small
          ><strong>{{ format(allTime.tile64) }}</strong>
        </div>
        <div>
          <small>{{ t("历史平均分", "ALL-TIME AVERAGE") }}</small
          ><strong>{{
            format(allTime.games ? Math.round(allTime.score_sum / allTime.games) : 0)
          }}</strong>
        </div>
        <div>
          <small>{{ t('得分中位数', 'MEDIAN SCORE') }}</small>
          <strong>{{ allTime.median_score == null ? '—' : format(allTime.median_score) }}</strong>
        </div>
        <div :title="t('通过次数 /（通过次数 + 死亡失败次数）', 'Passed stages / (passed stages + stages lost on death)')">
          <small>{{ t('32k 综率', '32k STAGE WIN RATE') }}</small>
          <strong>{{ allTime.stage32_rate == null ? '—' : `${(allTime.stage32_rate * 100).toFixed(2)}%` }}</strong>
        </div>
      </section>
        <div class="chat-panel">
          <RoomAudience :lang="lang" :count="viewers" @count="viewers=$event" @help="giftPanel?.showContributionHelp()" />
        <aside class="about">
          <h2>{{ t("2048 练习与对战", "2048 Practice & Battles") }}</h2>
          <p>
            {{
              t(
                "想试试自己的走法？在主站练习残局、复盘对局，或和其他玩家来一场对战。",
                "Try your own moves: practice endgames, review your games, or challenge other players on the main site.",
              )
            }}
          </p>
          <a
            class="visit"
            href="https://2048tables.online/"
            target="_blank"
            rel="noopener"
            ><span class="visit-label">{{ t("前往主站体验", "Explore the main site") }}<ArrowUpRight :size="18" /></span>
            <span class="visit-url">https://2048tables.online/</span>
          </a>
        </aside>
          <div ref="chatList" class="chat-list">
            <p v-if="!messages.length" class="empty">
              {{ t("来聊聊这局的走法吧", "What do you think of this run?") }}
            </p>
            <div v-for="message in messages" :key="message.id" :class="['chat-row', { 'chat-gold': liveSupporterLevel(message) === 2, 'chat-entrance': message.type === 'entrance', 'chat-gift': message.type === 'gift' }]">
              <div>
                <header>
                  <LiveIdentity :actor="message" :lang="lang" />
                  <small v-if="message.guest">{{ t("游客", "Guest") }}</small>
                </header>
                <p v-if="message.type === 'entrance'">{{ t('来到直播间，欢迎！', 'joined the stream. Welcome!') }}</p>
                <p v-else-if="message.type === 'gift'" class="chat-gift-content"><span>{{ t('送出', 'sent') }} {{ giftCatalog.find(item => item.id === message.gift_id)?.[lang] || message.gift_id }}</span><GiftIcon :id="message.gift_id" /><b>×{{ message.combo_count }}</b></p>
                <button v-else-if="message.type === 'red_envelope'" class="chat-red" @click="redEnvelopes?.open(message.envelope_id)"><span>{{ t('发了一个红包','sent a red envelope') }} · {{ message.amount.toLocaleString() }} Token</span><img src="/live-gifts/red-envelope.webp" alt="" /></button>
                <p v-else>{{ message.text }}</p>
              </div>
            </div>
          </div>
          <div class="chat-tools">
            <label class="effect-controls"><Sparkles :size="15" /><select v-model="effectsMode" :aria-label="t('礼物特效','Gift effects')"><option value="full">{{ t('特效','Effects') }}</option><option value="simple">{{ t('简洁','Simple') }}</option><option value="off">{{ t('关闭','Off') }}</option></select></label>
            <div class="like-control">
              <LikeReaction ref="likeReaction" />
              <button class="like-button" @click="like" :aria-label="t('点赞', 'Like')" :aria-busy="likes.pending">
                <Heart :size="18" /> {{ format(likes.count) }}
              </button>
            </div>
          </div>
          <form class="chat-form" @submit.prevent="sendChat">
            <input
              v-model="draft"
              maxlength="64"
              :placeholder="
                t('聊一句，最多 32 字', 'Say something, up to 32 characters')
              "
              :aria-label="t('聊天内容', 'Chat message')"
            /><small>{{ [...draft].length }}/32</small
            ><button
              type="submit"
              :disabled="sending || !draft.trim() || [...draft].length > 32"
              :title="t('发送', 'Send')"
            >
              <Send :size="18" />
            </button>
          </form>
          <p v-if="notice" class="notice" role="status">{{ notice }}</p>
        </div>
        <LiveMusicPlayer class="music-footer" :lang="lang" :extra-url="musicUrl" compact />
      </div>
    </main>
    <dialog
      ref="loginDialog"
      class="live-login"
      v-if="loginOpen"
      @cancel="loginOpen = false"
    >
      <form @submit.prevent="login">
        <header>
          <h2>{{ t("登录账号", "Sign in") }}</h2>
          <button type="button" @click="loginOpen = false" aria-label="Close">
            <X :size="20" />
          </button>
        </header>
        <input
          v-model="email"
          type="email"
          autocomplete="username"
          required
          :placeholder="t('邮箱', 'Email')"
        /><input
          v-model="password"
          type="password"
          autocomplete="current-password"
          required
          :placeholder="t('密码', 'Password')"
        />
        <p v-if="loginError" role="alert">{{ loginError }}</p>
        <button type="submit">{{ t("登录", "Sign in") }}</button>
        <a class="register-link" href="https://2048tables.online/?auth=register" target="_blank" rel="noopener">{{ t('没有账号？前往主站注册', 'No account? Register on the main site') }}</a>
      </form>
    </dialog>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, nextTick, watch } from "vue";
import {
  Radio,
  Users,
  Sun,
  Heart,
  Sparkles,
  Clock,
  History,
  RefreshCw,
  PanelLeftOpen,
  PanelLeftClose,
  PanelRightOpen,
  PanelRightClose,
  Copy,
  ExternalLink,
  Trophy,
  WifiOff,
  LoaderCircle,
  MessageCircle,
  Send,
  ArrowUpRight,
  X,
} from "@lucide/vue";
import BaseBoard from "../components/BaseBoard.vue";
import NativeLandscapeButton from '../components/NativeLandscapeButton.vue';
import LiveIdentity from './LiveIdentity.vue';
import LiveAccountMenu from './LiveAccountMenu.vue';
import { liveSupporterLevel, entranceChat } from './supporterIdentity.js';
import { createSnapshotBoardFrame } from "../components/boardFrame.js";
import { applyLiveStep } from "./liveEngine.js";
import LiveTile from './LiveTile.vue';
import LiveMusicPlayer from './LiveMusicPlayer.vue';
import { LikeFeedback } from './likeFeedback.js';
import LikeReaction from './LikeReaction.vue';
import GiftPanel from './GiftPanel.vue';
import RoomAudience from './RoomAudience.vue';
import LuckyBags from './LuckyBags.vue';
import RedEnvelopes from './RedEnvelopes.vue';
import LuckyBagIcon from './LuckyBagIcon.vue';
import GiftEffects from './GiftEffects.vue';
import GiftIcon from './GiftIcon.vue';
import { mergeLiveChat } from './giftArtwork.js';
import { useI18n } from 'vue-i18n';
import { useLiveLayoutScale } from './liveLayout.js';
import { liveConnectionState } from './connectionState.js';

useLiveLayoutScale();
const stageColumn = ref(null);

const lang = ref(navigator.language.startsWith("zh") ? "zh" : "en");
const { locale } = useI18n();
watch(lang, value => { locale.value = value; }, { immediate: true });
const giftEffects = ref(null), giftCatalog = ref([]);
const effectsMode = ref('full');
const timingCollapsed = ref(false), historyCollapsed = ref(false);
const giftPanel = ref(null), luckyState = ref(null);
const redEnvelopes = ref(null), redState = ref(null);
const t = (zh, en) => (lang.value === "zh" ? zh : en);
const run = ref(null),
  frame = ref(createSnapshotBoardFrame("empty", Array(16).fill(0)));
const online = ref(false),
  connected = ref(false),
  viewers = ref(0),
  best = ref(0),
  history = ref([]),
  allTime = ref({});
const synchronized = ref(false), seenSnapshot = ref(false), paused = ref(false);
const streamState = computed(() => liveConnectionState({ connected:connected.value, synchronized:synchronized.value, seenSnapshot:seenSnapshot.value, online:online.value, paused:paused.value }));
const likes = reactive(new LikeFeedback()), likeReaction = ref(null);
let likeFlushTimer, likeNoticeAt = 0, actorPromise;
const messages = ref([]),
  draft = ref(""),
  sending = ref(false),
  notice = ref(""),
  chatList = ref(null);
const user = ref(null),
  loginDialog = ref(null),
  loginOpen = ref(false),
  email = ref(""),
  password = ref(""),
  loginError = ref("");
watch(loginOpen, async (open) => {
  await nextTick();
  if (open) loginDialog.value?.showModal();
});
const musicUrl = ref(""),
  now = ref(Date.now());
const milestones = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
let socket,
  backgroundTimer,
  backgroundDeadline = 0,
  retry,
  ping,
  tick,
  noticeTimer,
  stopped = false,
  failures = 0;
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
  Number(n || 0).toLocaleString(lang.value === "zh" ? "zh-CN" : "en-US");
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
  `https://2048tables.online/verse-replay/?live=${encodeURIComponent(id)}`;
const showNotice = (text) => {
  notice.value = text;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => (notice.value = ""), 5000);
};
async function api(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
  const response = await fetch(path, {
    signal: controller.signal,
    credentials: "same-origin",
    cache: "no-store",
    ...(body !== undefined
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {}),
  });
  if (!response.ok)
    throw Object.assign(Error("request_failed"), { status: response.status });
  return await response.json();
  } finally { clearTimeout(timeout); }
}
let chatHistoryLoaded = false;
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
    const data = await api(`/api/live/history?page=${page}`);
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
async function refreshSummary() {
  try {
    const data = await api("/api/live/state");
    best.value = data.best;
    likes.update(data.likes);
    updateHistorySummary(data);
    allTime.value = data.all_time || {};
    const firstLoad = !chatHistoryLoaded;
    const el = chatList.value;
    const followLatest = !chatHistoryLoaded || !el || el.scrollHeight - el.scrollTop - el.clientHeight < 50;
    messages.value = mergeLiveChat(messages.value, [...(data.chat || []), ...(data.gifts || [])]);
    chatHistoryLoaded = true;
    if (followLatest) {
      await nextTick();
      if (firstLoad && document.fonts) await document.fonts.ready;
      if (firstLoad) await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      if (chatList.value) chatList.value.scrollTop = chatList.value.scrollHeight;
    }
    if (!musicUrl.value) musicUrl.value = data.music_url || "";
    return data;
  } catch {
    showNotice(
      t("暂时无法更新，请稍后重试。", "Could not refresh. Please try again."),
    );
  }
}
function installSnapshot(data) {
  if (data.red_envelopes) redState.value = { ...data.red_envelopes, server_time: data.server_time };
  if (data.lucky_bags) luckyState.value = { bags: data.lucky_bags, server_time: data.server_time };
  online.value = data.online;
  paused.value = Boolean(data.paused);
  viewers.value = data.viewers;
  run.value = data.run;
  frame.value = createSnapshotBoardFrame(
    `${data.run?.run_id}:${data.run?.seq}:sync`,
    data.run?.board || Array(16).fill(0),
  );
}
async function receive(event) {
  if (document.hidden && backgroundDeadline && Date.now() >= backgroundDeadline) {
    socket?.close();
    return;
  }
  if (event.data instanceof ArrayBuffer) {
    try {
      const next = applyLiveStep(run.value, event.data);
      run.value = next.run;
      if (!document.hidden) frame.value = next.frame;
    } catch {
      socket?.close();
    }
    return;
  }
  let data;
  try { data = JSON.parse(event.data); }
  catch { event.target?.close(); return; }
  if (data.type === "snapshot") {
    installSnapshot(data);
    synchronized.value = true;
    seenSnapshot.value = true;
  }
  else if (data.type === 'lucky_bags') luckyState.value = data;
  else if (data.type === 'red_envelopes') redState.value = data;
  else if (data.type === 'red_envelope') await appendChat(data);
  else if (data.type === 'gift' || data.type === 'entrance') {
    if (!document.hidden) giftEffects.value?.receive(data);
    if (data.type === 'gift') await appendChat(data);
    else await appendChat(entranceChat(data));
  }
  else if (data.type === "source" && run.value) run.value.source = data.source;
  else if (data.type === "presence") {
    online.value = data.online;
    paused.value = Boolean(data.paused);
    viewers.value = data.viewers;
  } else if (data.type === "summary") {
    updateHistorySummary(data);
    allTime.value = data.all_time || {};
    best.value = data.best;
    likes.update(data.likes);
  } else if (data.type === "likes") likes.update(data.count);
  else if (data.type === "chat") {
    await appendChat(data);
  }
}
async function appendChat(data) {
    const el = chatList.value,
      nearBottom = !el || el.scrollHeight - el.scrollTop - el.clientHeight < 50;
    messages.value = mergeLiveChat(messages.value, [data]);
    if (nearBottom && !document.hidden) {
      await nextTick();
      if (el) el.scrollTop = el.scrollHeight;
    }
}
function connect() {
  if (stopped || document.hidden) return;
  if (socket && socket.readyState < 2) return;
  clearTimeout(retry);
  clearInterval(ping);
  synchronized.value = false;
  const ws = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/live/watch`,
  );
  socket = ws;
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    if (socket !== ws) return;
    connected.value = true;
    failures = 0;
    ping = setInterval(() => {
      if (document.hidden && backgroundDeadline && Date.now() >= backgroundDeadline) {
        ws.close();
        return;
      }
      if (ws.readyState === 1) ws.send("ping");
    }, 10000);
    refreshSummary();
  };
  ws.onmessage = event => { if (socket === ws) receive(event); };
  ws.onclose = () => {
    if (socket !== ws) return;
    socket = null;
    connected.value = false;
    synchronized.value = false;
    clearInterval(ping);
    if (!stopped && !document.hidden)
      retry = setTimeout(
        connect,
        Math.min(15000, 1000 * 2 ** failures++) + Math.random() * 300,
      );
  };
}
async function ensureActor() {
  if (!user.value) {
    actorPromise ||= api("/api/guest/session", {}).catch(error => { actorPromise = null; throw error; });
    await actorPromise;
  }
}
async function sendChat() {
  if (sending.value) return;
  sending.value = true;
  try {
    await ensureActor();
    await api("/api/live/chat", { text: draft.value });
    draft.value = "";
  } catch (e) {
    showNotice(
      e.status === 429
        ? t(
            "发言太快了，请稍后再聊。",
            "Please slow down and try again shortly.",
          )
        : t("发送失败，请稍后重试。", "Message not sent. Please try again."),
    );
  } finally {
    sending.value = false;
  }
}
function likeLimitNotice() {
  if (Date.now() - likeNoticeAt < 10000) return;
  likeNoticeAt = Date.now();
  showNotice(t('谢谢支持，稍后再点吧。', 'Thanks! Try another like in a moment.'));
}
function scheduleLikes() {
  if (stopped || likeFlushTimer || likes.inFlight || !likes.queued) return;
  likeFlushTimer = setTimeout(flushLikes, 500);
}
function like() {
  if (!likes.begin()) { likeLimitNotice(); return; }
  likeReaction.value?.play();
  scheduleLikes();
}
async function flushLikes() {
  likeFlushTimer = null;
  const amount = likes.takeBatch();
  if (!amount) return;
  try {
    await ensureActor();
    const data = await api("/api/live/like", { count: amount });
    likes.finish(data.count);
  } catch (e) {
    likes.reject();
    if (e.status === 429) likeLimitNotice();
    else showNotice(t("暂时无法点赞。", "Could not send your like."));
  } finally { scheduleLikes(); }
}
async function login() {
  try {
    const data = await api("/api/auth/login", {
      email: email.value,
      password: password.value,
    });
    user.value = data.user;
    password.value = "";
    loginOpen.value = false;
    loginError.value = "";
    socket?.close();
  } catch {
    loginError.value = t(
      "登录失败，请检查邮箱和密码。",
      "Sign-in failed. Check your email and password.",
    );
  }
}
async function logout() {
  try {
    await api("/api/auth/logout", {});
    user.value = null;
    actorPromise = null;
    await ensureActor().catch(() => {});
    socket?.close();
  } catch {
    showNotice(t("退出失败，请重试。", "Could not sign out."));
  }
}
function toggleTheme() {
  document.documentElement.dataset.theme =
    document.documentElement.dataset.theme === "dark" ? "light" : "dark";
}
async function copyHex() {
  try {
    await navigator.clipboard.writeText(hex.value);
    showNotice(t("盘面已复制", "Board copied"));
  } catch {
    showNotice(t("请选择盘面编码复制", "Select the board code to copy it"));
  }
}
async function refreshIdentity() {
  try {
    const previousUserId = user.value?.id;
    user.value = (await api('/api/auth/me')).user;
    actorPromise = null;
    if (previousUserId !== user.value?.id) socket?.close();
  } catch {}
}
async function visibility() {
  clearTimeout(retry);
  clearTimeout(backgroundTimer);
  if (document.hidden) {
    backgroundDeadline = Date.now() + 180000;
    backgroundTimer = setTimeout(() => {
      if (document.hidden) socket?.close();
    }, 180000);
  }
  else {
    const expired = backgroundDeadline && Date.now() >= backgroundDeadline;
    backgroundDeadline = 0;
    frame.value = createSnapshotBoardFrame(
      `${run.value?.run_id}:${run.value?.seq}:resume`,
      run.value?.board || Array(16).fill(0),
    );
    const previousUserId = user.value?.id;
    if (expired) socket?.close();
    else if (socket?.readyState === 1) socket.send('ping');
    connect();
    await refreshIdentity();
    if (stopped || document.hidden) return;
    await ensureActor().catch(() => {});
    if (stopped || document.hidden) return;
    if (previousUserId !== user.value?.id) socket?.close();
    connect();
    refreshSummary();
  }
}
onMounted(async () => {
  document.documentElement.dataset.theme = "dark";
  tick = setInterval(() => (now.value = Date.now()), 500);
  document.addEventListener("visibilitychange", visibility);
  const data = await refreshSummary();
  if (data) {
    installSnapshot(data);
  }
  await refreshIdentity();
  await ensureActor().catch(() => {});
  connect();
});
onUnmounted(() => {
  stopped = true;
  clearTimeout(backgroundTimer);
  clearTimeout(retry);
  clearTimeout(noticeTimer);
  clearTimeout(likeFlushTimer);
  clearInterval(ping);
  clearInterval(tick);
  socket?.close();
  document.removeEventListener("visibilitychange", visibility);
});
</script>

<style scoped>
:global(body.live-document) { --live-scale:1;overflow:auto;zoom:var(--live-scale);background:var(--bg-main); }
:global(body.live-document) { -webkit-text-size-adjust:100%;text-size-adjust:100%; }
.live-page {
  -webkit-text-size-adjust:100%;
  text-size-adjust:100%;
  --live-board-size:480px;
  min-width:1500px;
  width:var(--live-page-width,100%);
  margin-inline:auto;
  background: var(--bg-main);
  color: var(--text-main);
  font-size: 14px;
  letter-spacing: 0;
}
.live-page button,
.icon-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-width: 36px;
  min-height: 36px;
  border: 1px solid var(--border-main);
  border-radius: 6px;
  padding: 6px 10px;
  background: var(--bg-card);
  cursor: pointer;
}
.live-page button:hover,
.icon-button:hover {
  border-color: var(--accent);
}
button:disabled {
  opacity: 0.4;
  cursor: default;
}
.live-page input {
  min-width: 0;
  border: 1px solid var(--border-main);
  background: var(--bg-input);
  color: var(--text-main);
  border-radius: 5px;
  padding: 9px;
}
.live-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 16px 28px;
  border-bottom: 1px solid var(--border-main);
}
.register-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 9px;
  border: 1px solid var(--border-main);
  border-radius: 5px;
  background: var(--bg-input);
  color: var(--text-main);
  text-decoration: none;
}
.register-link:hover {
  border-color: var(--accent);
  color: var(--accent);
}
.brand,
.live-header nav,
.viewer-count {
  display: flex;
  gap: 12px;
  align-items: center;
}
.brand {
  color: var(--text-main);
  text-decoration: none;
}
.brand span {
  color: var(--accent);
  font-size: 13px;
}
.live-header nav {
  gap: 8px;
}
.live-landscape { zoom:calc(1 / var(--live-scale)); }
.live-status {
  color: var(--text-secondary);
  font-size: 12px;
}
.live-status.on {
  color: #34d399;
}
.live-status.on:before {
  content: "";
  display: inline-block;
  width: 7px;
  height: 7px;
  background: #34d399;
  border-radius: 50%;
  margin-right: 6px;
}
main {
  max-width: calc(2048px / var(--live-scale));
  margin: auto;
  padding: 26px 20px;
}
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
.live-page .like-button {
  color: #fb7185;
  position: relative;
  min-width: 96px;
  flex: 0 0 96px;
  font-variant-numeric: tabular-nums;
}
.like-control { position:relative;flex:0 0 auto; }
.broadcast-grid {
  display: grid;
  grid-template-columns: var(--timing-track, 200px) minmax(480px, 1fr) var(--history-track, 280px);
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
.live-page .collapse-panel,.live-page .expand-panel { padding:4px;min-width:30px;min-height:30px;flex-shrink:0; }
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
.score-strip small,
.history-stats-strip small {
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
.history-stats-strip {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  border-top: 1px solid var(--border-main);
  border-bottom: 1px solid var(--border-main);
  padding: 24px 0;
  margin: 28px 0;
  gap: 20px;
}
.history-stats-strip strong {
  font-size: 23px;
}
.live-layout { display:grid;grid-template-columns:minmax(0,1fr) clamp(300px,22%,520px);gap:20px;align-items:start; }
.stage-column { min-width:0;position:relative; }
.lucky-strip-entry { display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0;width:78px;gap:2px;background:transparent;border:0;border-right:1px solid var(--border-main);padding:4px;color:var(--text-main); }
.lucky-strip-entry :deep(svg) { width:40px;height:44px; }.lucky-strip-entry b,.lucky-strip-entry small { font-size:11px;line-height:1.4; }
.stage-column :deep(.gift-effects) { height:0;border:0;margin:0;z-index:25;pointer-events:none; }
.stage-column :deep(.effect-lanes) { position:absolute;top:0;left:0;right:0; }
.stage-column :deep(.gift-ceremony) { top:68px; }
.chat-panel { grid-column:2;grid-row:1;align-self:stretch;min-height:0;contain:size;position:relative;display:flex;flex-direction:column;border-left:1px solid var(--border-main);padding-left:12px;font-size:16px; }
.chat-panel small { font-size:13px; }
.chat-panel .empty,.chat-panel .notice { font-size:15px; }
.history-stats-strip { grid-column:1 / -1;grid-row:2;margin:8px 0; }
.music-footer { grid-column:1 / -1;grid-row:3; }
.chat-list {
  height:0;
  flex:1;
  min-height:0;
  overflow: auto;
  border-top: 1px solid var(--border-main);
  padding: 8px 0;
}
.chat-row {
  display: flex;
  gap: 6px;
  padding: 5px 6px;
  font-size:16px;
}
.chat-row { border-left:2px solid transparent; }
.chat-entrance { border-left-color:#54ac94;background:color-mix(in srgb,#54ac94 6%,transparent); }
.chat-gold { border-left-color:#bb9645;background:color-mix(in srgb,#d6b461 8%,transparent); }
.chat-row header > .live-identity { flex:0 1 auto;min-width:0; }
.chat-row :deep(.live-identity) { gap:5px;font-size:15px; }
.chat-row :deep(.account-avatar-shell) { width:24px;height:24px;flex:0 0 24px;font-size:10px; }
.chat-row :deep(.account-supporter-mark) { width:7px;height:7px;border-width:1px;right:-1px;bottom:-1px; }
.chat-row > div {
  min-width: 0;
  flex: 1;
}
.chat-row header {
  display: flex;
  gap: 5px;
  align-items: center;
}
.chat-row p {
  margin: 1px 0 0;
  padding-left:29px;
  line-height:22px;
  overflow-wrap: anywhere;
}
.chat-tools { display:flex;align-items:center;justify-content:space-between;gap:12px;flex-shrink:0;padding:6px 0;border-top:1px solid var(--border-main); }
.effect-controls { display:flex;align-items:center;gap:5px;color:var(--text-secondary); }
.effect-controls select { background:var(--bg-card);color:var(--text-main);border:1px solid var(--border-main);border-radius:4px;font-size:14px;padding:5px; }
.effect-controls option { background:var(--bg-main);color:var(--text-main); }
.chat-tools .like-button { min-height:30px;padding:4px 8px; }
.chat-form {
  display: flex;
  gap: 10px;
  align-items: center;
  padding-top: 4px;
}
.chat-form input {
  flex: 1;
  min-width:0;
  font-size:16px;
}
.chat-form button {
  color: var(--accent);
}
.notice {
  color: #e9be69;
  font-size: 13px;
}
.chat-gift { border-left-color:var(--accent); }
.chat-gift-content { display:flex;align-items:center;gap:5px; }
.chat-gift-content > span { min-width:0;overflow-wrap:anywhere; }
.chat-gift-content > b { white-space:nowrap;font-size:15px;color:var(--text-main); }
.chat-gift-content :deep(.gift-icon) { width:24px;height:24px; }
.chat-red { display:flex;align-items:center;gap:4px;width:100%;text-align:left;color:var(--text-main);background:transparent;border:0;padding:0;font-size:14px;cursor:pointer; }.chat-red span { min-width:0;overflow-wrap:anywhere; }.chat-red img { width:35px;height:35px;flex-shrink:0; }.chat-red:hover { color:#ef997c; }
.about {
  flex-shrink:0;
  padding:14px 8px 12px;
}
.about h2 { font-size:17px; }.about p { font-size:15px;margin:8px 0; }
.about p {
  color: var(--text-secondary);
}
.visit {
  display: inline-flex;
  flex-direction: column;
  align-items: flex-start;
  max-width: 100%;
  gap: 4px;
  color: var(--accent);
  text-decoration: none;
  font-weight: 700;
  margin-bottom: 0;
}
.visit-label { display: inline-flex; align-items: center; gap: 6px;font-size:15px; }
.visit-url { font-size: 15px; font-weight: 400; overflow-wrap: anywhere; user-select: text; }
.live-login {
  position: fixed;
  inset: 50% auto auto 50%;
  transform: translate(-50%, -50%);
  z-index: 100;
  width: min(360px, calc(100vw - 32px));
  padding: 24px;
  background: var(--bg-card);
  color: var(--text-main);
  border: 1px solid var(--border-main);
  border-radius: 8px;
  box-shadow: 0 0 0 100vmax #0008;
}
.live-login form {
  display: grid;
  gap: 16px;
}
.live-login header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
</style>
