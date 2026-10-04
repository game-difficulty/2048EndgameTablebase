<template>
  <div :class="['live-page', { 'room-focus-active': focusActive }]">
    <header v-show="!focusActive" class="live-header">
      <a href="https://2048tables.online/" class="brand"
        ><Radio :size="24" /><strong>2048 <span>LIVE</span></strong></a
      >
      <nav>
        <a href="/lobby">{{ t('直播大厅', 'Live lobby') }}</a>
        <a class="room-address" :href="room.path">{{ room.title[lang] || room.title.en }}</a>
        <RoomPipControls ref="roomPip" :get-surface="() => roomStage?.element()" :get-frame="() => content?.getPipFrame?.()"
          :lang="lang" :title="room.title[lang] || room.title.en" @active="setPipActive" @surface="setSurfaceDetached" />
        <RoomFocusControls v-model:active="focusActive" :disabled="pipDetached" :lang="lang" />
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
      <aside v-show="!focusActive" id="room-activity-dock" class="room-activity-dock" :aria-label="t('直播间活动','Room activities')"></aside>
      <div class="stage-column">
      <div class="room-stage-home">
      <div v-show="pipDetached" class="room-stage-placeholder"><p>{{ t('直播内容正在小窗中显示','The stream is playing in the mini player') }}</p><button @click="roomPip?.close()">{{ t('返回页面观看','Watch here') }}</button></div>
      <RoomStage ref="roomStage" :immersive="focusActive"><div class="content-stage">
        <component :is="contentComponent" ref="content" :lang="lang" :stream-state="streamState" :pip-active="pipActive" @notice="showNotice" @resync="snapshotRecovery.refresh()" />
      </div><template #overlays>
        <GiftEffects v-if="room.capabilities.gifts" ref="giftEffects" overlay v-model:mode="effectsMode" :lang="lang" :catalog="giftCatalog" />
      </template></RoomStage></div>
      <RoomActivities v-if="hasRoomActivities" v-show="!focusActive" ref="redEnvelopes" :room="room" :transport="{api,url}" dock-target="#room-activity-dock" prediction-target="#room-prediction-entry" :lucky-state="luckyState" :red-state="redState" :prediction-state="predictionState" :user="user" :connected="interactionConnected" :online="interactionOnline" :lang="lang" @login="loginOpen = true" @balance="giftPanel?.refreshBalance()">
        <template #default="{ bag, open, caption }">
          <GiftPanel v-if="room.capabilities.gifts" :red-envelopes="room.capabilities.red_envelopes" ref="giftPanel" :user="user" :online="interactionOnline" :lang="lang" @login="loginOpen = true" @catalog="giftCatalog = $event" @red-envelope="redEnvelopes?.compose()">
            <template #leading>
              <div v-if="room.capabilities.predictions" id="room-prediction-entry" class="room-prediction-entry"></div>
              <button v-if="bag" class="lucky-strip-entry" @click="open(bag)" :title="t('福袋','Lucky bags')"><LuckyBagIcon /><b>{{ t('福袋','Lucky bags') }}</b><small>{{ caption }}</small></button>
            </template>
          </GiftPanel>
        </template>
      </RoomActivities>
      </div>
      <section v-if="room.capabilities.statistics" v-show="!focusActive" class="history-stats-strip">
        <label class="stats-range">
          <span>{{ t('统计范围', 'STATISTICS') }}</span>
          <UiSelect
            v-model="statsRange"
            :options="statsRangeOptions"
            :aria-label="t('选择统计范围', 'Choose statistics range')"
            trigger-class="stats-range-select"
            menu-class="stats-range-menu"
            option-class="stats-range-option"
            @change="refreshSummary"
          />
        </label>
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
        <div v-show="!focusActive" class="chat-panel">
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
                <header v-if="message.type !== 'prediction_reward'">
                  <LiveIdentity :actor="message" :lang="lang" />
                  <small v-if="message.guest">{{ t("游客", "Guest") }}</small>
                </header>
                <PredictionAnnouncement v-if="message.type === 'prediction_reward'" :event="message" :lang="lang" />
                <p v-else-if="message.type === 'entrance'">{{ t('来到直播间，欢迎！', 'joined the stream. Welcome!') }}</p>
                <p v-else-if="message.type === 'gift'" class="chat-gift-content"><span>{{ t('送出', 'sent') }} {{ giftCatalog.find(item => item.id === message.gift_id)?.[lang] || message.gift_id }}</span><GiftIcon :id="message.gift_id" /><b>×{{ message.combo_count }}</b></p>
                <button v-else-if="message.type === 'red_envelope'" class="chat-red" @click="redEnvelopes?.open(message.envelope_id)"><span>{{ t('发了一个红包','sent a red envelope') }} · {{ message.amount.toLocaleString() }} Token</span><img src="/live-gifts/red-envelope.webp" alt="" /></button>
                <p v-else>{{ message.text }}</p>
              </div>
            </div>
          </div>
          <div class="chat-tools">
            <label v-if="room.capabilities.gifts" class="effect-controls"><Sparkles :size="15" /><select v-model="effectsMode" :aria-label="t('礼物特效','Gift effects')"><option value="full">{{ t('特效','Effects') }}</option><option value="simple">{{ t('简洁','Simple') }}</option><option value="off">{{ t('关闭','Off') }}</option></select></label>
            <div v-if="room.capabilities.likes" class="like-control">
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
                t('聊一句', 'Say something')
              "
              :aria-label="t('聊天内容', 'Chat message')"
            /><small>{{ chatLength(draft) }}/{{ CHAT_LIMIT }}</small
            ><button
              type="submit"
              :disabled="sending || !draft.trim() || chatLength(draft) > CHAT_LIMIT"
              :title="t('发送', 'Send')"
            >
              <Send :size="18" />
            </button>
          </form>
          <p v-if="notice" class="notice" role="status">{{ notice }}</p>
        </div>
        <LiveMusicPlayer v-show="!focusActive" class="music-footer" :lang="lang" :extra-url="musicUrl" compact />
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
import { serverErrorText } from '../services/errors/serverErrorText.js';
import { chatLength, CHAT_LIMIT } from './chatLength.js';
import { ref, reactive, computed, onMounted, onUnmounted, nextTick, watch } from "vue";
import { liveLanguage, saveLiveLanguage } from './language.js';
import {
  Radio,
  Users,
  Sun,
  Heart,
  Sparkles,
  MessageCircle,
  Send,
  ArrowUpRight,
  X,
} from "@lucide/vue";
import UiSelect from "../components/UiSelect.vue";
import NativeLandscapeButton from '../components/NativeLandscapeButton.vue';
import LiveIdentity from '../features/auth/AudienceIdentity.vue';
import LiveAccountMenu from './LiveAccountMenu.vue';
import { liveSupporterLevel, entranceChat } from './supporterIdentity.js';
import LiveMusicPlayer from './LiveMusicPlayer.vue';
import { LikeFeedback } from './likeFeedback.js';
import LikeReaction from './LikeReaction.vue';
import GiftPanel from '../features/gifts/GiftPanel.vue';
import RoomAudience from './RoomAudience.vue';
import RoomActivities from '../features/roomActivities/RoomActivities.vue';
import LuckyBagIcon from '../features/roomActivities/LuckyBagIcon.vue';
import GiftEffects from '../features/gifts/GiftEffects.vue';
import GiftIcon from '../features/gifts/GiftIcon.vue';
import { mergeLiveChat } from '../features/gifts/giftArtwork.js';
import PredictionAnnouncement from '../features/roomActivities/PredictionAnnouncement.vue';
import { provideRoom } from './roomContext.js';
import { contentRegistry } from './content/registry.js';
import { createGiftClient } from '../features/gifts/giftApi.js';
import { provideGiftClient } from '../features/gifts/context.js';
import { useI18n } from 'vue-i18n';
import { useLiveLayoutScale } from './liveLayout.js';
import { liveConnectionState } from './connectionState.js';
import { createWatchConnection } from './watchConnection.js';
import { createSnapshotRecovery } from './snapshotRecovery.js';
import { createSharedRefresh } from './sharedRefresh.js';
import { createMatchDeltaDecoder } from './matchDelta.js';
import { projectionIsOlder } from '../../../competition/shared/projectStateOrder.mjs';
import { isRoomEndedEvent } from './roomLifecycle.js';
import { canConnectLive, backgroundExpired } from './pipPolicy.js';
import RoomStage from './RoomStage.vue';
import RoomPipControls from './pip/RoomPipControls.vue';
import RoomFocusControls from './RoomFocusControls.vue';
const pipActive = ref(false);
const roomPip = ref(null), roomStage = ref(null), pipDetached = ref(false);
const focusActive = ref(false);
function setSurfaceDetached(detached) {
  pipDetached.value = detached;
  roomStage.value?.refreshLayout();
}

const props = defineProps({ room: { type: Object, required: true } });
const emit = defineEmits(['room-ended']);
useLiveLayoutScale();
const room = props.room;
const splitStreams = room.content_kind === 'competition-match';
const hasRoomActivities = computed(() => Boolean(
  room.capabilities.gifts || room.capabilities.red_envelopes
  || room.capabilities.lucky_bags || room.capabilities.predictions
));
const contentComponent = contentRegistry[room.content_kind];
const content = ref(null);
const { api, url } = provideRoom(room);
provideGiftClient(createGiftClient({ base: `${room.api_base}/gifts`, target: `live:${room.id}` }));

const lang = ref(liveLanguage());
const { locale } = useI18n();
watch(lang, value => { locale.value = value; saveLiveLanguage(value); }, { immediate: true });
const giftEffects = ref(null), giftCatalog = ref([]);
const effectsMode = ref('full');
const giftPanel = ref(null), luckyState = ref(null);
const redEnvelopes = ref(null), redState = ref(null);
const predictionState = ref(null);
const t = (zh, en) => (lang.value === "zh" ? zh : en);
watch(focusActive, async value => {
  document.body.classList.toggle('live-focus-document', value);
  await nextTick();
  roomStage.value?.refreshLayout();
});
const online = ref(false),
  connected = ref(false),
  viewers = ref(0),
  allTime = ref({});
const synchronized = ref(false), seenSnapshot = ref(false), paused = ref(false);
const socialReady = ref(false), socialOnline = ref(false);
const interactionConnected = computed(() => splitStreams ? socialReady.value : connected.value);
const interactionOnline = computed(() => splitStreams ? socialReady.value && socialOnline.value : online.value && connected.value && synchronized.value);
const statsRange = ref('all');
const statsRangeOptions = computed(() => [
  { value: '24h', label: t('24小时内', 'Last 24 hours') },
  { value: 'recent100', label: t('最近100局', 'Last 100 runs') },
  { value: 'all', label: t('历史以来', 'All time') },
]);
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
const musicUrl = ref("");
let backgroundTimer,
  backgroundDeadline = 0,
  noticeTimer,
  stopped = false;
let roomEnded = false, installedMatch = null;
const snapshotRecovery = createSnapshotRecovery({ url:url('/snapshot'), install:installSnapshot });
const watchUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${room.api_base}/watch`;
const watchConnection = createWatchConnection({
  room,
  url: watchUrl + (splitStreams ? '?channel=board' : ''),
  channel: splitStreams ? 'board' : 'all',
  decoder: splitStreams ? createMatchDeltaDecoder(room) : undefined,
  canConnect: () => !stopped && canConnectLive(document.hidden, pipActive.value),
  expired: () => backgroundExpired(document.hidden, pipActive.value, backgroundDeadline, Date.now()),
  onOpen: () => { connected.value = true; if (!splitStreams) void refreshSummary(); },
  onDisconnect: () => { connected.value = false; synchronized.value = false; },
  onMessage: receive,
  onResync: () => snapshotRecovery.refresh(),
  onSnapshot: () => { synchronized.value = true; seenSnapshot.value = true; },
  onEnded: endRoom,
  onActivity: () => roomPip.value?.refresh(),
});
const socialConnection = splitStreams ? createWatchConnection({
  room, url: watchUrl + '?channel=social', channel: 'social',
  canConnect: () => !stopped && canConnectLive(document.hidden, pipActive.value),
  expired: () => backgroundExpired(document.hidden, pipActive.value, backgroundDeadline, Date.now()),
  onOpen: () => { void refreshSummary(); },
  onDisconnect: () => { socialReady.value = false; },
  onMessage: receive,
  onSnapshot: () => { socialReady.value = true; },
  onEnded: endRoom,
}) : null;
function connections(method) {
  watchConnection[method]();
  socialConnection?.[method]();
}
const format = (n) =>
  Number(n || 0).toLocaleString(lang.value === "zh" ? "zh-CN" : "en-US");
const showNotice = (text) => {
  notice.value = text;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => (notice.value = ""), 5000);
};
let chatHistoryLoaded = false;
let summaryRequest = 0;
const sharedHumanSummary = createSharedRefresh(loadSummary);
function refreshSummary() {
  return room.content_kind === 'human-play' ? sharedHumanSummary() : loadSummary();
}
async function loadSummary() {
  const request = ++summaryRequest;
  try {
    const data = await api(['competition-match','human-play'].includes(room.content_kind) ? '/social-state' : `/state?stats_range=${encodeURIComponent(statsRange.value)}`);
    if (stopped || request !== summaryRequest) return;
    likes.update(data.likes);
    if (room.content_kind === 'competition-match' && data.match) installSnapshot({ ...data, type:'snapshot' });
    else if (room.content_kind === 'competition-match') await receive(data);
    else content.value?.receive({ ...data, type: 'summary' });
    allTime.value = data.all_time || {};
    statsRange.value = data.stats_range || statsRange.value;
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
  if (data.match) {
    if (projectionIsOlder(installedMatch, data.match)) return;
    installedMatch = data.match;
    seenSnapshot.value = true;
  }
  watchConnection.observeSnapshot(data);
  if (!splitStreams) installSocialState(data);
  online.value = data.online;
  paused.value = Boolean(data.paused);
  viewers.value = data.viewers;
  content.value?.receive(data);
}
function installSocialState(data) {
  if (data.predictions) predictionState.value = { ...data.predictions, server_time:data.server_time };
  if (data.red_envelopes) redState.value = { ...data.red_envelopes, server_time:data.server_time };
  if (data.lucky_bags) luckyState.value = { bags:data.lucky_bags, server_time:data.server_time };
  if (data.likes != null) likes.update(data.likes);
  socialOnline.value = Boolean(data.online);
  viewers.value = data.viewers;
}
async function receive(data) {
  if (data instanceof ArrayBuffer) {
    content.value?.receive(data);
    return;
  }
  if (data.type === 'room_ended') {
    if (isRoomEndedEvent(data, room.id)) endRoom();
  }
  else if (data.type === "snapshot") {
    installSnapshot(data);
  }
  else if (data.type === 'social_snapshot') installSocialState(data);
  else if (data.type === 'lucky_bags') luckyState.value = data;
  else if (data.type === 'red_envelopes') redState.value = data;
  else if (data.type === 'predictions') predictionState.value = data;
  else if (data.type === 'red_envelope') await appendChat(data);
  else if (data.type === 'prediction_reward') {
    await appendChat(data);
    giftPanel.value?.refreshBalance();
  }
  else if (data.type === 'gift' || data.type === 'entrance') {
    if (!document.hidden || pipDetached.value) giftEffects.value?.receive(data);
    if (data.type === 'gift') await appendChat(data);
    else await appendChat(entranceChat(data));
  }
  else if (data.type === "presence") {
    socialOnline.value = Boolean(data.online);
    online.value = data.online;
    paused.value = Boolean(data.paused);
    viewers.value = data.viewers;
  } else if (data.type === "summary") {
    content.value?.receive({ ...data, type: 'summary' });
    if (statsRange.value === (data.stats_range || 'all')) allTime.value = data.all_time || {};
    else refreshSummary();
    likes.update(data.likes);
  } else if (data.type === "likes") likes.update(data.count);
  else if (data.type === "chat") {
    await appendChat(data);
  } else {
    content.value?.receive(data);
  }
}

function endRoom() {
  if (roomEnded) return;
  roomEnded = true;
  stopped = true;
  connections('stop');
  snapshotRecovery.stop();
  emit('room-ended');
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
  connections('connect');
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
    await api("/chat", { text: draft.value });
    draft.value = "";
  } catch (e) {
    showNotice(
      e.status === 429
        ? t(
            "发言太快了，请稍后再聊。",
            "Please slow down and try again shortly.",
          )
        : e.status === 400
          ? t("消息未通过内容检查，请修改后重试。", "Message blocked by content rules. Please edit it and try again.")
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
    const data = await api("/like", { count: amount });
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
    connections('reconnect');
  } catch (error) {
    loginError.value = serverErrorText(error, lang.value);
  }
}
async function logout() {
  try {
    await api("/api/auth/logout", {});
    user.value = null;
    actorPromise = null;
    await ensureActor().catch(() => {});
    connections('reconnect');
  } catch {
    showNotice(t("退出失败，请重试。", "Could not sign out."));
  }
}
function toggleTheme() {
  document.documentElement.dataset.theme =
    document.documentElement.dataset.theme === "dark" ? "light" : "dark";
}
async function refreshIdentity() {
  try {
    const previousUserId = user.value?.id;
    user.value = (await api('/api/auth/me')).user;
    actorPromise = null;
    if (previousUserId !== user.value?.id) connections('reconnect');
  } catch {}
}
function setPipActive(value) {
  if (stopped) return;
  pipActive.value = value;
  clearTimeout(backgroundTimer);
  if (value) { backgroundDeadline = 0; connect(); }
  else if (document.hidden) visibility();
}
async function visibility() {
  connections('cancelRetry');
  clearTimeout(backgroundTimer);
  if (document.hidden) {
    if (pipActive.value) { backgroundDeadline = 0; connect(); return; }
    backgroundDeadline = Date.now() + 180000;
    backgroundTimer = setTimeout(() => {
      if (backgroundExpired(document.hidden, pipActive.value, backgroundDeadline, Date.now())) connections('disconnect');
    }, 180000);
  }
  else {
    const expired = backgroundDeadline && Date.now() >= backgroundDeadline;
    backgroundDeadline = 0;
    content.value?.resume();
    const previousUserId = user.value?.id;
    if (expired) connections('disconnect');
    else connections('check');
    connect();
    await refreshIdentity();
    if (stopped || document.hidden) return;
    await ensureActor().catch(() => {});
    if (stopped || document.hidden) return;
    if (previousUserId !== user.value?.id) connections('reconnect');
    connect();
    refreshSummary();
  }
}
onMounted(async () => {
  document.documentElement.dataset.theme = "dark";
  document.addEventListener("visibilitychange", visibility);
  if (room.content_kind === 'competition-match') connect();
  const data = room.content_kind === 'human-play' ? undefined : await refreshSummary();
  if (stopped) return;
  if (data && !splitStreams) {
    installSnapshot(data);
  }
  await refreshIdentity();
  await ensureActor().catch(() => {});
  connect();
});
onUnmounted(() => {
  stopped = true;
  clearTimeout(backgroundTimer);
  clearTimeout(noticeTimer);
  clearTimeout(likeFlushTimer);
  connections('stop');
  snapshotRecovery.stop();
  document.removeEventListener("visibilitychange", visibility);
  document.body.classList.remove('live-focus-document');
});
</script>

<style scoped>
:global(body.live-document) { --live-scale:1;overflow:auto;zoom:var(--live-scale);background:var(--bg-main); }
:global(body.live-document) { -webkit-text-size-adjust:100%;text-size-adjust:100%; }
:global(body.live-document.live-focus-document) { overflow:hidden;zoom:1; }
.live-page {
  -webkit-text-size-adjust:100%;
  text-size-adjust:100%;
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
.room-focus-active { min-width:0;width:100vw;height:100vh;height:100dvh;margin:0;overflow:hidden; }
.room-focus-active main { width:100%;height:100%;max-width:none;margin:0;padding:0;box-sizing:border-box; }
.room-focus-active .live-layout { display:flex;width:100%;height:100%;align-items:center;justify-content:center;gap:0; }
.room-focus-active .stage-column,.room-focus-active .room-stage-home { width:100%;height:100%; }
.room-focus-active .room-stage-home { display:flex;align-items:center;justify-content:center; }
h2 { font-size:15px;margin:0; }
p { line-height:1.65; }
.live-page .like-button {
  color: #fb7185;
  position: relative;
  min-width: 96px;
  flex: 0 0 96px;
  font-variant-numeric: tabular-nums;
}
.like-control { position:relative;flex:0 0 auto; }
.stats-range { grid-column:1 / -1;display:flex;align-items:center;justify-content:flex-end;gap:10px;margin-bottom:2px;color:var(--text-secondary);font-size:15px;font-weight:600;line-height:1.3; }
.stats-range > :deep(.ui-popover-select) { min-width:142px; }
.stats-range :deep(.stats-range-select) { min-height:38px;padding:7px 10px;border:1px solid var(--border-main);border-radius:8px;background:var(--bg-input);color:var(--text-main);font-size:15px;font-weight:600; }
.stats-range :deep(.stats-range-select:hover) { border-color:color-mix(in srgb,var(--accent) 55%,var(--border-main)); }
.stats-range :deep(.stats-range-menu) { border-radius:10px; }
.stats-range :deep(.stats-range-option) { min-height:36px;border-radius:7px;font-size:15px; }
small { font-size:11px;color:var(--text-secondary); }
.history-stats-strip small { display:block;margin-bottom:6px; }
.empty { color:var(--text-secondary);padding:28px 0;font-size:13px; }
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
.live-layout { display:grid;grid-template-columns:minmax(0,1fr) clamp(300px,22%,520px) 96px;gap:20px;align-items:start; }
.room-activity-dock { grid-column:3;grid-row:1;display:flex;flex-direction:column;gap:16px;padding-top:12px; }
.stage-column { grid-column:1;grid-row:1; }
.stage-column { min-width:0; }
.content-stage { position:relative;min-width:0; }
.room-stage-home { min-width:0; }
.room-stage-placeholder { aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:12px;background:var(--bg-card);border:1px solid var(--border-main);border-radius:12px; }
.room-address { font-size:13px;color:var(--text-secondary);text-decoration:none; }
.lucky-strip-entry { display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0;width:78px;gap:2px;background:transparent;border:0;border-right:1px solid var(--border-main);padding:4px;color:var(--text-main); }
.room-prediction-entry { display:flex;flex-shrink:0; }
.lucky-strip-entry :deep(svg) { width:40px;height:44px; }.lucky-strip-entry b,.lucky-strip-entry small { font-size:11px;line-height:1.4; }
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
