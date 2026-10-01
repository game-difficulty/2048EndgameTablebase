<template>
  <HumanLiveLobby v-if="lobby" />
  <LivePage v-else-if="room" :key="room.id" :room="room" @room-ended="showRoomUnavailable" />
  <div v-else class="room-state-shell">
    <header class="room-state-header">
      <a class="brand" href="/lobby" aria-label="2048 LIVE">
        <Radio :size="24" aria-hidden="true" />
        <span>2048 <b>LIVE</b></span>
      </a>
      <a class="lobby-link" href="/lobby">{{ t('直播大厅', 'Live lobby') }}</a>
    </header>

    <main class="room-state-main" :aria-busy="!errorKind">
      <section class="room-state-card" role="status">
        <div class="offline-visual" aria-hidden="true">
          <div class="ghost-board">
            <i v-for="cell in ghostCells" :key="cell.index" :class="{ filled: cell.value }">
              {{ cell.value || '' }}
            </i>
          </div>
          <span class="offline-mark">
            <LoaderCircle v-if="!errorKind" :size="27" class="spin" />
            <WifiOff v-else :size="27" />
          </span>
        </div>

        <div class="room-state-copy">
          <span class="eyebrow">{{ errorKind ? t('直播已离线', 'Stream offline') : t('正在连接', 'Connecting') }}</span>
          <h1>{{ stateTitle }}</h1>
          <p>{{ stateDescription }}</p>
          <div v-if="errorKind" class="room-state-actions">
            <a class="primary-action" href="/lobby">
              <ArrowLeft :size="18" aria-hidden="true" />
              {{ t('返回直播大厅', 'Back to live lobby') }}
            </a>
            <button v-if="retryable" class="secondary-action" type="button" @click="load">
              <RefreshCw :size="17" aria-hidden="true" />
              {{ t('重试', 'Retry') }}
            </button>
          </div>
          <small v-if="errorKind === 'missing'">{{ t('你可以在直播大厅查看当前正在进行的对局。', 'See every active stream in the live lobby.') }}</small>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, onUnmounted } from 'vue';
import { ArrowLeft, LoaderCircle, Radio, RefreshCw, WifiOff } from '@lucide/vue';
import LivePage from './LivePage.vue';
import HumanLiveLobby from './HumanLiveLobby.vue';
import { liveLanguage } from './language.js';
import { roomIdFromPath, isLobbyPath } from './roomRoute.js';
import { contentProtocols } from './content/registry.js';
import { requestJson } from './roomContext.js';

const lang = ref(liveLanguage());
const room = ref(null);
const errorKind = ref('');
const retryable = ref(false);
const lobby = ref(isLobbyPath(location.pathname));
const ghostCells = [2, 4, 0, 0, 0, 8, 16, 0, 0, 0, 32, 64, 0, 0, 0, 128].map((value, index) => ({ value, index }));
let generation = 0;

const t = (zh, en) => lang.value === 'zh' ? zh : en;
const stateTitle = computed(() => {
  if (!errorKind.value) return t('正在连接直播间', 'Connecting to the stream');
  if (errorKind.value === 'outdated') return t('需要刷新页面', 'Page update required');
  if (errorKind.value === 'network') return t('暂时无法连接', 'Unable to connect');
  return t('房间不可用', 'Room unavailable');
});
const stateDescription = computed(() => {
  if (!errorKind.value) return t('正在读取房间状态，请稍候。', 'Loading the room. This should only take a moment.');
  if (errorKind.value === 'outdated') return t('直播内容已经更新，请刷新页面后再次进入。', 'The stream format has changed. Refresh the page and try again.');
  if (errorKind.value === 'network') return t('连接直播服务器失败，请检查网络后重试。', 'Could not reach the live server. Check your connection and try again.');
  return t('主播当前不在线。直播可能已经结束，或房间地址无效。', 'The broadcaster is offline. This stream may have ended, or the room link is invalid.');
});

function showRoomUnavailable() {
  generation++;
  room.value = null;
  errorKind.value = 'missing';
  retryable.value = false;
  document.title = `${t('房间不可用', 'Room unavailable')} · 2048 LIVE`;
}

async function load() {
  const current = ++generation;
  lobby.value = isLobbyPath(location.pathname);
  if (lobby.value) {
    room.value = null;
    errorKind.value = '';
    document.title = '2048 LIVE';
    return;
  }
  room.value = null;
  errorKind.value = '';
  retryable.value = false;
  const id = roomIdFromPath(location.pathname);
  if (!id) {
    errorKind.value = 'missing';
    document.title = `${t('房间不可用', 'Room unavailable')} · 2048 LIVE`;
    return;
  }
  try {
    const data = await requestJson(`/api/live/rooms/${id}`);
    if (generation !== current) return;
    if (contentProtocols[data.content_kind] !== data.protocol) {
      errorKind.value = 'outdated';
      retryable.value = true;
      return;
    }
    room.value = Object.freeze(data);
    document.title = data.title[lang.value] || data.title.en;
  } catch (reason) {
    if (generation !== current) return;
    errorKind.value = reason.status === 404 ? 'missing' : 'network';
    retryable.value = reason.status !== 404;
    document.title = `${stateTitle.value} · 2048 LIVE`;
  }
}

onMounted(() => {
  document.documentElement.dataset.theme = 'dark';
  fetch('/api/auth/me', { credentials: 'same-origin', cache: 'no-store' }).catch(() => {});
  load();
  window.addEventListener('popstate', load);
});
onUnmounted(() => {
  generation++;
  window.removeEventListener('popstate', load);
});
</script>

<style scoped>
.room-state-shell{min-height:100vh;background:radial-gradient(circle at 15% 3%,rgba(56,189,248,.11),transparent 34%),radial-gradient(circle at 86% 84%,rgba(251,113,133,.07),transparent 30%),var(--bg-main);color:var(--text-main)}
.room-state-header{height:72px;padding:0 clamp(22px,4vw,68px);display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border-main);background:rgba(15,23,42,.72);backdrop-filter:blur(14px)}
.brand{display:flex;align-items:center;gap:10px;color:#fff;font-size:22px;font-weight:800;text-decoration:none}.brand b{color:#fb7185}.lobby-link{padding:8px 13px;border:1px solid var(--border-main);border-radius:6px;color:var(--text-secondary);font-size:14px;text-decoration:none;transition:border-color .15s,color .15s}.lobby-link:hover{border-color:var(--accent);color:var(--text-main)}
.room-state-main{min-height:calc(100vh - 73px);display:grid;place-items:center;padding:48px 24px;box-sizing:border-box}.room-state-card{display:grid;grid-template-columns:280px minmax(0,390px);align-items:center;gap:clamp(38px,6vw,74px);width:min(820px,100%);padding:clamp(34px,5vw,62px);box-sizing:border-box;border:1px solid var(--border-main);border-radius:16px;background:color-mix(in srgb,var(--bg-card) 82%,transparent);box-shadow:0 28px 70px rgba(2,6,23,.28);backdrop-filter:blur(18px)}
.offline-visual{position:relative;display:grid;place-items:center;width:280px;height:280px}.offline-visual::before{content:"";position:absolute;inset:16px;border-radius:50%;background:radial-gradient(circle,rgba(56,189,248,.16),rgba(56,189,248,0) 68%);filter:blur(8px)}
.ghost-board{position:relative;display:grid;width:220px;height:220px;padding:10px;box-sizing:border-box;grid-template-columns:repeat(4,1fr);grid-template-rows:repeat(4,1fr);gap:8px;border-radius:13px;background:color-mix(in srgb,var(--board-bg) 88%,#0b1221);transform:rotate(-4deg);box-shadow:0 22px 42px rgba(2,6,23,.32)}.ghost-board i{display:grid;place-items:center;border-radius:5px;background:rgba(148,163,184,.08);color:transparent;font-style:normal;font-size:20px;font-weight:800}.ghost-board i.filled{background:rgba(148,163,184,.14);color:rgba(248,250,252,.28)}.ghost-board i:nth-child(6),.ghost-board i:nth-child(7){background:rgba(56,189,248,.13)}.ghost-board i:nth-child(11),.ghost-board i:nth-child(12){background:rgba(251,113,133,.12)}
.offline-mark{position:absolute;display:grid;width:66px;height:66px;place-items:center;border:1px solid rgba(148,163,184,.18);border-radius:50%;background:#172033;color:#cbd5e1;box-shadow:0 12px 28px rgba(2,6,23,.42)}
.room-state-copy{display:flex;min-width:0;flex-direction:column;align-items:flex-start}.eyebrow{margin-bottom:12px;color:#7dd3fc;font-size:12px;font-weight:800;letter-spacing:.15em;text-transform:uppercase}.room-state-copy h1{margin:0;color:var(--text-main);font-size:clamp(30px,5vw,46px);line-height:1.12}.room-state-copy p{max-width:370px;margin:17px 0 0;color:var(--text-secondary);font-size:16px;line-height:1.75}.room-state-copy small{margin-top:20px;color:color-mix(in srgb,var(--text-secondary) 72%,transparent);font-size:12px}
.room-state-actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:27px}.primary-action,.secondary-action{display:inline-flex;min-height:42px;box-sizing:border-box;align-items:center;justify-content:center;gap:8px;padding:0 17px;border-radius:7px;font:inherit;font-size:14px;font-weight:700;text-decoration:none;cursor:pointer}.primary-action{border:1px solid rgba(56,189,248,.52);background:linear-gradient(135deg,#0ea5e9,#2563eb);color:#fff;box-shadow:0 9px 24px rgba(14,165,233,.2)}.secondary-action{border:1px solid var(--border-main);background:var(--bg-card);color:var(--text-main)}.primary-action:hover,.secondary-action:hover{filter:brightness(1.08)}.spin{animation:spin 1.1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:720px){.room-state-header{height:64px}.room-state-main{min-height:calc(100vh - 65px);padding:28px 18px}.room-state-card{grid-template-columns:1fr;gap:15px;padding:30px 24px;text-align:center}.offline-visual{width:210px;height:210px;margin:auto}.ghost-board{width:174px;height:174px;padding:8px;gap:6px}.offline-mark{width:56px;height:56px}.room-state-copy{align-items:center}.room-state-copy p{font-size:15px}.room-state-actions{justify-content:center}.lobby-link{font-size:13px}}
@media(prefers-reduced-motion:reduce){.spin{animation:none}}
</style>
