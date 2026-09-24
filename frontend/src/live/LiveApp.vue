<template>
  <LivePage v-if="room" :key="room.id" :room="room" />
  <main v-else class="room-loading" role="status">
    <h1>{{ error ? '直播间暂不可用 / Room unavailable' : '正在连接直播间 / Connecting' }}</h1>
    <p v-if="error">{{ error }}</p>
    <button v-if="retryable" @click="load">重试 / Retry</button>
    <a v-if="error" href="/rooms/ai-classic">返回 AI 直播间 / AI room</a>
  </main>
</template>
<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import LivePage from './LivePage.vue';
import { roomIdFromPath } from './roomRoute.js';
import { contentProtocols } from './content/registry.js';
import { requestJson } from './roomContext.js';
const room = ref(null), error = ref(''), retryable = ref(false);
let generation = 0;
async function load() {
  const current = ++generation;
  room.value = null; error.value = ''; retryable.value = false;
  const id = roomIdFromPath(location.pathname);
  if (!id) { error.value = '房间不存在 / Room not found'; return; }
  try {
    const data = await requestJson(`/api/live/rooms/${id}`);
    if (generation !== current) return;
    if (contentProtocols[data.content_kind] !== data.protocol) {
      error.value = '此直播内容需要更新页面 / This content requires a page update'; return;
    }
    room.value = Object.freeze(data);
    document.title = data.title[navigator.language.startsWith('zh') ? 'zh' : 'en'] || data.title.en;
  } catch (reason) {
    if (generation !== current) return;
    retryable.value = reason.status !== 404;
    error.value = reason.status === 404 ? '房间不存在 / Room not found' : '连接失败，请重试 / Connection failed. Please retry.';
  }
}
onMounted(() => { load(); window.addEventListener('popstate', load); });
onUnmounted(() => { generation++; window.removeEventListener('popstate', load); });
</script>
<style scoped>
.room-loading { margin:12vh auto;padding:24px;max-width:650px;color:var(--text-main); }
.room-loading a,.room-loading button { display:inline-block;margin:16px 16px 0 0;color:var(--accent); }
</style>
