<template>
  <aside class="run-history" :class="{ 'run-history-horizontal': horizontal }">
    <header><h2>{{ t('最近对局', 'Recent runs') }}</h2><button @click="collapsed = !collapsed" :aria-expanded="!collapsed" :aria-label="collapsed ? t('展开最近对局', 'Expand recent runs') : t('收起最近对局', 'Collapse recent runs')">{{ collapsed ? '+' : '−' }}</button></header>
    <div v-show="!collapsed" class="history-body">
      <div class="history-list">
      <p v-if="!history.length">{{ t('等待第一局完成', 'Waiting for the first completed run') }}</p>
      <a v-for="game in history" :key="game.id" :href="replayUrl(game.id)" target="_blank" rel="noopener">
        <span><b>{{ participantName(game) }} · {{ Number(game.score).toLocaleString() }}</b><small>{{ game.max_tile >= 1024 ? `${game.max_tile / 1024}K` : game.max_tile }}</small></span>
        <time>{{ new Date(game.ended * 1000).toLocaleString([], { month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit' }) }}</time>
      </a>
      </div>
      <nav :aria-label="t('最近对局分页', 'Recent runs pagination')"><button :disabled="page <= 1 || loading" :aria-label="t('上一页最近对局', 'Previous page of recent runs')" @click="load(page - 1)">‹</button><span>{{ page }} / {{ pages }}</span><button :disabled="page >= pages || loading" :aria-label="t('下一页最近对局', 'Next page of recent runs')" @click="load(page + 1)">›</button><button :disabled="loading" @click="load(page)">{{ t('刷新', 'Refresh') }}</button></nav>
    </div>
  </aside>
</template>
<script setup>
import { ref, computed, onUnmounted } from 'vue';
import { useRoom } from '../roomContext.js';
import { participantName } from './participants.js';
const props = defineProps({ lang: String, horizontal: Boolean });
const emit = defineEmits(['notice']);
const { room, api } = useRoom();
const t = (zh, en) => props.lang === 'zh' ? zh : en;
const history = ref([]), page = ref(1), total = ref(0), loading = ref(false), collapsed = ref(false);
const pages = computed(() => Math.max(1, Math.ceil(total.value / 10)));
const replayUrl = id => `https://2048tables.online/verse-replay/?live=${encodeURIComponent(id)}&room=${encodeURIComponent(room.id)}`;
let request = 0;
function receive(data) {
  if (!Array.isArray(data.history)) return;
  total.value = data.history_total ?? data.history.length;
  if (page.value === 1 && !loading.value) history.value = data.history;
}
async function load(nextPage) {
  const id = ++request; loading.value = true;
  try {
    const data = await api(`/history?page=${nextPage}`);
    if (id !== request) return;
    history.value = data.history; page.value = data.page; total.value = data.total;
  } catch { if (id === request) emit('notice', t('历史加载失败，请重试', 'Could not load history')); }
  finally { if (id === request) loading.value = false; }
}
defineExpose({ receive });
onUnmounted(() => { request++; });
</script>
<style scoped>
.run-history { min-width:0;padding:16px;border:1px solid var(--border-main);border-radius:16px;background:var(--bg-card);align-self:start; }
header { display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px; }h2 { font-size:16px;margin:0; }
a { display:flex;justify-content:space-between;align-items:center;gap:10px;padding:11px 0;border-bottom:1px solid var(--border-main);text-decoration:none;color:inherit; }
a span { display:flex;flex-direction:column;gap:4px; }b { font-size:13px; }small,time,p { font-size:11px;color:var(--text-secondary); }time { text-align:right; }
nav { display:flex;gap:8px;align-items:center;justify-content:center;margin-top:14px;font-size:12px; }button { padding:4px 9px; }button:disabled { opacity:.35; }
.run-history-horizontal { position:relative;display:flex;flex-direction:column;padding:10px 12px; }
.run-history-horizontal header { flex-shrink:0;height:28px;margin-bottom:8px; }
.run-history-horizontal h2 { font-size:14px; }
.run-history-horizontal header > button { min-width:28px;min-height:28px;height:28px;padding:0; }
.run-history-horizontal .history-body { flex:1;min-height:0; }
.run-history-horizontal .history-list { display:grid;grid-template-columns:repeat(5,minmax(0,1fr));grid-template-rows:repeat(2,minmax(0,1fr));gap:6px 8px;height:100%; }
.run-history-horizontal .history-list > p { grid-column:1 / -1;grid-row:1 / -1;align-self:center;text-align:center;margin:0; }
.run-history-horizontal a { position:relative;display:block;min-width:0;padding:5px 8px;border:1px solid var(--border-main);border-radius:6px; }
.run-history-horizontal a:hover,.run-history-horizontal a:focus-visible { border-color:var(--accent);background:var(--bg-main); }
.run-history-horizontal a span { min-width:0;gap:3px; }
.run-history-horizontal b { overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;line-height:14px; }
.run-history-horizontal small,.run-history-horizontal time { font-size:10px;line-height:12px; }
.run-history-horizontal time { position:absolute;right:8px;bottom:5px;white-space:nowrap; }
.run-history-horizontal nav { position:absolute;right:48px;top:10px;height:28px;margin:0;gap:6px;font-size:11px; }
.run-history-horizontal nav button { min-width:24px;min-height:26px;padding:3px 6px; }
</style>
