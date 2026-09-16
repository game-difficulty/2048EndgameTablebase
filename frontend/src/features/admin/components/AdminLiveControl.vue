<template>
  <section class="live-control">
    <div>
      <h2>{{ t('title') }}</h2>
      <p role="status">{{ statusText }}</p>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
    </div>
    <button type="button" class="action-btn-small" :disabled="busy" :aria-label="t('refresh')" @click="refresh">
      <RefreshCw :size="16" />
    </button>
    <button type="button" class="action-btn-small" :disabled="!state || busy || (state.connected && !state.supported)" @click="toggle">
      <Pause v-if="state?.enabled" :size="16" /><Play v-else :size="16" />
      {{ state?.enabled ? t('pause') : t('start') }}
    </button>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { Pause, Play, RefreshCw } from '@lucide/vue';
import { useI18n } from 'vue-i18n';
import { adminClient } from '../../../services/admin/adminClient';

const props = defineProps({ active: Boolean });
const { t } = useI18n({ useScope: 'local', messages: {
  zh: { title: 'AI 直播', refresh: '刷新状态', pause: '暂停直播', start: '开启直播', loading: '正在读取状态',
    live: '直播中', paused: '已暂停，开启后继续当前对局', pausing: '正在暂停直播', starting: '正在开启直播',
    offline: '等待本机连接', saved: '已设置暂停；本机连接后保持暂停', update: '请先更新本地直播运行器', failed: '操作未完成，请刷新状态后重试' },
  en: { title: 'AI livestream', refresh: 'Refresh status', pause: 'Pause stream', start: 'Start stream', loading: 'Loading status',
    live: 'Live', paused: 'Paused. Starting resumes this game.', pausing: 'Pausing stream', starting: 'Starting stream',
    offline: 'Waiting for the local broadcaster', saved: 'Pause saved. The broadcaster will stay paused on reconnect.',
    update: 'Update the local broadcaster first', failed: 'Request failed. Refresh the status and retry.' },
} });
const state = ref(null), busy = ref(false), error = ref('');
let timer, generation = 0, disposed = false;
const statusText = computed(() => {
  const value = state.value;
  if (!value) return t('loading');
  if (!value.connected) return t(value.enabled ? 'offline' : 'saved');
  if (!value.supported) return t('update');
  return t(value.applied ? (value.enabled ? 'live' : 'paused') : (value.enabled ? 'starting' : 'pausing'));
});
function schedule() {
  clearTimeout(timer);
  if (!disposed && props.active) timer = setTimeout(refresh, state.value?.connected && !state.value?.applied ? 1000 : 5000);
}
async function refresh() {
  if (busy.value) return;
  const current = ++generation;
  try {
    const result = await adminClient.liveStatus();
    if (current === generation) { state.value = result; error.value = ''; }
  } catch {
    if (current === generation) error.value = t('failed');
  } finally { if (current === generation) schedule(); }
}
async function toggle() {
  busy.value = true;
  ++generation;
  clearTimeout(timer);
  error.value = '';
  try { state.value = await adminClient.setLiveEnabled(!state.value.enabled); }
  catch { error.value = t('failed'); }
  finally { busy.value = false; schedule(); }
}
watch(() => props.active, active => {
  clearTimeout(timer);
  ++generation;
  if (active) refresh();
}, { immediate: true });
onBeforeUnmount(() => { disposed = true; ++generation; clearTimeout(timer); });
</script>

<style scoped>
.live-control { display:flex; align-items:center; gap:12px; padding:16px 0; border-bottom:1px solid var(--border-main); }
.live-control > div { flex:1; min-width:0; }
h2 { font-size:18px; font-weight:800; }
p { margin-top:4px; font-size:14px; color:var(--text-secondary); }
.error { color:#ef4444; }
button { width:auto; flex-shrink:0; display:flex; align-items:center; gap:6px; letter-spacing:0; }
</style>
