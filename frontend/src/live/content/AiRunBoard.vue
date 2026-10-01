<template>
  <article class="ai-run-card" :class="{ leader, compact, primary }" :data-lane="slot.lane">
    <header>
      <div><span class="ai-name">{{ label }}</span><span v-if="leader" class="leader-badge">{{ t('最高分', 'LEADER') }}</span></div>
      <strong class="score">{{ format(slot.run?.score) }}</strong>
    </header>
    <div class="run-meta"><span :title="source">{{ slot.run?.ended_at ? t('已结束，等待本批', 'Finished · Waiting') : source === 'AI' ? t('AI 搜索', 'AI search') : source }}</span><span>#{{ format(slot.run?.seq) }} · {{ elapsed }}</span></div>
    <div ref="boardSpace" class="board-wrap" :class="{ selectable: compact }" :role="compact ? 'button' : undefined"
      :tabindex="compact ? 0 : undefined" :aria-label="compact ? t(`切换到 ${label}`, `Watch ${label}`) : undefined"
      @click="compact && emit('select', slot.lane)" @keydown.enter.prevent="compact && emit('select', slot.lane)"
      @keydown.space.prevent="compact && emit('select', slot.lane)">
      <div class="board-square" :style="squareStyle">
        <BaseBoard :frame="frame" />
        <div v-if="overlay" class="ai-overlay" role="status"><strong>{{ overlay }}</strong>
          <span v-if="slot.run?.ended_at">{{ t('等待其他选手结束', 'Waiting for the other players') }}</span>
        </div>
      </div>
    </div>
    <footer>
      <button type="button" @click="copy" :aria-label="t(`复制 ${label} 盘面`, `Copy ${label} board`)">{{ t('复制盘面', 'Copy board') }}</button>
      <button v-if="!primary" type="button" @click="emit('select', slot.lane)">{{ t('在主屏查看', 'Watch on main') }}</button>
    </footer>
  </article>
</template>
<script setup>
import { computed, ref, watch } from 'vue';
import BaseBoard from '../../components/BaseBoard.vue';
import { participantName, showEndNotice } from './participants.js';
import { useSurfaceBox } from '../roomSurfaceSize.js';
const props = defineProps({ slot: Object, frame: Object, source: String, lang: String, streamState: String,
  leader: Boolean, compact: Boolean, primary: Boolean, now: Number });
const emit = defineEmits(['select', 'notice']);
const boardSpace = ref(null), side = ref(0);
const refreshSize = useSurfaceBox(boardSpace, (width, height) => {
  side.value = Math.max(0, Math.min(width, height));
});
// Also handles layout switches in older engines without ResizeObserver.
watch(() => [props.compact, props.primary, props.lang], refreshSize, { flush: 'post' });
const squareStyle = computed(() => ({
  width: `${side.value}px`, height: `${side.value}px`,
  '--tile-label-small': `${side.value * .08}px`,
  '--tile-label-medium': `${side.value * .065}px`,
  '--tile-label-large': `${side.value * .05}px`,
  '--board-notice-size': `${Math.max(12, Math.min(24, side.value * .05))}px`,
}));
const t = (zh, en) => props.lang === 'zh' ? zh : en;
const label = computed(() => participantName(props.slot));
const format = n => Number(n || 0).toLocaleString();
const elapsed = computed(() => { const s = Math.floor((props.slot.run?.elapsed_ms || 0) / 1000);
  return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; });
const overlay = computed(() => {
  if (props.slot.run?.ended_at) return showEndNotice(props.slot.run, props.now) ? t('本局结束', 'Game over') : '';
  if (props.streamState === 'loading' || props.streamState === 'reconnecting') return t('正在连接', 'Connecting');
  if (props.streamState === 'paused') return t('直播已暂停', 'Stream paused');
  if (props.streamState === 'offline') return t('等待主播连接', 'Broadcaster offline');
  if (props.slot.status === 'recovering' || !props.slot.run) return t('AI 正在准备', 'Preparing AI');
  return '';
});
async function copy() {
  const code = (props.slot.run?.board || Array(16).fill(0)).map(v => v ? Math.min(15, Math.log2(v)).toString(16) : '0').join('');
  try { await navigator.clipboard.writeText(code); emit('notice', t('盘面已复制', 'Board copied')); }
  catch { emit('notice', code); }
}
</script>
<style scoped>
.ai-run-card { min-width:0;min-height:0;box-sizing:border-box;display:grid;grid-template-rows:32px 22px minmax(0,1fr) 32px;gap:6px;padding:14px; border:1px solid var(--border-main); border-radius:16px; background:var(--bg-card); }
.ai-run-card.leader { border-color:var(--accent); }
header { display:flex;justify-content:space-between;align-items:center;gap:8px;min-height:0;overflow:hidden;line-height:1.2; }
header > div { min-width:0;display:flex;align-items:center;overflow:hidden;white-space:nowrap; }
.ai-name { font-weight:800;font-size:17px;white-space:nowrap; }
.leader-badge { margin-left:8px;font-size:10px;color:var(--accent);font-weight:800;white-space:nowrap; }
.score { flex:none;font-size:24px;line-height:1;white-space:nowrap;font-variant-numeric:tabular-nums; }
.run-meta { display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:11px;line-height:1.2;color:var(--text-secondary);margin:0;overflow:hidden; }
.run-meta span:first-child { overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }
.run-meta span:last-child { flex-shrink:0; }
.board-wrap { position:relative;flex:1;min-height:0;width:100%; }
.board-square { position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);isolation:isolate; }
.board-square :deep(.board-stage) { position:relative;z-index:0;width:100%;height:100%;max-width:none; }
.board-square :deep(.board-stage)::before { display:none; }
.selectable { cursor:pointer; }
.selectable:focus-visible { outline:3px solid var(--accent);outline-offset:4px; }
.ai-overlay { position:absolute;inset:0;z-index:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;background:#0f172ab8;color:white;border-radius:12px;pointer-events:none; }
.ai-overlay strong { font-size:var(--board-notice-size,24px); }.ai-overlay span { font-size:12px; }
footer { display:flex;align-items:stretch;justify-content:space-between;gap:6px;margin:0;min-height:0;overflow:hidden; }
footer button { min-width:0;min-height:0;max-width:100%;font-size:11px;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:5px 8px; }
.compact { padding:10px;grid-template-rows:24px 16px minmax(0,1fr) 26px;gap:4px; }.compact .score { font-size:17px; }.compact .ai-name { font-size:13px; }
.compact .leader-badge { font-size:9px;margin-left:4px;overflow:hidden;text-overflow:ellipsis; }.compact .run-meta { font-size:10px;margin:0; }
.compact footer { margin:0; }.compact footer button { font-size:10px;padding:3px 5px; }
</style>
