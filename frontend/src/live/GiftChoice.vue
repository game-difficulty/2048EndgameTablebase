<template>
  <article ref="card" class="gift-choice" :class="{ highlighted: active }" @mouseenter="activate" @pointermove="!active && activate()" @mouseleave="leave" @focusin="activate" @focusout="leave">
    <button class="gift-art" :disabled="disabled" @click="send(1)" :aria-label="`${t('投喂 1 个','Send 1')} ${gift[lang]}`"><GiftIcon :id="gift.id" /></button>
    <strong>{{ gift[lang] }}</strong><small>{{ tokens(giftPrice(gift,1)) }} Token</small>
    <button class="feed-one" :disabled="disabled" @click="send(1)" @mouseenter="expand">{{ t('投喂 1 个','Send 1') }}</button>
    <Teleport to="body">
      <div v-if="expanded" class="live-gift-popover" :style="position" @mouseenter="hold" @mouseleave="leave" @focusin="hold" @focusout="leave" @keydown.esc="close">
        <header><GiftIcon :id="gift.id" /><strong>{{ gift[lang] }}</strong><button @click="close" :aria-label="t('关闭','Close')"><X :size="14" /></button></header>
        <div class="bulk-options"><button v-for="n in [10,100,1000]" :key="n" :disabled="disabled" :title="`${tokens(giftPrice(gift,n))} Token`" @click="send(n)">{{ n }}{{ t(' 个','') }}</button></div>
        <form @submit.prevent="send(Number(custom))"><label class="custom-quantity"><span class="sr-only">{{ t('自定义数量','Custom quantity') }}</span><BattleNumberInput v-model="custom" :min="1" :max="1000" :step="1" /></label><button :disabled="disabled || customCost === null" type="submit">{{ t('投喂','Send') }}</button></form>
      </div>
    </Teleport>
  </article>
</template>
<script setup>
import { computed, ref, onMounted, onUnmounted } from 'vue';
import { X } from '@lucide/vue';
import GiftIcon from './GiftIcon.vue';
import BattleNumberInput from '../features/battle/components/BattleNumberInput.vue';
import { giftPrice } from './giftPrice.js';
import { liveLayoutViewport } from './liveLayout.js';
const props = defineProps({gift:Object,lang:String,disabled:Boolean});
const emit = defineEmits(['send']);
const card = ref(null), active = ref(false), expanded = ref(false), custom = ref(1), position = ref({});
const t = (zh,en) => props.lang === 'zh' ? zh : en;
const tokens = n => ((n || 0)/1000).toLocaleString(undefined,{maximumFractionDigits:3});
const customCost = computed(() => giftPrice(props.gift, Number(custom.value)));
let enterTimer, leaveTimer;
function hold() { clearTimeout(leaveTimer); }
function expand() { hold(); const { rect, width }=liveLayoutViewport(card.value); position.value={left:`${Math.max(8,Math.min(width-268,rect.left))}px`,top:`${Math.max(8,rect.top-156)}px`}; expanded.value=true; }
function activate() { hold(); active.value=true; clearTimeout(enterTimer); enterTimer=setTimeout(expand,350); }
function leave() { clearTimeout(enterTimer); leaveTimer=setTimeout(close,160); }
function close() { clearTimeout(enterTimer); clearTimeout(leaveTimer); expanded.value=false; active.value=false; }
function send(n) { if (!props.disabled && giftPrice(props.gift,n) !== null) emit('send',props.gift.id,n); }
function outside(event) { if (!card.value?.contains(event.target) && !event.target.closest('.live-gift-popover')) close(); }
function onScroll(){if(expanded.value)close();}
onMounted(()=>{ window.addEventListener('resize',close); window.addEventListener('scroll',onScroll,true); document.addEventListener('pointerdown',outside); });
onUnmounted(()=>{close();window.removeEventListener('resize',close);window.removeEventListener('scroll',onScroll,true);document.removeEventListener('pointerdown',outside);});
</script>
<style scoped>
.gift-choice { position:relative;min-width:0;height:112px;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding:5px 3px;border-radius:6px;border:1px solid transparent; }
.gift-choice.highlighted { background:var(--bg-card);border-color:var(--border-main);box-shadow:0 3px 16px #0004;z-index:2; }
.gift-art { display:flex;align-items:center;justify-content:center;width:52px;height:52px;padding:0;border:0;background:transparent;cursor:pointer; }
.gift-art :deep(.gift-icon) { width:48px;height:48px; }
.gift-art:disabled { opacity:.5;cursor:default; }
strong { font-size:11px;line-height:14px;text-align:center;min-height:28px;display:flex;align-items:center;justify-content:center;overflow-wrap:anywhere; }
small { font-size:10px;color:var(--text-secondary);white-space:nowrap; }
.feed-one { position:absolute;bottom:0;left:0;width:100%;height:26px;background:var(--accent);color:var(--text-on-accent,#10202f);border:0;border-radius:0 0 5px 5px;visibility:hidden;font-size:11px;cursor:pointer; }
.highlighted .feed-one { visibility:visible; }.feed-one:disabled { opacity:.5;cursor:default; }
</style>
<style>
.live-gift-popover { position:fixed;z-index:90;width:260px;padding:12px;background:var(--bg-main);color:var(--text-main);border:1px solid var(--border-main);border-radius:7px;box-shadow:0 6px 24px #0005; }
.live-gift-popover header { display:flex;align-items:center;gap:8px;margin-bottom:10px; }.live-gift-popover header strong { font-size:13px;flex:1; }.live-gift-popover header .gift-icon { width:28px;height:28px; }
.live-gift-popover button { border:1px solid var(--border-main);background:var(--bg-card);color:var(--text-main);border-radius:4px;padding:5px 8px;cursor:pointer;font-size:12px; }
.live-gift-popover button:disabled { opacity:.45;cursor:default; }.live-gift-popover .bulk-options { display:flex;gap:6px; }.live-gift-popover .bulk-options button { flex:1; }
.live-gift-popover form { display:flex;gap:8px;margin:8px 0 0; }.live-gift-popover .custom-quantity { flex:1;min-width:0; }
.live-gift-popover .battle-number-input { background:var(--bg-input); }
.live-gift-popover .battle-number-steppers button { padding:0;min-height:0;background:transparent;color:var(--text-secondary);border:0;border-radius:0; }
.live-gift-popover .battle-number-steppers button+button { border-top:1px solid var(--border-main); }
.live-gift-popover .battle-number-steppers button:hover { background:var(--btn-bg);color:var(--text-on-accent,#fff); }
.live-gift-popover form button { flex-shrink:0;background:var(--accent);color:var(--text-on-accent,#10202f); }.live-gift-popover small { font-size:11px;color:var(--text-secondary); }
</style>
