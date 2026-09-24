<template>
  <RoomActivityEntry v-if="claimable && !dismissed.includes(active.id)" :target="dockTarget" kind="red"
    :label="t('抢红包','Red envelope')" :caption="remaining + 's'" :dismiss-label="t('收起红包','Dismiss envelope')"
    @dismiss="dismiss(active.id)" @open="open(active.id)"><img :src="art(false)" alt="" /></RoomActivityEntry>
  <div v-if="opened" ref="mask" class="red-mask" :style="bounds" @click.self="close">
    <section ref="dialog" class="red-dialog" role="dialog" aria-modal="true" aria-labelledby="red-title" tabindex="-1">
      <header><h2 id="red-title">{{ composing ? t('发红包', 'Send a red envelope') : t('直播间红包', 'Room red envelope') }}</h2><button :aria-label="t('关闭','Close')" @click="close"><X :size="20" /></button></header>
      <form v-if="composing" @submit.prevent="send">
        <img class="red-art" :src="art(false)" alt="" />
        <label>{{ t('总额（常驻额度）', 'Total (permanent balance)') }}<input v-model.number="amount" type="number" min="1000" max="50000" step="1" required :disabled="busy || !!pending" /></label>
        <label>{{ t('红包个数', 'Number of shares') }}<input v-model.number="count" type="number" min="5" max="20" step="1" required :disabled="busy || !!pending" /></label>
        <fieldset :disabled="busy || !!pending"><legend>{{ t('分配方式', 'Split') }}</legend><label><input v-model="mode" type="radio" value="random" />{{ t('拼手气', 'Random') }}</label><label><input v-model="mode" type="radio" value="equal" />{{ t('等额', 'Equal') }}</label></fieldset>
        <p class="red-note">{{ t('5–20 个，合计 1,000–50,000 Token。仅使用常驻额度，展示 60 秒后未领取部分自动退回。发红包的人也可参与，每人限领一次，先到先得。', '5–20 shares, 1,000–50,000 permanent Tokens. Unclaimed shares return after 60 seconds on display. The sender can also claim: one share per person, first come, first served.') }}</p>
        <p v-if="mode === 'equal'" class="red-note">{{ t('每份', 'Each') }} {{ Math.floor(amount / count) || 0 }} Token<span v-if="amount % count"> · {{ t('余数', 'Remainder') }} {{ amount % count }} Token {{ t('不退还', 'is not returned') }}</span></p>
        <p v-if="paid != null" class="red-note">{{ t('常驻额度', 'Permanent balance') }}: {{ format(paid) }} Token</p>
        <button class="red-primary" type="submit" :disabled="busy || !connected">{{ busy ? t('正在确认…', 'Confirming…') : pending ? t('核对并重试（不会重复扣费）', 'Check and retry (no duplicate charge)') : t('发放红包', 'Send envelope') }}</button>
      </form>
      <div v-else-if="selected" class="red-result">
        <p class="red-sender">{{ selected.actor.name }} {{ t('的红包', 'sent an envelope') }}</p>
        <img :key="selected.id + ':' + (selected.award || 0)" class="red-art" :src="art(!reduced && !selected.award && selected.status === 'active')" alt="" />
        <template v-if="selected.award > 0"><strong>{{ format(selected.award) }} <small>Token</small></strong><p>{{ t('已加入你的常驻额度', 'Added to your permanent balance') }}</p></template>
        <template v-else>
          <strong>{{ format(selected.amount) }} <small>Token</small></strong>
          <p>{{ selected.count }} {{ t('个红包', 'shares') }} · {{ selected.mode === 'equal' ? t('等额', 'Equal') : t('拼手气', 'Random') }}</p>
          <button v-if="selected.status === 'active'" class="red-open" :aria-busy="busy" :disabled="busy || !connected" @click="claim">{{ busy ? t('正在打开…', 'Opening…') : t('开', 'Open') }}</button>
        </template>
        <p v-if="!selected.award && ['active','queued'].includes(selected.status)">{{ t('发红包的人也可参与，每人限领一次，先到先得。', 'The sender can also claim: one share per person, first come, first served.') }}</p>
        <p v-if="selected.status === 'queued'">{{ t('已排队，轮到后开始领取', 'Queued. Claims open when this envelope is displayed.') }}</p>
        <p v-else-if="selected.status === 'exhausted' || (selected.status === 'closed' && selected.claimed === selected.count)">{{ t('红包已抢光', 'All shares have been claimed') }}</p>
        <p v-else-if="selected.status === 'expired' || selected.status === 'closed'">{{ t('红包已结束，未领取部分自动退回', 'This envelope ended. Unclaimed shares are returned automatically.') }}</p>
        <small>{{ t('已领取', 'Claimed') }} {{ selected.claimed }}/{{ selected.count }}<template v-if="selected.sender_id === user?.id && selected.refunded"> · {{ t('已退回', 'Returned') }} {{ format(selected.refunded) }} Token</template></small>
      </div>
      <p v-else>{{ t('正在加载…', 'Loading…') }}</p>
      <p v-if="error" class="red-error" role="alert">{{ error }}<button v-if="!composing && !selected" @click="refresh">{{ t('重试','Retry') }}</button></p>
    </section>
  </div>
</template>

<script setup>
import { useActivities } from './context.js';
const { url, room } = useActivities();
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue';
import { X } from '@lucide/vue';
import RoomActivityEntry from './RoomActivityEntry.vue';
import { useActivityDismissals } from './useActivityDismissals.js';
import { envelopeIsClaimable } from './activityAvailability.js';
import { layoutViewport as liveLayoutViewport } from '../../utils/layoutViewport.js';
const props = defineProps({ state: Object, user: Object, connected: Boolean, lang: String, dockTarget: String });
const emit = defineEmits(['login', 'balance', 'open']);
const t = (zh,en) => props.lang === 'zh' ? zh : en;
const format = n => Number(n).toLocaleString(props.lang === 'zh' ? 'zh-CN' : 'en-US', { maximumFractionDigits: 1 });
const art = motion => `/live-gifts/red-envelope${motion ? '-motion' : ''}.webp`;
const active = computed(() => props.state?.active);
const now = ref(Date.now()/1000), offset = ref(0);
const remaining = computed(() => Math.max(0, Math.ceil((active.value?.expires_at || now.value) - now.value)));
const { dismissed, dismiss } = useActivityDismissals('red');
const claimedIds = ref(new Set());
const claimable = computed(() => envelopeIsClaimable(active.value, now.value, props.connected, claimedIds.value.has(active.value?.id)));
const opened = ref(false), composing = ref(false), selected = ref(null), selectedId = ref('');
const amount = ref(1000), count = ref(5), mode = ref('random'), paid = ref(null), pending = ref(null);
const busy = ref(false), error = ref(''), dialog = ref(null), mask = ref(null), bounds = ref({}), reduced = ref(false);
let timer, media, restoreFocus, generation = 0, requestVersion = 0, lastRefresh = 0;
const key = () => room.id === 'ai-classic' ? `live:red-pending:${props.user?.id}` : `live:red-pending:${room.id}:${props.user?.id}`;
function persist() { try { if(pending.value) sessionStorage.setItem(key(),JSON.stringify(pending.value)); else sessionStorage.removeItem(key()); } catch {} }
function loadPending() {
  pending.value=null;
  try { const data=JSON.parse(sessionStorage.getItem(key()) || 'null'); if(data?.request_id) { pending.value=data;amount.value=data.amount;count.value=data.count;mode.value=data.mode; } } catch {}
}
async function api(path='', body) {
  const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),12000);
  try {
    const response=await fetch(url('/red-envelopes'+path),{credentials:'same-origin',cache:'no-store',signal:controller.signal,...(body !== undefined ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {})});
    const data=await response.json();
    if(!response.ok)throw Object.assign(new Error('red_failed'),{status:response.status,detail:data.detail});
    return data;
  } finally { clearTimeout(timeout); }
}
function fail(e) {
  const messages={red_paid_balance:['常驻额度不足，周度额度不能用于红包。','Not enough permanent Tokens. Weekly allowances cannot fund envelopes.'],red_cooldown:['每 15 秒可发一个红包，请稍后再发。','You can send one envelope every 15 seconds. Try again shortly.'],red_not_present:['直播间正在连接，请稍后再试。','The room is reconnecting. Please try again.'],red_invalid:['请检查总额和红包个数。','Check the amount and number of shares.'],red_not_found:['这个红包已无法查看。','This envelope is no longer available.']};
  error.value=e.status===401 ? t('请重新登录。','Please sign in again.') : t(...(messages[e.detail] || ['暂未确认，请重试；不会重复扣费或领取。','Not yet confirmed. Retry safely without duplicate charges or claims.']));
}
function place() {
  const stage=mask.value?.closest('.stage-column');if(!stage)return;
  const {rect:r,width,height}=liveLayoutViewport(stage),top=Math.max(8,r.top),bottom=Math.min(height-8,r.bottom);
  bounds.value={left:`${Math.max(8,r.left)}px`,width:`${Math.min(r.width,width-16)}px`,top:`${top}px`,height:`${Math.max(100,bottom-top)}px`};
}
async function show() { emit('open');restoreFocus=document.activeElement;opened.value=true;await nextTick();place();dialog.value?.focus(); }
function close() { if(!opened.value)return;opened.value=false;restoreFocus?.focus?.({preventScroll:true}); }
async function compose() {
  if(busy.value)return;
  if(!props.user){emit('login');return;}
  composing.value=true;error.value='';selected.value=null;requestVersion++;loadPending();await show();
  const epoch=generation;
  try { const response=await fetch('/api/auth/me',{credentials:'same-origin',cache:'no-store'});const data=await response.json();if(epoch===generation)paid.value=data.user?.token_balance?.paid ?? data.token_balance?.paid ?? null; } catch {}
}
async function open(id) {
  if(!props.user){emit('login');return;}
  if(busy.value)return;
  composing.value=false;selectedId.value=id;selected.value=active.value?.id===id ? {...active.value} : null;error.value='';await show();await refresh();
}
async function refresh() {
  if(!selectedId.value || composing.value || busy.value)return;
  const version=++requestVersion, epoch=generation;lastRefresh=Date.now();
  try { const data=await api('/'+encodeURIComponent(selectedId.value));if(version===requestVersion && epoch===generation)selected.value=data; }
  catch(e){if(version===requestVersion && epoch===generation)fail(e);}
}
async function send() {
  if(busy.value || !props.user || !props.connected)return;
  const epoch=generation;
  if(!pending.value){pending.value={request_id:crypto.randomUUID(),amount:amount.value,count:count.value,mode:mode.value};persist();}
  busy.value=true;error.value='';
  try {
    const result=await api('',pending.value);if(epoch!==generation)return;
    pending.value=null;persist();selectedId.value=result.id;selected.value=result;composing.value=false;emit('balance');
  } catch(e) {
    if(epoch===generation){
      // Only validation failures after server idempotency checks prove no debit.
      // Authentication, presence and generic rate limits may precede that check.
      if(['red_invalid','red_paid_balance','red_cooldown'].includes(e.detail)){pending.value=null;persist();}
      fail(e);
    }
  }
  finally { if(epoch===generation)busy.value=false; }
}
async function claim() {
  if(busy.value || !selected.value)return;
  const epoch=generation, id=selected.value.id;requestVersion++;busy.value=true;error.value='';
  try { const result=await api('/'+encodeURIComponent(id)+'/claim',{});if(epoch===generation){selected.value=result;if(result.award)emit('balance');} }
  catch(e){if(epoch===generation)fail(e);}
  finally{if(epoch===generation)busy.value=false;}
}
watch(()=>props.state,(data,previous)=>{
  if(data?.server_time)offset.value=data.server_time-Date.now()/1000;
  now.value=Date.now()/1000+offset.value;
  if(previous?.active && previous.active.sender_id===props.user?.id && previous.active.id!==data?.active?.id)emit('balance');
  if(opened.value && !composing.value)refresh();
},{immediate:true});
watch(()=>props.user?.id,()=>{generation++;requestVersion++;busy.value=false;opened.value=false;paid.value=null;selected.value=null;claimedIds.value=new Set();loadPending();});
watch(selected,value=>{if(value?.award>0)claimedIds.value.add(value.id);});
// Restore personal eligibility once per envelope/reconnect, including after reload.
watch([()=>active.value?.id,()=>props.user?.id,()=>props.connected],async()=>{
  if(!props.user || !claimable.value)return;
  const epoch=generation,id=active.value.id;
  try { const value=await api('/'+encodeURIComponent(id));if(epoch===generation && value.award>0)claimedIds.value.add(id); } catch {}
},{immediate:true});
watch(()=>props.connected,value=>{if(value && opened.value)refresh();});
function keydown(event) {
  if(!opened.value)return;
  if(event.key==='Escape'){event.preventDefault();event.stopPropagation();close();}
  if(event.key==='Tab'){
    const items=[...dialog.value.querySelectorAll('button:not(:disabled),input:not(:disabled),[tabindex="0"]')],first=items[0],last=items.at(-1);
    if(event.shiftKey && (document.activeElement===first || document.activeElement===dialog.value)){event.preventDefault();last?.focus();}
    else if(!event.shiftKey && (document.activeElement===last || !dialog.value.contains(document.activeElement))){event.preventDefault();first?.focus();}
  }
}
const motionChange=()=>{reduced.value=media.matches;};
onMounted(()=>{
  media=matchMedia('(prefers-reduced-motion: reduce)');motionChange();media.addEventListener('change',motionChange);
  timer=setInterval(()=>{now.value=Date.now()/1000+offset.value;if(opened.value && !composing.value && props.connected && Date.now()-lastRefresh>5000)refresh();},500);
  document.addEventListener('keydown',keydown,true);window.addEventListener('resize',place);window.addEventListener('scroll',place,true);
});
onUnmounted(()=>{generation++;clearInterval(timer);media?.removeEventListener('change',motionChange);document.removeEventListener('keydown',keydown,true);window.removeEventListener('resize',place);window.removeEventListener('scroll',place,true);});
defineExpose({compose,open,close});
</script>

<style scoped>
.red-mask{position:fixed;z-index:95;box-sizing:border-box;padding:14px;background:#07111caa;display:flex;justify-content:center;align-items:center}.red-dialog{width:410px;max-width:100%;max-height:100%;box-sizing:border-box;overflow:auto;background:var(--bg-main);color:var(--text-main);border:1px solid #dd7052;border-radius:8px;padding:20px;box-shadow:0 20px 60px #0005;outline:none}.red-dialog header{display:flex;align-items:center;justify-content:space-between}.red-dialog h2{font-size:21px;margin:0}.red-dialog header button{background:transparent;color:var(--text-main);border:0;padding:5px}.red-dialog form{display:grid;gap:10px}.red-art{width:138px;height:138px;object-fit:contain;display:block;margin:0 auto}.red-dialog label{font-size:14px}.red-dialog input[type=number]{display:block;box-sizing:border-box;width:100%;padding:9px;margin-top:6px;border:1px solid var(--border-main);border-radius:4px;color:var(--text-main);background:var(--bg-input);color-scheme:dark}.red-dialog fieldset{display:flex;gap:24px;border:0;padding:8px 0;margin:0}.red-dialog legend{font-size:13px;color:var(--text-secondary)}.red-dialog fieldset label{display:flex;align-items:center;gap:6px}.red-dialog input[type=radio]{accent-color:#d95745}.red-note{font-size:12px;line-height:1.5;margin:0;color:var(--text-secondary)}.red-primary,.red-open{border:1px solid #ef9570;border-radius:5px;background:#b63126;color:#fff4df;font-weight:700;min-height:42px;padding:10px;cursor:pointer}.red-primary:disabled,.red-open:disabled{opacity:.55;cursor:default}.red-result{text-align:center}.red-result strong{display:block;font-size:30px;color:#eeac67}.red-result strong small{font-size:15px}.red-result p{font-size:14px;line-height:1.5}.red-result>small{font-size:12px;color:var(--text-secondary)}.red-sender{overflow-wrap:anywhere}.red-open{display:flex;align-items:center;justify-content:center;box-sizing:border-box;width:90px;height:52px;margin:12px auto;padding:10px 6px;font-size:23px;line-height:1;white-space:nowrap}.red-open[aria-busy="true"]{font-size:13px}.red-error{font-size:13px;line-height:1.5;color:var(--text-main)}
:global([data-theme=light]) .red-dialog input[type=number]{color-scheme:light}
</style>
