<template>
  <slot :bag="visible[0]" :open="openBag" :caption="visible[0] ? caption(visible[0]) : ''" />
  <div v-if="floating" class="lucky-floating">
    <button class="lucky-dismiss" @click="dismiss(floating.id)" :aria-label="t('收起福袋', 'Dismiss lucky bag')"><X :size="14" /></button>
    <button class="lucky-entry" @click="openBag(floating)"><LuckyBagIcon /><b>{{ floating.milestone === 65536 ? '65k' : '32K' }} {{ t('福袋','Lucky bag') }}</b><span>{{ caption(floating) }}</span></button>
  </div>
  <div v-if="opened" ref="mask" class="lucky-mask" :style="bounds" @click.self="close">
    <section ref="dialog" class="lucky-dialog" role="dialog" aria-modal="true" aria-labelledby="lucky-title" tabindex="-1">
      <header><span class="lucky-eyebrow">{{ selected?.milestone === 65536 ? '65k' : '32K' }} MILESTONE</span><button @click="close" :aria-label="t('关闭','Close')"><X :size="20" /></button></header>
      <h2 id="lucky-title">{{ t('合出惊喜，一起分享','A milestone worth sharing') }}</h2>
      <nav v-if="visible.length > 1" class="lucky-tabs" :aria-label="t('选择福袋','Choose a lucky bag')"><button v-for="bag in visible" :key="bag.id" :aria-pressed="bag.id===selectedId" @click="select(bag)">{{ bag.milestone===65536?'65k':'32K' }} · {{ caption(bag) }}</button></nav>
      <template v-if="selected">
        <div class="lucky-prize"><LuckyBagIcon /><strong>{{ format(selected.pool) }} <small>Token</small></strong><span>{{ t('福袋总额','Prize pool') }}</span></div>
        <div class="lucky-countdown"><span>{{ selected.drawn_at != null ? t('已开奖','Draw complete') : t('距离开奖','Draw in') }}</span><b>{{ caption(selected) }}</b></div>
        <div class="lucky-rules">
          <p><UserRound :size="16" /><span>{{ t('登录即可免费参与','Free to enter when signed in') }}</span></p>
          <p><Gift :size="16" /><span>{{ t('最多 10 人中奖，每人','Up to 10 winners, each receiving') }} {{ format(selected.minimum) }}–{{ format(selected.maximum) }} Token</span></p>
          <p><Radio :size="16" /><span>{{ t('开奖时请留在直播间，奖励自动到账','Stay in the room for the draw. Prizes are credited automatically.') }}</span></p>
        </div>
        <template v-if="selected.drawn_at != null">
          <div class="lucky-result" role="status">
            <template v-if="user && !selected.resultKnown">{{ t('正在查询开奖结果…','Checking your result…') }}<button @click="refresh">{{ t('刷新','Refresh') }}</button></template>
            <template v-else-if="selected.award > 0"><b>{{ t('恭喜！获得','You won') }} {{ format(selected.award) }} Token</b><p>{{ t('已加入你的充值额度','Added to your paid Token balance') }}</p></template>
            <template v-else-if="selected.joined && !selected.present">{{ t('开奖时你已离开直播间，未参与本次抽奖','You were away from the room at the draw.') }}</template>
            <template v-else-if="selected.joined">{{ t('这次未中奖，下个福袋再见！','Not this time. See you at the next lucky bag!') }}</template>
            <template v-else>{{ t('本次福袋已开奖','This lucky bag has been drawn') }}</template>
          </div>
          <small class="lucky-distributed">{{ selected.winners }} {{ t('人中奖，共送出','winners · Awarded') }} {{ format(selected.distributed) }} Token</small>
        </template>
        <button v-else class="lucky-join" :disabled="busy || (user && selected.joined) || now >= selected.draw_at || !connected" @click="join">
          {{ !user ? t('登录参与','Sign in to join') : busy ? t('参与中…','Joining…') : selected.joined ? t('已参与，等待开奖','Joined · Waiting for the draw') : t('免费参与','Join for free') }}
        </button>
        <small v-if="selected.drawn_at == null" class="lucky-distributed">{{ selected.participants }} {{ t('人已参与','joined') }}<template v-if="!connected"> · {{ t('正在重新连接','Reconnecting') }}</template></small>
      </template>
      <p v-else>{{ t('本次福袋已结束','This lucky bag has ended') }}</p>
      <p v-if="error" class="lucky-error" role="alert">{{ error }}</p>
    </section>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue';
import { X, Gift, Radio, UserRound } from '@lucide/vue';
import LuckyBagIcon from './LuckyBagIcon.vue';
import { bagCaption, mergeBags, visibleBags } from './luckyBagState.js';
import { liveLayoutViewport } from './liveLayout.js';
const props = defineProps({ state: Object, user: Object, connected: Boolean, lang: String });
const emit = defineEmits(['login', 'balance']);
const t = (zh,en) => props.lang === 'zh' ? zh : en;
const bags = ref([]), now = ref(Date.now()/1000), dismissed = ref([]), opened = ref(false), selectedId = ref('');
const busy = ref(false), error = ref(''), dialog = ref(null), mask = ref(null), bounds = ref({});
let offset=0, revision=0, timer, request, generation=0, lastRefresh=0, restoreFocus;
const visible = computed(() => visibleBags(bags.value, now.value));
const floating = computed(() => visible.value.find(bag => !dismissed.value.includes(bag.id)));
const selected = computed(() => bags.value.find(bag => bag.id===selectedId.value));
const caption = bag => bagCaption(bag,now.value,props.lang);
const format = value => Number(value || 0).toLocaleString(props.lang==='zh'?'zh-CN':'en-US');
function install(data, personal=false) {
  if (!data || !Array.isArray(data.bags) || data.server_time < revision) return;
  revision=data.server_time; offset=data.server_time-Date.now()/1000; now.value=Date.now()/1000+offset;
  const won = personal && data.bags.some(bag => bag.award>0 && !bags.value.find(old=>old.id===bag.id && old.award>0));
  bags.value=mergeBags(bags.value,data.bags,personal);
  if(won)emit('balance');
}
async function api(path, post=false) {
  const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),10000);
  try {
    const response=await fetch(`/api/live/lucky-bags${path}`,{credentials:'same-origin',cache:'no-store',signal:controller.signal,...(post?{method:'POST'}:{})});
    const data=await response.json();
    if(!response.ok)throw Object.assign(new Error('request_failed'),{status:response.status,detail:data.detail});
    return data;
  } finally { clearTimeout(timeout); }
}
async function refresh() {
  if(request)return request;
  const current=generation;
  lastRefresh=Date.now();
  const pending=api('').then(data=>{if(current===generation)install(data,true);}).catch(()=>{
    if(opened.value && current===generation)error.value=t('暂时无法查询，请稍后重试。','Could not check the result. Please try again.');
  }).finally(()=>{if(request===pending)request=null;});
  request=pending;
  return pending;
}
watch(()=>props.state,data=>{
  const needsResult=data?.bags?.some(bag=>bag.drawn_at!=null && !bags.value.find(old=>old.id===bag.id && old.resultKnown && old.drawn_at===bag.drawn_at));
  install(data);
  if(props.user && needsResult)refresh();
},{immediate:true});
watch(()=>props.user?.id,()=>{generation++; request=null; bags.value=mergeBags([],bags.value); error.value=''; refresh();});
watch(()=>props.connected,value=>{if(value)refresh();});
function dismiss(id) {
  dismissed.value=[...dismissed.value.filter(value=>value!==id),id].slice(-30);
  try{sessionStorage.setItem('live:lucky-dismissed:v1',JSON.stringify(dismissed.value));}catch{}
}
function place() {
  const stage=mask.value?.closest('.stage-column');
  if(!stage)return;
  const { rect:r, width, height }=liveLayoutViewport(stage);
  const top=Math.max(8,r.top), bottom=Math.min(height-8,r.bottom);
  bounds.value=bottom-top<Math.min(640,height-16) ? {inset:'8px'} : {left:`${Math.max(8,r.left)}px`,width:`${Math.min(r.width,width-16)}px`,top:`${top}px`,height:`${bottom-top}px`};
}
function select(bag){selectedId.value=bag.id;error.value='';}
async function openBag(bag=visible.value[0]) {
  if(!bag)return;
  if(!props.user){emit('login');return;}
  select(bag);restoreFocus=document.activeElement;opened.value=true;
  await nextTick();place();dialog.value?.focus();refresh();
}
function close(){opened.value=false;error.value='';restoreFocus?.focus?.({preventScroll:true});}
async function join() {
  if(!props.user){close();emit('login');return;}
  if(busy.value || !selected.value)return;
  const current=generation, id=selected.value.id;
  busy.value=true;error.value='';
  try{const data=await api(`/${encodeURIComponent(id)}/join`,true);if(current===generation)install(data,true);}
  catch(e){
    if(current!==generation)return;
    if(e.status===401){close();emit('login');}
    else if(e.detail==='lucky_bag_closed'){error.value=t('报名已结束，正在开奖。','Entries have closed. The draw is starting.');refresh();}
    else error.value=e.detail==='lucky_bag_not_present' ? t('直播间正在连接，请稍后再试。','The room is reconnecting. Please try again shortly.') : t('未能确认参与，请点击重试。','Could not confirm your entry. Please try again.');
  }finally{busy.value=false;}
}
function keydown(event) {
  if(!opened.value)return;
  if(event.key==='Escape'){event.preventDefault();event.stopPropagation();close();}
  if(event.key==='Tab'){
    const items=[...dialog.value.querySelectorAll('button:not(:disabled),[tabindex="0"]')];
    const first=items[0],last=items.at(-1);
    if(event.shiftKey && (document.activeElement===first || document.activeElement===dialog.value)){event.preventDefault();last?.focus();}
    else if(!event.shiftKey && (document.activeElement===last || !dialog.value.contains(document.activeElement))){event.preventDefault();first?.focus();}
  }
}
onMounted(()=>{
  try{const stored=JSON.parse(sessionStorage.getItem('live:lucky-dismissed:v1')||'[]');if(Array.isArray(stored))dismissed.value=stored.filter(v=>typeof v==='string').slice(-30);}catch{}
  refresh();
  timer=setInterval(()=>{
    now.value=Date.now()/1000+offset;
    if(props.connected && Date.now()-lastRefresh>5000 && bags.value.some(bag=>(bag.drawn_at==null && bag.draw_at<=now.value)||(props.user && bag.drawn_at!=null && !bag.resultKnown)))refresh();
  },500);
  document.addEventListener('keydown',keydown,true);window.addEventListener('resize',place);window.addEventListener('scroll',place,true);
});
onUnmounted(()=>{generation++;clearInterval(timer);document.removeEventListener('keydown',keydown,true);window.removeEventListener('resize',place);window.removeEventListener('scroll',place,true);});
</script>

<style scoped>
.lucky-floating { position:absolute;top:165px;left:8px;z-index:46;width:96px;text-align:center; }
.lucky-entry { display:flex;flex-direction:column;align-items:center;gap:3px;width:100%;padding:8px 4px;border:1px solid #efbf67;background:var(--bg-main);border-radius:8px;box-shadow:0 6px 20px #0003; }
.lucky-entry b { color:var(--text-main);font-size:11px; }.lucky-entry span { font:700 13px ui-monospace,monospace;color:var(--text-main); }
.lucky-dismiss { position:absolute;right:-7px;top:-7px;z-index:1;display:grid;place-items:center;padding:3px;border-radius:50%;background:var(--bg-main);border:1px solid var(--border-main);color:var(--text-main); }
.lucky-mask { position:fixed;z-index:90;box-sizing:border-box;padding:16px;display:flex;align-items:center;justify-content:center;background:#07111caa;border-radius:8px; }
.lucky-dialog { box-sizing:border-box;width:420px;max-width:100%;max-height:100%;overflow:auto;background:var(--bg-main);color:var(--text-main);border:1px solid #d2a14b;border-radius:8px;padding:22px;box-shadow:0 20px 64px #0005;outline:none; }
.lucky-dialog header { display:flex;align-items:center;justify-content:space-between;gap:10px; }.lucky-eyebrow { font-size:11px;color:var(--text-secondary);font-weight:700; }
.lucky-dialog header button { padding:4px;display:grid;place-items:center;background:transparent;border:0;color:var(--text-secondary); }
.lucky-dialog h2 { margin:8px 0 14px;font-size:21px;line-height:1.35; }
.lucky-tabs { display:flex;gap:6px;overflow:auto;padding:0 0 8px; }.lucky-tabs button { flex-shrink:0;padding:6px 9px;font-size:11px;background:var(--bg-input);color:var(--text-main);border:1px solid var(--border-main);border-radius:4px; }.lucky-tabs button[aria-pressed=true] { border-color:#d4a345; }
.lucky-prize { display:flex;flex-direction:column;align-items:center;gap:5px;padding:4px 0 18px; }.lucky-prize svg { width:85px;height:93px; }.lucky-prize strong { font-size:30px;color:var(--text-main);font-variant-numeric:tabular-nums; }.lucky-prize small { font-size:15px; }.lucky-prize>span { font-size:12px;color:var(--text-secondary); }
.lucky-countdown { display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 0;border-top:1px solid var(--border-main);border-bottom:1px solid var(--border-main);font-size:13px; }.lucky-countdown b { font:700 23px ui-monospace,monospace;color:var(--text-main); }
.lucky-rules { margin:14px 0;color:var(--text-secondary);font-size:12px;line-height:1.6; }.lucky-rules p { display:flex;gap:9px;align-items:flex-start;margin:8px 0; }.lucky-rules svg { flex-shrink:0;margin-top:2px;color:#c59748; }
.lucky-join { width:100%;min-height:42px;background:#cb9d3d;color:#16130c;border:1px solid #e6bf64;font-weight:700;border-radius:5px;padding:9px 12px;line-height:1.4; }.lucky-join:disabled { background:var(--bg-input);color:var(--text-secondary);border-color:var(--border-main);cursor:default; }
.lucky-result { text-align:center;line-height:1.6;font-size:14px;padding:10px 0; }.lucky-result b { font-size:19px;color:var(--text-main); }.lucky-result p { color:var(--text-secondary);margin:3px 0;font-size:12px; }
.lucky-distributed { display:block;text-align:center;color:var(--text-secondary);font-size:11px;margin-top:10px; }.lucky-error { font-size:12px;line-height:1.5;color:var(--text-main); }
</style>
