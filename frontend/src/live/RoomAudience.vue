<template>
  <div class="room-audience" @mouseenter="hoverOpen" @mouseleave="hoverClose" @keydown.esc="pinned=false;open=false">
    <button class="audience-toggle" @click="toggle" :aria-expanded="open"><Users :size="17" />{{ t('房间观众','Room viewers') }} <b>{{ count }}</b><ChevronDown :size="15" :class="{up:open}" /></button>
    <section v-if="open" class="audience-list" :aria-label="t('观众贡献榜','Audience contributions')">
      <header><span>{{ t('在线榜','Online') }}</span><small>{{ t('贡献度','Contribution') }} <button class="contribution-help" @click="showHelp" :aria-label="t('贡献度说明','How contribution works')"><CircleHelp :size="15" /></button></small></header>
      <p v-if="error" role="status">{{ t('暂时无法加载','Could not load viewers') }}</p>
      <p v-else-if="!viewers.length">{{ t('等待观众加入','Waiting for viewers') }}</p>
      <ol><li v-for="viewer in viewers" :key="viewer.id" :class="{ gold:viewer.supporter_level===2 }"><span class="rank">{{ viewer.rank }}</span><LiveIdentity :actor="viewer" :lang="lang" /><b>{{ (viewer.contribution_units/1000).toLocaleString(undefined,{maximumFractionDigits:3}) }}</b></li></ol>
    </section>
  </div>
</template>
<script setup>
import { ref,watch,onUnmounted } from 'vue';
import { Users,ChevronDown,CircleHelp } from '@lucide/vue';
import LiveIdentity from './LiveIdentity.vue';
const props=defineProps({lang:String,count:Number});
const emit=defineEmits(['count','help']);
const t=(zh,en)=>props.lang==='zh'?zh:en;
const open=ref(false),pinned=ref(false),viewers=ref([]),error=ref(false);
let timer,closeTimer,controller;
function hoverOpen(){clearTimeout(closeTimer);open.value=true;}
function hoverClose(){closeTimer=setTimeout(()=>{if(!pinned.value)open.value=false;},180);}
function toggle(){pinned.value=!pinned.value;open.value=pinned.value;}
function showHelp(){pinned.value=false;open.value=false;emit('help');}
async function refresh(){
  if(document.hidden || controller)return;
  const current=new AbortController();controller=current;
  const timeout=setTimeout(()=>current.abort(),8000);
  try {const response=await fetch('/api/live/audience',{signal:current.signal});if(!response.ok)throw new Error();const data=await response.json();viewers.value=data.viewers;emit('count',data.viewers.length);error.value=false;}
  catch {if(open.value)error.value=true;}finally{clearTimeout(timeout);if(controller===current)controller=null;}
}
watch(open,value=>{clearInterval(timer);if(value){refresh();timer=setInterval(refresh,5000);}else{controller?.abort();}});
onUnmounted(()=>{clearTimeout(closeTimer);clearInterval(timer);controller?.abort();});
</script>
<style scoped>
.room-audience { position:relative;flex-shrink:0;z-index:40; }
.audience-toggle { width:100%;min-height:46px;display:flex;align-items:center;justify-content:center;gap:8px;background:transparent;border:0;border-bottom:1px solid var(--border-main);color:var(--text-main);cursor:pointer;font-size:16px; }
.up { transform:rotate(180deg); }.audience-list { position:absolute;top:100%;left:0;right:0;background:var(--bg-main);border:1px solid var(--border-main);border-radius:0 0 6px 6px;box-shadow:0 10px 24px #0005;padding:10px; }
header { display:flex;justify-content:space-between;gap:8px;align-items:center;font-size:16px; }.audience-list small { display:flex;gap:4px;align-items:center;font-size:14px;color:var(--text-secondary); }
.contribution-help { display:grid;place-items:center;width:24px;height:24px;padding:0;background:transparent;border:0;color:var(--text-secondary);cursor:pointer; }.contribution-help:hover { color:var(--accent); }
ol { list-style:none;padding:0;margin:10px 0 0;max-height:370px;overflow:auto; }li { display:flex;align-items:center;gap:7px;min-height:46px;border-bottom:1px solid var(--border-main); }li .live-identity { flex:1;min-width:0;font-size:15px; }li b { font-size:15px;font-variant-numeric:tabular-nums; }.rank { width:20px;flex-shrink:0;font-size:14px;color:var(--text-secondary); }.gold { background:color-mix(in srgb,#d6b461 8%,transparent); }.gold .rank { color:#d6b461; }p { font-size:15px;color:var(--text-secondary); }
</style>
