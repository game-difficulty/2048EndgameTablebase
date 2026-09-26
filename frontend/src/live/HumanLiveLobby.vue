<template>
  <div class="lobby-shell">
    <header><a class="brand" href="https://2048tables.online/"><Radio :size="25" />2048 <b>LIVE</b></a><div><span>{{ t('直播大厅','LIVE LOBBY') }}</span><button @click="load">{{ t('刷新','Refresh') }}</button></div></header>
    <main>
      <div class="intro"><div><h1>{{ t('正在直播','Live now') }}</h1><p>{{ t('观看玩家和 AI 正在进行的 2048 对局。','Watch live 2048 games from players and AI.') }}</p></div><span>{{ rooms.length }} {{ t('个直播间','rooms') }}</span></div>
      <p v-if="error" class="error">{{ t('直播大厅暂时无法更新。','The lobby could not be refreshed.') }}</p>
      <section v-if="rooms.length" class="room-grid">
        <a v-for="room in rooms" :key="room.id" :href="room.path" class="room-card">
          <div class="preview">
            <div class="preview-board" :style="boardVars(room)">
              <i v-for="(value,index) in board(room)" :key="index" :style="tile(room,value)"><span :style="tileLabel(value)">{{ value || '' }}</span></i>
            </div>
            <em>{{ room.content_kind === 'human-play' ? t('玩家直播','PLAYER') : t('AI 直播','AI') }}</em>
          </div>
          <div class="card-copy"><div class="identity"><img v-if="room.streamer?.avatar_url" :src="room.streamer.avatar_url" alt="" /><span v-else>{{ room.content_kind === 'human-play' ? 'P' : 'AI' }}</span><div><strong>{{ title(room) }}</strong><small>{{ room.variant }} · {{ Number(room.score || 0).toLocaleString() }}</small></div></div><small class="watch"><Users :size="15" />{{ room.viewers }}</small></div>
        </a>
      </section>
      <div v-else-if="!error" class="empty">{{ t('当前没有正在直播的对局。','No streams are live right now.') }}</div>
    </main>
  </div>
</template>
<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { Radio, Users } from '@lucide/vue';
import { liveEmptyTileColors, liveTileColors } from './tilePalette.js';
import { liveAppearanceTileStyle } from '../human/liveAppearance.js';
import { getTileLabelStyle } from '../components/tileLabelStyle.js';
const lang = ref(navigator.language.startsWith('zh') ? 'zh' : 'en');
const rooms = ref([]), error = ref(false); let timer, stopped = false;
const t = (zh,en) => lang.value === 'zh' ? zh : en;
const title = room => room.title?.[lang.value] || room.title?.en || room.streamer?.display_name || room.id;
const dims = variant => ({'4x4':[4,4],'3x4':[3,4],'2x4':[2,4],'3x3':[3,3]}[variant] || [4,4]);
const board = room => room.board || Array(dims(room.variant).reduce((a,b)=>a*b)).fill(0);
const boardVars = room => { const [rows,cols]=dims(room.variant); return {
  '--rows':rows,
  '--cols':cols,
  '--board-width':`${Math.min(90,50*cols/rows)}%`,
  '--tile-label-small':'clamp(11px,2.1vw,25px)',
  '--tile-label-medium':'clamp(9px,1.68vw,20px)',
  '--tile-label-large':'clamp(7px,1.26vw,15px)',
}; };
const tile = (room,value) => value
  ? (liveAppearanceTileStyle(room.appearance,value) || liveTileColors(value))
  : liveEmptyTileColors();
const tileLabel = value => value ? getTileLabelStyle({ value }) : undefined;
async function load(){ try { const response=await fetch('/api/live/lobby',{cache:'no-store'}); if(!response.ok) throw Error(); const data=await response.json(); if(!stopped){rooms.value=data.rooms||[];error.value=false;} } catch { if(!stopped) error.value=true; } }
onMounted(()=>{document.documentElement.dataset.theme='dark';load();timer=setInterval(load,10000);});
onUnmounted(()=>{stopped=true;clearInterval(timer);});
</script>
<style scoped>
.lobby-shell{min-height:100vh;background:radial-gradient(circle at 12% 0,#223251 0,transparent 38%),var(--bg-main);color:var(--text-main)}
header{height:72px;padding:0 clamp(22px,4vw,68px);display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border-main);background:#0f172acc}header>div,.brand{display:flex;align-items:center;gap:18px}.brand{font-size:23px;font-weight:800;color:#fff;text-decoration:none}.brand b{color:#fb7185}button{border:1px solid var(--border-main);border-radius:6px;background:var(--bg-card);color:var(--text-main);padding:8px 14px;cursor:pointer}
main{max-width:1400px;margin:auto;padding:42px clamp(22px,4vw,68px) 80px}.intro{display:flex;align-items:end;justify-content:space-between;margin-bottom:26px}.intro h1{font-size:38px;margin:0 0 8px}.intro p,.intro>span{color:var(--text-secondary)}
.room-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:22px}.room-card{overflow:hidden;border:1px solid var(--border-main);border-radius:12px;background:var(--bg-card);color:inherit;text-decoration:none;transition:transform .16s,border-color .16s}.room-card:hover{transform:translateY(-3px);border-color:var(--accent)}
.preview{position:relative;display:grid;place-items:center;aspect-ratio:16/9;background:#111827;overflow:hidden}.preview-board{display:grid;width:var(--board-width);grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:5px}.preview i{display:flex;aspect-ratio:1;align-items:center;justify-content:center;border-radius:4px;min-width:0;font-style:normal;font-weight:800}.preview em{position:absolute;left:12px;top:12px;padding:4px 8px;border-radius:4px;background:#020617cc;color:#fff;font-size:11px;font-style:normal;letter-spacing:.08em}
.card-copy{display:flex;align-items:center;justify-content:space-between;padding:15px}.identity{display:flex;align-items:center;gap:11px;min-width:0}.identity>img,.identity>span{width:40px;height:40px;border-radius:50%;object-fit:cover;background:var(--accent);display:grid;place-items:center;font-weight:800}.identity div{display:grid;min-width:0}.identity strong{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.identity small,.watch{color:var(--text-secondary)}.watch{display:flex;gap:5px;align-items:center}.empty,.error{padding:60px;text-align:center;border:1px dashed var(--border-main);border-radius:12px;color:var(--text-secondary)}
</style>
