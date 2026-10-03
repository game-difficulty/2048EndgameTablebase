<script setup>
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import { api, commandId } from './api.js';
import { language, t } from './i18n.js';
import { userFacingError } from './errorMessages.js';
import { TimeAttackRuntime, VARIANTS } from './timeAttackRuntime.js';
import TournamentBoard from './projects/TournamentBoard.vue';
import DuelWaiting from './DuelWaiting.vue';
import { buildReplay } from '../../../frontend/src/human/engine.js';
const props=defineProps({room:Object,busy:Boolean});
const emit=defineEmits(['update','claim','leave','ready']);
const en=computed(()=>language.value==='en'), data=computed(()=>props.room.time_attack), config=computed(()=>data.value.configuration);
const side=computed(()=>props.room.me.seat?.side), attempt=computed(()=>side.value && data.value.players[side.value].attempt);
const runtime=shallowRef(null), revision=ref(0), now=ref(Date.now()), sending=ref(false), stalled=ref(false), error=ref('');
const replay=shallowRef(null), replayStep=ref(0), replayBusy=ref(false);
const replaySnapshot=computed(()=>{
  if(!replay.value)return null;
  const [rows,cols]=VARIANTS[replay.value.header.variant];
  return {board:replay.value.seek(Number(replayStep.value)).board,rows,cols,revision:`${replay.value.id}:${replayStep.value}`};
});
async function review(playerSide){
  replayBusy.value=true;
  try {const record=await api.timeAttackBest(props.room.room_code,playerSide);if(alive){replay.value={...buildReplay(record),id:record.id,pb_ms:record.pb_ms};replayStep.value=replay.value.total;}}
  catch(e){error.value=t(userFacingError(e));}finally{replayBusy.value=false;}
}
let anchorServer=Date.now(), anchorLocal=performance.now(), anchored=false, packet=null, alive=true, timer, poller;
const clock=()=>anchorServer+performance.now()-anchorLocal;
const waiting=computed(()=>['SEATING','READY_CHECK'].includes(props.room.status));
const remaining=computed(()=>Math.max(0,Date.parse(data.value.deadline_at)-now.value));
const active=computed(()=>props.room.status==='GAME_A_PLAYING' && remaining.value>0);
const ownElapsed=computed(()=>attempt.value ? Math.max(0,Math.min(now.value,Date.parse(data.value.deadline_at))-Date.parse(attempt.value.started_at)):0);
const playable=computed(()=>{revision.value; return active.value && !!runtime.value && runtime.value.status==='playing' && !stalled.value;});
function format(ms){if(ms==null || !Number.isFinite(ms))return '—'; ms=Math.max(0,Math.floor(ms)); return `${String(Math.floor(ms/60000)).padStart(2,'0')}:${String(Math.floor(ms/1000)%60).padStart(2,'0')}.${String(ms%1000).padStart(3,'0')}`;}
function label(status){return ({playing:en.value?'Playing':'进行中',reached:en.value?'Target reached · restart to improve':'已达标，可重开刷新 PB',overshot:en.value?'Target sum exceeded · restart':'超过目标盘面和，请重开',no_moves:en.value?'No moves · restart':'无路可走，请重开',expired:en.value?'Time is up':'时间已到',abandoned:en.value?'Restarted':'已重开'})[status]||status;}
function sync(room){
  const sample=Date.parse(room.server_time);
  // Never rewind countdowns when delayed snapshots arrive.
  anchorServer=anchored?Math.max(clock(),sample):sample; anchored=true; anchorLocal=performance.now(); now.value=clock();
  const a=room.me.seat && room.time_attack.players[room.me.seat.side].attempt;
  if(a?.state && !sending.value && !packet && !runtime.value?.pending.length &&
      (!runtime.value || runtime.value.id!==a.id || a.sequence>=runtime.value.state.seq)){
    runtime.value=new TimeAttackRuntime(a,room.time_attack.configuration); revision.value++;
  }
}
watch(()=>props.room,sync,{immediate:true});
function snapshot(playerSide){
  revision.value;
  const a=data.value.players[playerSide].attempt;
  const local=playerSide===side.value && runtime.value?.id===a?.id ? runtime.value.state:null;
  const [rows,cols]=VARIANTS[config.value.variant];
  const moves=local?.seq ?? a?.sequence ?? 0;
  return {board:local?.board || a?.board || Array(rows*cols).fill(0),rows,cols,moves,revision:`${a?.id}:${moves}`};
}
function move(direction){
  if(!playable.value)return;
  const d=typeof direction==='number'?direction:['up','right','down','left'].indexOf(direction);
  if(d<0)return;
  try { if(runtime.value.move(d,clock()-Date.parse(attempt.value.started_at))){revision.value++;pump();} }
  catch(e){stalled.value=true;error.value=en.value?'Unable to continue. Reload the confirmed attempt.':'无法继续，请恢复已确认的尝试。';}
}
async function pump(){
  if(sending.value || !alive || !runtime.value || (!packet && !runtime.value.pending.length))return;
  packet ||= runtime.value.batch(commandId());
  const current=packet, run=runtime.value;
  sending.value=true;
  try {
    const result=await api.timeAttackCommand(props.room.room_code,current);
    if(!alive)return;
    if(current.action==='submit')run.acknowledge(current);
    packet=null; stalled.value=false; error.value='';
    sending.value=false;
    if(current.action==='restart' || result.competition.status!=='GAME_A_PLAYING')runtime.value=null;
    emit('update',result.competition); sync(result.competition); revision.value++;
  } catch(e){
    if(!alive)return;
    stalled.value=true;
    error.value=e.status ? t(userFacingError(e)) : (en.value?'Connection interrupted. Retry to confirm the pending operation; the match clock keeps running.':'连接中断。请重试确认待提交操作，比赛计时不会暂停。');
    if(e.status && e.status<500){packet=null;runtime.value=null;}
  } finally{sending.value=false;}
  if(alive && !stalled.value && runtime.value?.pending.length)void pump();
}
async function restart(){
  if(!active.value || !attempt.value || sending.value || packet || runtime.value?.pending.length || stalled.value)return;
  packet={action:'restart',attempt_id:attempt.value.id,command_id:commandId()};
  await pump();
}
async function recover(){
  if(sending.value)return;
  if(packet){await pump();return;}
  try {const result=await api.room(props.room.room_code,{timeoutMs:5000}); if(alive){stalled.value=false;error.value='';emit('update',result.competition);sync(result.competition);}}
  catch(e){error.value=t(userFacingError(e));}
}
function key(e){
  if(replay.value||e.ctrlKey||e.metaKey||e.altKey||e.target?.closest('input,textarea,select,[contenteditable="true"]'))return;
  if(e.code==='KeyR'&&!e.repeat&&active.value&&side.value){e.preventDefault();restart();return;}
  const directions={ArrowUp:0,KeyW:0,ArrowRight:1,KeyD:1,ArrowDown:2,KeyS:2,ArrowLeft:3,KeyA:3};
  if(directions[e.code]!==undefined && playable.value){e.preventDefault();move(directions[e.code]);}
}
let polling=false;
onMounted(()=>{
  window.addEventListener('keydown',key);
  timer=setInterval(()=>now.value=clock(),50);
  poller=setInterval(async()=>{
    if(polling || waiting.value || props.room.status==='FINISHED')return;
    polling=true;
    try { const result=await api.room(props.room.room_code,{timeoutMs:4000}); if(alive)emit('update',result.competition); } catch{} finally{polling=false;}
  },2000);
});
onBeforeUnmount(()=>{alive=false;clearInterval(timer);clearInterval(poller);window.removeEventListener('keydown',key);});
</script>
<template>
  <section class="time-room">
    <div class="panel time-summary"><strong>{{config.variant.replace('x',' × ')}} · {{config.target_kind==='tile'?(en?'Target tile ≥':'目标数字 ≥'):(en?'Exact board sum =':'精确盘面和 =')}} {{config.target_value}}</strong>
      <p>{{en?'Unlimited restarts · No undo · Best verified run wins':'无限重开 · 不可悔棋 · 比较最快有效单局'}}</p>
      <details :open="waiting"><summary>{{en?'Timing and network rules':'计时与网络规则'}}</summary><p class="muted">{{en?'Runs start when the server creates the board. PB includes network latency; moves must arrive before the deadline. Disconnection does not pause time.':'单局从服务端生成棋盘时计时。PB 包含网络耗时，操作须在截止前送达。断线不暂停计时。'}}</p></details></div>
    <DuelWaiting v-if="waiting" :room="room" :busy="busy" :time-attack="true" :project-name="()=>config.variant.replace('x',' × ')" @claim="(...args)=>emit('claim',...args)" @leave="emit('leave')" @ready="emit('ready')" />
    <template v-else>
      <div class="time-clock panel"><span>{{en?'Match remaining':'比赛剩余时间'}}</span><strong>{{format(remaining)}}</strong>
        <h2 v-if="room.status==='FINISHED'">{{data.winner_side==='draw'?(en?'Draw':'平局'):`${room.seats.find(s=>s.side===data.winner_side)?.display_name || data.winner_side} ${en?'wins':'获胜'}`}}</h2></div>
      <p v-if="error" class="alert" role="alert">{{error}} <button class="secondary-button" :disabled="sending" @click="recover">{{en?'Retry / recover':'重试 / 恢复'}}</button></p>
      <div class="time-boards"><article v-for="playerSide in ['yellow','white']" :key="playerSide" class="panel time-player" :class="[playerSide,{own:playerSide===side}]">
        <header><h2>{{room.seats.find(s=>s.side===playerSide)?.display_name}}</h2><span>{{playerSide===side?(en?'You':'你'):(en?'Opponent':'对方')}}</span></header>
        <div class="time-metrics"><div><small>{{en?'Personal best':'最佳 PB'}}</small><strong>{{format(data.players[playerSide].best?.pb_ms)}}</strong></div><div><small>{{en?'Attempt':'尝试次数'}}</small><strong>{{data.players[playerSide].attempt?.number || 0}}</strong></div><div v-if="playerSide===side"><small>{{en?'Current run (local)':'本次用时（本地）'}}</small><strong>{{format(['playing','expired'].includes(runtime?.status) ? ownElapsed : data.players[playerSide].attempt?.pb_ms ?? runtime?.state.elapsed)}}</strong></div></div>
        <TournamentBoard :snapshot="snapshot(playerSide)" :disabled="playerSide!==side || !playable" @move="move" />
        <p class="time-run-status">{{label(playerSide===side && runtime ? runtime.status : data.players[playerSide].attempt?.status)}}</p>
        <p>{{en?'Valid finishes':'有效完成'}}: {{data.players[playerSide].completed}} <span v-if="data.players[playerSide].best">· PB #{{data.players[playerSide].best.number}}</span></p>
        <button v-if="data.players[playerSide].best" class="secondary-button" :disabled="replayBusy" @click="review(playerSide)">{{en?'Review best run':'查看最佳局'}}</button>
        <button v-if="playerSide===side && active" class="primary-button" :disabled="sending || stalled || !!packet || !!runtime?.pending.length || ownElapsed<500" @click="restart">{{en?'Restart (R)':'重新开始（R）'}}</button>
        <small v-if="playerSide===side && sending">{{en?'Confirming…':'正在确认…'}}</small>
      </article></div>
      <section v-if="replay" class="panel time-replay" aria-label="PB replay">
        <h2>{{en?'Verified best run':'已验证最佳局'}} · {{format(replay.pb_ms)}}</h2>
        <p>{{en?'The match clock keeps running while reviewing.':'查看回放时比赛倒计时仍会继续。'}}</p>
        <TournamentBoard :snapshot="replaySnapshot" disabled />
        <label>{{en?'Move':'步数'}} {{replayStep}} / {{replay.total}}<input v-model="replayStep" type="range" min="0" :max="replay.total" /></label>
        <button class="secondary-button" @click="replay=null">{{en?'Close replay':'关闭回放'}}</button>
      </section>
    </template>
  </section>
</template>
<style scoped>
@media(max-width:720px){.time-player.own{order:-1}}
.time-summary details{margin-top:10px;font-size:13px}.time-summary summary{cursor:pointer}.time-player button{margin-right:8px;margin-bottom:6px}
.time-replay{padding:20px;max-width:560px;width:100%;box-sizing:border-box;margin:auto}.time-replay label{display:grid;gap:10px;margin:15px 0}.time-replay input{width:100%}
.time-room{display:grid;gap:18px}.time-summary,.time-player,.time-clock{padding:20px}.time-summary p{line-height:1.65;margin-bottom:0}.time-clock{text-align:center;display:grid;gap:8px}.time-clock>strong{font-size:32px;font-variant-numeric:tabular-nums}.time-clock h2{margin:8px}.time-boards{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}.time-player{min-width:0}.time-player header{display:flex;justify-content:space-between;align-items:center;gap:12px}.time-player h2{font-size:20px;overflow-wrap:anywhere}.time-metrics{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}.time-metrics>div{flex:1;min-width:96px}.time-metrics small,.time-metrics strong{display:block}.time-metrics strong{font-size:18px;font-variant-numeric:tabular-nums;margin-top:5px;white-space:nowrap}.time-metrics small{font-size:12px;line-height:1.4}.time-run-status{min-height:1.5em;line-height:1.5}.time-player button{min-height:44px}.time-player .tournament-board{max-width:520px;margin:auto}.time-player>small{display:block;margin-top:8px}.time-player.yellow{border-top:3px solid #b39555}.time-player.white{border-top:3px solid #8ca3b5}@media(max-width:720px){.time-boards{grid-template-columns:1fr}.time-summary,.time-player,.time-clock{padding:16px}.time-metrics{gap:8px}.time-clock>strong{font-size:28px}}
</style>
