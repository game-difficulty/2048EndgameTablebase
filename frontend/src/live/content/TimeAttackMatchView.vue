<script setup>
import {computed} from 'vue';
import TournamentBoard from '../../../../competition/frontend/src/projects/TournamentBoard.vue';
const props=defineProps({match:Object,lang:String,now:Number});
const data=computed(()=>props.match.time_attack),config=computed(()=>data.value.configuration);
const t=(zh,en)=>props.lang==='zh'?zh:en;
const remaining=computed(()=>data.value.deadline_at?Math.max(0,Date.parse(data.value.deadline_at)-props.now):null);
function time(ms){if(ms==null||!Number.isFinite(ms))return '—';return `${Math.floor(ms/60000).toString().padStart(2,'0')}:${(Math.floor(ms/1000)%60).toString().padStart(2,'0')}.${Math.floor(ms%1000).toString().padStart(3,'0')}`;}
function board(side){const a=data.value.players[side].attempt;const [rows,cols]=config.value.variant.split('x').map(Number);return {rows,cols,board:a?.board||Array(rows*cols).fill(0),revision:`${a?.id}:${a?.sequence}`};}
const name=side=>props.match.teams[side]?.roster?.[0]?.display_name || (side==='yellow'?t('黄方','Yellow'):t('白方','White'));
</script>
<template><div class="pb-live">
  <header><h2>{{match.name}}</h2><p>{{config.variant.replace('x',' × ')}} · {{config.target_kind==='tile'?t('目标数字 ≥','Target tile ≥'):t('精确盘面和 =','Exact board sum =')}} {{config.target_value}}</p>
    <strong v-if="match.phase==='FINISHED'">{{data.winner_side==='draw'?t('平局','Draw'):`${name(data.winner_side)} ${t('获胜','wins')}`}}</strong>
    <strong v-else-if="match.phase==='CANCELLED'">{{t('房间已关闭 · 下注退款','Room closed · Stakes refunded')}}</strong>
    <strong v-else-if="match.prediction_window?.open">{{t('下注开放，距开赛','Entries open; starts in')}} {{Math.max(0,Math.ceil((Date.parse(match.prediction_window.minimum_until)-now)/1000))}}s</strong>
    <strong v-else>{{t('比赛剩余','Match remaining')}} {{time(remaining)}}</strong>
  </header>
  <div class="pb-players"><article v-for="side in ['yellow','white']" :key="side"><h3>{{name(side)}}</h3><div class="pb-metrics"><span>{{t('最佳 PB','Best PB')}} <b>{{time(data.players[side].best?.pb_ms)}}</b></span><span>{{t('尝试次数','Attempts')}} <b>{{data.players[side].attempt?.number||0}}</b></span></div><TournamentBoard :snapshot="board(side)" disabled /></article></div>
  <p>{{t('无限重开；时间结束时比较双方最快有效单局。仅展示服务端确认的棋盘和 PB。','Unlimited restarts; fastest valid run wins at the deadline. Boards and PBs are server-confirmed.')}}</p>
</div></template>
<style scoped>
.pb-live{width:100%;min-height:0;color:var(--match-text)}
.pb-live header{text-align:center}.pb-live h2{margin:4px 0;font-size:22px}
.pb-live header strong{font-size:24px;color:var(--match-accent)}
.pb-players{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px}
.pb-players article{min-width:0;background:var(--match-card);padding:14px;border:1px solid var(--match-line)}
.pb-players h3{margin:0 0 12px}
.pb-metrics{display:flex;justify-content:space-between;gap:12px;margin-bottom:12px}
.pb-metrics span{font-size:13px}.pb-metrics b{display:block;font-size:22px}
.pb-live :deep(.tournament-board){width:100%;max-width:400px;margin:auto;background:var(--match-tint)}
.pb-live :deep(.board-cell){background:var(--match-cell)}
.pb-live p{font-size:14px;line-height:1.6}
@media(max-width:600px){.pb-players{gap:8px}.pb-players article{padding:8px}.pb-metrics{flex-wrap:wrap}.pb-metrics b{font-size:18px}}
</style>
