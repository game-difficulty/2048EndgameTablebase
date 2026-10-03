<template>
  <section ref="layoutRoot" :class="['competition-content', {'showing-settlement':stage==='result'||match?.room_kind==='time_attack'}]">
    <header v-if="stage!=='result' && match?.room_kind!=='time_attack'" class="scorebar">
      <div class="team yellow"><span>{{ team('yellow').name || t('黄方','Yellow') }}</span><b>{{ score.yellow || 0 }}</b><time>{{ clock('yellow') }}</time></div>
      <div class="match-title"><small>{{ t('团队赛','TEAM MATCH') }} · {{ gameLabel }}</small><strong>{{ match?.name || t('比赛直播','Competition') }}</strong><em>{{ phaseLabel }}</em></div>
      <div class="team white"><time>{{ clock('white') }}</time><b>{{ score.white || 0 }}</b><span>{{ team('white').name || t('白方','White') }}</span></div>
    </header>

    <div v-if="!match" class="waiting">{{ t('正在读取比赛公开状态…','Loading public match state…') }}</div>
    <TimeAttackMatchView v-else-if="match.time_attack" :match="match" :lang="lang" :now="now" />
    <div v-else-if="match.phase==='READY_CHECK' && match.prediction_window" class="waiting"><h2>{{t('双方已准备 · 观众下注开放','Both players ready · Entries open')}}</h2><strong>{{predictionWait}}s</strong><p>{{t('下注结束后开赛，目前不扣比赛用时。','Play starts after entries close. Match clocks are stopped.')}}</p><p v-for="side in sides" :key="side">{{sideLabel(side)}} · {{team(side).roster?.map(p=>p.display_name).join(' / ')}}</p></div>

    <DraftWorkflow v-else-if="stage==='draft' && draft?.workflow && match.phase!=='DRAW'" class="live-workflow" :workflow="draft.workflow" :projects="match.projects" :phase="match.phase" :lang="lang" :name-of="projectName" :icon-of="key=>gameIcon({project_key:key})" />
    <div v-else-if="stage==='draft'" class="draft-layout">
      <TeamRoster side="yellow" :team="team('yellow')" :lang="lang" />
      <main class="draft-main"><div class="section-title"><span>{{ t('项目选定','PROJECT DRAFT') }}</span><div class="phase-info"><em v-if="phaseActor">{{ phaseActor }}</em><b>{{ phaseLabel }}</b><time v-if="phaseCountdown">{{ phaseCountdown }}</time></div></div>
        <div v-if="match.phase==='DRAW' && draft?.first_side" class="draw-reveal" :class="draft.first_side"><small>{{ t('先手抽签结果','FIRST-PICK DRAW') }}</small><strong>{{ team(draft.first_side).name || sideLabel(draft.first_side) }}{{ t('获得先手',' takes first pick') }}</strong><span>{{ match.rules ? t(`首步选择 ${match.rules.steps[0].picks} 项，禁用 ${match.rules.steps[0].bans} 项`,`First step: ${match.rules.steps[0].picks} pick(s), ${match.rules.steps[0].bans} ban(s)`) : t('先选择项目 A，并 BAN 一项','Select project A and ban one project first') }}</span></div>
        <div v-if="match.phase==='C_DRAW'" class="candidate-reveal">
          <h2>{{ draft.c_reveal_at ? t('双方盲选候选','BLIND CANDIDATES') : t('抽签结果 · 项目 C','DRAW RESULT · GAME C') }}</h2>
          <div><article v-for="side in sides" :key="side">
            <img v-if="gameIcon({project_key:draft.blind_choices?.[side]})" :src="gameIcon({project_key:draft.blind_choices?.[side]})" alt=""/>
            <small>{{ sideLabel(side) }}</small><strong>{{ projectName(draft.blind_choices?.[side]) }}</strong>
            <b v-if="draft.project_c === draft.blind_choices?.[side]">{{ t('入选项目 C','SELECTED FOR C') }}</b>
          </article></div>
        </div>
        <div v-else class="project-grid"><article v-for="project in match.projects" :key="project.key" :class="projectClass(project.key)"><img v-if="projectIconUrl(project.project_ref, darkTheme ? 'dark' : 'light')" class="draft-icon" :src="projectIconUrl(project.project_ref, darkTheme ? 'dark' : 'light')" alt="" /><small>{{ String(project.sort_order).padStart(2,'0') }}</small><strong>{{ projectName(project.key) }}</strong><span>{{ projectMark(project.key) }}</span></article></div>
        <div class="draft-summary"><span>A · {{ projectName(draft?.project_a) }}</span><span>B · {{ projectName(draft?.project_b) }}</span><span>C · {{ projectName(draft?.project_c) }}</span></div>
        <div v-if="draft?.blind_choices" class="blind-reveal"><span>{{ t('黄方盲选','Yellow blind') }} · {{ projectName(draft.blind_choices.yellow) }}</span><span>{{ t('白方盲选','White blind') }} · {{ projectName(draft.blind_choices.white) }}</span></div>
        <p v-if="match.phase==='BLIND_PICK'">{{ blindStatus }}</p>
      </main>
      <TeamRoster side="white" :team="team('white')" :lang="lang" />
    </div>

    <div v-else-if="stage==='lineup'" class="lineup-layout">
      <div class="section-title wide"><span>{{ t('出战阵容','MATCH LINEUP') }}</span><div class="phase-info"><b>{{ phaseLabel }}</b><time v-if="phaseCountdown">{{ phaseCountdown }}</time></div></div>
      <article v-for="game in match.games" :key="game.game_key" class="game-card">
        <img v-if="gameIcon(game)" class="lineup-icon" :src="gameIcon(game)" alt="" /><small>GAME {{ game.game_key }}</small><h2>{{ projectName(game.project_key) }}</h2>
        <div><PlayerLine side="yellow" :player="game.players?.yellow" :sealed="sealed('yellow')" :lang="lang"/><i>VS</i><PlayerLine side="white" :player="game.players?.white" :sealed="sealed('white')" :lang="lang"/></div>
      </article>
    </div>

    <div v-else-if="stage==='ready'" class="ready-layout">
      <div class="section-title wide"><span>{{ t('开局检查','GAME READY CHECK') }}</span><div class="phase-info"><b>{{ phaseLabel }}</b></div></div>
      <div class="ready-project"><small>GAME {{ match.current_game }} · {{ t('即将开始','UP NEXT') }}</small><img v-if="gameIcon(current)" :src="gameIcon(current)" alt="" /><h2>{{ projectName(current?.project_key) }}</h2><p>{{ currentRule }}</p></div>
      <div class="ready-sides"><article v-for="side in sides" :key="side" class="ready-side" :class="side"><h3>{{ team(side).name || sideLabel(side) }}</h3><div><span>{{ t('出战者就绪','Player ready') }}</span><b :class="{confirmed:readiness(side).player_ready}">{{ readiness(side).player_ready ? t('已就绪','READY') : t('等待中','WAITING') }}</b></div><div><span>{{ t('队长确认','Captain confirms') }}</span><b :class="{confirmed:readiness(side).captain_ready}">{{ readiness(side).captain_ready ? t('已确认','CONFIRMED') : t('等待中','WAITING') }}</b></div></article></div>
      <p v-if="predictionWait > 0" class="ready-note">{{ t(`最短下注窗口剩余 ${predictionWait} 秒；此等待不扣比赛用时。`,`Minimum betting window: ${predictionWait}s remaining. Team clocks are stopped.`) }}</p>
      <div class="public-matchups"><p v-for="game in match.games" :key="game.game_key">{{ game.players?.yellow?.display_name }} — {{ game.game_key }} · {{ projectName(game.project_key) }} — {{ game.players?.white?.display_name }}</p></div>
      <p v-if="match.room_kind==='duel'" class="ready-note">{{t('等待双方主动确认，不设超时自动确认。','Waiting for both players to confirm. There is no automatic confirmation.')}}</p>
      <p v-else class="ready-note">{{ t('双方阵容已公开。队长和出战者在60秒内确认，超时自动确认后开局。','Lineups are public. Both teams confirm within 60 seconds; missing confirmations are automatic.') }} <span>{{ t('自动确认倒计时','Auto-confirm in') }} {{ readyWait }}s</span></p>
    </div>

    <div v-else-if="stage==='game'" class="game-layout">
      <CompetitionRosterHud v-model:collapsed="hudCollapsed.yellow" side="yellow" :team="team('yellow')" :assignments="playerProjects('yellow')" :active-player="currentPlayer('yellow')" :finished="session('yellow').finished" :lang="lang" />
      <ProjectPane side="yellow" :view="views.yellow" :status="session('yellow')" :player="currentPlayer('yellow')" :lang="lang" :suspended="match?.suspended" />
      <aside class="series-panel"><small>{{ t('当前项目','CURRENT PROJECT') }}</small><img v-if="gameIcon(current)" class="current-icon" :src="gameIcon(current)" alt="" /><h2>{{ projectName(current?.project_key) }}</h2><div class="series-track"><span v-for="game in match.games" :key="game.game_key" :class="{active:game.game_key===match.current_game,done:game.result}">{{ game.game_key }}</span></div><p>{{ gameState }}</p><dl><template v-for="game in match.games" :key="game.game_key"><dt>{{ game.game_key }}</dt><dd>{{ resultText(game.result) }}</dd></template></dl></aside>
      <ProjectPane side="white" :view="views.white" :status="session('white')" :player="currentPlayer('white')" :lang="lang" :suspended="match?.suspended" />
      <CompetitionRosterHud v-model:collapsed="hudCollapsed.white" side="white" :team="team('white')" :assignments="playerProjects('white')" :active-player="currentPlayer('white')" :finished="session('white').finished" :lang="lang" />
      <p class="game-rule-strip"><b>{{ t('当前玩法','RULES') }}</b><span>{{ currentRule }}</span></p>
    </div>

    <div v-else-if="stage==='game-result'" class="game-result-layout">
      <div class="section-title wide"><span>{{ t('单局结果','GAME RESULT') }}</span><b>{{ t('休整后自动继续','Continuing after rest') }}</b></div>
      <div class="game-result-heading"><small>GAME {{ match.current_game }}</small><img v-if="gameIcon(current)" :src="gameIcon(current)" alt="" /><h2>{{ projectName(current?.project_key) }}</h2><strong>{{ resultText(current?.result) }}</strong><p>{{ resultReason(current?.result) }}</p></div>
      <div class="game-result-scores"><article v-for="side in sides" :key="side" :class="side"><small>{{ team(side).name || sideLabel(side) }}</small><div><PlayerAvatar :player="currentPlayer(side)"/><span>{{ currentPlayer(side)?.display_name || '—' }}</span></div><strong>{{ resultScore(current?.result,side) }}</strong><b :class="{confirmed:confirmed(side)}">{{ current?.result?.[`${side}_refund_ms`] ? `${t('补时','REFUND')} +${(current.result[`${side}_refund_ms`]/1000).toFixed(2)}s` : t('本局已结束','FINISHED') }}</b></article></div>
      <p class="next-game" v-if="nextGame"><img v-if="gameIcon(nextGame)" :src="gameIcon(nextGame)" alt="" /><span>{{ t('下一局','NEXT') }} · GAME {{ nextGame.game_key }} · {{ projectName(nextGame.project_key) }}</span></p>
      <p class="next-game" v-else>{{ t('休整后进入全场结算','Final result follows the rest') }}</p>
      <p v-if="stageWait">{{ t('休整剩余','Rest remaining') }} {{ stageWait }}s</p>
      <div class="result-final-boards"><ProjectPane v-for="side in sides" :key="side" :side="side" :view="views[side]" :status="session(side)" :player="currentPlayer(side)" :lang="lang" :suspended="true" /></div>
    </div>

    <MatchSettlement v-else class="live-settlement" :dark="darkTheme" :games="settlementGames" :teams="{yellow:team('yellow'),white:team('white')}" :score="score" :points="match.series_points" :winner="match.public_result?.winner_side" :reason="match.public_result?.finish_reason" :lang="lang" :cancelled="match.phase==='CANCELLED'" />
    <div v-if="match?.suspended || match?.member_hold" class="suspended">{{ match?.member_hold ? t('参赛人员变更，等待赛事方处理','Roster intervention — awaiting match staff') : t('比赛已由裁判暂停','MATCH SUSPENDED') }}</div>
    <div v-if="match && streamState !== 'live'" class="signal-notice">{{ t('直播信号重连中，当前保留最后公开画面','Reconnecting — showing the last verified public frame') }}</div>
  </section>
</template>
<script setup>
import DraftWorkflow from '../../../../competition/shared/DraftWorkflow.vue';
import TimeAttackMatchView from './TimeAttackMatchView.vue';
import { shouldRefreshLiveClock } from '../displayClock.js';
import { observeAdaptiveBoards } from './adaptiveBoardLayout.js';
import MatchSettlement from '../../../../competition/shared/MatchSettlement.vue';
import { computed, defineComponent, h, onMounted, onUnmounted, ref, watch } from 'vue';
import { projectViewRenderer } from './projectViewRegistry.js';
import { projectIconUrl } from '../../../../competition/shared/projectIcons.js';
import { projectionIsOlder, receivedProjectView } from '../../../../competition/shared/projectStateOrder.mjs';
import { ServerClock } from '../../../../competition/shared/serverClock.mjs';
import { competitionPhaseLabel, competitionProjectLabel, competitionProjectDescription } from '../../../../competition/shared/projectLabels.mjs';
import { projectResultValue } from '../../../../competition/shared/projectMetrics.mjs';
import CompetitionRosterHud from './CompetitionRosterHud.vue';
import './competitionTheme.css';
import { useLiveTheme } from '../useLiveTheme.js';
const darkTheme = useLiveTheme();
const props=defineProps({lang:String,streamState:String});
const layoutRoot=ref(null);let stopAdaptiveLayout;
onMounted(()=>{stopAdaptiveLayout=observeAdaptiveBoards(layoutRoot.value)});
onUnmounted(()=>stopAdaptiveLayout?.());
const hudCollapsed=ref({yellow:false,white:false});
const playbackPending=ref({yellow:false,white:false}),finishPlaybackUntil=ref(0);
function setPlaybackPending(side,pending){const wasPending=playbackPending.value[side];playbackPending.value[side]=pending;if(wasPending&&!pending&&match.value?.phase?.endsWith('_RESULT'))finishPlaybackUntil.value=Math.max(finishPlaybackUntil.value,serverClock.now()+300)}
let serverClock=new ServerClock();
const match=ref(null),now=ref(serverClock.now());let timer;
const t=(zh,en)=>props.lang==='zh'?zh:en;
const score=computed(()=>match.value?.score||{}),draft=computed(()=>match.value?.public_draft||{}),views=computed(()=>match.value?.project_public_views||{});
const current=computed(()=>match.value?.games?.find(item=>item.game_key===match.value.current_game));
const nextGame=computed(()=>{const index=(match.value?.games || []).map(g=>g.game_key).indexOf(match.value?.current_game);return index>=0?match.value?.games?.find(item=>item.game_key===(match.value?.games || []).map(g=>g.game_key)[index+1]):null});
const sides=['yellow','white'];
const stageWait=computed(()=>Math.max(0,Math.ceil((Date.parse((stage.value==='game-result'?match.value?.rest_until:match.value?.preview_until)||'')-currentServerNow())/1000))||0);
const readyWait=computed(()=>Math.max(0,Math.ceil((Date.parse(match.value?.ready_deadline_at||'')-currentServerNow())/1000))||0);
const predictionWait=computed(()=>Math.max(0,Math.ceil((Date.parse(match.value?.prediction_window?.minimum_until||'')-currentServerNow())/1000))||0);
const stage=computed(()=>{const value=match.value?.phase||'';if(['DRAW','DRAFT_STEP','FIRST_PICK_BAN','SECOND_PICK_BAN','BLIND_PICK','C_DRAW'].includes(value))return'draft';if(value==='LINEUP')return'lineup';if(value.endsWith('_READY'))return'ready';if(value.endsWith('_RESULT'))return now.value<finishPlaybackUntil.value||Object.values(playbackPending.value).some(Boolean)?'game':'game-result';if(value==='FINISHED'||value==='CANCELLED')return'result';return'game'});
const phaseLabels={DRAW:['抽签','DRAW'],FIRST_PICK_BAN:['先手选禁','FIRST PICK / BAN'],SECOND_PICK_BAN:['后手选禁','SECOND PICK / BAN'],BLIND_PICK:['双方盲选','BLIND PICK'],C_DRAW:['项目 C 抽签','PROJECT C DRAW'],LINEUP:['秘密布阵','SECRET LINEUP'],FINISHED:['全场结束','FINAL']};
const phaseLabel=computed(()=>{const value=match.value?.phase||'';if(value==='C_DRAW'&&draft.value?.workflow)return t('BP 已完成','DRAFT COMPLETE');const label=phaseLabels[value];if(label)return props.lang==='zh'?label[0]:label[1];const gamePhase=/^GAME_([A-O])_(READY|PLAYING|RESULT)$/.exec(value);if(gamePhase)return`${t('项目','GAME')} ${gamePhase[1]} · ${{READY:t('开局检查','READY CHECK'),PLAYING:t('对局进行中','LIVE'),RESULT:t('单局结果','RESULT')}[gamePhase[2]]}`;return competitionPhaseLabel(value,props.lang)});
const phaseActor=computed(()=>{const side=match.value?.phase_timing?.active_side;if(!side)return'';const name=team(side).name||(side==='yellow'?t('黄方','Yellow'):t('白方','White'));return`${name} · ${t('操作中','ON TURN')}`});
const currentServerNow=()=>now.value;
const phaseCountdown=computed(()=>{const deadline=Date.parse(match.value?.phase_timing?.deadline_at||'');if(!Number.isFinite(deadline))return'';const seconds=Math.max(0,Math.ceil((deadline-currentServerNow())/1000));return`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`});
const gameLabel=computed(()=>match.value?.current_game?`GAME ${match.value.current_game}`:phaseLabel.value);
const team=side=>{const value=match.value?.teams?.[side]||{};return {...value,name:!value.name||['黄方','白方'].includes(value.name)?sideLabel(side):value.name};};
const sideLabel=side=>side==='yellow'?t('黄方','Yellow'):t('白方','White');
const projectName=key=>key?competitionProjectLabel(match.value?.projects?.find(item=>item.key===key),props.lang)||key:t('待定','TBD');
const gameIcon=game=>projectIconUrl(match.value?.projects?.find(item=>item.key===(game?.project_key||game?.project_ref))?.project_ref||game?.project_ref,darkTheme.value?'dark':'light');
const currentRule=computed(()=>competitionProjectDescription(match.value?.projects?.find(item=>item.key===current.value?.project_key),props.lang)||t('玩法说明待公布','Rules pending'));
const projectMark=key=>key===draft.value.project_a?'A':key===draft.value.project_b?'B':key===draft.value.project_c?'C':key===draft.value.ban_m?`${sideLabel(draft.value.first_side)} BAN`:key===draft.value.ban_n?`${sideLabel(draft.value.first_side==='yellow'?'white':'yellow')} BAN`:'';
const projectClass=key=>({picked:(match.value?.games || []).map(g=>g.game_key).includes(projectMark(key)),banned:projectMark(key).includes('BAN')});
const blindStatus=computed(()=>`${t('黄方','Yellow')} ${draft.value.blind_submissions?.yellow?t('已密封','sealed'):t('等待','waiting')} · ${t('白方','White')} ${draft.value.blind_submissions?.white?t('已密封','sealed'):t('等待','waiting')}`);
const sealed=side=>match.value?.lineup_submission_status?.[side]?.submitted;
const session=side=>match.value?.session_status?.[side]||{};
const readiness=side=>match.value?.game_readiness?.[side]||{};
const confirmed=side=>Boolean(match.value?.captain_confirmation_status?.[side]);
const currentPlayer=side=>current.value?.players?.[side];
const playerProjects=side=>Object.fromEntries((match.value?.games||[]).filter(game=>game.players?.[side]?.position).map(game=>[
  game.players[side].position, `${game.game_key} · ${projectName(game.project_key).replace(/\s*[（(]\d+\s*[×x]\s*\d+[）)]\s*$/, '')}`,
]));
const clock=side=>{const value=match.value?.team_clocks?.[side];if(!value)return'30:00';const projectionTime=Date.parse(match.value?.server_time||'');const elapsed=value.state==='running'&&Number.isFinite(projectionTime)?Math.max(0,currentServerNow()-projectionTime):0;const ms=Math.max(0,Number(value.remaining_ms||0)-elapsed),seconds=Math.ceil(ms/1000);return`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`};
const gameState=computed(()=>match.value?.phase?.endsWith('_RESULT')?t('休整后自动继续','Continuing after rest'):Object.values(match.value?.session_status||{}).some(item=>item.finished)?t('等待另一方完成','Waiting for the other side'):t('双方对局进行中','Both players in progress'));
const resultText=result=>!result?t('待进行','Pending'):result.winner_side==='draw'?t('平局','Draw'):result.winner_side==='yellow'?t('黄方胜','Yellow win'):t('白方胜','White win');
const resultScore=(result,side)=>projectResultValue(result,side,props.lang);
const settlementGames=computed(()=>(match.value?.games||[]).map(game=>({...game,name:projectName(game.project_key),icon:gameIcon(game),project:match.value?.projects?.find(p=>p.key===game.project_key)})));
const resultReason=result=>{
  if(!result)return'';
  if(result.reason==='yellow_surrendered')return t('黄方认输，保留认输时成绩','Yellow conceded; score at concession retained');
  if(result.reason==='white_surrendered')return t('白方认输，保留认输时成绩','White conceded; score at concession retained');
  const labels={score:['按得分结算','Decided by score'],board_sum:['按盘面和结算','Decided by board sum'],race_target:['率先达成目标','First to target'],race_elapsed:['双方达标，按完成用时结算','Both finished; decided by elapsed time'],delivered_cargo:['按送出数量结算','Decided by cargo delivered'],yellow_clock_expired:['黄方包干时间耗尽','Yellow team clock expired'],white_clock_expired:['白方包干时间耗尽','White team clock expired'],both_clocks_expired:['双方包干时间耗尽','Both team clocks expired']};
  const label=labels[result.reason];return label?(props.lang==='zh'?label[0]:label[1]):'';
};
const winnerLabel=computed(()=>match.value?.public_result?.winner_side==='yellow'?t('黄方获胜','YELLOW WINS'):match.value?.public_result?.winner_side==='white'?t('白方获胜','WHITE WINS'):t('比赛完赛','MATCH COMPLETE'));
function receive(data){
  if(data.type!=='snapshot'||!data.match||projectionIsOlder(match.value,data.match))return;
  const incoming={...data.match,project_public_views:{...data.match.project_public_views}};
  const sameMatch=match.value?.match_public_key===incoming.match_public_key&&match.value?.generation===incoming.generation;
  if(!sameMatch){serverClock=new ServerClock();finishPlaybackUntil.value=0;playbackPending.value={yellow:false,white:false};}
  // competition_server_time extrapolates the source clock across cached snapshots;
  // the unrelated outer server_time is the live host's wall clock and is not used.
  serverClock.observe(incoming.server_time);
  const sourceNow=data.competition_server_time;
  if(typeof sourceNow==='number'&&Number.isFinite(sourceNow)&&Math.abs(sourceNow*1000)<=8.64e15)serverClock.observe(new Date(sourceNow*1000).toISOString());
  now.value=serverClock.now();
  if(sameMatch&&match.value?.phase?.endsWith('_PLAYING')&&incoming.phase?.endsWith('_RESULT'))finishPlaybackUntil.value=serverClock.now()+500;
  const sameGame=match.value?.match_public_key===incoming.match_public_key&&match.value?.generation===incoming.generation&&match.value?.current_game===incoming.current_game;
  for(const side of sides){const next=incoming.project_public_views[side];if(next)incoming.project_public_views[side]=receivedProjectView(sameGame?match.value?.project_public_views?.[side]:null,next)}
  match.value=incoming;
}
let resyncPending=false,lastResync=0;
async function requestProjectResync(){
  const key=match.value?.match_public_key;
  if(!key||resyncPending||Date.now()-lastResync<1000)return;
  resyncPending=true;lastResync=Date.now();
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),3000);
  try{const response=await fetch(`/api/live/rooms/competition-${encodeURIComponent(key)}/state`,{signal:controller.signal});if(response.ok){const data=await response.json();if(match.value?.match_public_key===key)receive({...data,type:'snapshot'})}}
  catch{/* Playback has its own bounded snapshot fallback. */}finally{clearTimeout(timeout);resyncPending=false}
}
function resume(){now.value=serverClock.now()}
function getPipFrame(){const currentMatch=match.value,lang=props.lang;return{key:`${currentMatch?.generation}:${currentMatch?.content_sequence}:${lang}`,width:960,height:540,draw(ctx){ctx.fillStyle='#111c30';ctx.fillRect(0,0,960,540);ctx.fillStyle='#f8fafc';ctx.textAlign='center';ctx.font='700 30px sans-serif';ctx.fillText(currentMatch?.name||(lang==='zh'?'2048 赛事':'2048 Competition'),480,105);ctx.font='800 86px sans-serif';ctx.fillText(`${currentMatch?.score?.yellow||0}  :  ${currentMatch?.score?.white||0}`,480,270);ctx.font='600 24px sans-serif';ctx.fillStyle='#aebbd0';ctx.fillText(competitionPhaseLabel(currentMatch?.phase,lang),480,345)}}}
onMounted(()=>timer=setInterval(()=>{if(shouldRefreshLiveClock())now.value=serverClock.now()},250));onUnmounted(()=>clearInterval(timer));defineExpose({receive,resume,getPipFrame});
const PlayerAvatar=defineComponent({props:['player'],setup(p){const failed=ref(false);watch(()=>p.player?.avatar_url,()=>{failed.value=false});return()=>h('span',{class:'live-player-avatar','aria-hidden':'true'},p.player?.avatar_url&&!failed.value?h('img',{src:p.player.avatar_url,alt:'',onError:()=>{failed.value=true}}):h('span',String(p.player?.display_name||'?').trim().slice(0,2).toUpperCase()))}});
const TeamRoster=defineComponent({props:['side','team','lang'],setup(p){return()=>h('aside',{class:['roster',p.side]},[h('small',p.side==='yellow'?(p.lang==='zh'?'黄方阵容':'YELLOW TEAM'):(p.lang==='zh'?'白方阵容':'WHITE TEAM')),...(p.team?.roster||[]).map(player=>h('div',{class:{captain:player.is_captain}},[h(PlayerAvatar,{player}),h('span',player.display_name),player.is_captain?h('em',p.lang==='zh'?'队长':'CPT'):null]))])}});
const PlayerLine=defineComponent({props:['side','player','sealed','lang'],setup(p){return()=>h('section',{class:p.side},p.player?[h(PlayerAvatar,{player:p.player}),h('span',p.player.display_name)]:[h('b','—'),h('span',p.sealed?(p.lang==='zh'?'已密封':'Sealed'):(p.lang==='zh'?'布阵中':'Selecting'))])}});
const ProjectPane=defineComponent({props:['side','view','status','player','lang','suspended'],setup(p){return()=>{
  const Renderer=projectViewRenderer(p.view);
  const waiting=p.view?.payload?.awaiting_client;
  const body=waiting?h('div',{class:'view-fallback'},p.lang==='zh'?'等待选手载入棋盘…':'Waiting for the player to load…')
    :Renderer?h(Renderer,{key:`${match.value?.match_public_key}:${match.value?.generation}:${match.value?.current_game}:${p.side}`,view:p.view,project:match.value?.projects?.find(item=>item.key===current.value?.project_key),lang:p.lang,suspended:p.suspended,onPending:value=>setPlaybackPending(p.side,value),onGap:requestProjectResync})
      :h('div',{class:'view-fallback'},[h('b',p.lang==='zh'?'项目画面暂不可用':'Project view unavailable'),h('small',p.view?`${p.view.view_kind} · ${p.view.view_protocol}`:'')]);
  return h('article',{class:['project-pane',p.side,p.status?.finished?'finished':'']},[
    h('header',[h('div',{class:'project-player'},[h(PlayerAvatar,{player:p.player}),h('span',[
      h('small',p.side==='yellow'?(p.lang==='zh'?'黄方选手':'YELLOW PLAYER'):(p.lang==='zh'?'白方选手':'WHITE PLAYER')),
      h('strong',p.player?.display_name||'—')])]),
      h('span',p.status?.finished?(p.lang==='zh'?'已完成':'FINISHED'):waiting?(p.lang==='zh'?'载入中':'LOADING'):(p.lang==='zh'?'进行中':'PLAYING'))]),body]);
}}});
</script>
<style scoped>
.competition-content .live-workflow{max-height:calc(100% - 100px);overflow:auto;margin-top:12px;box-sizing:border-box}
.competition-content .lineup-layout{height:calc(100% - 100px);overflow:auto;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));grid-template-rows:max-content;grid-auto-rows:max-content;align-content:start}
.competition-content .lineup-layout .game-card{min-width:0;display:grid;grid-template-columns:56px minmax(0,1fr);gap:8px;padding:16px;text-align:left;align-content:start}
.competition-content .lineup-layout .lineup-icon{grid-column:1;grid-row:1/3;width:56px;height:56px;margin:0}
.competition-content .lineup-layout .game-card>small{grid-column:2;align-self:end}
.competition-content .lineup-layout .game-card>h2{grid-column:2;height:auto;margin:0;font-size:16px;line-height:1.4;overflow-wrap:anywhere}
.competition-content .lineup-layout .game-card>div{grid-column:1/-1;gap:6px}
.competition-content .lineup-layout .game-card :deep(section){padding:8px;font-size:13px}
.competition-content .lineup-layout .game-card :deep(section .live-player-avatar){width:28px;height:28px}
.competition-content .series-track{flex-wrap:wrap;gap:4px}
.competition-content .series-panel{min-height:0;overflow:auto}
.competition-content .roster{min-height:0;overflow:auto}
.competition-content.showing-settlement{overflow:auto;display:block;padding:12px 18px}
.competition-content.showing-settlement .live-settlement{height:auto;min-height:100%;padding:12px 8px}
.live-settlement :deep(.settlement-row){padding:9px 0}
.live-settlement :deep(.settlement-heading){padding-bottom:12px}
.live-settlement :deep(.settlement-project img){width:64px;height:64px}
.candidate-reveal{text-align:center;padding:20px}.candidate-reveal>div{display:grid;grid-template-columns:1fr 1fr;gap:24px}.candidate-reveal article{display:flex;flex-direction:column;align-items:center;gap:12px}.candidate-reveal img{width:150px;height:150px;object-fit:contain}.candidate-reveal b{color:var(--match-accent,#f4cd69)}.competition-content .game-result-layout{overflow:auto;height:calc(100% - 90px)}.result-final-boards{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:16px auto;max-width:840px}.result-final-boards :deep(.project-pane){min-width:0}.public-matchups{text-align:center;font-size:13px}
.draft-icon,.lineup-icon,.current-icon{display:block;border-radius:6px;object-fit:cover}
.project-grid article:has(.draft-icon){grid-template-columns:42px minmax(0,1fr);column-gap:8px;align-content:center}
.draft-icon{grid-row:span 2;width:42px;height:42px}
.project-grid article:has(.draft-icon)>strong{font-size:12px}
.lineup-icon{width:70px;height:70px;margin:0 auto 10px}
.current-icon{width:66px;height:66px;margin:10px auto 4px}
.competition-content{position:relative;height:100%;box-sizing:border-box;padding:18px 22px;color:var(--match-text,#f8fafc);background:var(--match-bg,#111c30);overflow:hidden}.scorebar{height:84px;display:grid;grid-template-columns:1fr 330px 1fr;align-items:stretch;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a)}.team{display:flex;align-items:center;gap:18px;padding:0 24px}.team.white{justify-content:flex-end;text-align:right}.team span{font-size:20px;font-weight:700}.team b{font-size:42px}.team time{color:var(--match-accent,#d8bd69);font-size:23px;font-variant-numeric:tabular-nums}.match-title{display:grid;place-items:center;align-content:center;border-inline:1px solid var(--match-line,#334155);text-align:center}.match-title small{color:var(--match-accent,#d8bd69);letter-spacing:.12em}.match-title strong{max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:19px}.match-title em{color:var(--match-muted,#94a3b8);font-size:12px;font-style:normal}.waiting,.result-layout{height:580px;display:grid;place-items:center}.section-title{display:flex;align-items:center;justify-content:space-between;height:44px;padding:0 14px;border-bottom:1px solid var(--match-line,#334155)}.section-title span{color:var(--match-muted,#94a3b8);font-size:12px;letter-spacing:.12em}.section-title b{color:var(--match-accent,#d8bd69)}.phase-info{display:flex;align-items:center;gap:12px}.phase-info time{min-width:54px;padding:4px 7px;border:1px solid var(--match-border,#475569);background:var(--match-bg,#111c30);color:var(--match-text,#f8fafc);font-size:13px;font-variant-numeric:tabular-nums;text-align:center}.draft-layout{height:580px;display:grid;grid-template-columns:220px 1fr 220px;gap:16px;padding-top:16px}.roster{padding:14px;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a)}.roster>small{display:block;margin-bottom:14px;color:var(--match-muted,#94a3b8);letter-spacing:.1em}.roster>div{display:grid;grid-template-columns:28px 1fr auto;align-items:center;gap:8px;height:58px;border-top:1px solid #2a3850}.roster>div>b{display:grid;width:24px;height:24px;place-items:center;background:var(--match-tint,#25334b)}.roster em{color:var(--match-accent,#d8bd69);font-size:10px;font-style:normal}.draft-main{border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a)}.project-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;padding:13px}.project-grid article{position:relative;display:grid;min-height:92px;align-content:center;gap:4px;padding:10px;border:1px solid var(--match-line,#334155);background:var(--match-bg,#111c30)}.project-grid article>small{color:var(--match-dim,#64748b)}.project-grid article>strong{font-size:14px}.project-grid article>span{position:absolute;right:7px;top:7px;color:var(--match-accent,#d8bd69);font-weight:800}.project-grid article.banned{opacity:.46}.project-grid article.banned:after{content:"";position:absolute;inset:50% 7px auto;border-top:2px solid #ef6b68;transform:rotate(-8deg)}.project-grid article.picked{border-color:#a78b3a}.draft-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;padding:0 13px}.draft-summary span{padding:10px;background:var(--match-tint,#25334b);text-align:center}.draft-main>p{text-align:center;color:var(--match-muted,#94a3b8)}.lineup-layout{height:580px;display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:50px 1fr;gap:14px;padding-top:16px}.section-title.wide{grid-column:1/-1}.game-card{padding:28px 22px;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a);text-align:center}.game-card>small{color:var(--match-accent,#d8bd69);letter-spacing:.12em}.game-card h2{height:58px;margin:12px 0 24px}.game-card>div{display:grid;gap:13px}.game-card section{display:flex;align-items:center;gap:12px;padding:14px;background:var(--match-bg,#111c30);text-align:left}.game-card section.white{flex-direction:row-reverse;text-align:right}.game-card section b{display:grid;width:34px;height:34px;place-items:center;background:var(--match-tint,#25334b)}.game-card i{color:var(--match-dim,#64748b);font-size:11px}.game-layout{height:580px;display:grid;grid-template-columns:minmax(0,1fr) 210px minmax(0,1fr);gap:14px;padding-top:16px}.project-pane{display:grid;grid-template-rows:58px minmax(0,1fr);padding:14px;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a)}.project-pane>header{display:flex;justify-content:space-between;border-bottom:1px solid var(--match-line,#334155)}.project-pane>header div{display:grid}.project-pane>header small{color:var(--match-muted,#94a3b8);font-size:10px;letter-spacing:.1em}.project-pane>header strong{font-size:18px}.project-pane>header span{color:var(--match-success,#78c59b);font-size:11px;font-weight:800}.project-pane.finished>header span{color:var(--match-accent,#d8bd69)}.series-panel{padding:20px 14px;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a);text-align:center}.series-panel>small{color:var(--match-muted,#94a3b8);letter-spacing:.1em}.series-panel h2{min-height:54px;font-size:17px}.series-track{display:flex;justify-content:center;gap:9px}.series-track span{display:grid;width:34px;height:34px;place-items:center;border:1px solid var(--match-border,#475569)}.series-track span.active{border-color:var(--match-accent,#d8bd69);color:var(--match-accent,#d8bd69)}.series-track span.done{background:var(--match-tint,#25334b)}.series-panel p{min-height:42px;margin:24px 0;color:var(--match-accent,#d8bd69)}.series-panel dl{display:grid;grid-template-columns:30px 1fr;text-align:left}.series-panel dt,.series-panel dd{margin:0;padding:8px;border-top:1px solid var(--match-line,#334155)}.view-fallback{height:100%;display:grid;place-items:center;align-content:center;gap:8px;color:var(--match-muted,#94a3b8);text-align:center}.result-layout{align-content:center;gap:18px}.result-layout>small{color:var(--match-accent,#d8bd69);letter-spacing:.16em}.result-layout h1{margin:0;font-size:48px}.result-layout>strong{font-size:72px}.result-layout>div{display:flex;gap:10px}.result-layout>div span{padding:10px 16px;background:var(--match-card,#17243a)}.suspended{position:absolute;inset:84px 22px auto;padding:10px;background:#9f3d3a;color:#fff;text-align:center;font-weight:800;letter-spacing:.08em}.signal-notice{position:absolute;right:30px;bottom:12px;padding:7px 10px;border:1px solid #6b7280;background:var(--match-card,#17243a);color:var(--match-copy,#cbd5e1);font-size:12px}
.phase-info em{color:var(--match-copy,#cbd5e1);font-size:11px;font-style:normal}.blind-reveal{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:9px 13px 0}.blind-reveal span{padding:8px;border:1px solid var(--match-line,#334155);color:var(--match-copy,#cbd5e1);font-size:12px;text-align:center}
.live-player-avatar{display:grid;flex:none;place-items:center;width:34px;height:34px;overflow:hidden;border:1px solid #506077;border-radius:50%;background:var(--match-tint,#25334b);color:var(--match-accent,#d8bd69);font-size:11px;font-weight:700;line-height:1}
.live-player-avatar img{display:block;width:100%;height:100%;object-fit:cover}
.roster>div{grid-template-columns:34px 1fr auto}
.game-card section .live-player-avatar{width:34px;height:34px}
.project-pane>header .project-player{display:flex;align-items:center;gap:9px}
.project-pane>header .project-player>span:last-child{display:grid;gap:2px}
.draw-reveal{display:grid;justify-items:center;gap:3px;margin:12px 13px 0;padding:10px 14px;border:1px solid #b89442;background:linear-gradient(90deg,#3b301e,#263148);text-align:center}
.draw-reveal.white{border-color:#a9b3c1;background:linear-gradient(90deg,var(--match-line,#334155),#263148)}
.draw-reveal small{color:var(--match-accent,#d8bd69);font-size:10px;letter-spacing:.13em}.draw-reveal strong{font-size:22px}.draw-reveal span{color:var(--match-copy,#cbd5e1);font-size:11px}
.ready-layout,.game-result-layout{height:580px;display:grid;grid-template-rows:50px auto 1fr auto;gap:14px;padding-top:16px;box-sizing:border-box}
.ready-project,.game-result-heading{display:grid;justify-items:center;align-content:center;gap:6px;min-height:170px;padding:13px;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a);text-align:center}
.ready-project small,.game-result-heading small{color:var(--match-accent,#d8bd69);letter-spacing:.12em}.ready-project img,.game-result-heading img{width:80px;height:80px;object-fit:cover;border-radius:7px}
.ready-project h2,.game-result-heading h2{margin:0;font-size:23px}.ready-project p,.game-result-heading p{max-width:780px;margin:2px 0;color:var(--match-copy,#cbd5e1);font-size:13px;line-height:1.5}
.ready-sides,.game-result-scores{display:grid;grid-template-columns:1fr 1fr;gap:14px}.ready-side,.game-result-scores article{padding:18px 22px;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a)}.ready-side.yellow,.game-result-scores article.yellow{border-top:3px solid var(--match-accent,#d8bd69)}.ready-side.white,.game-result-scores article.white{border-top:3px solid #a9b3c1}.ready-side h3{margin:0 0 18px;font-size:21px}.ready-side>div{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 0;border-top:1px solid var(--match-line,#334155);color:var(--match-copy,#cbd5e1)}.ready-side b,.game-result-scores article>b{color:var(--match-muted,#94a3b8);font-size:12px}.ready-side b.confirmed,.game-result-scores article>b.confirmed{color:var(--match-success,#78c59b)}.ready-note,.next-game{margin:0;padding:12px;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a);color:var(--match-copy,#cbd5e1);text-align:center;font-size:13px}
.game-layout{grid-template-columns:94px minmax(0,1fr) 174px minmax(0,1fr) 94px;grid-template-rows:minmax(0,1fr) auto;gap:8px;height:580px;box-sizing:border-box}
.game-layout>*{min-width:0;min-height:0}.game-layout .project-pane{padding:10px}.game-layout .series-panel{padding:16px 8px}.game-layout .series-panel h2{min-height:40px;margin:12px 0;font-size:16px}.game-layout .current-icon{width:72px;height:72px}.game-layout .series-panel p{margin:16px 0}.game-layout .series-panel dt,.game-layout .series-panel dd{padding:6px}
.match-roster{display:grid;grid-template-rows:30px repeat(3,1fr);gap:8px}.match-roster>small{align-self:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--match-copy,#cbd5e1);font-size:12px;text-align:center}.match-roster>div{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px;overflow:hidden;padding:6px 2px;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a);text-align:center}.match-roster>div.active{border-color:var(--match-accent,#d8bd69);background:var(--match-active,#28334a)}.match-roster.white>div.active{border-color:#a9b3c1}.match-roster .live-player-avatar{width:40px;height:40px}.match-roster>div>span:not(.live-player-avatar){max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px}.match-roster b{color:var(--match-accent,#d8bd69);font-size:10px}.match-roster .finished b{color:var(--match-success,#78c59b)}
.game-rule-strip{grid-column:1/-1;display:flex;align-items:baseline;justify-content:center;gap:14px;margin:0;padding:10px 14px;border-top:1px solid var(--match-line,#334155);color:var(--match-copy,#e2e8f0);font-size:24px;font-weight:500;line-height:1.5;text-align:left}.game-rule-strip b{flex:none;color:var(--match-accent,#d8bd69);letter-spacing:.08em}.game-rule-strip span{min-width:0;white-space:normal;overflow-wrap:anywhere}
.game-result-heading{min-height:185px}.game-result-heading img{width:78px;height:78px}.game-result-heading strong{color:var(--match-accent,#d8bd69);font-size:25px}.game-result-scores article{display:grid;grid-template-rows:auto auto 1fr auto;gap:9px;justify-items:center;align-items:center}.game-result-scores article>small{color:var(--match-copy,#cbd5e1)}.game-result-scores article>div{display:flex;align-items:center;gap:9px;font-size:16px}.game-result-scores article>strong{font-size:47px;font-variant-numeric:tabular-nums}.game-result-scores article>b{padding:7px 12px;border:1px solid var(--match-border,#475569)}.game-result-scores article>b.confirmed{border-color:#4b9271}.next-game{display:flex;justify-content:center;align-items:center;gap:9px}.next-game img{width:28px;height:28px;object-fit:cover;border-radius:3px}
.result-layout{height:580px;display:grid;grid-template-rows:auto auto auto 1fr;align-content:center;justify-items:center;gap:10px;padding-top:16px;box-sizing:border-box}.result-layout h1{font-size:40px}.result-layout>strong{font-size:56px;line-height:1}.result-layout>.final-games{display:grid;width:min(100%,1000px);grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:8px}.final-games article{display:grid;grid-template-columns:64px 1fr;grid-template-rows:auto 1fr;gap:8px;padding:14px;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a)}.final-games article>img{grid-row:1/3;width:64px;height:64px;object-fit:cover;border-radius:5px}.final-game-title{display:grid;gap:3px;align-content:start}.final-game-title small{color:var(--match-accent,#d8bd69)}.final-game-title b{font-size:15px}.final-game-title em{color:var(--match-copy,#cbd5e1);font-size:12px;font-style:normal}.final-game-scores{grid-column:1/-1;display:grid;gap:5px;margin-top:8px;padding-top:8px;border-top:1px solid var(--match-line,#334155)}.result-layout .final-game-scores span{display:flex;justify-content:space-between;gap:8px;padding:0;background:none;color:var(--match-copy,#cbd5e1);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.final-game-scores b{color:var(--match-text,#f8fafc);font-size:15px;font-variant-numeric:tabular-nums}
.project-grid{grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;padding:11px}
.project-grid article{min-height:78px;padding:7px}
.project-grid article:has(.draft-icon){grid-template-columns:62px minmax(0,1fr);column-gap:8px}
.draft-icon{width:62px;height:62px}
.lineup-icon{width:112px;height:112px}
.ready-project{min-height:206px}
.ready-project img{width:126px;height:126px}
.game-result-layout{grid-template-rows:50px 160px minmax(0,1fr) 54px}
.game-result-heading{display:grid;grid-template-columns:128px minmax(0,460px);grid-template-rows:repeat(4,auto);justify-content:center;align-content:center;justify-items:start;column-gap:24px;row-gap:3px;min-height:0;box-sizing:border-box;text-align:left}
.game-result-heading img{grid-column:1;grid-row:1/5;width:128px;height:128px}
.game-result-heading small,.game-result-heading h2,.game-result-heading strong,.game-result-heading p{grid-column:2}
.game-result-heading h2{font-size:25px}
.game-result-heading p{margin:0}
.game-layout{grid-template-columns:120px minmax(0,1fr) 124px minmax(0,1fr) 120px}
.game-layout .series-panel{padding:12px 5px}
.game-layout .current-icon{width:104px;height:104px;margin:10px auto 6px}
.game-layout .series-panel h2{font-size:14px;overflow-wrap:anywhere}
.game-layout .series-track{gap:5px}
.game-layout .series-track span{width:28px;height:28px}
.game-layout .series-panel dl{grid-template-columns:20px 1fr;font-size:11px}
.game-layout{position:relative;grid-template-columns:minmax(0,1fr) 124px minmax(0,1fr);padding-inline:132px}
.next-game img{width:38px;height:38px}
.final-games article{grid-template-columns:92px 1fr;gap:10px}
.final-games article>img{width:92px;height:92px}
.result-layout .final-game-scores span{align-items:center;justify-content:flex-start;gap:7px}
.final-game-scores span>b{margin-left:auto}
</style>
<style>
/* These rows are rendered by local render-function components. Keep their rules
   under the competition root so scoped-style boundaries do not hide avatars. */
.competition-content .live-player-avatar{display:grid;flex:none;place-items:center;width:48px;height:48px;overflow:hidden;border:2px solid #68788e;border-radius:50%;background:var(--match-line,#334155);color:var(--match-accent,#f1cf78);font-size:13px;font-weight:800;line-height:1;box-sizing:border-box}
.competition-content .live-player-avatar img{display:block;width:100%;height:100%;object-fit:cover}
.competition-content .roster>small{display:block;margin-bottom:14px;color:var(--match-muted,#94a3b8);letter-spacing:.1em}
.competition-content .roster>div{display:grid;grid-template-columns:52px minmax(0,1fr) auto;align-items:center;gap:9px;height:76px;border-top:1px solid #2a3850}
.competition-content .roster>div>span:not(.live-player-avatar){overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:14px}
.competition-content .roster em{color:var(--match-accent,#d8bd69);font-size:10px;font-style:normal}
.competition-content .match-roster{display:grid;grid-template-rows:24px repeat(3,minmax(0,1fr));gap:8px}
.competition-content .match-roster>small{align-self:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--match-copy,#cbd5e1);font-size:12px;text-align:center}
.competition-content .match-roster>div{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;overflow:hidden;padding:7px 4px;border:1px solid var(--match-line,#334155);background:var(--match-card,#17243a);text-align:center}
.competition-content .match-roster>div.active{border-color:var(--match-accent,#d8bd69);background:var(--match-active,#28334a)}
.competition-content .match-roster.white>div.active{border-color:#a9b3c1}
.competition-content .match-roster .live-player-avatar{width:58px;height:58px}
.competition-content .match-roster>div.active .live-player-avatar{width:68px;height:68px;border-color:var(--match-accent,#d8bd69)}
.competition-content .match-roster.white>div.active .live-player-avatar{border-color:var(--match-white,#d8e0e9)}
.competition-content .match-roster>div>span:not(.live-player-avatar){max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;font-weight:700}
.competition-content .match-roster b{color:var(--match-accent,#d8bd69);font-size:10px}
.competition-content .match-roster .finished b{color:var(--match-success,#78c59b)}
.competition-content .match-roster em{color:var(--match-muted,#94a3b8);font-size:10px;font-style:normal}
.competition-content .project-pane{grid-template-rows:68px minmax(0,1fr)}
.competition-content .project-pane>header{display:flex;align-items:center;justify-content:space-between;gap:6px;border-bottom:1px solid var(--match-line,#334155)}
.competition-content .project-pane>header .project-player{display:flex;align-items:center;gap:9px;min-width:0}
.competition-content .project-pane>header .project-player .live-player-avatar{width:46px;height:46px}
.competition-content .project-pane>header .project-player>span:last-child{display:grid;gap:2px;min-width:0}
.competition-content .project-pane>header small{color:var(--match-muted,#94a3b8);font-size:10px;letter-spacing:.08em}
.competition-content .project-pane>header strong{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:16px}
.competition-content .project-pane>header>span{flex:none;color:var(--match-success,#78c59b);font-size:11px;font-weight:800}
.competition-content .project-pane.finished>header>span{color:var(--match-accent,#d8bd69)}
.competition-content .game-card section{display:flex;align-items:center;gap:12px;padding:14px;background:var(--match-bg,#111c30);text-align:left}
.competition-content .game-card section.white{flex-direction:row-reverse;text-align:right}
.competition-content .game-card section .live-player-avatar{width:50px;height:50px}
.competition-content .game-card section>span:last-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.competition-content .game-result-scores .live-player-avatar{width:56px;height:56px}
.competition-content .final-game-scores .live-player-avatar{width:28px;height:28px;border-width:1px;font-size:9px}
.competition-content .final-game-scores span{min-width:0}
.competition-content .ready-layout,.competition-content .game-result-layout{display:flex;flex-direction:column;gap:14px;height:calc(100% - 90px);overflow:auto;min-height:0}
.competition-content .ready-layout>*,.competition-content .game-result-layout>*{flex-shrink:0}
.competition-content .result-final-boards{width:min(100%,840px);box-sizing:border-box;height:550px;min-height:550px}
.competition-content .result-final-boards .project-pane{min-width:0;min-height:0}
</style>
