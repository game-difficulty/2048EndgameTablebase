<template>
  <div class="project-board-shell">
    <div class="project-score">
      <span><small>{{ metric.label }}</small><strong>{{ metric.value }}</strong></span>
      <span><small>{{ payload.time_limit_ms ? t('倒计时', 'COUNTDOWN') : t('用时', 'ELAPSED') }}</small><strong>{{ projectTime }}</strong></span>
    </div>
    <div class="project-board" :class="{ mirror: payload.mirror_portals, irregular: payload.shape_shifter }" :style="boardStyle">
      <i
        v-for="(value,index) in cells"
        :key="`cell-${index}`"
        :class="['cell',{blocked:value===-1}]"
        :style="position(index)"
      />
      <b
        v-for="(value,index) in cells"
        v-show="value && !(payload.shape_shifter && value===-1) && !hidden.has(index)"
        :key="`tile-${index}-${value}`"
        :class="['tile', { wall: value === -1, island: value === -3, pop: pops.has(index), appear: appear === index }]"
        :style="{ ...position(index), ...tile(value), ...labelSize(value) }"
      >{{ tileLabel(value) }}</b>
      <b v-for="item in moving" :key="item.id" :class="['tile','moving',{island:item.value===-3}]" :style="movingStyle(item)">{{ tileLabel(item.value) }}</b>
      <div v-for="index in payload.sealed_cells || []" :key="`seal-${index}`" class="live-seal" :style="position(index)" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg></div>
      <div v-if="payload.mirror_portals" class="mirror-cross"><i></i><b></b></div>
    </div>
    <small>{{ number(payload.move_count) }} {{ t('步','moves') }}<template v-if="payload.dice"> · 🎲 {{ payload.dice }}</template><template v-if="payload.target_sum"> · {{ t('盘面和','sum') }} {{ payload.board_sum }}/{{ payload.target_sum }}</template><template v-if="payload.current_target_count != null"> · {{ currentTargetCount }}/{{ payload.target_count }}</template><template v-if="payload.next_seal_in != null"> · {{ payload.next_seal_in }} {{ t('步后轮换','to rotation') }}</template><template v-if="payload.no_moves && !payload.finished"> · {{ t('待撤销或重开','undo or restart') }}</template></small>
  </div>
</template>

<script setup>
import { shouldRefreshLiveClock } from '../displayClock.js';
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { projectPerformanceMetric } from '../../../../competition/shared/projectMetrics.mjs';
import { liveEmptyTileColors, liveTileColors } from '../tilePalette.js';
import { tileLabelSize } from '../../../../competition/frontend/src/projects/practiceAppearance.js';

const props=defineProps({view:{type:Object,required:true},lang:{type:String,default:'zh'},suspended:Boolean});
const payload=computed(()=>props.view?.payload||{});
const metric=computed(()=>projectPerformanceMetric(props.view,props.lang));
const currentTargetCount=computed(()=>Number(payload.value.current_target_count||0));
const cells=computed(()=>{const board=payload.value.board||[];return Array.isArray(board[0])?board.flat():board});
const rows=computed(()=>Number(payload.value.rows||Math.sqrt(cells.value.length)||4));
const cols=computed(()=>Number(payload.value.cols||Math.sqrt(cells.value.length)||4));
const IRREGULAR_GAP_UNITS=.1;
const irregularGeometry=computed(()=>payload.value.shape_shifter?{width:cols.value+IRREGULAR_GAP_UNITS*(cols.value+1),height:rows.value+IRREGULAR_GAP_UNITS*(rows.value+1)}:null);
const boardStyle=computed(()=>{const geometry=irregularGeometry.value;return{'--rows':rows.value,'--cols':cols.value,'--cell-width':geometry?`${100/geometry.width}%`:undefined,'--cell-height':geometry?`${100/geometry.height}%`:undefined,aspectRatio:geometry?`${geometry.width} / ${geometry.height}`:`${cols.value} / ${rows.value}`}});
const hidden=ref(new Set()),pops=ref(new Set()),appear=ref(null),moving=ref([]),started=ref(false),now=ref(performance.now());
const elapsedAnchor=ref(0),receivedAt=ref(performance.now());
let interval,timers=[];
const clearTimers=()=>{timers.forEach(clearTimeout);timers=[]};
const row=index=>Math.floor(index/cols.value),col=index=>index%cols.value;
function positionPoint(r,c){const geometry=irregularGeometry.value;if(geometry)return{left:`${100*(IRREGULAR_GAP_UNITS+c*(1+IRREGULAR_GAP_UNITS))/geometry.width}%`,top:`${100*(IRREGULAR_GAP_UNITS+r*(1+IRREGULAR_GAP_UNITS))/geometry.height}%`};const gap=2.25,w=(100-gap*(cols.value+1))/cols.value,h=(100-gap*(rows.value+1))/rows.value;return{left:`${gap+c*(w+gap)}%`,top:`${gap+r*(h+gap)}%`}}
const position=index=>positionPoint(row(index),col(index));
const tile=value=>value<0?{}:value?liveTileColors(value):liveEmptyTileColors();
const labelSize=value=>value>0?{fontSize:tileLabelSize(value,cols.value)}:{};
const tileLabel=value=>value<0?'':value;
const number=value=>Number(value||0).toLocaleString(props.lang==='zh'?'zh-CN':'en-US');
const t=(zh,en)=>props.lang==='zh'?zh:en;
const format=value=>{const ms=Math.max(0,Number(value)||0),m=String(Math.floor(ms/60000)).padStart(2,'0'),s=String(Math.floor(ms/1000)%60).padStart(2,'0'),cs=String(Math.floor(ms/10)%100).padStart(2,'0');return`${m}:${s}.${cs}`};
const projectTime=computed(()=>{const elapsed=elapsedAnchor.value+(payload.value.finished||props.suspended?0:Math.max(0,now.value-receivedAt.value));const limit=Number(payload.value.time_limit_ms||0);return format(limit>0?Math.max(0,limit-elapsed):elapsed)});
function wraps(from,to,direction){if(!payload.value.mirror_portals)return false;if(direction==='left')return col(to)>col(from);if(direction==='right')return col(to)<col(from);if(direction==='up')return row(to)>row(from);return row(to)<row(from)}
function buildMoving(transition){const result=[];(transition.movements||[]).forEach((item,index)=>{if(!wraps(item.from,item.to,transition.direction)){result.push({...item,id:`m-${index}`,fr:row(item.from),fc:col(item.from),tr:row(item.to),tc:col(item.to),delay:0,duration:100});return}const horizontal=['left','right'].includes(transition.direction),negative=['left','up'].includes(transition.direction);result.push({...item,id:`x-${index}`,fr:row(item.from),fc:col(item.from),tr:horizontal?row(item.from):(negative?-1:rows.value),tc:horizontal?(negative?-1:cols.value):col(item.from),delay:0,duration:50});result.push({...item,id:`e-${index}`,fr:horizontal?row(item.to):(negative?rows.value:-1),fc:horizontal?(negative?cols.value:-1):col(item.to),tr:row(item.to),tc:col(item.to),delay:50,duration:50})});return result}
function movingStyle(item){const point=started.value?positionPoint(item.tr,item.tc):positionPoint(item.fr,item.fc);return{...point,...tile(item.value),...labelSize(item.value),transition:`left ${item.duration}ms ease-in-out ${item.delay}ms, top ${item.duration}ms ease-in-out ${item.delay}ms`}}
watch(()=>[payload.value.elapsed_ms,props.suspended],()=>{elapsedAnchor.value=Number(payload.value.elapsed_ms||0);receivedAt.value=performance.now();now.value=receivedAt.value},{immediate:true});
watch(()=>props.view?.sequence,()=>{clearTimers();hidden.value=new Set();pops.value=new Set();appear.value=null;moving.value=[];started.value=false;const transition=payload.value.last_transition;if(transition?.kind!=='move')return;const destinations=new Set((transition.movements||[]).map(item=>item.to));if(transition.spawn?.index!=null)destinations.add(transition.spawn.index);hidden.value=destinations;moving.value=buildMoving(transition);requestAnimationFrame(()=>requestAnimationFrame(()=>{started.value=true}));timers.push(setTimeout(()=>{const spawn=transition.spawn?.index;hidden.value=new Set(spawn==null?[]:[spawn]);pops.value=new Set((transition.movements||[]).filter(item=>item.merged).map(item=>item.to));moving.value=[]},100));timers.push(setTimeout(()=>{hidden.value=new Set();appear.value=transition.spawn?.index??null},125));timers.push(setTimeout(()=>{pops.value=new Set();appear.value=null},300))},{immediate:true});
onMounted(()=>interval=setInterval(()=>{if(!payload.value.finished&&!props.suspended&&shouldRefreshLiveClock())now.value=performance.now()},50));
onBeforeUnmount(()=>{clearInterval(interval);clearTimers()});
</script>

<style scoped>
.project-board-shell{height:100%;display:grid;grid-template-rows:48px minmax(0,1fr) 22px;gap:8px;align-items:center}.project-score{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));align-items:end;gap:8px;border-bottom:1px solid #334155;padding:0 2px 7px}.project-score span{display:grid;gap:3px;min-width:0}.project-score span:last-child{text-align:right}.project-score small{color:#94a3b8;font-size:10px;letter-spacing:.08em}.project-score strong{overflow:hidden;color:#f8fafc;font-size:24px;font-variant-numeric:tabular-nums;line-height:1;text-overflow:ellipsis;white-space:nowrap}.project-board{--gap:2.25%;position:relative;width:min(100%,330px);aspect-ratio:1;margin:auto;overflow:hidden;border-radius:8px;background:#25334b}.cell{float:left;width:calc((100% - (var(--cols) + 1)*var(--gap))/var(--cols));height:calc((100% - (var(--rows) + 1)*var(--gap))/var(--rows));margin:var(--gap) 0 0 var(--gap);border-radius:4px;background:#3b4960}.tile{position:absolute;z-index:3;display:grid;place-items:center;width:calc((100% - (var(--cols) + 1)*var(--gap))/var(--cols));height:calc((100% - (var(--rows) + 1)*var(--gap))/var(--rows));border-radius:4px;font-size:clamp(13px,2vw,29px);font-weight:800}.tile.wall{background:repeating-linear-gradient(135deg,#596474 0 7px,#465162 7px 14px)}.tile.moving{z-index:5}.mirror-cross{position:absolute;inset:0;z-index:2;pointer-events:none}.mirror-cross i{left:calc(50% - var(--gap)/2);width:var(--gap);height:100%}.mirror-cross b{top:calc(50% - var(--gap)/2);width:100%;height:var(--gap)}.mirror-cross i,.mirror-cross b{position:absolute;background:#111c30}.pop{animation:pop .2s ease}.appear{animation:appear .2s ease backwards}.project-board-shell>small{text-align:center;color:#94a3b8}@keyframes pop{50%{transform:scale(1.2)}}@keyframes appear{from{transform:scale(0);opacity:0}}
.project-board.irregular{background:transparent}.project-board.irregular .cell{box-shadow:0 0 0 1px #25334b}.project-board.irregular .cell.blocked{visibility:hidden}.tile.island{background:#172f3d url('https://2048tables.online/minigames-assets/portal.png?v=minigames-img-20260710b') center/cover no-repeat;box-shadow:0 0 10px rgba(56,189,248,.22)}
.cell{position:absolute;float:none;margin:0}.cell,.tile{width:var(--cell-width,calc((100% - (var(--cols) + 1)*var(--gap))/var(--cols)));height:var(--cell-height,calc((100% - (var(--rows) + 1)*var(--gap))/var(--rows)))}.project-board.irregular{background:#25334b}.project-board.irregular .cell{box-shadow:none}
.live-seal{position:absolute;z-index:7;display:grid;place-items:start end;box-sizing:border-box;width:var(--cell-width,calc((100% - (var(--cols) + 1)*var(--gap))/var(--cols)));height:var(--cell-height,calc((100% - (var(--rows) + 1)*var(--gap))/var(--rows)));padding:3px 5px;border-radius:4px;background:rgba(8,12,18,.52);box-shadow:inset 0 0 0 2px #111827,inset 0 8px 13px rgba(0,0,0,.6);color:#f8fafc;font-size:14px;pointer-events:none}
.live-seal svg{width:16px;height:16px}
.project-board{container-type:inline-size}
</style>
