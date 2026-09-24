<template>
  <Teleport v-if="entryTarget" defer :to="entryTarget">
    <button ref="entry" class="prediction-entry prediction-strip-entry" @click="open"><img :src="predictionArtwork" alt="" width="40" height="40" draggable="false" /><b>{{ t('下注','Predict') }}</b><small>{{ caption }}</small></button>
  </Teleport>
  <RoomActivityEntry v-if="dockOpen && !dismissed.includes(state.market.id)" :target="dockTarget" kind="prediction"
    :label="t('下注','Predict')" :caption="marketCaption(state.market)" :dismiss-label="t('收起下注','Dismiss predictions')"
    @dismiss="dismiss(state.market.id)" @open="open"><img :src="predictionArtwork" alt="" draggable="false" /></RoomActivityEntry>
  <dialog ref="dialog" class="prediction-dialog" aria-labelledby="prediction-title" @cancel.prevent="close" @click="backdrop">
    <header><h2 id="prediction-title">{{ t('本批下注','Batch predictions') }}</h2><button ref="help" :aria-label="t('下注规则','Prediction rules')" @click="showRules"><CircleHelp :size="20" /></button><button :aria-label="t('关闭','Close')" @click="close"><X :size="20" /></button></header>
    <p class="deadline" role="status">{{ caption }}<span v-if="market?.status === 'open' && !online"> · {{ t('直播暂停或离线，暂不接单','Paused or offline · Entries suspended') }}</span></p>
    <div v-if="market" class="prediction-columns">
      <section class="winner-bet">
      <h3>{{ t('本批最高分','Highest score this batch') }}</h3>
      <fieldset :disabled="busy || !!pending || !canBet"><legend>{{ t('选择选手','Choose a player') }}</legend>
        <button v-for="player in market.options" :key="player.id" type="button" :aria-pressed="selection === player.id" :disabled="!!market.mine && market.mine.option_id !== player.id" @click="selection=player.id"><b>{{ player.name }}</b><small>{{ tokens(player.units) }} Token · {{ player.count }}{{ t(' 人',' viewers') }}</small></button>
      </fieldset>
      <dl><div><dt>{{ t('常驻额度','Permanent quota') }}</dt><dd>{{ user ? tokens(data.paid_balance_units) : '—' }} Token</dd></div>
        <div><dt>{{ t('个人下注','Your stake') }}</dt><dd>{{ market.mine ? `${name(market.mine.option_id)} · ${tokens(market.mine.units)} Token` : t('尚未下注','No stake yet') }}</dd></div>
        <div><dt>{{ t('总奖池','Total pool') }}</dt><dd>{{ tokens(market.pool_units) }} Token</dd></div></dl>
      <p class="note">{{ t('奖池中超出可赔付范围的金额，将退还原下注者。','Unmatched stakes are returned to their owners.') }}</p>
      <fieldset class="amounts" :disabled="busy || !!pending || !canBet"><legend>{{ t('下注金额（Token）','Stake (Token)') }}</legend><button v-for="value in amounts" :key="value" type="button" :aria-pressed="amount === value" @click="amount=value">{{ value.toLocaleString() }}</button></fieldset>
      <p v-if="selection && canBet" class="note">{{ name(selection) }} · {{ amount.toLocaleString() }} Token · {{ t('最多损失本次投入','Maximum loss: this stake') }}</p>
      <button v-if="!user" class="submit" @click="close();emit('login')">{{ t('登录下注','Sign in to predict') }}</button>
      <button v-else-if="canBet || (pending && pending.kind !== 'target65536')" class="submit" :disabled="busy || !connected || (pending && pending.kind === 'target65536') || (!pending && (!selection || data.paid_balance_units < amount*1000))" @click="submit('winner')">{{ busy ? t('正在确认…','Confirming…') : pending && pending.kind !== 'target65536' ? t('核对并重试（不会重复扣款）','Check and retry (no duplicate charge)') : t('确认最高分下注','Confirm highest-score stake') }}</button>
      <section v-if="market.result" class="result" role="status"><h3>{{ t('最高分结算','Highest-score result') }}</h3><p>{{ t('本金返还','Principal') }}: {{ tokens(market.result.principal) }} · {{ t('净收益','Profit') }}: {{ tokens(market.result.profit) }}</p><p>{{ t('退款','Refund') }}: {{ tokens(market.result.refund) }} · {{ t('实际损失','Loss') }}: {{ tokens(market.result.loss) }} Token</p></section>
      </section>
      <section v-if="targetBet" class="target-bet" aria-labelledby="prediction-target-title">
        <h3 id="prediction-target-title">{{ t('独立加注 · 65536','Side bet · 65536') }}</h3>
        <p class="note">{{ t('给已下注的同一位 AI 加注。合成 65k，奖励 8 倍；合成 65k+32k，奖励升级为 50 倍。两档不叠加、本金另退。','Back the same AI you selected. Make 65k for an 8× reward, or 65k+32k for a 50× reward. Tiers do not stack; principal is returned separately.') }}</p>
        <dl><div><dt>{{ t('加注选手','Side-bet player') }}</dt><dd>{{ market.mine ? name(market.mine.option_id) : t('请先完成最高分下注','Place a highest-score stake first') }}</dd></div>
          <div><dt>{{ t('个人加注','Your side stake') }}</dt><dd>{{ tokens(targetBet.mine?.units) }} Token</dd></div>
          <div v-if="targetOutcome"><dt>{{ t('达标状态','Target status') }}</dt><dd>{{ targetOutcome === 'reached_combo' ? t('65536＋32768 · 50 倍奖励','65536 + 32768 · 50× reward') : targetOutcome === 'reached' ? t('65536 · 8 倍奖励','65536 · 8× reward') : t('已结束，未达标','Ended without reaching target') }}{{ !targetBet.result ? t(' · 待本批结算',' · Awaiting batch settlement') : '' }}</dd></div></dl>
        <p class="note">{{ t('独立扣除常驻额度，不计入最高分奖池。截止时间相同，已达标或已结束的选手不可再加注。','Charged separately from permanent Tokens; excluded from the highest-score pool. The same deadline applies. No more side bets once this AI reaches the target or ends.') }}</p>
        <fieldset class="amounts" :disabled="busy || !!pending || !canTargetBet"><legend>{{ t('65536 加注金额（Token）','65536 side stake (Token)') }}</legend><button v-for="value in amounts" :key="value" type="button" :aria-pressed="targetAmount === value" @click="targetAmount=value">{{ value.toLocaleString() }}</button></fieldset>
        <p v-if="canTargetBet" class="note">{{ t('本次加注','This side stake') }} {{ targetAmount.toLocaleString() }} Token<br />{{ t('65536 总到账','65536 total return') }} {{ (targetAmount*((targetBet.reward_multiplier ?? 8)+1)).toLocaleString() }} Token<br />{{ t('65536＋32768 总到账','65536 + 32768 total return') }} {{ (targetAmount*((targetBet.bonus_reward_multiplier ?? 50)+1)).toLocaleString() }} Token</p>
        <button v-if="user && (canTargetBet || pending?.kind === 'target65536')" class="submit" :disabled="busy || !connected || (pending && pending.kind !== 'target65536') || (!pending && data.paid_balance_units < targetAmount*1000)" @click="submit('target65536')">{{ busy ? t('正在确认…','Confirming…') : pending?.kind === 'target65536' ? t('核对并重试加注（不会重复扣款）','Check and retry side bet (no duplicate charge)') : t('确认 65536 加注','Confirm 65536 side bet') }}</button>
        <section v-if="targetBet.result" class="result" role="status"><h3>{{ t('65536 加注结算','65536 side-bet result') }}</h3><p>{{ t('本金返还','Principal') }}: {{ tokens(targetBet.result.principal) }} · {{ t('额外奖励','Extra reward') }}: {{ tokens(targetBet.result.profit) }}</p><p>{{ t('退款','Refund') }}: {{ tokens(targetBet.result.refund) }} · {{ t('实际损失','Loss') }}: {{ tokens(targetBet.result.loss) }} Token</p></section>
      </section>
    </div>
    <p v-else>{{ t('当前为过渡批或准备阶段。下一批三位选手同时起跑后开放下注。','The room is preparing or finishing a transition batch. Entries open when all three players start the next batch.') }}</p>
    <details v-if="data.recent?.length" class="recent-results"><summary>{{ t('最近个人结算','Recent results') }}</summary>
      <table><thead><tr><th>{{ t('开局时间','Started') }}</th><th>{{ t('本金','Stake') }} (Token)</th><th>{{ t('净收益','Net profit') }} (Token)</th></tr></thead>
        <tbody><tr v-for="result in data.recent.slice(0,5)" :key="result.id"><td>{{ startedTime(result.started_at) }}</td><td>{{ tokens(result.stake_units) }}</td><td>{{ result.net_profit_units > 0 ? '+' : '' }}{{ tokens(result.net_profit_units) }}</td></tr></tbody>
      </table>
    </details>
    <p v-if="error" class="error" role="alert">{{ error }}</p><button class="refresh" :disabled="busy" @click="refresh">{{ t('刷新','Refresh') }}</button>
  </dialog>
  <dialog ref="rules" class="prediction-dialog rules-dialog" aria-labelledby="prediction-rules-title" @cancel.prevent="hideRules" @click="rulesBackdrop">
    <header><h2 id="prediction-rules-title">{{ t('下注规则','Prediction rules') }}</h2><button :aria-label="t('返回下注','Back to predictions')" @click="hideRules"><X :size="20" /></button></header>
    <h3>{{ t('最高分下注','Highest-score prediction') }}</h3>
    <ol>
      <li>{{ t('每批开始后 10 分钟内，选择最终得分最高的选手。若胜负提前确定则提前封盘。暂停或离线期间不接单，恢复后不延长截止时间。','Choose the final highest scorer within 10 minutes of batch start. Entries close early if the outcome is certain. Pauses and disconnects suspend entries without extending the deadline.') }}</li>
      <li>{{ t('仅使用常驻额度。每人每批选择一位选手，可追加，不能换选手或撤回。同一账户的追加合并计算。','Permanent Tokens only. One player per account per batch; top-ups are allowed, switching and withdrawals are not. Top-ups are combined per account.') }}</li>
      <li>{{ t('赢家本金另退。从每个输家手里，最多净赢自己的累计下注金额。每个输家最多赔付赢家本金总额，未匹配部分退回本人。','Winners receive their principal back. Profit from each losing account is capped by your stake. Each loser pays at most the combined winning stakes; unmatched funds are refunded.') }}</li>
      <li>{{ t('奖金按所有赢家的下注金额比例分配。并列最高的选手都算获胜，其下注者不互相赔付，只分配其他选项可赔付的金额。','Prizes are proportional to winning stakes. All tied highest scorers win; their backers share only the matched stakes of losing options.') }}</li>
      <li>{{ t('例如你押 100，唯一对手押 10000，你赢：净赢 100，加本金到账 200，对手退回 9900。','You stake 100 and the only opponent stakes 10000. If you win, profit is 100, total return is 200, and the opponent gets 9900 back.') }}</li>
      <li>{{ t('无人押中或技术性作废：全额退款。三位全部并列或没有输家下注：各自取回本金，净收益为零。系统不抽成。','No winning bets or a technical void: full refunds. All three tie, or no losing bets: stakes return with zero profit. No platform fee.') }}</li>
    </ol>
    <h3>{{ t('65536 独立加注','65536 side bet') }}</h3>
    <ol>
      <li>{{ t('先完成最高分下注，才可为同一位 AI 单独加注。金额为 100／500／1000／5000 Token，可追加；仅扣除常驻额度，不能撤回或换选手。','First place a highest-score stake, then optionally back the same AI with a separate side bet. Amounts: 100 / 500 / 1000 / 5000 permanent Tokens; top-ups are allowed, withdrawals and switching are not.') }}</li>
      <li>{{ t('沿用开局后 10 分钟及提前封盘规则。暂停或离线不接单；该 AI 已合成 65536 棋块或本局已结束时，不再接受加注。65536 指棋块数值，不是得分。','The same 10-minute window and early closure rules apply. No entries while paused or offline, or once this AI creates a 65536 tile or ends. The target is a tile value, not the score.') }}</li>
      <li>{{ t('本局达到 65536 即成功，与最高分名次无关。之后若同一盘面同时包含一个 65536 棋块和一个 32768 棋块，奖励升级到组合档；曾分别达到两个数值不算组合达标。达标记录会保留，后续继续合并不影响资格。每位 AI 独立判定，不限本批第一位达标者。','Reaching 65536 is a hit regardless of final ranking. Having a 65536 tile and a 32768 tile together on the same board upgrades the reward. Reaching the two values at separate times does not qualify. Once earned, eligibility survives later merges. Each AI qualifies independently, not just the first one.') }}</li>
      <li>{{ t('65536 档：返还本金，另赠送累计加注金额的 8 倍常驻额度。加注 100，到账 900（本金 100＋奖励 800）。','65536 tier: return principal plus an 8× permanent-Token reward. Stake 100, receive 900 (100 principal + 800 reward).') }}</li>
      <li>{{ t('65536＋32768 档：返还本金，另赠送累计加注金额的 50 倍常驻额度。加注 100，到账 5100（本金 100＋奖励 5000）。两档取最高档，不叠加；本批结束时统一结算一次。未达到 65536 则加注不返还。','65536 + 32768 tier: return principal plus a 50× permanent-Token reward. Stake 100, receive 5100 (100 principal + 5000 reward). Only the highest tier pays, without stacking, once at batch end. No 65536 tile means no return.') }}</li>
      <li>{{ t('加注独立核算，不进入或消耗最高分奖池；奖励由系统发放，不受最高分下注的赔付上限约束。最高分无人押中不影响加注结果；技术性作废时，两项下注均全额退款。','Side bets neither fund nor draw from the highest-score pool. Rewards come from the system and are not subject to that pool’s payout caps. No highest-score winner does not void side bets. A technical void refunds both stakes in full.') }}</li>
    </ol><button class="submit" @click="hideRules">{{ t('返回下注','Back to predictions') }}</button>
  </dialog>
</template>
<script setup>
import { computed, ref, watch, nextTick, onMounted, onUnmounted } from 'vue';
import { CircleHelp, X } from '@lucide/vue';
import predictionArtwork from './assets/prediction-colored.webp';
import { useActivities } from './context.js';
import RoomActivityEntry from './RoomActivityEntry.vue';
import { useActivityDismissals } from './useActivityDismissals.js';
import { predictionIsOpen } from './activityAvailability.js';
const props=defineProps({state:Object,user:Object,connected:Boolean,online:Boolean,lang:String,dockTarget:String,entryTarget:String});
const emit=defineEmits(['login','balance','open']);
const {url,room}=useActivities();
const { dismissed, dismiss } = useActivityDismissals('prediction');
const dockOpen = computed(() => predictionIsOpen(props.state?.market, now.value, props.connected, props.online));
const t=(zh,en)=>props.lang==='zh'?zh:en;
const data=ref({market:null}),dialog=ref(null),rules=ref(null),help=ref(null),entry=ref(null),opened=ref(false),rulesOpen=ref(false);
const amount=ref(100),targetAmount=ref(100),selection=ref(''),pending=ref(null),busy=ref(false),error=ref(''),now=ref(Date.now()/1000);
let offset=0,timer,generation=0,version=0,lastRefresh=0,fetching=false,refreshAgain=false,restoreFocus;
const market=computed(()=>data.value.market),amounts=computed(()=>data.value.amounts || [100,500,1000,5000]);
const remaining=computed(()=>Math.max(0,Math.ceil((market.value?.deadline || 0)-now.value)));
const canBet=computed(()=>predictionIsOpen(market.value, now.value, props.connected, props.online));
const targetBet=computed(()=>market.value?.target_bet);
const targetOutcome=computed(()=>targetBet.value?.outcomes?.[market.value?.mine?.option_id]);
const canTargetBet=computed(()=>canBet.value && !!market.value?.mine && !!targetBet.value && !targetOutcome.value);
function marketCaption(value) {
  const seconds=Math.max(0,Math.ceil((value?.deadline || 0)-now.value));
  return !value?t('等待下一批','Waiting for next batch'):value.status==='void'?t('本批作废 · 已退款','Voided · Refunded'):value.status==='settled'?t('本批已结算','Batch settled'):value.status==='closed'||!seconds?t('已封盘','Entries closed'):`${t('截止','Closes in')} ${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`;
}
const caption=computed(()=>marketCaption(market.value));
const startedTime=value=>value == null ? '—' : new Date(value*1000).toLocaleString(props.lang==='zh'?'zh-CN':'en-GB',{year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
const tokens=units=>Number((Number(units || 0)/1000).toFixed(3)).toLocaleString(undefined,{maximumFractionDigits:3});
const name=id=>market.value?.options.find(p=>p.id===id)?.name || room.participants?.find(p=>p.id===id)?.name || id;
const storageKey=()=>`room:prediction-pending:${room.id}:${props.user?.id}`;
function persist(){try{if(pending.value)sessionStorage.setItem(storageKey(),JSON.stringify(pending.value));else sessionStorage.removeItem(storageKey());}catch{}}
function loadPending(){pending.value=null;try{const saved=JSON.parse(sessionStorage.getItem(storageKey())||'null');if(saved?.request_id){pending.value=saved;selection.value=saved.option_id;if(saved.kind==='target65536')targetAmount.value=saved.amount;else amount.value=saved.amount;}}catch{}}
async function api(body){
  const abort=new AbortController(),timeout=setTimeout(()=>abort.abort(),12000);
  try{
    const path=url('/predictions')+(body===undefined && pending.value?`?market_id=${encodeURIComponent(pending.value.market_id)}`:'');
    const response=await fetch(path,{credentials:'same-origin',cache:'no-store',signal:abort.signal,...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});
    const result=await response.json();if(!response.ok)throw Object.assign(Error('prediction_failed'),{status:response.status,detail:result.detail});return result;
  }finally{clearTimeout(timeout);}
}
function install(result){
  const before=market.value?.id;
  data.value=result;offset=result.server_time-Date.now()/1000;now.value=Date.now()/1000+offset;
  if(result.market?.mine)selection.value=result.market.mine.option_id;
  else if(before!==result.market?.id && !pending.value)selection.value='';
}
async function refresh(){
  if(fetching){refreshAgain=true;return;}
  fetching=true;lastRefresh=Date.now();const current=generation,request=++version;
  try{const result=await api();if(current===generation && request===version){install(result);if(result.market?.result)emit('balance');}}
  catch{if(current===generation)error.value=t('暂时无法查询，请刷新重试。','Could not refresh. Please try again.');}
  finally{if(current===generation){fetching=false;if(refreshAgain){refreshAgain=false;refresh();}}}
}
async function open(){restoreFocus=document.activeElement;emit('open');opened.value=true;error.value='';loadPending();await nextTick();dialog.value?.showModal();await refresh();}
function close(){if(!opened.value)return;dialog.value?.close();rules.value?.close();opened.value=false;rulesOpen.value=false;(restoreFocus?.isConnected?restoreFocus:entry.value)?.focus();}
async function showRules(){dialog.value?.close();rulesOpen.value=true;await nextTick();rules.value?.showModal();}
async function hideRules(){rules.value?.close();rulesOpen.value=false;await nextTick();dialog.value?.showModal();help.value?.focus();}
const backdrop=e=>{if(e.target===dialog.value && (e.clientX<dialog.value.getBoundingClientRect().left||e.clientX>dialog.value.getBoundingClientRect().right||e.clientY<dialog.value.getBoundingClientRect().top||e.clientY>dialog.value.getBoundingClientRect().bottom))close();};
const rulesBackdrop=e=>{if(e.target===rules.value && (e.clientX<rules.value.getBoundingClientRect().left||e.clientX>rules.value.getBoundingClientRect().right||e.clientY<rules.value.getBoundingClientRect().top||e.clientY>rules.value.getBoundingClientRect().bottom))hideRules();};
async function submit(kind='winner'){
  if(busy.value || rulesOpen.value || !props.user || (pending.value && (pending.value.kind || 'winner')!==kind) || (!pending.value && !(kind==='target65536'?canTargetBet.value:canBet.value)))return;
  if(!pending.value){pending.value={request_id:crypto.randomUUID(),market_id:market.value.id,option_id:kind==='target65536'?market.value.mine.option_id:selection.value,amount:kind==='target65536'?targetAmount.value:amount.value,kind};persist();}
  const current=generation;busy.value=true;error.value='';++version;
  try{await api(pending.value);if(current!==generation)return;pending.value=null;persist();emit('balance');await refresh();}
  catch(e){if(current!==generation)return;
    const definitive=['prediction_closed','prediction_insufficient_permanent','prediction_cannot_switch','prediction_invalid_stake','prediction_invalid_option','prediction_not_found','prediction_main_required','prediction_target_closed','prediction_invalid_kind'];
    if(definitive.includes(e.detail)){pending.value=null;persist();}
    const messages={prediction_closed:['下注已截止或直播暂停，请刷新。','Entries are closed or suspended. Please refresh.'],prediction_insufficient_permanent:['常驻额度不足。','Not enough permanent Tokens.'],prediction_cannot_switch:['本批只能追加已选选手。','You can only add to your chosen player.'],prediction_main_required:['请先完成最高分下注，再为同一位 AI 加注。','Place a highest-score stake before backing the same AI with a side bet.'],prediction_target_closed:['该 AI 已达标或已结束，不能再加注。','This AI has reached the target or ended; side bets are closed.']};
    error.value=t(...(messages[e.detail]||['尚未确认，请使用同一请求重试，不会重复扣款。','Not yet confirmed. Retry the same request safely without duplicate charges.']));
  }finally{if(current===generation)busy.value=false;}
}
watch(()=>props.state,state=>{if(!state)return;if(opened.value){refresh();}else if(!pending.value)install(state);},{immediate:true});
watch(()=>props.user?.id,()=>{generation++;version++;busy.value=false;fetching=false;close();data.value={market:props.state?.market||null};loadPending();});
watch(()=>props.connected,connected=>{if(connected && opened.value)refresh();});
onMounted(()=>{loadPending();timer=setInterval(()=>{now.value=Date.now()/1000+offset;if(opened.value && !busy.value && props.connected && Date.now()-lastRefresh>5000)refresh();},500);});
onUnmounted(()=>{generation++;clearInterval(timer);});
defineExpose({close});
</script>
<style scoped>
.target-bet{margin-top:24px;padding-top:18px;border-top:1px solid var(--border-main)}.prediction-dialog h3{font-size:15px;margin:12px 0}.target-bet .note{margin:10px 0}
.prediction-entry{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px;width:96px;padding:12px 4px;border:1px solid var(--border-main);border-radius:8px;background:var(--bg-main);color:var(--text-main);cursor:pointer}.prediction-entry img{display:block;flex-shrink:0;width:40px;height:40px;object-fit:contain;border-radius:4px}.prediction-entry b{font-size:13px}.prediction-entry small{font-size:11px;font-variant-numeric:tabular-nums}
.prediction-entry.prediction-strip-entry{flex-shrink:0;width:78px;justify-content:flex-start;gap:0;padding:5px 4px;background:transparent;border:0;border-right:1px solid var(--border-main);border-radius:0}.prediction-strip-entry img{width:52px;height:52px}.prediction-strip-entry b,.prediction-strip-entry small{font-size:11px;line-height:1.4}.prediction-strip-entry b{min-height:28px;display:flex;align-items:center;justify-content:center}
.prediction-dialog{box-sizing:border-box;width:920px;max-width:calc(100% - 24px);max-height:calc(100% - 24px);margin:auto;padding:24px;border:1px solid var(--border-main);border-radius:14px;background:var(--bg-main);color:var(--text-main);overflow:auto;box-shadow:0 20px 70px #0005}.prediction-dialog::backdrop{background:#07111caa}.prediction-dialog header{display:flex;align-items:center;gap:8px;margin-bottom:16px}.prediction-dialog h2{font-size:20px;flex:1;margin:0}.prediction-dialog header button{display:grid;place-items:center;background:transparent;border:0;color:var(--text-secondary);padding:5px}.prediction-dialog button{cursor:pointer}.prediction-dialog button:disabled{opacity:.45;cursor:default}.prediction-dialog fieldset{border:0;padding:0;display:flex;gap:8px;margin:18px 0}.prediction-dialog legend{font-size:13px;color:var(--text-secondary);margin-bottom:8px}.prediction-dialog fieldset button{flex:1;min-width:0;padding:12px 6px;border:1px solid var(--border-main);border-radius:8px;background:var(--bg-card);color:var(--text-main)}.prediction-dialog fieldset small{display:block;font-size:10px;margin-top:7px}.prediction-dialog button[aria-pressed=true]{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 12%,var(--bg-card))}.prediction-dialog dl{margin:18px 0;display:grid;gap:12px;font-size:14px}.prediction-dialog dl div{display:flex;justify-content:space-between;gap:12px}.prediction-dialog dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}.prediction-dialog dt,.note{color:var(--text-secondary)}.prediction-dialog .note,.prediction-dialog details,.prediction-dialog .result{font-size:12px;line-height:1.65}.deadline{font-size:13px;font-variant-numeric:tabular-nums}.submit{width:100%;padding:12px;border:1px solid var(--accent);border-radius:8px;background:var(--accent);color:#071524;font-weight:700}.prediction-dialog details{margin-top:18px}.prediction-dialog .refresh{font-size:12px;margin-top:12px}.prediction-dialog .error{font-size:13px;color:var(--text-main);line-height:1.6}.rules-dialog ol{padding-left:20px;font-size:14px;line-height:1.75}.rules-dialog li{margin-bottom:14px}.result h3{font-size:14px}

.prediction-columns { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px; }
.prediction-columns>section { min-width:0; }
.prediction-columns .target-bet { margin:0;padding:0 0 0 24px;border:0;border-left:1px solid var(--border-main); }
.prediction-columns h3 { margin:0 0 16px;font-size:16px; }
.prediction-dialog.rules-dialog { width:600px; }
.recent-results table { width:100%;border-collapse:collapse;table-layout:fixed;margin-top:12px; }
.recent-results th,.recent-results td { padding:9px 8px;border-bottom:1px solid var(--border-main);text-align:right;overflow-wrap:anywhere;font-variant-numeric:tabular-nums; }
.recent-results th:first-child,.recent-results td:first-child { width:45%;text-align:left; }
@media(max-width:600px) { .prediction-dialog { padding:14px; }.prediction-columns { gap:12px; }.prediction-columns .target-bet { padding-left:12px; }.prediction-columns fieldset { flex-wrap:wrap; }.prediction-columns dl div { flex-wrap:wrap; }.prediction-columns dd { overflow-wrap:anywhere; } }
</style>
