<template>
  <Teleport v-if="entryTarget" defer :to="entryTarget">
    <button ref="entry" class="prediction-entry" @click="open"><img :src="artwork" alt="" /><b>{{ t('赛事下注','Match bets') }}</b><small>{{ caption }}</small></button>
  </Teleport>
  <dialog ref="dialog" class="competition-bets" aria-labelledby="competition-bets-title" @cancel.prevent="close" @click="backdrop">
    <header><h2 id="competition-bets-title">{{ t('赛事下注','Match predictions') }}</h2><button @click="close" :aria-label="t('关闭','Close')">×</button></header>
    <p class="timing" role="status">{{ caption }} · {{ t('首局开始前统一封盘，比赛结束结算','Both markets close before game A; settle at match end') }}</p>
    <p v-if="remaining > 0" class="note">{{ t(`最早开赛还需 ${remaining} 秒；BP 和就绪未完成时仍继续开放。`,`Earliest start in ${remaining}s; entries remain open until drafting and readiness are complete.`) }}</p>
    <p v-if="!connected || !online || !data.available" class="note">{{ t('暂停、断线或尚未开放时不能下注。','Entries are disabled while paused, disconnected or not yet open.') }}</p>
    <div class="market-columns">
      <section v-for="kind in kinds" :key="kind">
        <h3>{{ kind === 'winner' ? t('全场胜方','Match winner') : t('前两局比分 · 黄 : 白','First two games · Yellow : White') }}</h3>
        <template v-if="market(kind)">
          <span class="status">{{ statusText(market(kind)) }}</span>
          <fieldset :disabled="busy || !!pending || !canBet(kind)"><legend>{{ t('选择结果（选定后不能换边）','Choose an outcome (cannot switch after betting)') }}</legend>
            <button v-for="option in market(kind).options" :key="option.id" class="option" :class="option.id" :aria-pressed="selection[kind] === option.id" :disabled="!!market(kind).mine && market(kind).mine.option_id !== option.id" @click="selection[kind] = option.id">
              <b>{{ optionLabel(kind, option) }}</b><small>{{ t('当前参考赔率','Marginal odds') }} ×{{ option.marginal_odds.toFixed(3) }}</small>
            </button>
          </fieldset>
          <fieldset class="amounts" :disabled="busy || !!pending || !canBet(kind)"><legend>{{ t('本次金额','Stake amount') }} · Token</legend>
            <button v-for="amount in amounts" :key="amount" :aria-pressed="amountsByKind[kind] === amount" @click="amountsByKind[kind] = amount">{{ amount.toLocaleString() }}</button>
          </fieldset>
          <dl>
            <div><dt>{{ t('已下注 / 上限','Staked / limit') }}</dt><dd>{{ tokens(market(kind).mine?.stake) }} / 60,000</dd></div>
            <div><dt>{{ t('已锁定获胜返还','Locked return if correct') }}</dt><dd>{{ tokens(market(kind).mine?.shares) }} Token</dd></div>
            <div v-if="quote(kind) && market(kind).status === 'open'"><dt>{{ t('本笔成交赔率','This order’s effective odds') }}</dt><dd>×{{ (quote(kind)/(amountsByKind[kind]*1000)).toFixed(3) }}</dd></div>
            <div v-if="quote(kind) && market(kind).status === 'open'"><dt>{{ t('本笔获胜返还（含本金）','This order’s return if correct (includes stake)') }}</dt><dd>{{ tokens(quote(kind)) }} Token</dd></div>
          </dl>
          <p v-if="overLimit(kind)" class="error">{{ t('每个注题累计最多 60,000 Token，请降低本次金额。','Limit: 60,000 Tokens per market. Reduce the amount.') }}</p>
          <button v-if="!user" class="submit" @click="emit('login')">{{ t('登录后下注','Sign in to bet') }}</button>
          <button v-else-if="pending?.market_id === market(kind).id" class="submit" :disabled="busy || !connected" @click="submit(kind)">{{ t('核对并重试，不会重复扣款','Check and retry — no duplicate charge') }}</button>
          <button v-else class="submit" :disabled="busy || !!pending || !canBet(kind) || !selection[kind] || overLimit(kind)" @click="submit(kind)">{{ busy ? t('正在确认…','Confirming…') : t(`确认下注 ${amountsByKind[kind].toLocaleString()} Token`,`Confirm ${amountsByKind[kind].toLocaleString()} Tokens`) }}</button>
          <p v-if="market(kind).status === 'void'" class="result">{{ t('本题作废，已原额退款。','Market voided; stakes refunded in full.') }}</p>
          <p v-if="market(kind).status === 'settled'" class="result">{{ t('正确结果','Winning outcome') }}：{{ winningLabel(kind) }}<br v-if="market(kind).mine"/><template v-if="market(kind).mine">{{ t('实际返还','Paid out') }} {{ tokens(market(kind).mine.payout) }} Token</template></p>
        </template>
        <p v-else class="note">{{ t('先后手抽签完成后开放。','Opens after the first-pick draw completes.') }}</p>
      </section>
    </div>
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <details><summary>{{ t('赔率与结算规则','Pricing and settlement rules') }}</summary>
      <p>{{ t('采用固定乘积 AMM，无手续费。全场胜方初始虚拟储备各 10,000 Token，参考赔率均为 ×2。前两局比分按两局独立、双方每局胜率相同且无平局的假设，以 1:2:1 初始化概率权重；2:0、1:1、0:2 的虚拟储备分别为 10,000、5,000、10,000 Token，参考赔率分别为 ×4、×2、×4。新比例适用于新开及尚未下注的开放市场，已有下注的市场保留原储备。参考赔率不是本笔成交赔率：下注金额越大，滑点越明显。每笔买入的兑付份额锁定，后续下注不会改变已成交返还。','A fixed-product AMM is used, with no fees. Match-winner outcomes each start with 10,000 virtual Tokens and marginal odds of ×2. Assuming independent games, equal win chances and no draws, the first-two-games scores start with probability weights of 1:2:1. Virtual reserves for 2:0, 1:1 and 0:2 are 10,000, 5,000 and 10,000 Tokens, giving marginal odds of ×4, ×2 and ×4. The new weights apply to new markets and open markets with no bets; markets with existing bets retain their reserves. Marginal odds are not your execution odds: larger orders incur more slippage. Each order locks its payout shares; later orders do not change them.') }}</p>
      <p>{{ t('仅扣常驻 Token；每个注题只能选一个选项，可追加，累计最多 60,000 Token。两题额度独立。返还含本金，押错不返还。全场平局退全场胜方题；前两局含平局且不属于三个选项时退比分题。比赛取消或无法确定结果时退还对应题本金。虚拟储备不是玩家资金，平台承担赔付风险。','Only permanent Tokens can be used. One outcome per market, top-ups allowed up to 60,000 Tokens independently for each market. Returns include the stake; incorrect selections pay zero. A drawn match voids the winner market. A draw in the first two games that falls outside the three options voids that market. Cancellation or indeterminate results refund the affected stakes. Opening reserves are virtual; the platform bears payout risk.') }}</p>
    </details>
  </dialog>
</template>
<script setup>
import { computed, ref, reactive, watch, onMounted, onUnmounted } from 'vue';
import { useActivities } from './context.js';
import artwork from './assets/prediction-colored.webp';
const props = defineProps({ user:Object, connected:Boolean, online:Boolean, lang:String, state:Object, entryTarget:String });
const emit = defineEmits(['login','balance','open']);
const { room, url } = useActivities();
const kinds = ['winner','first_two'], amounts = [500,1000,5000,10000];
const data = ref({markets:[],available:false}), dialog = ref(null), entry = ref(null), opened = ref(false), busy = ref(false), error = ref(''), pending = ref(null), now = ref(Date.now());
const selection = reactive({winner:'',first_two:''}), amountsByKind = reactive({winner:500,first_two:500});
let timer, focusBefore, epoch=0, fetching=false, refreshAfter=false, balanceVersion='';
const serverOffset=ref(0);
const t=(zh,en)=>props.lang==='zh'?zh:en;
const tokens=value=>(Number(value||0)/1000).toLocaleString(undefined,{maximumFractionDigits:3});
const market=kind=>data.value.markets.find(item=>item.kind===kind);
const remaining=computed(()=>Math.max(0,Math.ceil((Date.parse(data.value.markets[0]?.minimum_until||'')-now.value-serverOffset.value)/1000))||0);
const caption=computed(()=>data.value.markets.some(item=>item.status==='open')?t('下注开放','Open'):data.value.markets.length?t('已封盘','Closed'):t('等待抽签','Awaiting draw'));
const statusText=item=>item.status==='open'?t('开放','Open'):item.status==='void'?t('已退款','Refunded'):item.status==='settled'?t('已结算','Settled'):t('已封盘','Closed');
const canBet=kind=>props.connected && props.online && data.value.available && market(kind)?.status==='open';
const quote=kind=>market(kind)?.options.find(item=>item.id===selection[kind])?.quotes?.[amountsByKind[kind]]||0;
const overLimit=kind=>Number(market(kind)?.mine?.stake||0)+amountsByKind[kind]*1000>60000000;
const optionLabel=(kind,option)=>kind==='winner'?`${option.id==='yellow'?t('黄方','Yellow'):t('白方','White')} · ${option.name}`:option.name;
const winningLabel=kind=>{const item=market(kind);const option=item?.options.find(option=>option.id===item.winner);return option?optionLabel(kind,option):'—';};
const storageKey=()=>`competition-bet-pending:${room.id}:${props.user?.id}`;
function persist(){try{if(pending.value)sessionStorage.setItem(storageKey(),JSON.stringify(pending.value));else sessionStorage.removeItem(storageKey());}catch{}}
function load(){pending.value=null;try{const value=JSON.parse(sessionStorage.getItem(storageKey())||'null');if(value?.request_id)pending.value=value;}catch{}}
async function api(body){const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),12000);try{const response=await fetch(url('/predictions'),{credentials:'same-origin',cache:'no-store',signal:controller.signal,...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})});const result=await response.json();if(!response.ok)throw Object.assign(Error('request_failed'),{detail:result.detail});return result;}finally{clearTimeout(timeout);}}
async function refresh(){if(fetching){refreshAfter=true;return;}fetching=true;const generation=epoch;try{const result=await api();if(generation!==epoch)return;data.value=result;serverOffset.value=Number(result.server_time)*1000-Date.now();for(const kind of kinds)if(market(kind)?.mine)selection[kind]=market(kind).mine.option_id;const version=JSON.stringify(result.markets.map(item=>[item.id,item.mine?.payout]));if(version!==balanceVersion){balanceVersion=version;emit('balance');}}catch{if(generation===epoch)data.value.available=false;}finally{if(generation===epoch){fetching=false;if(refreshAfter){refreshAfter=false;refresh();}}}}
async function open(){focusBefore=document.activeElement;opened.value=true;emit('open');load();dialog.value?.showModal();await refresh();}
function close(){dialog.value?.close();opened.value=false;(focusBefore?.isConnected?focusBefore:entry.value)?.focus();}
function backdrop(event){if(event.target!==dialog.value)return;const rect=dialog.value.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)close();}
async function submit(kind){
  if(busy.value||!props.user||(!pending.value&&(!canBet(kind)||overLimit(kind)||!selection[kind])))return;
  if(!pending.value){pending.value={request_id:crypto.randomUUID(),market_id:market(kind).id,option_id:selection[kind],amount:amountsByKind[kind],revision:market(kind).revision};persist();}
  const generation=epoch;busy.value=true;error.value='';
  try{await api(pending.value);if(generation!==epoch)return;pending.value=null;persist();emit('balance');await refresh();}
  catch(cause){if(generation!==epoch)return;const messages={
    competition_prediction_price_changed:['赔率已变化，请核对最新返还后重新确认。','Odds changed. Review the new return and confirm again.'],
    competition_prediction_limit:['本题最多下注 60,000 Token。','Maximum stake is 60,000 Tokens per market.'],
    competition_prediction_cannot_switch:['本题只能追加已选选项。','Top up only your existing selection.'],
    prediction_insufficient_permanent:['常驻 Token 不足。','Insufficient permanent Tokens.'],
    competition_prediction_closed:['已封盘或暂停，本次未扣款。','Closed or paused. This order was not charged.'],
    competition_prediction_invalid:['下注参数无效，请重新选择。','Invalid order; choose again.'],
    competition_prediction_request_conflict:['请求与已成交记录不一致，请刷新核对。','Request conflicts with a prior order. Refresh to check.']};
    if(messages[cause.detail]){error.value=t(...messages[cause.detail]);pending.value=null;persist();await refresh();}
    else error.value=t('尚未确认，请使用原请求核对并重试，不会重复扣款。','Not yet confirmed. Check and retry the same request without duplicate charges.');
  }finally{if(generation===epoch)busy.value=false;}
}
watch(()=>props.user?.id,()=>{epoch++;busy.value=false;fetching=false;close();data.value={markets:[],available:false};selection.winner='';selection.first_two='';load();refresh();});
watch(()=>props.state,()=>{if(props.state?.protocol==='competition-fpmm-v1')refresh();});
onMounted(()=>{load();refresh();timer=setInterval(()=>{now.value=Date.now();if(props.connected&&!busy.value)refresh();},3000);});
onUnmounted(()=>{epoch++;clearInterval(timer);});
defineExpose({close});
</script>
<style scoped>
.prediction-entry{width:78px;display:flex;flex-direction:column;align-items:center;gap:3px;border:0;border-right:1px solid var(--border-main);background:transparent;color:var(--text-main);padding:5px;cursor:pointer}.prediction-entry img{width:52px;height:52px}.prediction-entry b,.prediction-entry small{font-size:11px}
.competition-bets{box-sizing:border-box;width:940px;max-width:calc(100% - 24px);max-height:calc(100% - 24px);margin:auto;padding:24px;border:1px solid var(--border-main);border-radius:14px;background:var(--bg-main);color:var(--text-main);overflow:auto;box-shadow:0 20px 70px #0005}.competition-bets::backdrop{background:#07111caa}.competition-bets header{display:flex;align-items:center;gap:16px}.competition-bets h2{flex:1;font-size:20px;margin:0}.competition-bets header button{border:0;background:none;font-size:28px;color:inherit;cursor:pointer}.market-columns{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:20px}.market-columns section{min-width:0}.market-columns section+section{border-left:1px solid var(--border-main);padding-left:24px}h3{font-size:16px;margin:0 0 8px}.status,.timing{font-size:13px}.status{color:var(--accent)}fieldset{border:0;padding:0;margin:16px 0;display:flex;flex-wrap:wrap;gap:8px}legend{font-size:12px;margin-bottom:9px;color:var(--text-secondary)}fieldset button{min-width:0;border:1px solid var(--border-main);border-radius:8px;background:var(--bg-card);color:inherit;cursor:pointer;padding:10px}.option{width:100%;text-align:left;display:flex;flex-direction:column;gap:6px;overflow-wrap:anywhere}.option small{color:var(--text-secondary)}button[aria-pressed=true]{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 12%,var(--bg-card))}.amounts button{flex:1;font-variant-numeric:tabular-nums}.competition-bets button:disabled{opacity:.45;cursor:default}dl{display:grid;gap:12px;font-size:13px}dl div{display:flex;justify-content:space-between;gap:8px}dt{color:var(--text-secondary)}dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}.submit{width:100%;padding:12px;background:var(--accent);border:0;border-radius:8px;color:#071524;font-weight:700;cursor:pointer}.note,.result,details{font-size:12px;line-height:1.7;color:var(--text-secondary)}details{margin-top:20px}.error{color:#f09b82;font-size:13px;line-height:1.6}
@media(max-width:640px){.competition-bets{padding:16px}.market-columns{grid-template-columns:1fr;gap:24px}.market-columns section+section{padding:22px 0 0;border-left:0;border-top:1px solid var(--border-main)}.amounts button{padding:10px 5px}}
</style>
