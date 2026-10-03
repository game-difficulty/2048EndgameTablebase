<template>
  <details v-if="results?.length" class="prediction-history" open>
    <summary>{{ t('最近个人结算','Recent personal results') }}</summary>
    <div class="history-scroll"><table><thead><tr><th>{{ t('开局时间 / 注题','Started / market') }}</th><th>{{ t('本金','Stake') }}</th><th>{{ t('返还','Paid out') }}</th><th>{{ t('净收益','Net profit') }}</th></tr></thead>
      <tbody><tr v-for="result in results" :key="result.id"><td>{{ time(result.started_at) }}<small v-if="result.kind">{{ marketTitle(result.kind,lang) }} · {{ result.selection }} · {{ result.status === 'void' ? t('已退款','Refunded') : t('已结算','Settled') }}</small></td><td>{{ tokens(result.stake_units) }}</td><td>{{ tokens(result.payout_units ?? (result.stake_units + result.net_profit_units)) }}</td><td :class="{positive:result.net_profit_units>0,negative:result.net_profit_units<0}">{{ result.net_profit_units>0 ? '+' : '' }}{{ tokens(result.net_profit_units) }}</td></tr></tbody>
    </table></div><small>{{ t('金额单位：Token；返还含本金。','Amounts in Tokens; payouts include stake.') }}</small>
  </details>
</template>
<script setup>
import { marketTitle } from './matchPredictionLabels.js';
const props=defineProps({results:Array,lang:String});
const t=(zh,en)=>props.lang==='zh'?zh:en;
const tokens=value=>(Number(value||0)/1000).toLocaleString(undefined,{maximumFractionDigits:3});
const time=value=>new Date(typeof value==='number'?value*1000:value).toLocaleString(props.lang==='zh'?'zh-CN':'en-GB',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
</script>
<style scoped>
.prediction-history{margin-top:20px;font-size:13px;color:var(--text-main)}summary{cursor:pointer;font-weight:700}.history-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:12px 0}th,td{padding:9px 7px;border-bottom:1px solid var(--border-main);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}th:first-child,td:first-child{text-align:left}td small{display:block;white-space:normal;margin-top:4px;color:var(--text-secondary)}.positive{color:#299669}.negative{color:#d76652}small{color:var(--text-secondary)}
</style>
