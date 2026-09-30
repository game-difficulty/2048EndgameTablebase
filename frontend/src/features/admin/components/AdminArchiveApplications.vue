<template>
  <section class="admin-panel mt-4">
    <div class="admin-panel-head"><h2>{{ zh ? '对局补录审核' : 'Manual archive review' }}</h2><button class="action-btn-small" type="button" :disabled="busy" @click="load">{{ zh ? '刷新' : 'Refresh' }}</button></div>
    <p class="px-4 pb-3 text-text-secondary">{{ zh ? '批准后进入总榜、PB、B10 和个人统计；不进入近 168 小时榜及每周 Token 结算。' : 'Approved games enter all-time rankings, PB, B10 and profile statistics, but never the rolling 168-hour ranking or weekly Token settlement.' }}</p>
    <p v-if="error" class="admin-alert error">{{ error }}</p>
    <div v-if="!applications.length" class="p-4 text-text-secondary">{{ zh ? '此用户暂无补录申请' : 'This user has no manual archive requests' }}</div>
    <div v-for="item in applications" :key="item.id" class="border-t border-border-main p-4">
      <div class="flex flex-wrap items-baseline justify-between gap-2"><strong class="text-text-main">#{{ item.id }} · {{ item.variant }} · {{ number(item.score) }}</strong><span class="text-text-secondary">{{ item.status }}</span></div>
      <div class="mt-1 text-text-secondary">{{ item.started_at ? date(item.started_at) : '—' }} → {{ date(item.ended_at) }} · {{ number(item.moves) }} {{ zh ? '步' : 'moves' }} · {{ item.game_over ? (zh ? '死亡终盘' : 'game over') : (zh ? '未死亡局面' : 'unfinished position') }}</div>
      <div class="mt-2 grid max-w-[23rem] grid-cols-4 gap-1 rounded-lg bg-bg-main p-2">
        <span v-for="(tile,index) in item.board" :key="index" class="grid aspect-square place-items-center rounded bg-bg-card text-xs font-bold text-text-main">{{ tile || '' }}</span>
      </div>
      <p v-if="item.review_note" class="mt-2 text-text-secondary">{{ item.review_note }}</p>
      <div v-if="item.status === 'pending' || item.status === 'approved'" class="mt-3 flex flex-wrap items-end gap-2">
        <label class="admin-form-row min-w-[18rem] flex-1"><span>{{ zh ? '审核备注（必填）' : 'Review note (required)' }}</span><textarea v-model.trim="notes[item.id]" class="admin-input min-h-[4rem]" /></label>
        <button v-if="item.status === 'pending'" class="action-btn-small" type="button" :disabled="busy || !notes[item.id]" @click="decide(item,true)">{{ zh ? '批准归档' : 'Approve archive' }}</button>
        <button v-if="item.status === 'pending'" class="action-btn-small" type="button" :disabled="busy || !notes[item.id]" @click="decide(item,false)">{{ zh ? '拒绝' : 'Reject' }}</button>
        <button v-if="item.status === 'approved'" class="action-btn-small" type="button" :disabled="busy || !notes[item.id]" @click="revoke(item)">{{ zh ? '撤销归档资格' : 'Revoke archive' }}</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { userError } from '../../../services/errors/userError.js';
import { computed, reactive, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { adminClient } from '../../../services/admin/adminClient';
const { locale }=useI18n();
const props=defineProps({active:Boolean,userId:{type:Number,default:null}});
const emit=defineEmits(['changed']);
const zh=computed(()=>String(locale.value).startsWith('zh'));
const applications=ref([]),busy=ref(false),error=ref('');const notes=reactive({});
const number=value=>new Intl.NumberFormat(zh.value?'zh-CN':'en-US').format(Number(value)||0);
const date=value=>value?new Date(value*1000).toLocaleString(zh.value?'zh-CN':'en-US'):'—';
async function load(){busy.value=true;error.value='';try{applications.value=(await adminClient.archiveApplications(props.userId)).applications||[];}catch(e){error.value=userError(e);}finally{busy.value=false;}}
async function decide(item,approved){busy.value=true;error.value='';try{await adminClient.decideArchiveApplication(item.id,approved,notes[item.id]||'');delete notes[item.id];await load();emit('changed');}catch(e){error.value=userError(e);busy.value=false;}}
async function revoke(item){busy.value=true;error.value='';try{await adminClient.revokeArchiveApplication(item.id,notes[item.id]||'');delete notes[item.id];await load();emit('changed');}catch(e){error.value=userError(e);busy.value=false;}}
watch(()=>[props.active,props.userId],([active])=>{if(active)load();},{immediate:true});
</script>

<style scoped>
.admin-panel{border:1px solid var(--border-main);border-radius:1rem;background:color-mix(in srgb,var(--bg-card) 94%,transparent);color:var(--text-main);overflow:hidden}.admin-panel-head{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:1rem}.admin-panel-head h2{color:var(--text-main);font-size:var(--font-ui-lg);font-weight:900}.admin-form-row{display:grid;gap:.4rem}.admin-form-row span{color:var(--text-secondary);font-size:var(--font-ui-xs);font-weight:900}.admin-input{width:100%;border:1px solid var(--border-main);border-radius:.8rem;background:color-mix(in srgb,var(--bg-main) 82%,transparent);color:var(--text-main);padding:.65rem .8rem;resize:vertical}.admin-alert{margin:0 1rem 1rem;border-radius:.8rem;padding:.7rem .85rem}.admin-alert.error{border:1px solid rgba(239,68,68,.35);background:rgba(239,68,68,.12);color:rgb(239,68,68)}
</style>
