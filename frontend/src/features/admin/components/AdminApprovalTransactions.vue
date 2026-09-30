<template>
  <section class="approval-panel">
    <div class="approval-head">
      <div>
        <h2>{{ copy.title }}</h2>
        <p>{{ copy.subtitle }}</p>
      </div>
      <button class="action-btn-small" type="button" :disabled="busy" @click="load">{{ copy.refresh }}</button>
    </div>

    <form class="approval-toolbar" @submit.prevent="search">
      <select v-model="kind" class="approval-input" :aria-label="copy.kind">
        <option value="all">{{ copy.kindAll }}</option>
        <option value="verse">{{ copy.kindVerse }}</option>
        <option value="archive">{{ copy.kindArchive }}</option>
      </select>
      <select v-model="stage" class="approval-input" :aria-label="copy.stage">
        <option value="all">{{ copy.stageAll }}</option>
        <option value="pending">{{ copy.stagePending }}</option>
        <option value="processing">{{ copy.stageProcessing }}</option>
        <option value="approved">{{ copy.stageApproved }}</option>
        <option value="rejected">{{ copy.stageRejected }}</option>
        <option value="revoked">{{ copy.stageRevoked }}</option>
      </select>
      <input v-model.trim="query" class="approval-input approval-search" :placeholder="copy.searchPlaceholder" />
      <button class="action-btn-small" type="submit" :disabled="busy">{{ copy.search }}</button>
    </form>

    <div v-if="error" class="approval-alert">{{ error }}</div>
    <div class="approval-table-wrap">
      <table class="approval-table">
        <thead><tr><th>{{ copy.updated }}</th><th>{{ copy.kind }}</th><th>{{ copy.user }}</th><th>{{ copy.subject }}</th><th>{{ copy.stage }}</th><th>{{ copy.action }}</th></tr></thead>
        <tbody>
          <tr v-if="!busy && !items.length"><td colspan="6" class="approval-empty">{{ copy.empty }}</td></tr>
          <tr v-for="item in items" :key="item.key">
            <td class="approval-time">{{ date(item.updated_at) }}</td>
            <td><span class="kind-pill">{{ kindLabel(item.kind) }}</span></td>
            <td><strong>{{ item.user?.display_name || `#${item.user_id}` }}</strong><small>{{ item.user?.email || '' }}</small></td>
            <td>
              <strong>{{ subject(item) }}</strong>
              <small>{{ subjectDetail(item) }}</small>
            </td>
            <td><span :class="['stage-pill', `is-${item.stage}`, { 'is-failed': item.raw_status === 'failed' }]">{{ stageLabel(item) }}</span></td>
            <td><button class="row-action" type="button" @click="open(item)">{{ item.actions.length ? copy.review : copy.details }}</button></td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="approval-pagination">
      <button type="button" :disabled="busy || page.page <= 1" @click="go(page.page - 1)">&lt;</button>
      <span>{{ copy.pagination.replace('{page}', page.page).replace('{pages}', page.page_count).replace('{total}', page.total) }}</span>
      <button type="button" :disabled="busy || page.page >= page.page_count" @click="go(page.page + 1)">&gt;</button>
    </div>

    <div v-if="selected" class="approval-modal" @click.self="close">
      <section class="approval-dialog">
        <header>
          <div><span>{{ kindLabel(selected.kind) }} · #{{ selected.transaction_id }}</span><h3>{{ subject(selected) }}</h3></div>
          <button class="action-btn-small" type="button" :disabled="acting" @click="close">{{ copy.close }}</button>
        </header>
        <dl class="approval-detail-grid">
          <div><dt>{{ copy.user }}</dt><dd>{{ selected.user?.display_name || `#${selected.user_id}` }}<small>{{ selected.user?.email || '' }}</small></dd></div>
          <div><dt>{{ copy.stage }}</dt><dd>{{ stageLabel(selected) }}</dd></div>
          <div><dt>{{ copy.requested }}</dt><dd>{{ date(selected.requested_at) }}</dd></div>
          <div><dt>{{ copy.updated }}</dt><dd>{{ date(selected.updated_at) }}</dd></div>
          <div v-if="selected.kind === 'archive'"><dt>{{ copy.game }}</dt><dd>{{ selected.variant }} · {{ number(selected.score) }} · {{ number(selected.moves) }} {{ copy.moves }}<small>{{ selected.started_at ? date(selected.started_at) : '—' }} → {{ date(selected.ended_at) }}</small></dd></div>
          <div v-else><dt>{{ copy.records }}</dt><dd>{{ counts(selected.details) }}</dd></div>
        </dl>
        <p v-if="selected.error" class="approval-alert">{{ selected.error }}</p>
        <p v-if="selected.review_note" class="approval-note"><strong>{{ copy.previousNote }}</strong>{{ selected.review_note }}</p>
        <label v-if="selected.actions.length" class="approval-note-field"><span>{{ noteRequired ? copy.noteRequired : copy.noteOptional }}</span><textarea v-model.trim="note" class="approval-input" rows="3" /></label>
        <p v-if="actionError" class="approval-alert">{{ actionError }}</p>
        <div v-if="selected.actions.length" class="approval-actions">
          <button v-if="selected.actions.includes('reject')" type="button" class="action-btn-small danger" :disabled="cannotAct" @click="act('reject')">{{ copy.reject }}</button>
          <button v-if="selected.actions.includes('retry')" type="button" class="action-btn-small" :disabled="cannotAct" @click="act('retry')">{{ copy.retry }}</button>
          <button v-if="selected.actions.includes('revoke')" type="button" class="action-btn-small danger" :disabled="cannotAct" @click="act('revoke')">{{ copy.revoke }}</button>
          <button v-if="selected.actions.includes('approve')" type="button" class="action-btn-small btn-prominent" :disabled="cannotAct" @click="act('approve')">{{ copy.approve }}</button>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
import { userError } from '../../../services/errors/userError.js';
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { adminClient } from '../../../services/admin/adminClient';

const props = defineProps({ active: Boolean });
const emit = defineEmits(['changed']);
const { locale } = useI18n();
const zh = computed(() => String(locale.value).startsWith('zh'));
const text = {
  zh: { title:'审批事务',subtitle:'查询近期 Verse 继承与对局补录申请，并在此完成审核。',refresh:'刷新',kind:'事务类型',kindAll:'全部类型',kindVerse:'Verse 继承',kindArchive:'对局补录',stage:'状态',stageAll:'全部状态',stagePending:'待审批',stageProcessing:'处理中',stageApproved:'已批准',stageRejected:'已拒绝',stageRevoked:'已撤销',statusApproved:'等待导入',statusImporting:'导入中',statusFailed:'导入失败',statusComplete:'已完成',searchPlaceholder:'用户名、邮箱、外站账号、用户 ID 或事务 ID',search:'查询',updated:'最近更新',user:'申请用户',subject:'申请内容',action:'操作',empty:'没有符合条件的审批事务',review:'处理',details:'详情',pagination:'第 {page} / {pages} 页，共 {total} 条',close:'关闭',requested:'申请时间',game:'对局',records:'记录数量',moves:'步',previousNote:'最近备注：',noteRequired:'审核备注（必填）',noteOptional:'审核备注（可选）',reject:'拒绝',retry:'重试导入',revoke:'撤销资格',approve:'批准',verseSubject:'Verse 账号' },
  en: { title:'Approval transactions',subtitle:'Search recent Verse claims and manual archive requests, then review them here.',refresh:'Refresh',kind:'Type',kindAll:'All types',kindVerse:'Verse claim',kindArchive:'Manual archive',stage:'Status',stageAll:'All statuses',stagePending:'Pending',stageProcessing:'Processing',stageApproved:'Approved',stageRejected:'Rejected',stageRevoked:'Revoked',statusApproved:'Awaiting import',statusImporting:'Importing',statusFailed:'Import failed',statusComplete:'Complete',searchPlaceholder:'Name, email, external account, user ID, or transaction ID',search:'Search',updated:'Updated',user:'Applicant',subject:'Request',action:'Action',empty:'No approval transactions match these filters',review:'Review',details:'Details',pagination:'Page {page} / {pages}, {total} total',close:'Close',requested:'Requested',game:'Game',records:'Record counts',moves:'moves',previousNote:'Latest note: ',noteRequired:'Review note (required)',noteOptional:'Review note (optional)',reject:'Reject',retry:'Retry import',revoke:'Revoke',approve:'Approve',verseSubject:'Verse account' },
};
const copy = computed(() => zh.value ? text.zh : text.en);
const items = ref([]), page = ref({ page:1,page_count:1,total:0,page_size:30 });
const query = ref(''), kind = ref('all'), stage = ref('all'), busy = ref(false), error = ref('');
const selected = ref(null), note = ref(''), acting = ref(false), actionError = ref('');
const noteRequired = computed(() => selected.value?.kind === 'archive');
const cannotAct = computed(() => acting.value || (noteRequired.value && !note.value));
const number = value => Number(value || 0).toLocaleString(zh.value ? 'zh-CN' : 'en-US');
const date = value => value ? new Date(Number(value) * 1000).toLocaleString(zh.value ? 'zh-CN' : 'en-US') : '—';
const kindLabel = value => value === 'verse' ? copy.value.kindVerse : copy.value.kindArchive;
const stageLabel = item => {
  if (item.kind === 'verse') {
    const verseStatus = { approved:copy.value.statusApproved, importing:copy.value.statusImporting, failed:copy.value.statusFailed, complete:copy.value.statusComplete }[item.raw_status];
    if (verseStatus) return verseStatus;
  }
  return {pending:copy.value.stagePending,processing:copy.value.stageProcessing,approved:copy.value.stageApproved,rejected:copy.value.stageRejected,revoked:copy.value.stageRevoked}[item.stage] || item.raw_status;
};
const subject = item => item.kind === 'verse' ? `${copy.value.verseSubject} · ${item.subject}` : `${item.variant} · ${number(item.score)}`;
const subjectDetail = item => item.kind === 'archive' ? `${date(item.ended_at)} · ${number(item.moves)} ${copy.value.moves}` : counts(item.details);
const counts = details => Object.entries(details || {}).map(([key,value]) => `${key}: ${number(value)}`).join(' · ') || '—';
async function load(){busy.value=true;error.value='';try{const result=await adminClient.approvalTransactions({q:query.value,kind:kind.value,stage:stage.value,page:page.value.page,pageSize:30});items.value=result.transactions||[];page.value=result.page||page.value;}catch(e){error.value=userError(e);}finally{busy.value=false;}}
function search(){page.value={...page.value,page:1};load();}
function go(value){page.value={...page.value,page:Math.max(1,Number(value)||1)};load();}
function open(item){selected.value=item;note.value='';actionError.value='';}
function close(){if(!acting.value)selected.value=null;}
async function act(action){if(!selected.value||cannotAct.value)return;acting.value=true;actionError.value='';try{const item=selected.value;if(item.kind==='verse'){if(action==='approve'||action==='reject')await adminClient.decideVerseClaim(item.transaction_id,action==='approve',note.value);else if(action==='retry')await adminClient.retryVerseClaim(item.transaction_id,note.value);else await adminClient.revokeVerseClaim(item.transaction_id,note.value);}else if(action==='approve'||action==='reject')await adminClient.decideArchiveApplication(item.transaction_id,action==='approve',note.value);else await adminClient.revokeArchiveApplication(item.transaction_id,note.value);selected.value=null;await load();emit('changed');}catch(e){actionError.value=userError(e);}finally{acting.value=false;}}
watch(() => props.active, active => { if(active && !items.value.length) load(); }, { immediate:true });
defineExpose({ load });
</script>

<style scoped>
.approval-panel{border:1px solid var(--border-main);border-radius:1.25rem;background:color-mix(in srgb,var(--bg-card) 90%,transparent);box-shadow:0 14px 34px rgba(15,23,42,.08);overflow:hidden}.approval-head{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;padding:1.25rem}.approval-head h2{font-size:var(--font-ui-lg);font-weight:900;color:var(--text-main)}.approval-head p{margin-top:.3rem;color:var(--text-secondary);font-size:var(--font-ui-sm);font-weight:650}.approval-toolbar{display:grid;grid-template-columns:10rem 10rem minmax(16rem,1fr) auto;gap:.65rem;padding:0 1.25rem 1rem}.approval-input{min-width:0;border:1px solid var(--border-main);border-radius:.8rem;background:color-mix(in srgb,var(--bg-main) 78%,transparent);color:var(--text-main);font:700 var(--font-ui-sm)/1.4 inherit;padding:.65rem .75rem;outline:none}.approval-input:focus{border-color:var(--accent)}.approval-table-wrap{overflow:auto;border-top:1px solid var(--border-main)}.approval-table{width:100%;min-width:850px;border-collapse:collapse;color:var(--text-main);font-size:var(--font-ui-sm)}.approval-table th,.approval-table td{padding:.8rem 1rem;border-bottom:1px solid color-mix(in srgb,var(--border-main) 75%,transparent);text-align:left;vertical-align:middle}.approval-table th{color:var(--text-secondary);font-size:var(--font-ui-xs);font-weight:900;text-transform:uppercase}.approval-table strong,.approval-table small{display:block}.approval-table small{max-width:22rem;margin-top:.15rem;color:var(--text-secondary);font-size:.72rem}.approval-time{white-space:nowrap;color:var(--text-secondary)}.kind-pill,.stage-pill{display:inline-flex;border-radius:999px;padding:.28rem .55rem;font-size:.72rem;font-weight:900;white-space:nowrap}.kind-pill{background:color-mix(in srgb,var(--accent) 12%,var(--bg-main));color:var(--accent)}.stage-pill{background:var(--bg-main);color:var(--text-secondary)}.stage-pill.is-pending{background:rgba(245,158,11,.14);color:rgb(217,119,6)}.stage-pill.is-approved{background:rgba(34,197,94,.14);color:rgb(22,163,74)}.stage-pill.is-processing{background:rgba(59,130,246,.14);color:rgb(37,99,235)}.stage-pill.is-rejected,.stage-pill.is-revoked{background:rgba(239,68,68,.12);color:rgb(220,38,38)}.row-action{border:0;background:transparent;color:var(--accent);font-weight:900;cursor:pointer}.approval-empty{text-align:center;color:var(--text-secondary)}.approval-alert{margin:0 1.25rem 1rem;border:1px solid rgba(239,68,68,.3);border-radius:.75rem;background:rgba(239,68,68,.1);color:rgb(220,38,38);padding:.7rem .8rem;font-size:var(--font-ui-sm);font-weight:750}.approval-pagination{display:flex;align-items:center;justify-content:center;gap:1rem;padding:1rem;color:var(--text-secondary);font-size:var(--font-ui-xs);font-weight:900}.approval-pagination button{width:2.2rem;height:2.2rem;border:1px solid var(--border-main);border-radius:.7rem;background:var(--bg-main);color:var(--text-main);font-weight:900}.approval-pagination button:disabled{opacity:.4}.approval-modal{position:fixed;inset:0;z-index:80;display:grid;place-items:center;padding:1rem;background:rgba(15,23,42,.48);backdrop-filter:blur(5px)}.approval-dialog{width:min(42rem,calc(100vw - 2rem));max-height:calc(100vh - 2rem);overflow:auto;border:1px solid var(--border-main);border-radius:1.2rem;background:var(--bg-card);box-shadow:0 24px 80px rgba(15,23,42,.3);padding:1.25rem}.approval-dialog header{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem}.approval-dialog header span{color:var(--text-secondary);font-size:var(--font-ui-xs);font-weight:900;text-transform:uppercase}.approval-dialog h3{margin-top:.25rem;color:var(--text-main);font-size:1.4rem;font-weight:900}.approval-detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-top:1rem}.approval-detail-grid>div{border-radius:.8rem;background:var(--bg-main);padding:.75rem}.approval-detail-grid dt{color:var(--text-secondary);font-size:.72rem;font-weight:900}.approval-detail-grid dd{margin-top:.25rem;color:var(--text-main);font-weight:800}.approval-detail-grid small{display:block;color:var(--text-secondary);font-size:.72rem}.approval-note{margin-top:1rem;color:var(--text-secondary);font-size:var(--font-ui-sm)}.approval-note strong{color:var(--text-main)}.approval-note-field{display:grid;gap:.4rem;margin-top:1rem}.approval-note-field span{color:var(--text-secondary);font-size:var(--font-ui-xs);font-weight:900}.approval-actions{display:flex;justify-content:flex-end;gap:.55rem;margin-top:1rem}.action-btn-small.danger{color:rgb(220,38,38)}
@media(max-width:760px){.approval-toolbar{grid-template-columns:1fr 1fr}.approval-search{grid-column:1/-1}.approval-toolbar .action-btn-small{grid-column:1/-1}.approval-detail-grid{grid-template-columns:1fr}.approval-head{align-items:center}}
.stage-pill.is-failed{background:rgba(239,68,68,.12);color:rgb(220,38,38)}
</style>
