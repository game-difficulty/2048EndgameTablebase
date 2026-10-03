<template>
  <section class="admin-panel">
    <div class="admin-panel-head">
      <h2>{{ zh ? '2048Verse 历史继承审核' : '2048Verse history claims' }}</h2>
      <button class="action-btn-small" type="button" :disabled="busy" @click="load">{{ zh ? '刷新' : 'Refresh' }}</button>
    </div>
    <p v-if="error" class="admin-alert error">{{ error }}</p>
    <div v-if="!claims.length" class="p-4 text-text-secondary">{{ zh ? '此用户暂无 Verse 绑定申请' : 'This user has no Verse account claims' }}</div>
    <div v-for="claim in claims" :key="claim.id" class="border-t border-border-main p-4">
      <div class="font-bold text-text-main">{{ zh ? '申请' : 'Claim' }} #{{ claim.id }}</div>
      <div class="mt-1 text-text-secondary">
        {{ zh ? 'Verse 账号' : 'Verse account' }}：<strong class="text-text-main">{{ claim.username }}</strong>
        · {{ zh ? '绑定到本站用户' : 'Bind to site user' }} #{{ claim.user_id }}
      </div>
      <div class="text-text-secondary">{{ claim.status }} · {{ new Date(claim.requested_at * 1000).toLocaleString() }}</div>
      <div v-if="claim.error" class="admin-alert error">{{ claim.error }}</div>
      <div v-if="claim.counts && Object.keys(claim.counts).length" class="text-text-secondary">
        {{ Object.entries(claim.counts).map(([mode,count])=>mode+': '+count).join(' · ') }}
      </div>
      <div v-if="claim.can_manage !== false && (['pending','approved','failed','complete'].includes(claim.status) || stale(claim))" class="mt-3 flex flex-wrap items-end gap-2">
        <label class="admin-form-row min-w-[18rem] flex-1">
          <span>{{ zh ? '审核备注（可选）' : 'Review note (optional)' }}</span>
          <textarea v-model.trim="notes[claim.id]" class="admin-input min-h-[4rem]" />
        </label>
        <button v-if="['pending','approved'].includes(claim.status)" class="action-btn-small" type="button" :disabled="busy" @click="decide(claim,true)">{{ zh ? '批准并导入' : 'Approve and import' }}</button>
        <button v-if="['pending','approved'].includes(claim.status)" class="action-btn-small" type="button" :disabled="busy" @click="decide(claim,false)">{{ zh ? '拒绝' : 'Reject' }}</button>
        <button v-if="claim.status === 'failed' || stale(claim)" class="action-btn-small" type="button" :disabled="busy" @click="act(claim,'retry')">{{ zh ? '重试导入' : 'Retry import' }}</button>
        <button v-if="claim.status === 'complete'" class="action-btn-small" type="button" :disabled="busy" @click="act(claim,'revoke')">{{ zh ? '撤销归属及排名' : 'Revoke ownership and ranks' }}</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { userError } from '../../../services/errors/userError.js';
import { adminClient } from '../../../services/admin/adminClient';

const { locale } = useI18n();
const props = defineProps({ active: Boolean, userId: { type: Number, default: null } });
const emit = defineEmits(['changed']);
const zh = computed(() => String(locale.value).startsWith('zh'));
const claims = ref([]);
const notes = reactive({});
const busy = ref(false);
const error = ref('');
const stale = claim => claim.status === 'importing' && Date.now() / 1000 - claim.updated_at >= 7200;
async function load() {
  busy.value = true; error.value = '';
  try { claims.value = (await adminClient.verseClaims(props.userId)).claims || []; }
  catch (e) { error.value = userError(e); }
  finally { busy.value = false; }
}
async function decide(claim, approved) {
  busy.value = true; error.value = '';
  try { await adminClient.decideVerseClaim(claim.id, approved, notes[claim.id] || ''); delete notes[claim.id]; await load(); emit('changed'); }
  catch (e) { error.value = userError(e); busy.value = false; }
}
async function act(claim, action) {
  busy.value = true; error.value = '';
  try {
    if (action === 'retry') await adminClient.retryVerseClaim(claim.id, notes[claim.id] || '');
    else await adminClient.revokeVerseClaim(claim.id, notes[claim.id] || '');
    delete notes[claim.id]; await load(); emit('changed');
  } catch (e) { error.value = userError(e); busy.value = false; }
}
watch(() => [props.active, props.userId], ([active]) => { if (active) load(); }, { immediate: true });
</script>

<style scoped>
.admin-panel {
  border: 1px solid var(--border-main);
  border-radius: 1rem;
  background: color-mix(in srgb, var(--bg-card) 94%, transparent);
  color: var(--text-main);
  overflow: hidden;
}

.admin-panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 1rem;
}

.admin-panel-head h2 {
  color: var(--text-main);
  font-size: var(--font-ui-lg);
  font-weight: 900;
}

.admin-form-row {
  display: grid;
  gap: 0.4rem;
}

.admin-form-row span {
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 900;
}

.admin-input {
  width: 100%;
  border: 1px solid var(--border-main);
  border-radius: 0.8rem;
  background: color-mix(in srgb, var(--bg-main) 82%, transparent);
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  font-weight: 750;
  line-height: 1.5;
  outline: none;
  padding: 0.65rem 0.8rem;
  resize: vertical;
}

.admin-input:focus {
  border-color: var(--accent);
}

.admin-alert {
  margin: 0 1rem 1rem;
  border-radius: 0.8rem;
  padding: 0.7rem 0.85rem;
  font-size: var(--font-ui-sm);
  font-weight: 800;
}

.admin-alert.error {
  border: 1px solid rgba(239, 68, 68, 0.35);
  background: rgba(239, 68, 68, 0.12);
  color: rgb(239, 68, 68);
}
</style>
