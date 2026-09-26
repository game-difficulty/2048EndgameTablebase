<template>
  <section class="profile-reviews">
    <header>
      <h2>{{ zh ? '头像与昵称备查' : 'Profile Changes' }}</h2>
      <select v-model="status" :aria-label="zh ? '状态' : 'Status'" @change="page = 1; load()">
        <option value="pending">{{ zh ? '待检查' : 'Pending' }}</option>
        <option value="all">{{ zh ? '全部' : 'All' }}</option>
      </select>
      <button :disabled="busy" :title="zh ? '刷新' : 'Refresh'" @click="load"><RefreshCw :size="16" /></button>
    </header>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-if="!items.length && !busy">{{ zh ? '暂无记录' : 'No records' }}</p>
    <article v-for="item in items" :key="item.id">
      <div class="review-content">
        <strong>#{{ item.user_id }} · {{ item.display_name }}</strong>
        <small>{{ item.created_at }} · {{ stateLabel(item.status) }}<template v-if="item.ip_address"> · {{ item.ip_address }}</template></small>
        <img v-if="item.change_type === 'avatar' && item.is_current" :src="getBackendUrl('/media/avatars/' + item.new_value)" :alt="zh ? '修改后的头像' : 'Changed avatar'" />
        <span v-else-if="item.change_type === 'display_name'">{{ item.old_value || '—' }} → {{ item.new_value }}</span>
        <small v-if="!item.is_current">{{ zh ? '已被后续修改替代' : 'Superseded by a later change' }}</small>
      </div>
      <div v-if="item.is_current && ['pending', 'reviewed'].includes(item.status)" class="review-actions">
        <button v-if="item.status === 'pending'" :disabled="busy" @click="decide(item, 'keep')"><Check :size="16" />{{ zh ? '已检查' : 'Reviewed' }}</button>
        <button :disabled="busy" @click="confirmItem = item"><Undo2 :size="16" />{{ zh ? '恢复默认' : 'Reset to default' }}</button>
      </div>
    </article>
    <footer><button :disabled="busy || page <= 1" :aria-label="zh ? '上一页' : 'Previous'" @click="page--; load()"><ChevronLeft :size="16" /></button>
      <span>{{ page }} / {{ Math.max(1, Math.ceil(total / 20)) }}</span>
      <button :disabled="busy || page * 20 >= total" :aria-label="zh ? '下一页' : 'Next'" @click="page++; load()"><ChevronRight :size="16" /></button></footer>
    <div v-if="confirmItem" class="review-confirm" role="dialog" aria-modal="true" :aria-label="zh ? '确认恢复默认' : 'Confirm reset'">
      <div><p>{{ zh ? '确认将这次修改恢复为默认头像或默认用户名？原有修改冷却保留。' : 'Reset this change to a default avatar or username? The existing cooldown remains.' }}</p>
        <button :disabled="busy" @click="decide(confirmItem, 'revoke')">{{ zh ? '确认撤销' : 'Confirm reset' }}</button>
        <button :disabled="busy" @click="confirmItem = null">{{ zh ? '取消' : 'Cancel' }}</button>
      </div>
    </div>
  </section>
</template>
<script setup>
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { RefreshCw, Check, Undo2, ChevronLeft, ChevronRight } from '@lucide/vue';
import { adminClient } from '../../../services/admin/adminClient';
import { getBackendUrl } from '../../../services/runtime/backendUrl';
const props = defineProps({ active: Boolean });
const { locale } = useI18n();
const zh = computed(() => locale.value === 'zh');
const items = ref([]), page = ref(1), total = ref(0), status = ref('pending'), busy = ref(false), error = ref(''), confirmItem = ref(null);
const stateLabel = value => ({ pending: zh.value ? '待检查' : 'Pending', reviewed: zh.value ? '已检查' : 'Reviewed', revoked: zh.value ? '已撤销' : 'Revoked', superseded: zh.value ? '已被后续修改替代' : 'Superseded' }[value] || value);
async function load() {
  busy.value = true; error.value = '';
  try { const result = await adminClient.profileReviews(page.value, status.value); items.value = result.items; total.value = result.total; }
  catch (e) { error.value = e.message; }
  finally { busy.value = false; }
}
async function decide(item, action) {
  busy.value = true; error.value = '';
  try { await adminClient.reviewProfile(item.id, action); confirmItem.value = null; await load(); }
  catch (e) { error.value = e.message; }
  finally { busy.value = false; }
}
watch(() => props.active, value => { if (value) load(); }, { immediate: true });
</script>
<style scoped>
.profile-reviews { color:var(--text-main);margin-bottom:24px; }
header,footer,.review-actions { display:flex;align-items:center;gap:10px; }
h2 { font-size:18px;margin-right:auto; }
article { display:flex;justify-content:space-between;gap:16px;padding:14px 0;border-bottom:1px solid var(--border-main); }
.review-content { display:flex;flex-direction:column;gap:6px;overflow-wrap:anywhere;min-width:0; }
small { color:var(--text-secondary); }
img { width:64px;height:64px;object-fit:cover;border-radius:6px; }
button,select { display:inline-flex;align-items:center;gap:6px;background:var(--bg-card);color:var(--text-main);border:1px solid var(--border-main);border-radius:6px;padding:8px; }
button:disabled { opacity:.5; } footer { justify-content:flex-end;padding:12px 0; }
.review-confirm { position:fixed;inset:0;z-index:100;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.5); }
.review-confirm>div { background:var(--bg-card);padding:24px;border:1px solid var(--border-main);border-radius:8px;max-width:480px; }
</style>
