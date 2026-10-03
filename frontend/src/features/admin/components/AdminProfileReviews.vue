<template>
  <section class="review-panel">
    <header class="review-head">
      <div>
        <h2>{{ copy.title }}</h2>
        <p>{{ copy.subtitle }}</p>
      </div>
      <div class="review-head-actions">
        <button type="button" class="review-bulk" :disabled="busy || pendingTotal === 0" @click="confirmAll = true">
          <CheckCheck :size="16" />{{ copy.bulkApprove }}<span>{{ pendingTotal }}</span>
        </button>
        <button type="button" class="review-refresh" :disabled="busy" :title="copy.refresh" :aria-label="copy.refresh" @click="load">
          <RefreshCw :size="17" />
        </button>
      </div>
    </header>

    <form class="review-toolbar" @submit.prevent="search">
      <UiSelect
        class="review-filter"
        :model-value="status"
        :options="statusOptions"
        :aria-label="copy.status"
        :disabled="busy"
        trigger-class="review-filter-trigger"
        @change="setStatus"
      />
      <UiSelect
        class="review-filter"
        :model-value="changeType"
        :options="typeOptions"
        :aria-label="copy.type"
        :disabled="busy"
        trigger-class="review-filter-trigger"
        @change="setChangeType"
      />
      <input v-model="query" class="review-search" :aria-label="copy.search" :placeholder="copy.searchPlaceholder" />
      <button type="submit" class="action-btn-small review-search-button" :disabled="busy">
        <Search :size="16" />{{ copy.search }}
      </button>
    </form>

    <p v-if="error" class="review-error" role="alert">{{ error }}</p>
    <p v-if="feedback" class="review-feedback" role="status">{{ feedback }}</p>
    <div class="review-table-wrap">
      <table class="review-table">
        <thead>
          <tr>
            <th>{{ copy.changedAt }}</th>
            <th>{{ copy.user }}</th>
            <th>{{ copy.type }}</th>
            <th>{{ copy.change }}</th>
            <th>{{ copy.status }}</th>
            <th>{{ copy.ipAddress }}</th>
            <th>{{ copy.action }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-if="busy && !items.length"><td colspan="7" class="review-empty">{{ copy.loading }}</td></tr>
          <tr v-else-if="!items.length && !error"><td colspan="7" class="review-empty">{{ copy.empty }}</td></tr>
          <tr v-for="item in items" :key="item.id">
            <td class="review-time">{{ formatDate(item.created_at) }}</td>
            <td>
              <strong>{{ item.display_name || `#${item.user_id}` }}</strong>
              <small>#{{ item.user_id }} · {{ item.email || '—' }}</small>
            </td>
            <td><span class="review-type">{{ typeLabel(item.change_type) }}</span></td>
            <td>
              <div v-if="item.change_type === 'avatar'" class="review-avatar-change">
                <img v-if="item.is_current" :src="getBackendUrl(`/media/avatars/${item.new_value}`)" :alt="copy.newAvatar" loading="lazy" />
                <span>{{ item.is_current ? copy.newAvatar : copy.supersededHint }}</span>
              </div>
              <div v-else class="review-name-change">
                <span>{{ item.old_value || '—' }}</span><span aria-hidden="true">→</span><strong>{{ item.new_value }}</strong>
                <small v-if="!item.is_current">{{ copy.supersededHint }}</small>
              </div>
            </td>
            <td><span :class="['review-status', `is-${item.status}`]">{{ statusLabel(item.status) }}</span></td>
            <td class="review-ip">{{ item.ip_address || '—' }}</td>
            <td>
              <div v-if="item.can_manage !== false && (item.status === 'pending' || (item.is_current && item.status === 'reviewed'))" class="review-actions">
                <button v-if="item.status === 'pending'" type="button" :disabled="busy" @click="decide(item, 'keep')">
                  <Check :size="15" />{{ copy.markReviewed }}
                </button>
                <button v-if="item.is_current" type="button" class="is-danger" :disabled="busy" @click="confirmItem = item">
                  <Undo2 :size="15" />{{ copy.reset }}
                </button>
              </div>
              <span v-else class="review-muted">—</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <footer class="review-pagination">
      <span>{{ copy.pagination.replace('{page}', page).replace('{pages}', pageCount).replace('{total}', total) }}</span>
      <div>
        <button type="button" :disabled="busy || page <= 1" :aria-label="copy.previous" @click="go(page - 1)"><ChevronLeft :size="17" /></button>
        <button type="button" :disabled="busy || page >= pageCount" :aria-label="copy.next" @click="go(page + 1)"><ChevronRight :size="17" /></button>
      </div>
    </footer>

    <div v-if="confirmItem" class="review-overlay" @click.self="confirmItem = null">
      <section class="review-dialog" role="dialog" aria-modal="true" :aria-label="copy.confirmTitle">
        <h3>{{ copy.confirmTitle }}</h3>
        <p>{{ copy.confirmText }}</p>
        <div>
          <button type="button" class="action-btn-small" :disabled="busy" @click="confirmItem = null">{{ copy.cancel }}</button>
          <button type="button" class="action-btn-small is-danger" :disabled="busy" @click="decide(confirmItem, 'revoke')">{{ copy.confirm }}</button>
        </div>
      </section>
    </div>
    <div v-if="confirmAll" class="review-overlay" @click.self="!busy && (confirmAll = false)">
      <section class="review-dialog" role="dialog" aria-modal="true" :aria-label="copy.bulkConfirmTitle">
        <h3>{{ copy.bulkConfirmTitle }}</h3>
        <p>{{ copy.bulkConfirmText.replace('{count}', pendingTotal) }}</p>
        <div>
          <button type="button" class="action-btn-small" :disabled="busy" @click="confirmAll = false">{{ copy.cancel }}</button>
          <button type="button" class="action-btn-small" :disabled="busy" @click="approveAllPending">{{ copy.bulkConfirm }}</button>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { Check, CheckCheck, ChevronLeft, ChevronRight, RefreshCw, Search, Undo2 } from '@lucide/vue';
import UiSelect from '../../../components/UiSelect.vue';
import { adminClient } from '../../../services/admin/adminClient';
import { userError } from '../../../services/errors/userError.js';
import { getBackendUrl } from '../../../services/runtime/backendUrl';

const props = defineProps({ active: Boolean });
const { locale } = useI18n();
const zh = computed(() => String(locale.value).startsWith('zh'));
const labels = {
  zh: {
    title: '头像与昵称备查', subtitle: '按最近变更时间查看头像与昵称记录。', refresh: '刷新',
    status: '状态', statusAll: '全部状态', pending: '待检查', reviewed: '已检查', revoked: '已撤销', superseded: '已替代',
    type: '类型', typeAll: '全部类型', avatar: '头像', displayName: '昵称',
    search: '查询', searchPlaceholder: '用户 ID、昵称或邮箱', changedAt: '变更时间', user: '用户', change: '变更内容',
    ipAddress: '来源 IP', action: '操作', newAvatar: '新头像', supersededHint: '已被后续修改替代',
    markReviewed: '已检查', reset: '恢复默认', loading: '正在加载…', empty: '没有符合条件的记录',
    loadFailed: '无法加载记录', pagination: '第 {page} / {pages} 页，共 {total} 条', previous: '上一页', next: '下一页',
    confirmTitle: '确认恢复默认', confirmText: '确认将这次修改恢复为默认头像或默认昵称？原有修改冷却保留。',
    bulkApprove: '全部通过检查', bulkConfirmTitle: '全部通过待检查记录？',
    bulkConfirmText: '将全部 {count} 条待检查记录标记为已检查，不受当前筛选条件和分页影响；不会更改用户的头像或昵称。',
    bulkConfirm: '确认全部通过', bulkDone: '已通过 {count} 条记录的检查。',
    cancel: '取消', confirm: '确认撤销',
  },
  en: {
    title: 'Avatar and name reviews', subtitle: 'Review avatar and name changes, newest first.', refresh: 'Refresh',
    status: 'Status', statusAll: 'All statuses', pending: 'Pending', reviewed: 'Reviewed', revoked: 'Revoked', superseded: 'Superseded',
    type: 'Type', typeAll: 'All types', avatar: 'Avatar', displayName: 'Display name',
    search: 'Search', searchPlaceholder: 'User ID, name, or email', changedAt: 'Changed', user: 'User', change: 'Change',
    ipAddress: 'Source IP', action: 'Action', newAvatar: 'New avatar', supersededHint: 'Superseded by a later change',
    markReviewed: 'Mark reviewed', reset: 'Reset', loading: 'Loading…', empty: 'No records match these filters',
    loadFailed: 'Unable to load records', pagination: 'Page {page} / {pages}, {total} total', previous: 'Previous page', next: 'Next page',
    confirmTitle: 'Reset this change?', confirmText: 'Reset to the default avatar or display name? The existing cooldown remains.',
    bulkApprove: 'Approve all pending', bulkConfirmTitle: 'Approve all pending reviews?',
    bulkConfirmText: 'Mark all {count} pending records as reviewed, regardless of current filters or page. Avatars and names will not change.',
    bulkConfirm: 'Approve all', bulkDone: 'Marked {count} records as reviewed.',
    cancel: 'Cancel', confirm: 'Confirm reset',
  },
};
const copy = computed(() => zh.value ? labels.zh : labels.en);
const statusOptions = computed(() => [
  { value: 'all', label: copy.value.statusAll },
  { value: 'pending', label: copy.value.pending },
  { value: 'reviewed', label: copy.value.reviewed },
  { value: 'revoked', label: copy.value.revoked },
  { value: 'superseded', label: copy.value.superseded },
]);
const typeOptions = computed(() => [
  { value: 'all', label: copy.value.typeAll },
  { value: 'avatar', label: copy.value.avatar },
  { value: 'display_name', label: copy.value.displayName },
]);

const items = ref([]);
const page = ref(1);
const total = ref(0);
const pendingTotal = ref(0);
const pageCount = computed(() => Math.max(1, Math.ceil(total.value / 20)));
const status = ref('all');
const changeType = ref('all');
const query = ref('');
const appliedQuery = ref('');
const busy = ref(false);
const error = ref('');
const feedback = ref('');
const confirmItem = ref(null);
const confirmAll = ref(false);
let loadRevision = 0;

const statusLabel = value => ({ pending: copy.value.pending, reviewed: copy.value.reviewed,
  revoked: copy.value.revoked, superseded: copy.value.superseded })[value] || value;
const typeLabel = value => value === 'avatar' ? copy.value.avatar : copy.value.displayName;
const formatDate = value => {
  const date = new Date(value);
  return value && !Number.isNaN(date.getTime()) ? date.toLocaleString(zh.value ? 'zh-CN' : 'en-US') : '—';
};

async function load() {
  const revision = ++loadRevision;
  busy.value = true;
  error.value = '';
  feedback.value = '';
  try {
    const result = await adminClient.profileReviews({ page: page.value, status: status.value,
      changeType: changeType.value, query: appliedQuery.value });
    if (revision !== loadRevision) return;
    items.value = result.items || [];
    total.value = Number(result.total) || 0;
    pendingTotal.value = Number(result.pending_total) || 0;
    if (page.value > pageCount.value) {
      page.value = pageCount.value;
      await load();
    }
  } catch (requestError) {
    if (revision === loadRevision) {
      items.value = [];
      total.value = 0;
      pendingTotal.value = 0;
      error.value = userError(requestError, copy.value.loadFailed);
    }
  } finally {
    if (revision === loadRevision) busy.value = false;
  }
}

function setStatus(value) { status.value = value; page.value = 1; load(); }
function setChangeType(value) { changeType.value = value; page.value = 1; load(); }
function search() { appliedQuery.value = query.value.trim(); page.value = 1; load(); }
function go(value) { page.value = value; load(); }

async function decide(item, action) {
  busy.value = true;
  error.value = '';
  try {
    await adminClient.reviewProfile(item.id, action);
    confirmItem.value = null;
    await load();
  } catch (requestError) {
    error.value = userError(requestError);
  } finally {
    busy.value = false;
  }
}

async function approveAllPending() {
  busy.value = true;
  error.value = '';
  try {
    const result = await adminClient.approvePendingProfileReviews();
    confirmAll.value = false;
    await load();
    feedback.value = copy.value.bulkDone.replace('{count}', Number(result.updated) || 0);
  } catch (requestError) {
    error.value = userError(requestError);
  } finally {
    busy.value = false;
  }
}

watch(() => props.active, active => { if (active) load(); }, { immediate: true });
defineExpose({ load });
</script>

<style scoped>
.review-panel { border: 1px solid var(--border-main); border-radius: 1.25rem; background: var(--bg-card); color: var(--text-main); box-shadow: 0 14px 34px rgba(15, 23, 42, .08); }
.review-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; padding: 1.25rem; }
.review-head h2 { font-size: var(--font-ui-lg); font-weight: 900; }
.review-head p { margin-top: .3rem; color: var(--text-secondary); font-size: var(--font-ui-sm); }
.review-head-actions { display: flex; align-items: center; gap: .55rem; flex: 0 0 auto; }
.review-bulk { display: inline-flex; align-items: center; justify-content: center; gap: .45rem; min-height: 2.35rem; padding: .4rem .7rem; border: 1px solid var(--border-main); border-radius: .7rem; background: var(--bg-main); color: var(--text-main); font-size: var(--font-ui-sm); font-weight: 800; cursor: pointer; white-space: nowrap; }
.review-bulk span { display: inline-flex; align-items: center; justify-content: center; min-width: 1.45rem; height: 1.45rem; padding: 0 .3rem; border-radius: .4rem; background: var(--accent); color: #10202f; font-size: var(--font-ui-xs); }
.review-refresh, .review-pagination button { display: inline-flex; align-items: center; justify-content: center; width: 2.35rem; height: 2.35rem; border: 1px solid var(--border-main); border-radius: .7rem; background: var(--bg-main); color: var(--text-main); cursor: pointer; }
.review-toolbar { display: grid; grid-template-columns: minmax(9rem, 10rem) minmax(9rem, 10rem) minmax(12rem, 1fr) auto; gap: .65rem; padding: 0 1.25rem 1rem; }
.review-filter :deep(.review-filter-trigger), .review-search { min-width: 0; min-height: 2.4rem; border: 1px solid var(--border-main); border-radius: .8rem; background: var(--bg-main); color: var(--text-main); font-size: var(--font-ui-sm); font-weight: 750; padding: .55rem .75rem; }
.review-search { width: 100%; outline: none; }
.review-search:focus, .review-filter :deep(.review-filter-trigger:focus-visible) { border-color: var(--accent); }
.review-search-button { display: inline-flex; align-items: center; gap: .35rem; justify-content: center; }
.review-error { margin: 0 1.25rem 1rem; padding: .7rem .8rem; border: 1px solid rgba(239, 68, 68, .3); border-radius: .75rem; background: rgba(239, 68, 68, .1); color: #dc2626; font-size: var(--font-ui-sm); font-weight: 750; }
.review-feedback { margin: 0 1.25rem 1rem; padding: .7rem .8rem; border: 1px solid rgba(34, 197, 94, .3); border-radius: .75rem; background: rgba(34, 197, 94, .1); color: var(--text-main); font-size: var(--font-ui-sm); font-weight: 750; }
.review-table-wrap { overflow-x: auto; border-top: 1px solid var(--border-main); }
.review-table { width: 100%; min-width: 850px; border-collapse: collapse; font-size: var(--font-ui-sm); }
.review-table th, .review-table td { padding: .78rem 1rem; border-bottom: 1px solid var(--border-main); text-align: left; vertical-align: middle; }
.review-table th { color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 900; white-space: nowrap; }
.review-table strong, .review-table small { display: block; }
.review-table small, .review-time, .review-ip, .review-muted { color: var(--text-secondary); }
.review-table small { margin-top: .15rem; font-size: var(--font-ui-xs); }
.review-time, .review-ip { white-space: nowrap; }
.review-type, .review-status { display: inline-flex; padding: .3rem .55rem; border-radius: .5rem; background: var(--bg-main); color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 850; white-space: nowrap; }
.review-status.is-pending { background: rgba(245, 158, 11, .14); color: #b45309; }
.review-status.is-reviewed { background: rgba(34, 197, 94, .14); color: #15803d; }
.review-status.is-revoked { background: rgba(239, 68, 68, .13); color: #dc2626; }
.review-avatar-change { display: flex; align-items: center; gap: .65rem; min-width: 9rem; }
.review-avatar-change img { width: 2.75rem; height: 2.75rem; flex: 0 0 auto; border-radius: .45rem; object-fit: cover; }
.review-name-change { display: flex; align-items: center; flex-wrap: wrap; gap: .35rem; overflow-wrap: anywhere; }
.review-name-change small { flex-basis: 100%; }
.review-actions { display: flex; align-items: center; flex-wrap: wrap; gap: .35rem; }
.review-actions button { display: inline-flex; align-items: center; gap: .25rem; border: 0; background: transparent; color: var(--accent); font-size: var(--font-ui-xs); font-weight: 850; white-space: nowrap; cursor: pointer; }
.review-actions button.is-danger, .review-dialog .is-danger { color: #dc2626; }
.review-empty { padding: 1.5rem !important; text-align: center !important; color: var(--text-secondary); }
.review-pagination { display: flex; align-items: center; justify-content: flex-end; gap: 1rem; padding: .9rem 1.25rem; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 800; }
.review-pagination > div { display: flex; gap: .4rem; }
.review-pagination button { width: 2rem; height: 2rem; }
button:disabled { opacity: .45; cursor: not-allowed; }
.review-overlay { position: fixed; inset: 0; z-index: 100; display: grid; place-items: center; padding: 1rem; background: rgba(15, 23, 42, .55); backdrop-filter: blur(5px); }
.review-dialog { width: min(28rem, 100%); border: 1px solid var(--border-main); border-radius: 1rem; background: var(--bg-card); box-shadow: 0 24px 80px rgba(15, 23, 42, .3); padding: 1.25rem; }
.review-dialog h3 { font-size: var(--font-ui-lg); font-weight: 900; }
.review-dialog p { margin: .7rem 0 1.25rem; color: var(--text-secondary); font-size: var(--font-ui-sm); }
.review-dialog > div { display: flex; justify-content: flex-end; gap: .5rem; }
@media (max-width: 760px) { .review-head { flex-wrap: wrap; } .review-toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); } .review-search { grid-column: 1 / -1; } .review-search-button { grid-column: 1 / -1; } .review-pagination { justify-content: space-between; } }
</style>
