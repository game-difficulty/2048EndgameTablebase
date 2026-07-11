<template>
  <div class="page-root overflow-y-auto p-5">
    <div class="mx-auto flex w-full max-w-6xl flex-col gap-4">
      <header class="rounded-2xl border border-border-main bg-bg-card/88 p-5 shadow-sm backdrop-blur-md">
        <div class="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('admin.kicker') }}</div>
            <h1 class="mt-1 ui-metric font-black text-text-main">{{ $t('admin.title') }}</h1>
            <p class="mt-2 max-w-2xl ui-body text-text-secondary">{{ $t('admin.subtitle') }}</p>
          </div>
          <button type="button" class="action-btn-small justify-center" :disabled="loading" @click="refresh">
            {{ loading ? $t('common.updating') : $t('admin.refresh') }}
          </button>
        </div>
      </header>

      <div v-if="error" class="rounded-2xl border border-red-400/35 bg-red-500/10 p-4 ui-body font-bold text-red-500">
        {{ error }}
      </div>

      <section class="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
        <div v-for="card in summaryCards" :key="card.key" class="admin-card">
          <div class="ui-caption font-black uppercase text-text-secondary">{{ card.label }}</div>
          <div class="mt-2 text-2xl font-black text-text-main">{{ card.value }}</div>
        </div>
      </section>

      <section class="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <div class="admin-panel">
          <div class="admin-panel-head">
            <h2>{{ $t('admin.traffic.title') }}</h2>
            <span>{{ $t('admin.traffic.range') }}</span>
          </div>
          <div class="mt-4 grid gap-2">
            <div v-for="row in trafficRows" :key="row.date" class="admin-traffic-row">
              <span class="admin-date">{{ row.date.slice(5) }}</span>
              <div class="admin-bar-track">
                <div class="admin-bar" :style="{ width: `${trafficWidth(row)}%` }" />
              </div>
              <span class="admin-count">{{ rowTotal(row) }}</span>
              <span class="admin-row-note">
                {{ $t('admin.traffic.row', row) }}
              </span>
            </div>
          </div>
        </div>

        <div class="admin-panel">
          <div class="admin-panel-head">
            <h2>{{ $t('admin.operations.title') }}</h2>
            <span>{{ $t('admin.operations.range') }}</span>
          </div>
          <div class="mt-4 grid gap-2">
            <div v-if="!operations.length" class="admin-empty">{{ $t('admin.empty') }}</div>
            <div v-for="item in operations" :key="item.event_type" class="flex items-center justify-between gap-3 rounded-xl border border-border-main bg-bg-main/55 px-3 py-2">
              <span class="truncate ui-caption font-black text-text-secondary">{{ item.event_type }}</span>
              <span class="ui-body font-black text-text-main">{{ formatNumber(item.count) }}</span>
            </div>
          </div>
        </div>
      </section>

      <section class="admin-panel">
        <div class="admin-panel-head">
          <h2>{{ $t('admin.users.title') }}</h2>
          <div class="flex w-full flex-col gap-2 sm:w-auto sm:min-w-[20rem] sm:flex-row">
            <input
              v-model="query"
              class="admin-search"
              :placeholder="$t('admin.users.searchPlaceholder')"
              @keydown.enter.prevent="refresh"
            />
            <button type="button" class="action-btn-small justify-center" @click="refresh">
              {{ $t('admin.users.search') }}
            </button>
          </div>
        </div>
        <div class="mt-4 overflow-x-auto">
          <table class="admin-table">
            <thead>
              <tr>
                <th>{{ $t('admin.users.id') }}</th>
                <th>{{ $t('admin.users.user') }}</th>
                <th>{{ $t('admin.users.status') }}</th>
                <th>{{ $t('admin.users.tokens') }}</th>
                <th>{{ $t('admin.users.sessions') }}</th>
                <th>{{ $t('admin.users.usage') }}</th>
                <th>{{ $t('admin.users.created') }}</th>
                <th>{{ $t('admin.users.lastLogin') }}</th>
                <th>{{ $t('admin.users.actions') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="!users.length">
                <td colspan="9" class="text-center text-text-secondary">{{ $t('admin.empty') }}</td>
              </tr>
              <tr v-for="item in users" :key="item.id">
                <td>#{{ item.id }}</td>
                <td>
                  <div class="font-black text-text-main">{{ item.display_name || '-' }}</div>
                  <div class="text-[0.75rem] font-bold text-text-secondary">{{ item.email }}</div>
                </td>
                <td>
                  <span :class="['admin-status', item.status === 'active' ? 'active' : 'disabled']">
                    {{ item.status }}
                  </span>
                </td>
                <td>
                  <div>{{ formatTokens(item.token_balance?.total) }}</div>
                  <div class="text-[0.72rem] font-bold text-text-secondary">
                    {{ $t('admin.tokens.balanceDetail', {
                      bonus: formatTokens(item.token_balance?.bonus),
                      paid: formatTokens(item.token_balance?.paid)
                    }) }}
                  </div>
                </td>
                <td>{{ formatNumber(item.sessions) }}</td>
                <td>{{ formatNumber(item.usage_events) }}</td>
                <td>{{ formatDate(item.created_at) }}</td>
                <td>{{ formatDate(item.last_login_at) }}</td>
                <td>
                  <button type="button" class="action-btn-small justify-center" @click="openTokenAdjust(item)">
                    {{ $t('admin.tokens.adjust') }}
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>

    <div v-if="tokenAdjust.open" class="admin-modal">
      <div class="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" @click="closeTokenAdjust" />
      <section class="admin-modal-panel">
        <div class="flex items-start justify-between gap-4">
          <div class="min-w-0">
            <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('admin.tokens.kicker') }}</div>
            <h2 class="mt-1 ui-metric font-black text-text-main">{{ $t('admin.tokens.title') }}</h2>
            <div class="mt-1 truncate ui-body font-bold text-text-secondary">
              {{ tokenAdjust.user?.display_name || '-' }} · {{ tokenAdjust.user?.email }}
            </div>
          </div>
          <button type="button" class="action-btn-small admin-modal-close" @click="closeTokenAdjust">
            {{ $t('common.close') }}
          </button>
        </div>

        <div class="mt-4 grid gap-2 rounded-xl border border-border-main bg-bg-main/55 p-3">
          <div class="flex items-center justify-between gap-3">
            <span class="ui-caption font-black text-text-secondary">{{ $t('auth.account.bonusTokens') }}</span>
            <span class="ui-caption font-black text-text-main">{{ formatTokens(tokenAdjust.user?.token_balance?.bonus) }}</span>
          </div>
          <div class="flex items-center justify-between gap-3">
            <span class="ui-caption font-black text-text-secondary">{{ $t('auth.account.paidTokens') }}</span>
            <span class="ui-caption font-black text-text-main">{{ formatTokens(tokenAdjust.user?.token_balance?.paid) }}</span>
          </div>
          <div class="flex items-center justify-between gap-3 border-t border-border-main pt-2">
            <span class="ui-caption font-black text-text-secondary">{{ $t('auth.account.totalTokens') }}</span>
            <span class="ui-body font-black text-accent">{{ formatTokens(tokenAdjust.user?.token_balance?.total) }}</span>
          </div>
        </div>

        <div class="mt-4 grid grid-cols-2 gap-2">
          <button type="button" class="action-btn-small justify-center" @click="applyTokenPreset(100000, 9.9)">
            {{ $t('admin.tokens.presetSmall') }}
          </button>
          <button type="button" class="action-btn-small justify-center" @click="applyTokenPreset(2000000, 99)">
            {{ $t('admin.tokens.presetLarge') }}
          </button>
        </div>

        <form class="mt-4 grid gap-3" @submit.prevent="submitTokenAdjust">
          <label class="admin-form-row">
            <span>{{ $t('admin.tokens.mode') }}</span>
            <select v-model="tokenAdjust.mode" class="admin-input">
              <option value="add_paid">{{ $t('admin.tokens.modeAdd') }}</option>
              <option value="set_paid">{{ $t('admin.tokens.modeSet') }}</option>
            </select>
          </label>

          <label class="admin-form-row">
            <span>{{ $t('admin.tokens.amount') }}</span>
            <input
              v-model.trim="tokenAdjust.tokens"
              class="admin-input"
              inputmode="decimal"
              placeholder="100000"
            />
          </label>

          <label class="admin-form-row">
            <span>{{ $t('admin.tokens.paymentAmount') }}</span>
            <input
              v-model.trim="tokenAdjust.payment_amount_cny"
              class="admin-input"
              inputmode="decimal"
              placeholder="9.9"
            />
          </label>

          <label class="admin-form-row">
            <span>{{ $t('admin.tokens.paymentChannel') }}</span>
            <input v-model.trim="tokenAdjust.payment_channel" class="admin-input" placeholder="wechat_manual" />
          </label>

          <label class="admin-form-row">
            <span>{{ $t('admin.tokens.reason') }}</span>
            <textarea v-model.trim="tokenAdjust.reason" class="admin-input min-h-[5rem] resize-y" />
          </label>

          <div v-if="tokenAdjust.error" class="admin-alert error">{{ tokenAdjust.error }}</div>
          <div v-if="tokenAdjust.success" class="admin-alert success">{{ tokenAdjust.success }}</div>

          <button type="submit" class="action-btn-small justify-center" :disabled="tokenAdjust.submitting">
            {{ tokenAdjust.submitting ? $t('common.updating') : $t('admin.tokens.submit') }}
          </button>
        </form>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { adminClient } from '../../../services/admin/adminClient';

const props = defineProps({
  active: Boolean,
});

const { t } = useI18n();
const loading = ref(false);
const error = ref('');
const query = ref('');
const overview = ref(null);
const tokenAdjust = ref({
  open: false,
  user: null,
  mode: 'add_paid',
  tokens: '100000',
  reason: '',
  payment_amount_cny: '9.9',
  payment_channel: 'wechat_manual',
  submitting: false,
  error: '',
  success: '',
});

const formatNumber = (value) => Number(value || 0).toLocaleString();
const formatTokens = (value) => Number(value || 0).toLocaleString(undefined, {
  minimumFractionDigits: 0,
  maximumFractionDigits: 3,
});
const formatDate = (value) => {
  if (!value) return '-';
  return String(value).replace('T', ' ').slice(0, 16);
};

const summary = computed(() => overview.value?.summary || {});
const trafficRows = computed(() => overview.value?.traffic || []);
const operations = computed(() => overview.value?.operations || []);
const users = computed(() => overview.value?.users || []);

const summaryCards = computed(() => [
  { key: 'users_total', label: t('admin.summary.usersTotal'), value: formatNumber(summary.value.users_total) },
  { key: 'users_active', label: t('admin.summary.usersActive'), value: formatNumber(summary.value.users_active) },
  { key: 'new_users_24h', label: t('admin.summary.newUsers24h'), value: formatNumber(summary.value.new_users_24h) },
  { key: 'sessions_24h', label: t('admin.summary.sessions24h'), value: formatNumber(summary.value.sessions_24h) },
  { key: 'usage_events_24h', label: t('admin.summary.usage24h'), value: formatNumber(summary.value.usage_events_24h) },
  { key: 'tokens_spent_24h', label: t('admin.summary.tokens24h'), value: formatTokens(summary.value.tokens_spent_24h) },
]);

const rowTotal = (row) => Number(row.sessions || 0)
  + Number(row.usage_events || 0)
  + Number(row.uploads || 0)
  + Number(row.analysis_jobs || 0);

const maxTrafficTotal = computed(() => Math.max(1, ...trafficRows.value.map(rowTotal)));
const trafficWidth = (row) => Math.max(4, Math.round((rowTotal(row) / maxTrafficTotal.value) * 100));

const replaceUserInOverview = (updatedUser) => {
  if (!updatedUser || !overview.value) {
    return;
  }
  const replace = (items) => (Array.isArray(items)
    ? items.map((item) => (Number(item.id) === Number(updatedUser.id) ? updatedUser : item))
    : items);
  overview.value = {
    ...overview.value,
    users: replace(overview.value.users),
    recent_users: replace(overview.value.recent_users),
  };
};

const defaultTokenReason = () => t('admin.tokens.defaultReason');

const openTokenAdjust = (user) => {
  tokenAdjust.value = {
    open: true,
    user,
    mode: 'add_paid',
    tokens: '100000',
    reason: defaultTokenReason(),
    payment_amount_cny: '9.9',
    payment_channel: 'wechat_manual',
    submitting: false,
    error: '',
    success: '',
  };
};

const closeTokenAdjust = () => {
  tokenAdjust.value = {
    ...tokenAdjust.value,
    open: false,
    submitting: false,
    error: '',
    success: '',
  };
};

const applyTokenPreset = (tokens, amount) => {
  tokenAdjust.value = {
    ...tokenAdjust.value,
    mode: 'add_paid',
    tokens: String(tokens),
    payment_amount_cny: String(amount),
    payment_channel: 'wechat_manual',
    reason: defaultTokenReason(),
    error: '',
    success: '',
  };
};

const submitTokenAdjust = async () => {
  if (!tokenAdjust.value.user?.id) {
    return;
  }
  tokenAdjust.value = {
    ...tokenAdjust.value,
    submitting: true,
    error: '',
    success: '',
  };
  try {
    const response = await adminClient.adjustUserTokens(tokenAdjust.value.user.id, {
      mode: tokenAdjust.value.mode,
      tokens: tokenAdjust.value.tokens,
      reason: tokenAdjust.value.reason,
      payment_amount_cny: tokenAdjust.value.payment_amount_cny,
      payment_channel: tokenAdjust.value.payment_channel,
    });
    replaceUserInOverview(response.user);
    tokenAdjust.value = {
      ...tokenAdjust.value,
      user: response.user,
      submitting: false,
      success: t('admin.tokens.updated'),
    };
  } catch (requestError) {
    tokenAdjust.value = {
      ...tokenAdjust.value,
      submitting: false,
      error: requestError?.message || t('admin.tokens.updateFailed'),
    };
  }
};

const refresh = async () => {
  loading.value = true;
  error.value = '';
  try {
    overview.value = await adminClient.overview({ q: query.value, limit: 30 });
  } catch (requestError) {
    error.value = requestError?.message || t('admin.errors.loadFailed');
  } finally {
    loading.value = false;
  }
};

onMounted(() => {
  if (props.active) {
    refresh();
  }
});

watch(() => props.active, (active) => {
  if (active && !overview.value) {
    refresh();
  }
});
</script>

<style scoped>
.admin-card,
.admin-panel {
  border: 1px solid var(--border-main);
  background: color-mix(in srgb, var(--bg-card) 90%, transparent);
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(14px);
}

.admin-card {
  border-radius: 1rem;
  padding: 1rem;
}

.admin-panel {
  border-radius: 1.25rem;
  padding: 1.25rem;
}

.admin-panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.admin-panel-head h2 {
  font-size: var(--font-ui-lg);
  font-weight: 900;
  color: var(--text-main);
}

.admin-panel-head span {
  font-size: var(--font-ui-xs);
  font-weight: 900;
  color: var(--text-secondary);
  text-transform: uppercase;
}

.admin-traffic-row {
  display: grid;
  grid-template-columns: 3.2rem minmax(7rem, 1fr) 3.2rem minmax(11rem, 1.2fr);
  align-items: center;
  gap: 0.75rem;
  border: 1px solid var(--border-main);
  border-radius: 0.9rem;
  background: color-mix(in srgb, var(--bg-main) 55%, transparent);
  padding: 0.6rem 0.75rem;
}

.admin-date,
.admin-count,
.admin-row-note {
  font-size: var(--font-ui-xs);
  font-weight: 900;
  color: var(--text-secondary);
}

.admin-count {
  text-align: right;
  color: var(--text-main);
}

.admin-bar-track {
  height: 0.65rem;
  overflow: hidden;
  border-radius: 999px;
  background: color-mix(in srgb, var(--bg-main) 82%, var(--border-main));
}

.admin-bar {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--accent), var(--success));
}

.admin-search {
  min-height: 2.35rem;
  border: 1px solid var(--border-main);
  border-radius: 0.85rem;
  background: color-mix(in srgb, var(--bg-main) 82%, transparent);
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  font-weight: 800;
  padding: 0.55rem 0.8rem;
  outline: none;
}

.admin-search:focus {
  border-color: var(--accent);
}

.admin-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0 0.45rem;
}

.admin-table th,
.admin-table td {
  padding: 0.75rem;
  text-align: left;
  white-space: nowrap;
}

.admin-table th {
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 900;
  text-transform: uppercase;
}

.admin-table td {
  background: color-mix(in srgb, var(--bg-main) 58%, transparent);
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  font-weight: 800;
}

.admin-table td:first-child {
  border-radius: 0.85rem 0 0 0.85rem;
}

.admin-table td:last-child {
  border-radius: 0 0.85rem 0.85rem 0;
}

.admin-status {
  display: inline-flex;
  border-radius: 999px;
  padding: 0.2rem 0.55rem;
  font-size: var(--font-ui-xs);
  font-weight: 900;
  text-transform: uppercase;
}

.admin-status.active {
  background: color-mix(in srgb, var(--success) 18%, transparent);
  color: var(--success);
}

.admin-status.disabled {
  background: rgba(239, 68, 68, 0.14);
  color: rgb(239, 68, 68);
}

.admin-empty {
  border: 1px dashed var(--border-main);
  border-radius: 1rem;
  padding: 1rem;
  text-align: center;
  color: var(--text-secondary);
  font-size: var(--font-ui-sm);
  font-weight: 800;
}

.admin-modal {
  position: fixed;
  inset: 0;
  z-index: 118;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.25rem;
}

.admin-modal-panel {
  position: relative;
  z-index: 1;
  width: min(34rem, 100%);
  max-height: calc(100vh - 2rem);
  overflow-y: auto;
  border: 1px solid var(--border-main);
  border-radius: 1.35rem;
  background: color-mix(in srgb, var(--bg-card) 97%, transparent);
  padding: 1.35rem;
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.34);
}

.admin-modal-close {
  min-width: 4.5rem;
}

.admin-form-row {
  display: grid;
  gap: 0.35rem;
}

.admin-form-row span {
  font-size: var(--font-ui-xs);
  font-weight: 900;
  color: var(--text-secondary);
  text-transform: uppercase;
}

.admin-input {
  width: 100%;
  min-height: 2.7rem;
  border: 1px solid var(--border-main);
  border-radius: 0.85rem;
  background: color-mix(in srgb, var(--bg-main) 82%, transparent);
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  font-weight: 850;
  padding: 0.65rem 0.8rem;
  outline: none;
}

.admin-input:focus {
  border-color: var(--accent);
}

.admin-alert {
  border-radius: 0.85rem;
  padding: 0.7rem 0.85rem;
  font-size: var(--font-ui-sm);
  font-weight: 850;
}

.admin-alert.error {
  border: 1px solid rgba(239, 68, 68, 0.35);
  background: rgba(239, 68, 68, 0.12);
  color: rgb(239, 68, 68);
}

.admin-alert.success {
  border: 1px solid color-mix(in srgb, var(--success) 40%, transparent);
  background: color-mix(in srgb, var(--success) 16%, transparent);
  color: var(--success);
}

@media (max-width: 760px) {
  .admin-traffic-row {
    grid-template-columns: 3rem 1fr 2.7rem;
  }

  .admin-row-note {
    grid-column: 1 / -1;
  }

  .admin-panel-head {
    align-items: stretch;
    flex-direction: column;
  }

  .admin-modal {
    padding: 0.7rem;
  }

  .admin-modal-panel {
    border-radius: 1rem;
    padding: 1rem;
  }
}
</style>
