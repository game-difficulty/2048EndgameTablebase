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
              </tr>
            </thead>
            <tbody>
              <tr v-if="!users.length">
                <td colspan="8" class="text-center text-text-secondary">{{ $t('admin.empty') }}</td>
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
                <td>{{ formatTokens(item.token_balance?.total) }}</td>
                <td>{{ formatNumber(item.sessions) }}</td>
                <td>{{ formatNumber(item.usage_events) }}</td>
                <td>{{ formatDate(item.created_at) }}</td>
                <td>{{ formatDate(item.last_login_at) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
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
}
</style>
