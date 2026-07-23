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

      <section class="admin-panel">
        <div class="admin-panel-head">
          <div>
            <h2>{{ $t('admin.tokenActivity.title') }}</h2>
            <span>{{ $t('admin.tokenActivity.range') }}</span>
          </div>
          <div class="admin-chart-legend">
            <span><i class="legend-token" />{{ $t('admin.tokenActivity.tokens') }}</span>
            <span><i class="legend-users" />{{ $t('admin.tokenActivity.users') }}</span>
          </div>
        </div>
        <div class="admin-chart-wrap">
          <svg
            class="admin-line-chart"
            viewBox="0 0 960 340"
            role="img"
            :aria-label="$t('admin.tokenActivity.title')"
            @pointerleave="hideChartTooltip"
          >
            <g class="chart-grid">
              <line
                v-for="tick in tokenChart.leftTicks"
                :key="`grid-${tick.y}`"
                :x1="tokenChart.left"
                :x2="tokenChart.right"
                :y1="tick.y"
                :y2="tick.y"
              />
            </g>
            <g class="chart-axis">
              <line :x1="tokenChart.left" :x2="tokenChart.left" :y1="tokenChart.top" :y2="tokenChart.bottom" />
              <line :x1="tokenChart.right" :x2="tokenChart.right" :y1="tokenChart.top" :y2="tokenChart.bottom" />
              <line :x1="tokenChart.left" :x2="tokenChart.right" :y1="tokenChart.bottom" :y2="tokenChart.bottom" />
            </g>
            <g class="chart-labels">
              <text
                v-for="tick in tokenChart.leftTicks"
                :key="`left-${tick.y}`"
                :x="tokenChart.left - 12"
                :y="tick.y + 4"
                text-anchor="end"
              >
                {{ formatCompact(tick.value) }}
              </text>
              <text
                v-for="tick in tokenChart.rightTicks"
                :key="`right-${tick.y}`"
                :x="tokenChart.right + 12"
                :y="tick.y + 4"
              >
                {{ formatCompact(tick.value) }}
              </text>
              <text
                v-for="label in tokenChart.xLabels"
                :key="label.date"
                :x="label.x"
                :y="tokenChart.bottom + 30"
                text-anchor="middle"
              >
                {{ label.label }}
              </text>
            </g>
            <polyline class="chart-line token-line" :points="tokenChart.tokenPoints" />
            <polyline class="chart-line users-line" :points="tokenChart.userPoints" />
            <g v-if="activeChartPoint" class="chart-hover-layer">
              <line
                class="chart-hover-line"
                :x1="activeChartPoint.x"
                :x2="activeChartPoint.x"
                :y1="tokenChart.top"
                :y2="tokenChart.bottom"
              />
              <circle class="chart-point token-point active" :cx="activeChartPoint.x" :cy="activeChartPoint.tokenY" r="6" />
              <circle class="chart-point users-point active" :cx="activeChartPoint.x" :cy="activeChartPoint.userY" r="6" />
              <foreignObject
                class="chart-tooltip-object"
                :x="activeChartPoint.tooltipX"
                :y="activeChartPoint.tooltipY"
                width="188"
                height="88"
              >
                <div xmlns="http://www.w3.org/1999/xhtml" class="admin-chart-tooltip">
                  <div class="tooltip-date">{{ activeChartPoint.date }}</div>
                  <div class="tooltip-row">
                    <span><i class="legend-token" />{{ $t('admin.tokenActivity.tokens') }}</span>
                    <strong>{{ formatTokens(activeChartPoint.tokens) }}</strong>
                  </div>
                  <div class="tooltip-row">
                    <span><i class="legend-users" />{{ $t('admin.tokenActivity.users') }}</span>
                    <strong>{{ formatNumber(activeChartPoint.usersCount) }}</strong>
                  </div>
                </div>
              </foreignObject>
            </g>
            <g>
              <circle
                v-for="point in tokenChart.points"
                :key="`token-point-${point.date}`"
                class="chart-point token-point"
                :cx="point.x"
                :cy="point.tokenY"
                r="4"
              />
              <circle
                v-for="point in tokenChart.points"
                :key="`user-point-${point.date}`"
                class="chart-point users-point"
                :cx="point.x"
                :cy="point.userY"
                r="4"
              />
            </g>
            <g class="chart-hit-layer">
              <rect
                v-for="point in tokenChart.points"
                :key="`hit-${point.date}`"
                class="chart-hit-area"
                :x="point.hitX"
                :y="tokenChart.top"
                :width="point.hitWidth"
                :height="tokenChart.bottom - tokenChart.top"
                tabindex="0"
                :aria-label="`${point.date}: ${formatTokens(point.tokens)} tokens, ${formatNumber(point.usersCount)} users`"
                @pointerenter="showChartTooltip(point.index)"
                @pointermove="showChartTooltip(point.index)"
                @pointerdown.prevent="showChartTooltip(point.index)"
                @focus="showChartTooltip(point.index)"
              />
            </g>
          </svg>
        </div>
      </section>

      <section class="admin-panel">
        <div class="admin-panel-head">
          <h2>{{ $t('admin.users.title') }}</h2>
          <div class="flex w-full flex-col gap-2 sm:w-auto sm:min-w-[28rem] sm:flex-row sm:items-center">
            <div class="admin-segmented" role="group" :aria-label="$t('admin.users.tierFilter')">
              <button
                v-for="option in tierFilterOptions"
                :key="option.value"
                type="button"
                :class="['admin-segment-btn', tierFilter === option.value ? 'active' : '']"
                :disabled="loading"
                @click="setTierFilter(option.value)"
              >
                {{ option.label }}
              </button>
            </div>
            <input
              v-model="query"
              class="admin-search"
              :placeholder="$t('admin.users.searchPlaceholder')"
              @keydown.enter.prevent="runSearch"
            />
            <button type="button" class="action-btn-small justify-center" @click="runSearch">
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
                  <div v-if="item.entitlements?.is_supporter" class="mt-1">
                    <span class="admin-tier-pill">{{ $t('admin.entitlements.supporter') }}</span>
                  </div>
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
        <div class="admin-pagination">
          <button
            type="button"
            class="admin-page-nav-btn"
            :aria-label="$t('admin.users.prevPage')"
            :disabled="currentPage <= 1 || loading"
            @click="goToPage(currentPage - 1)"
          >
            &lt;
          </button>
          <div class="admin-pagination-center">
            <div class="ui-caption font-black text-text-secondary">
              {{ $t('admin.users.pagination', {
                start: userRangeStart,
                end: userRangeEnd,
                total: usersPage.total || 0,
                page: usersPage.page || 1,
                page_count: usersPage.page_count || 1
              }) }}
            </div>
            <div class="admin-pagination-controls">
              <template v-for="item in paginationItems" :key="item.key">
                <span v-if="item.type === 'ellipsis'" class="admin-page-ellipsis">...</span>
                <button
                  v-else
                  type="button"
                  :class="['admin-page-btn', item.page === currentPage ? 'active' : '']"
                  :disabled="loading || item.page === currentPage"
                  :aria-current="item.page === currentPage ? 'page' : undefined"
                  @click="goToPage(item.page)"
                >
                  {{ item.page }}
                </button>
              </template>
            </div>
          </div>
          <button
            type="button"
            class="admin-page-nav-btn"
            :aria-label="$t('admin.users.nextPage')"
            :disabled="currentPage >= pageCount || loading"
            @click="goToPage(currentPage + 1)"
          >
            &gt;
          </button>
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

          <label class="admin-check-row">
            <input v-model="tokenAdjust.set_supporter" type="checkbox" />
            <span>{{ $t('admin.tokens.setSupporter') }}</span>
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
const tierFilter = ref('all');
const currentPage = ref(1);
const pageSize = 20;
const overview = ref(null);
const activeChartIndex = ref(null);
const tokenAdjust = ref({
  open: false,
  user: null,
  mode: 'add_paid',
  tokens: '100000',
  reason: '',
  payment_amount_cny: '9.9',
  payment_channel: 'wechat_manual',
  set_supporter: true,
  submitting: false,
  error: '',
  success: '',
});

const formatNumber = (value) => Number(value || 0).toLocaleString();
const formatCompact = (value) => Intl.NumberFormat(undefined, {
  notation: Number(value || 0) >= 10000 ? 'compact' : 'standard',
  minimumFractionDigits: 0,
  maximumFractionDigits: Number(value || 0) >= 10000 ? 1 : 0,
}).format(Number(value || 0));
const formatTokens = (value) => Number(value || 0).toLocaleString(undefined, {
  minimumFractionDigits: 0,
  maximumFractionDigits: 3,
});
const formatDate = (value) => {
  if (!value) return '-';
  return String(value).replace('T', ' ').slice(0, 16);
};

const summary = computed(() => overview.value?.summary || {});
const activityRows = computed(() => overview.value?.token_activity || []);
const users = computed(() => overview.value?.users || []);
const usersPage = computed(() => overview.value?.users_page || {
  page: 1,
  page_size: pageSize,
  total: users.value.length,
  page_count: 1,
});
const pageCount = computed(() => Math.max(1, Number(usersPage.value.page_count || 1)));
const paginationItems = computed(() => {
  const totalPages = pageCount.value;
  const page = Math.max(1, Math.min(Number(currentPage.value || 1), totalPages));
  if (totalPages <= 7) {
    return Array.from({ length: totalPages }, (_item, index) => {
      const pageNumber = index + 1;
      return { key: `page-${pageNumber}`, type: 'page', page: pageNumber };
    });
  }

  const pages = new Set([1, totalPages, page - 1, page, page + 1]);
  if (page <= 4) {
    [2, 3, 4, 5].forEach((pageNumber) => pages.add(pageNumber));
  }
  if (page >= totalPages - 3) {
    [totalPages - 4, totalPages - 3, totalPages - 2, totalPages - 1].forEach((pageNumber) => pages.add(pageNumber));
  }

  const sortedPages = Array.from(pages)
    .filter((pageNumber) => pageNumber >= 1 && pageNumber <= totalPages)
    .sort((a, b) => a - b);
  const items = [];
  sortedPages.forEach((pageNumber, index) => {
    const previous = sortedPages[index - 1];
    if (previous && pageNumber - previous > 1) {
      items.push({ key: `ellipsis-${previous}-${pageNumber}`, type: 'ellipsis' });
    }
    items.push({ key: `page-${pageNumber}`, type: 'page', page: pageNumber });
  });
  return items;
});
const userRangeStart = computed(() => {
  if (!Number(usersPage.value.total || 0)) return 0;
  return ((Number(usersPage.value.page || 1) - 1) * Number(usersPage.value.page_size || pageSize)) + 1;
});
const userRangeEnd = computed(() => Math.min(
  Number(usersPage.value.total || 0),
  Number(userRangeStart.value || 0) + users.value.length - 1,
));

const summaryCards = computed(() => [
  { key: 'users_total', label: t('admin.summary.usersTotal'), value: formatNumber(summary.value.users_total) },
  { key: 'users_active', label: t('admin.summary.usersActive'), value: formatNumber(summary.value.users_active) },
  { key: 'new_users_24h', label: t('admin.summary.newUsers24h'), value: formatNumber(summary.value.new_users_24h) },
  { key: 'sessions_24h', label: t('admin.summary.sessions24h'), value: formatNumber(summary.value.sessions_24h) },
  { key: 'usage_events_24h', label: t('admin.summary.usage24h'), value: formatNumber(summary.value.usage_events_24h) },
  { key: 'tokens_spent_24h', label: t('admin.summary.tokens24h'), value: formatTokens(summary.value.tokens_spent_24h) },
]);
const tierFilterOptions = computed(() => [
  { value: 'all', label: t('admin.users.tierAll') },
  { value: 'supporter', label: t('admin.users.tierSupporter') },
  { value: 'free', label: t('admin.users.tierFree') },
]);

const linePoints = (points, key) => points.map((point) => `${point.x},${point[key]}`).join(' ');

const niceStep = (maxValue, targetTicks = 7) => {
  const rawStep = Math.max(1e-9, Number(maxValue || 0) / Math.max(1, targetTicks - 1));
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  let niceNormalized = 10;
  if (normalized <= 1) niceNormalized = 1;
  else if (normalized <= 2) niceNormalized = 2;
  else if (normalized <= 2.5) niceNormalized = 2.5;
  else if (normalized <= 5) niceNormalized = 5;
  return niceNormalized * magnitude;
};

const buildNiceTicks = (maxValue, top, bottom, { targetTicks = 7, integer = false } = {}) => {
  let step = niceStep(maxValue, targetTicks);
  if (integer) {
    step = Math.max(1, Math.ceil(step));
  }
  const axisMax = Math.max(step, Math.ceil(Number(maxValue || 0) / step) * step);
  const count = Math.max(2, Math.round(axisMax / step) + 1);
  const ticks = Array.from({ length: count }, (_item, index) => {
    const value = axisMax - (step * index);
    const y = top + ((bottom - top) * (axisMax - value)) / axisMax;
    return { value: integer ? Math.round(value) : value, y };
  });
  return { axisMax, ticks };
};

const tokenChart = computed(() => {
  const left = 72;
  const right = 890;
  const top = 30;
  const bottom = 282;
  const rows = activityRows.value.length ? activityRows.value : [];
  const tokenAxis = buildNiceTicks(
    Math.max(1, ...rows.map((row) => Number(row.tokens_spent || 0))),
    top,
    bottom,
    { targetTicks: 7 },
  );
  const userAxis = buildNiceTicks(
    Math.max(1, ...rows.map((row) => Number(row.spending_users || 0))),
    top,
    bottom,
    { targetTicks: 8, integer: true },
  );
  const span = Math.max(1, rows.length - 1);
  const rawPoints = rows.map((row, index) => {
    const x = left + ((right - left) * index) / span;
    const tokens = Number(row.tokens_spent || 0);
    const usersCount = Number(row.spending_users || 0);
    return {
      index,
      date: row.date,
      label: String(row.date || '').slice(5),
      x,
      tokens,
      usersCount,
      tokenY: bottom - ((bottom - top) * tokens) / tokenAxis.axisMax,
      userY: bottom - ((bottom - top) * usersCount) / userAxis.axisMax,
    };
  });
  const points = rawPoints.map((point, index) => {
    const previousX = rawPoints[index - 1]?.x ?? left;
    const nextX = rawPoints[index + 1]?.x ?? right;
    const hitX = index === 0 ? left : (previousX + point.x) / 2;
    const hitRight = index === rawPoints.length - 1 ? right : (point.x + nextX) / 2;
    const tooltipWidth = 188;
    const tooltipHeight = 88;
    const minY = Math.min(point.tokenY, point.userY);
    const tooltipX = Math.min(
      right - tooltipWidth,
      Math.max(left, point.x - tooltipWidth / 2),
    );
    const tooltipY = Math.max(
      top,
      Math.min(bottom - tooltipHeight, minY - tooltipHeight - 12),
    );
    return {
      ...point,
      hitX,
      hitWidth: Math.max(20, hitRight - hitX),
      tooltipX,
      tooltipY,
    };
  });
  return {
    left,
    right,
    top,
    bottom,
    points,
    leftTicks: tokenAxis.ticks,
    rightTicks: userAxis.ticks,
    xLabels: points.filter((_point, index) => rows.length <= 8 || index % 2 === 0 || index === rows.length - 1),
    tokenPoints: linePoints(points, 'tokenY'),
    userPoints: linePoints(points, 'userY'),
  };
});

const activeChartPoint = computed(() => {
  if (activeChartIndex.value === null) {
    return null;
  }
  return tokenChart.value.points[activeChartIndex.value] || null;
});

const showChartTooltip = (index) => {
  activeChartIndex.value = Number(index);
};

const hideChartTooltip = (event) => {
  if (event?.pointerType === 'touch') {
    return;
  }
  activeChartIndex.value = null;
};

const runSearch = () => {
  currentPage.value = 1;
  refresh();
};

const setTierFilter = (value) => {
  const nextValue = ['all', 'supporter', 'free'].includes(value) ? value : 'all';
  if (tierFilter.value === nextValue) {
    return;
  }
  tierFilter.value = nextValue;
  currentPage.value = 1;
  refresh();
};

const goToPage = (page) => {
  const nextPage = Math.max(1, Math.min(Number(page || 1), pageCount.value));
  if (nextPage === currentPage.value) {
    return;
  }
  currentPage.value = nextPage;
  refresh();
};

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
    set_supporter: true,
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
    set_supporter: true,
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
      set_supporter: tokenAdjust.value.set_supporter,
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
    overview.value = await adminClient.overview({
      q: query.value,
      page: currentPage.value,
      pageSize,
      tier: tierFilter.value,
    });
    currentPage.value = Number(overview.value?.users_page?.page || currentPage.value);
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

.admin-chart-legend {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.7rem;
}

.admin-chart-legend span {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  text-transform: none;
}

.admin-chart-legend i {
  width: 0.85rem;
  height: 0.85rem;
  border-radius: 999px;
}

.legend-token {
  background: var(--accent);
}

.legend-users {
  background: color-mix(in srgb, var(--text-main) 88%, black);
}

.admin-chart-wrap {
  margin-top: 1rem;
  overflow-x: auto;
  border: 1px solid var(--border-main);
  border-radius: 1rem;
  background: color-mix(in srgb, var(--bg-main) 58%, transparent);
  padding: 0.75rem;
}

.admin-line-chart {
  display: block;
  width: 100%;
  min-width: 48rem;
  height: auto;
}

.chart-grid line {
  stroke: color-mix(in srgb, var(--border-main) 72%, transparent);
  stroke-width: 1;
}

.chart-axis line {
  stroke: color-mix(in srgb, var(--text-secondary) 36%, transparent);
  stroke-width: 1.4;
}

.chart-labels text {
  fill: var(--text-secondary);
  font-size: 0.72rem;
  font-weight: 900;
}

.chart-line {
  fill: none;
  stroke-width: 4;
  stroke-linecap: round;
  stroke-linejoin: round;
  vector-effect: non-scaling-stroke;
}

.token-line {
  stroke: var(--accent);
}

.users-line {
  opacity: 0.9;
  stroke: color-mix(in srgb, var(--text-main) 88%, black);
  stroke-dasharray: 8 7;
  stroke-width: 3.5;
}

.chart-point {
  stroke: var(--bg-card);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}

.token-point {
  fill: var(--accent);
}

.users-point {
  fill: color-mix(in srgb, var(--text-main) 88%, black);
}

.chart-point.active {
  stroke-width: 3;
}

.chart-hover-line {
  opacity: 0.56;
  stroke: var(--text-secondary);
  stroke-dasharray: 5 6;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}

.chart-hit-area {
  cursor: crosshair;
  fill: transparent;
  outline: none;
  pointer-events: all;
}

.chart-hit-area:focus-visible {
  stroke: color-mix(in srgb, var(--accent) 62%, transparent);
  stroke-width: 2;
}

.chart-tooltip-object {
  overflow: visible;
  pointer-events: none;
}

.admin-chart-tooltip {
  min-height: 5.2rem;
  border: 1px solid color-mix(in srgb, var(--border-main) 76%, transparent);
  border-radius: 0.85rem;
  background: color-mix(in srgb, var(--bg-card) 96%, var(--bg-main) 4%);
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.2);
  color: var(--text-main);
  font-size: 0.75rem;
  font-weight: 850;
  padding: 0.7rem 0.8rem;
}

.tooltip-date {
  color: var(--text-main);
  font-weight: 950;
  margin-bottom: 0.45rem;
}

.tooltip-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.7rem;
  line-height: 1.5;
}

.tooltip-row span {
  display: inline-flex;
  align-items: center;
  color: var(--text-secondary);
  gap: 0.35rem;
}

.tooltip-row i {
  width: 0.65rem;
  height: 0.65rem;
  border-radius: 999px;
}

.tooltip-row strong {
  color: var(--text-main);
  font-weight: 950;
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

.admin-segmented {
  display: inline-grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  overflow: hidden;
  border: 1px solid var(--border-main);
  border-radius: 0.85rem;
  background: color-mix(in srgb, var(--bg-main) 74%, transparent);
}

.admin-segment-btn {
  min-height: 2.35rem;
  border: 0;
  border-right: 1px solid var(--border-main);
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 950;
  padding: 0.45rem 0.7rem;
  transition: background-color 0.16s ease, color 0.16s ease;
}

.admin-segment-btn:last-child {
  border-right: 0;
}

.admin-segment-btn:hover:not(:disabled),
.admin-segment-btn.active {
  background: color-mix(in srgb, var(--accent) 16%, transparent);
  color: var(--text-main);
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

.admin-pagination {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 1rem;
  border-top: 1px solid var(--border-main);
  margin-top: 1rem;
  padding-top: 1rem;
}

.admin-pagination-center {
  display: grid;
  min-width: 0;
  justify-items: center;
  gap: 0.55rem;
  text-align: center;
}

.admin-pagination-controls {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.admin-page-btn,
.admin-page-nav-btn,
.admin-page-ellipsis {
  display: inline-flex;
  min-width: 2.25rem;
  min-height: 2.25rem;
  align-items: center;
  justify-content: center;
  border-radius: 0.75rem;
  font-size: var(--font-ui-xs);
  font-weight: 950;
}

.admin-page-nav-btn {
  min-width: 2.65rem;
  min-height: 2.65rem;
  border-radius: 0.9rem;
  font-size: 1.05rem;
}

.admin-page-btn,
.admin-page-nav-btn {
  border: 1px solid var(--border-main);
  background: color-mix(in srgb, var(--bg-main) 68%, transparent);
  color: var(--text-main);
  transition: border-color 0.16s ease, background-color 0.16s ease, color 0.16s ease;
}

.admin-page-btn:hover:not(:disabled),
.admin-page-nav-btn:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent);
}

.admin-page-btn.active {
  border-color: color-mix(in srgb, var(--accent) 72%, var(--border-main));
  background: color-mix(in srgb, var(--accent) 18%, var(--bg-main));
  color: var(--text-main);
}

.admin-page-ellipsis {
  color: var(--text-secondary);
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

.admin-tier-pill {
  display: inline-flex;
  align-items: center;
  border: 1px solid color-mix(in srgb, var(--accent) 48%, transparent);
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 13%, transparent);
  color: var(--accent);
  font-size: 0.68rem;
  font-weight: 950;
  line-height: 1;
  padding: 0.18rem 0.45rem;
  text-transform: uppercase;
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

.admin-check-row {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  border: 1px solid var(--border-main);
  border-radius: 0.85rem;
  background: color-mix(in srgb, var(--bg-main) 66%, transparent);
  color: var(--text-main);
  cursor: pointer;
  font-size: var(--font-ui-sm);
  font-weight: 900;
  padding: 0.7rem 0.8rem;
}

.admin-check-row input {
  width: 1rem;
  height: 1rem;
  accent-color: var(--accent);
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

  .admin-chart-legend {
    justify-content: flex-start;
  }

  .admin-pagination {
    grid-template-columns: auto minmax(0, 1fr) auto;
    gap: 0.65rem;
  }

  .admin-pagination-controls {
    justify-content: center;
  }

  .admin-page-btn,
  .admin-page-ellipsis {
    min-width: 2rem;
    min-height: 2rem;
  }

  .admin-page-nav-btn {
    min-width: 2.4rem;
    min-height: 2.4rem;
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
