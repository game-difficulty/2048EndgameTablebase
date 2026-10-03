<template>
  <div class="page-root overflow-y-auto p-5" @click="actionMenuUserId = null">
    <div class="mx-auto flex w-full max-w-6xl flex-col gap-4">
      <header class="admin-page-header rounded-2xl border border-border-main bg-bg-card/88 p-5 shadow-sm backdrop-blur-md">
        <div class="flex flex-row items-end justify-between gap-4">
          <div>
            <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('admin.kicker') }}</div>
            <h1 class="mt-1 ui-metric font-black text-text-main">{{ $t(isOwner ? 'admin.title' : 'admin.moderator.title') }}</h1>
            <p v-if="isOwner" class="mt-2 max-w-2xl ui-body text-text-secondary">{{ $t('admin.subtitle') }}</p>
          </div>
          <button type="button" class="action-btn-small justify-center" :disabled="loading" @click="refreshCurrent">
            {{ loading ? $t('common.updating') : $t('admin.refresh') }}
          </button>
        </div>
      </header>

      <AdminLiveControl v-if="isOwner" :active="active" />
      <div v-if="error" class="rounded-2xl border border-red-400/35 bg-red-500/10 p-4 ui-body font-bold text-red-500">
        {{ error }}
      </div>

      <section v-if="isOwner" class="grid grid-cols-6 gap-3">
        <div v-for="card in summaryCards" :key="card.key" class="admin-card">
          <div class="ui-caption font-black uppercase text-text-secondary">{{ card.label }}</div>
          <div class="mt-2 text-2xl font-black text-text-main">{{ card.value }}</div>
        </div>
      </section>

      <section v-if="isOwner" class="admin-panel">
        <div class="admin-panel-head">
          <div>
            <h2>{{ $t('admin.tokenActivity.title') }}</h2>
            <span>{{ $t('admin.tokenActivity.range') }}</span>
          </div>
          <div class="admin-chart-legend">
            <span><i class="legend-token" />{{ $t('admin.tokenActivity.tokens') }}</span>
            <span><i class="legend-users" />{{ $t('admin.tokenActivity.users') }}</span>
            <span><i class="legend-active" />{{ $t('admin.tokenActivity.activeAccounts') }}</span>
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
            <polyline class="chart-line active-line" :points="tokenChart.activePoints" />
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
              <circle class="chart-point active-point active" :cx="activeChartPoint.x" :cy="activeChartPoint.activeY" r="6" />
              <foreignObject
                class="chart-tooltip-object"
                :x="activeChartPoint.tooltipX"
                :y="activeChartPoint.tooltipY"
                width="188"
                height="116"
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
                  <div class="tooltip-row">
                    <span><i class="legend-active" />{{ $t('admin.tokenActivity.activeAccounts') }}</span>
                    <strong>{{ formatNumber(activeChartPoint.activeAccounts) }}</strong>
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
              <circle
                v-for="point in tokenChart.points"
                :key="`active-point-${point.date}`"
                class="chart-point active-point"
                :cx="point.x"
                :cy="point.activeY"
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
                :aria-label="`${point.date}: ${formatTokens(point.tokens)} tokens, ${formatNumber(point.usersCount)} spending users, ${formatNumber(point.activeAccounts)} active accounts`"
                @pointerenter="showChartTooltip(point.index)"
                @pointermove="showChartTooltip(point.index)"
                @pointerdown.prevent="showChartTooltip(point.index)"
                @focus="showChartTooltip(point.index)"
              />
            </g>
          </svg>
        </div>
      </section>

      <nav class="admin-section-tabs" role="tablist" :aria-label="$t('admin.tabs.label')">
        <button type="button" role="tab" :aria-selected="adminSection === 'users'" :class="{ active: adminSection === 'users' }" @click="adminSection = 'users'">
          {{ $t('admin.tabs.users') }}
        </button>
        <button type="button" role="tab" :aria-selected="adminSection === 'approvals'" :class="{ active: adminSection === 'approvals' }" @click="adminSection = 'approvals'">
          {{ $t('admin.tabs.approvals') }}
        </button>
        <button type="button" role="tab" :aria-selected="adminSection === 'profileReviews'" :class="{ active: adminSection === 'profileReviews' }" @click="adminSection = 'profileReviews'">
          {{ $t('admin.tabs.profileReviews') }}
        </button>
      </nav>

      <section v-if="adminSection === 'users'" class="admin-panel">
        <div class="admin-panel-head admin-users-head">
          <h2>{{ $t('admin.users.title') }}</h2>
          <div class="admin-users-toolbar">
            <UiSelect
              class="admin-tier-select"
              :model-value="tierFilter"
              :options="tierFilterOptions"
              :aria-label="$t('admin.users.tierFilter')"
              :disabled="loading"
              align="right"
              trigger-class="admin-tier-select-trigger"
              option-class="font-black"
              @change="setTierFilter"
            />
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
                <th v-if="isOwner">{{ $t('admin.users.tokens') }}</th>
                <th v-if="isOwner">{{ $t('admin.users.sessions') }}</th>
                <th v-if="isOwner">{{ $t('admin.users.usage') }}</th>
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
                  <span v-if="item.is_owner || item.role === 'moderator'" class="admin-tier-pill">{{ $t(item.is_owner ? 'admin.moderator.owner' : 'admin.moderator.identity') }}</span>
                  <div class="text-[0.75rem] font-bold text-text-secondary">{{ item.email }}</div>
                  <div v-if="item.entitlements?.is_supporter" class="mt-1">
                    <span class="admin-tier-pill">{{ $t('admin.entitlements.supporter') }}</span>
                  </div>
                  <div v-if="item.managed_test_account" class="mt-1 ui-caption text-text-secondary">{{ $t('admin.actions.managedAccount') }}</div>
                </td>
                <td>
                  <span :class="['admin-status', item.status === 'active' ? 'active' : 'disabled']">
                    {{ item.status }}
                  </span>
                </td>
                <td v-if="isOwner">
                  <div>{{ formatTokens(item.token_balance?.total) }}</div>
                  <div class="text-[0.72rem] font-bold text-text-secondary">
                    {{ $t('admin.tokens.balanceDetail', {
                      bonus: formatTokens(item.token_balance?.bonus),
                      paid: formatTokens(item.token_balance?.paid)
                    }) }}
                  </div>
                </td>
                <td v-if="isOwner">{{ formatNumber(item.sessions) }}</td>
                <td v-if="isOwner">{{ formatNumber(item.usage_events) }}</td>
                <td>{{ formatDate(item.created_at) }}</td>
                <td>{{ formatDate(item.last_login_at) }}</td>
                <td class="admin-user-action-cell">
                  <div class="admin-user-actions" data-admin-user-actions @click.stop>
                    <button type="button" class="action-btn-small admin-action-trigger justify-center" :aria-expanded="actionMenuUserId === item.id" @click="toggleActionMenu(item.id, $event)">
                      {{ $t('admin.actions.open') }}
                    </button>
                    <span v-if="item.pending_approval" class="admin-approval-dot" aria-hidden="true" />
                  </div>
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
          <div v-if="isOwner" class="admin-pagination-center">
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
          <span v-else class="ui-caption">{{ $t('admin.moderator.page', { page: currentPage }) }}</span>
          <button
            type="button"
            class="admin-page-nav-btn"
            :aria-label="$t('admin.users.nextPage')"
            :disabled="(isOwner ? currentPage >= pageCount : !usersPage.has_more) || loading"
            @click="goToPage(currentPage + 1)"
          >
            &gt;
          </button>
        </div>
      </section>
      <AdminApprovalTransactions
        v-else-if="adminSection === 'approvals'"
        ref="approvalTransactionsPanel"
        :active="active && adminSection === 'approvals'"
        @changed="refresh"
      />
      <AdminProfileReviews
        v-else
        ref="profileReviewsPanel"
        :active="active && adminSection === 'profileReviews'"
      />
    </div>

    <Teleport to="body">
      <div v-if="activeMenuUser" class="admin-user-action-menu admin-floating-menu" :style="actionMenuStyle" role="menu" data-admin-user-actions @click.stop>
        <button v-if="isOwner" type="button" role="menuitem" @click="openTokenAdjust(activeMenuUser)">{{ $t('admin.tokens.adjust') }}</button>
        <button type="button" role="menuitem" @click="openApprovals(activeMenuUser)">{{ $t('admin.actions.approvals') }}</button>
        <button v-if="isOwner && activeMenuUser.managed_test_account" type="button" role="menuitem" @click="openManagedPassword(activeMenuUser)">{{ $t('admin.actions.resetPassword') }}</button>
        <button v-if="canChangeRole(activeMenuUser)" type="button" role="menuitem" @click="openRoleChange(activeMenuUser)">{{ $t(activeMenuUser.role === 'moderator' ? 'admin.moderator.revoke' : 'admin.moderator.appoint') }}</button>
        <button v-if="canChangeStatus(activeMenuUser)" type="button" role="menuitem" :class="{ danger: activeMenuUser.status === 'active' }" @click="openStatusChange(activeMenuUser)">
          {{ $t(activeMenuUser.status === 'active' ? 'admin.actions.disable' : 'admin.actions.enable') }}
        </button>
      </div>
    </Teleport>

    <div v-if="approvals.open" class="admin-modal">
      <div class="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" @click="closeApprovals" />
      <section class="admin-modal-panel admin-approval-modal">
        <div class="flex items-start justify-between gap-4">
          <div class="min-w-0">
            <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('admin.actions.approvals') }}</div>
            <h2 class="mt-1 ui-metric font-black text-text-main">{{ approvals.user?.display_name || '-' }}</h2>
            <div class="mt-1 truncate ui-body font-bold text-text-secondary">#{{ approvals.user?.id }} · {{ approvals.user?.email }}</div>
          </div>
          <button type="button" class="action-btn-small admin-modal-close" @click="closeApprovals">{{ $t('common.close') }}</button>
        </div>
        <AdminVerseClaims :active="approvals.open" :user-id="Number(approvals.user?.id || 0) || null" @changed="refresh" />
        <AdminArchiveApplications :active="approvals.open" :user-id="Number(approvals.user?.id || 0) || null" @changed="refresh" />
      </section>
    </div>

    <div v-if="statusChange.open" class="admin-modal">
      <div class="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" @click="closeStatusChange" />
      <section class="admin-modal-panel">
        <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('admin.actions.accountStatus') }}</div>
        <h2 class="mt-1 ui-metric font-black text-text-main">
          {{ $t(statusChange.target === 'disabled' ? 'admin.actions.disableTitle' : 'admin.actions.enableTitle') }}
        </h2>
        <p class="mt-2 ui-body text-text-secondary">{{ statusChange.user?.display_name || '-' }} · {{ statusChange.user?.email }}</p>
        <p class="mt-4 ui-body text-text-secondary">
          {{ $t(statusChange.target === 'disabled' ? 'admin.actions.disableHint' : 'admin.actions.enableHint') }}
        </p>
        <div v-if="statusChange.error" class="admin-alert error">{{ statusChange.error }}</div>
        <div class="mt-5 grid grid-cols-2 gap-2">
          <button type="button" class="action-btn-small justify-center" :disabled="statusChange.submitting" @click="closeStatusChange">{{ $t('common.cancel') }}</button>
          <button type="button" :class="['action-btn-small justify-center', statusChange.target === 'disabled' ? 'admin-danger-action' : 'btn-prominent']" :disabled="statusChange.submitting" @click="submitStatusChange">
            {{ statusChange.submitting ? $t('common.updating') : $t(statusChange.target === 'disabled' ? 'admin.actions.confirmDisable' : 'admin.actions.confirmEnable') }}
          </button>
        </div>
      </section>
    </div>

    <div v-if="roleChange.user && isOwner" class="admin-modal">
      <div class="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" @click="!roleChange.submitting && (roleChange.user = null)" />
      <section class="admin-modal-panel" role="dialog" aria-modal="true" :aria-label="$t('admin.moderator.identity')">
        <h2 class="ui-metric font-black">{{ $t(roleChange.user.role === 'moderator' ? 'admin.moderator.revoke' : 'admin.moderator.appoint') }}</h2>
        <p class="mt-2 ui-body">{{ roleChange.user.display_name }} · {{ roleChange.user.email }}</p>
        <p class="mt-4 ui-body text-text-secondary">{{ $t('admin.moderator.confirmHint') }}</p>
        <p v-if="roleChange.error" class="admin-alert error">{{ roleChange.error }}</p>
        <div class="mt-5 grid grid-cols-2 gap-2">
          <button class="action-btn-small justify-center" :disabled="roleChange.submitting" @click="roleChange.user = null">{{ $t('common.cancel') }}</button>
          <button class="action-btn-small btn-prominent justify-center" :disabled="roleChange.submitting" @click="submitRoleChange">{{ $t('admin.moderator.confirm') }}</button>
        </div>
      </section>
    </div>

    <div v-if="managedPassword.open && isOwner" class="admin-modal">
      <div class="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" @click="closeManagedPassword" />
      <section class="admin-modal-panel">
        <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('admin.actions.managedAccount') }}</div>
        <h2 class="mt-1 ui-metric font-black text-text-main">{{ $t('admin.actions.resetPassword') }}</h2>
        <p class="mt-2 ui-body text-text-secondary">{{ managedPassword.user?.display_name }} · {{ managedPassword.user?.email }}</p>
        <p class="mt-4 ui-body text-text-secondary">{{ $t('admin.actions.resetPasswordHint') }}</p>
        <form class="mt-4 grid gap-3" @submit.prevent="submitManagedPassword">
          <label class="admin-form-row">
            <span>{{ $t('admin.actions.newPassword') }}</span>
            <input v-model="managedPassword.password" type="password" autocomplete="new-password" minlength="8" required class="admin-input" />
          </label>
          <div v-if="managedPassword.error" class="admin-alert error">{{ managedPassword.error }}</div>
          <div class="grid grid-cols-2 gap-2">
            <button type="button" class="action-btn-small justify-center" :disabled="managedPassword.submitting" @click="closeManagedPassword">{{ $t('common.cancel') }}</button>
            <button type="submit" class="action-btn-small btn-prominent justify-center" :disabled="managedPassword.submitting">{{ $t('admin.actions.confirmResetPassword') }}</button>
          </div>
        </form>
      </section>
    </div>

    <div v-if="tokenAdjust.open && isOwner" class="admin-modal">
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
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { userError } from '../../../services/errors/userError.js';

import { adminClient } from '../../../services/admin/adminClient';
import UiSelect from '../../../components/UiSelect.vue';
import AdminLiveControl from '../components/AdminLiveControl.vue';
import AdminVerseClaims from '../components/AdminVerseClaims.vue';
import AdminArchiveApplications from '../components/AdminArchiveApplications.vue';
import AdminApprovalTransactions from '../components/AdminApprovalTransactions.vue';
import AdminProfileReviews from '../components/AdminProfileReviews.vue';

const props = defineProps({
  active: Boolean,
});

const { t } = useI18n();
const permissions = ref(null);
const isOwner = computed(() => Boolean(permissions.value?.owner));
const roleChange = ref({ user: null, submitting: false, error: '' });
const canChangeStatus = user => user.id !== permissions.value?.user_id && (isOwner.value
  || (!user.is_owner && !['admin', 'moderator'].includes(user.role)));
const canChangeRole = user => isOwner.value && !user.is_owner && user.id !== permissions.value?.user_id
  && ['user', 'moderator'].includes(user.role) && (user.status === 'active' || user.role === 'moderator');
const openRoleChange = user => {
  actionMenuUserId.value = null;
  roleChange.value = { user, submitting: false, error: '' };
};
const submitRoleChange = async () => {
  roleChange.value.submitting = true;
  try {
    const { user } = await adminClient.setModerator(roleChange.value.user.id, roleChange.value.user.role !== 'moderator');
    replaceUserInOverview(user);
    roleChange.value.user = null;
  } catch (error) {
    roleChange.value.error = userError(error);
  } finally {
    roleChange.value.submitting = false;
  }
};
const loading = ref(false);
const adminSection = ref('users');
const approvalTransactionsPanel = ref(null);
const profileReviewsPanel = ref(null);
const error = ref('');
const query = ref('');
const tierFilter = ref('all');
const currentPage = ref(1);
const pageSize = 20;
const overview = ref(null);
const activeChartIndex = ref(null);
const actionMenuUserId = ref(null);
const actionMenuStyle = ref({});
const activeMenuUser = computed(() => users.value.find(user => user.id === actionMenuUserId.value));
const approvals = ref({ open: false, user: null });
const statusChange = ref({ open: false, user: null, target: 'disabled', submitting: false, error: '' });
const managedPassword = ref({ open: false, user: null, password: '', submitting: false, error: '' });
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
  maximumFractionDigits: 1,
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
  { value: 'moderator', label: t('admin.moderator.identity') },
  { value: 'supporter', label: t('admin.users.tierSupporter') },
  { value: 'free', label: t('admin.users.tierFree') },
  { value: 'pending', label: t('admin.users.tierPending') },
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
    Math.max(1, ...rows.map((row) => Math.max(Number(row.spending_users || 0), Number(row.active_accounts || 0)))),
    top,
    bottom,
    { targetTicks: 8, integer: true },
  );
  const span = Math.max(1, rows.length - 1);
  const rawPoints = rows.map((row, index) => {
    const x = left + ((right - left) * index) / span;
    const tokens = Number(row.tokens_spent || 0);
    const usersCount = Number(row.spending_users || 0);
    const activeAccounts = Number(row.active_accounts || 0);
    return {
      index,
      date: row.date,
      label: String(row.date || '').slice(5),
      x,
      tokens,
      usersCount,
      activeAccounts,
      tokenY: bottom - ((bottom - top) * tokens) / tokenAxis.axisMax,
      userY: bottom - ((bottom - top) * usersCount) / userAxis.axisMax,
      activeY: bottom - ((bottom - top) * activeAccounts) / userAxis.axisMax,
    };
  });
  const points = rawPoints.map((point, index) => {
    const previousX = rawPoints[index - 1]?.x ?? left;
    const nextX = rawPoints[index + 1]?.x ?? right;
    const hitX = index === 0 ? left : (previousX + point.x) / 2;
    const hitRight = index === rawPoints.length - 1 ? right : (point.x + nextX) / 2;
    const tooltipWidth = 188;
    const tooltipHeight = 116;
    const minY = Math.min(point.tokenY, point.userY, point.activeY);
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
    activePoints: linePoints(points, 'activeY'),
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
  const nextValue = ['all', 'supporter', 'free', 'pending', 'moderator'].includes(value) ? value : 'all';
  if (tierFilter.value === nextValue) {
    return;
  }
  tierFilter.value = nextValue;
  currentPage.value = 1;
  refresh();
};

const goToPage = (page) => {
  const nextPage = Math.max(1, Math.min(Number(page || 1), isOwner.value ? pageCount.value : currentPage.value + (usersPage.value.has_more ? 1 : 0)));
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
    ? items.map((item) => (Number(item.id) === Number(updatedUser.id)
      ? { ...item, ...updatedUser, pending_approval: updatedUser.pending_approval ?? item.pending_approval }
      : item))
    : items);
  overview.value = {
    ...overview.value,
    users: replace(overview.value.users),
    recent_users: replace(overview.value.recent_users),
  };
};

const defaultTokenReason = () => t('admin.tokens.defaultReason');

const toggleActionMenu = (userId, event) => {
  const rect = event.currentTarget.getBoundingClientRect();
  actionMenuStyle.value = {
    left: `${Math.max(8, Math.min(rect.right - 200, window.innerWidth - 208))}px`,
    ...(rect.bottom + 240 > window.innerHeight
      ? { bottom: `${window.innerHeight - rect.top + 6}px` }
      : { top: `${rect.bottom + 6}px` }),
  };
  actionMenuUserId.value = actionMenuUserId.value === userId ? null : userId;
};
const dismissActionMenu = event => {
  if (event.type === 'keydown' && event.key !== 'Escape') return;
  if (event.type === 'pointerdown' && event.target.closest?.('[data-admin-user-actions]')) return;
  actionMenuUserId.value = null;
};

const openApprovals = (user) => {
  actionMenuUserId.value = null;
  approvals.value = { open: true, user };
};

const closeApprovals = () => { approvals.value = { open: false, user: null }; };

const openStatusChange = (user) => {
  actionMenuUserId.value = null;
  statusChange.value = {
    open: true,
    user,
    target: user.status === 'active' ? 'disabled' : 'active',
    submitting: false,
    error: '',
  };
};

const closeStatusChange = () => {
  if (statusChange.value.submitting) return;
  statusChange.value = { ...statusChange.value, open: false, error: '' };
};

const submitStatusChange = async () => {
  if (!statusChange.value.user?.id) return;
  statusChange.value = { ...statusChange.value, submitting: true, error: '' };
  try {
    const response = await adminClient.updateUserStatus(statusChange.value.user.id, statusChange.value.target);
    replaceUserInOverview(response.user);
    statusChange.value = { ...statusChange.value, open: false, submitting: false, user: response.user };
  } catch (requestError) {
    statusChange.value = { ...statusChange.value, submitting: false, error: userError(requestError, t('admin.actions.statusFailed')) };
  }
};

const openManagedPassword = (user) => {
  actionMenuUserId.value = null;
  managedPassword.value = { open: true, user, password: '', submitting: false, error: '' };
};

const closeManagedPassword = () => {
  if (managedPassword.value.submitting) return;
  managedPassword.value = { open: false, user: null, password: '', submitting: false, error: '' };
};

const submitManagedPassword = async () => {
  const { user, password } = managedPassword.value;
  if (!user?.managed_test_account) return;
  if (password.length < 8) {
    managedPassword.value.error = t('admin.actions.passwordTooShort');
    return;
  }
  managedPassword.value.submitting = true;
  managedPassword.value.error = '';
  try {
    await adminClient.resetManagedPassword(user.id, password);
    managedPassword.value.submitting = false;
    closeManagedPassword();
  } catch (requestError) {
    managedPassword.value.error = userError(requestError, t('admin.actions.resetPasswordFailed'));
  } finally {
    managedPassword.value.submitting = false;
  }
};

const openTokenAdjust = (user) => {
  actionMenuUserId.value = null;
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
      error: userError(requestError, t('admin.tokens.updateFailed')),
    };
  }
};

const refresh = async () => {
  loading.value = true;
  error.value = '';
  try {
    permissions.value = await adminClient.permissions();
    overview.value = await (isOwner.value ? adminClient.overview : adminClient.users)({
      q: query.value,
      page: currentPage.value,
      pageSize,
      tier: tierFilter.value,
    });
    currentPage.value = Number(overview.value?.users_page?.page || currentPage.value);
  } catch (requestError) {
    if ([401, 403].includes(requestError.status)) {
      permissions.value = null;
      overview.value = null;
    }
    error.value = userError(requestError, t('admin.errors.loadFailed'));
  } finally {
    loading.value = false;
  }
};

const refreshCurrent = () => {
  if (adminSection.value === 'approvals') {
    approvalTransactionsPanel.value?.load?.();
    return;
  }
  if (adminSection.value === 'profileReviews') {
    profileReviewsPanel.value?.load?.();
    return;
  }
  refresh();
};

onMounted(() => {
  document.addEventListener('pointerdown', dismissActionMenu);
  document.addEventListener('keydown', dismissActionMenu);
  window.addEventListener('scroll', dismissActionMenu, true);
  window.addEventListener('resize', dismissActionMenu);
  if (props.active) {
    refresh();
  }
});

onUnmounted(() => {
  document.removeEventListener('pointerdown', dismissActionMenu);
  document.removeEventListener('keydown', dismissActionMenu);
  window.removeEventListener('scroll', dismissActionMenu, true);
  window.removeEventListener('resize', dismissActionMenu);
});

watch(() => props.active, (active) => {
  if (active) {
    refresh();
  }
});
</script>

<style scoped>
.admin-page-header h1 { white-space: nowrap; }
.admin-page-header .action-btn-small { width: auto; flex: 0 0 auto; }
.admin-floating-menu.admin-user-action-menu { position: fixed; top: auto; bottom: auto; right: auto; width: 200px; z-index: 450; max-height: calc(100vh - 16px); overflow-y: auto; background: var(--bg-main); }
.admin-section-tabs {
  display: inline-flex;
  align-self: flex-start;
  max-width: 100%;
  overflow-x: auto;
  gap: 0.3rem;
  border: 1px solid var(--border-main);
  border-radius: 0.9rem;
  background: color-mix(in srgb, var(--bg-card) 90%, transparent);
  padding: 0.3rem;
}

.admin-section-tabs button {
  flex: 0 0 auto;
  white-space: nowrap;
  border: 0;
  border-radius: 0.65rem;
  background: transparent;
  color: var(--text-secondary);
  font-size: var(--font-ui-sm);
  font-weight: 900;
  padding: 0.62rem 1rem;
  cursor: pointer;
}

.admin-section-tabs button.active {
  background: var(--accent);
  color: var(--accent-contrast, #fff);
  box-shadow: 0 5px 14px color-mix(in srgb, var(--accent) 22%, transparent);
}

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

.legend-active {
  background: #16a385;
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

.active-line {
  stroke: #16a385;
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

.active-point {
  fill: #16a385;
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

.admin-tier-select {
  width: 8rem;
  flex: 0 0 auto;
}

.admin-tier-select :deep(.admin-tier-select-trigger) {
  min-height: 2.35rem;
  border: 1px solid var(--border-main);
  border-radius: 0.85rem;
  background: color-mix(in srgb, var(--bg-main) 82%, transparent);
  color: var(--text-secondary);
  font-size: var(--font-ui-sm);
  font-weight: 850;
  padding: 0.5rem 0.7rem;
}

.admin-users-head h2 {
  flex: 0 0 auto;
  white-space: nowrap;
}

.admin-users-toolbar {
  display: flex;
  flex: 0 1 auto;
  min-width: 0;
  flex-wrap: nowrap;
  align-items: center;
  justify-content: flex-end;
  gap: 0.5rem;
}

.admin-users-toolbar .admin-search {
  width: 14rem;
  min-width: 6rem;
  flex: 0 1 14rem;
}

.admin-users-toolbar .action-btn-small {
  width: auto;
  min-height: 2.35rem;
  flex: 0 0 auto;
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

.admin-user-action-cell {
  min-width: 7.25rem;
}

.admin-user-actions {
  position: relative;
  display: inline-block;
  overflow: visible;
}

.admin-action-trigger {
  position: relative;
}

.admin-approval-dot {
  position: absolute;
  top: -0.3rem;
  right: -0.3rem;
  width: 0.58rem;
  height: 0.58rem;
  border: 2px solid var(--bg-card);
  border-radius: 999px;
  background: rgb(239, 68, 68);
  box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.22);
  pointer-events: none;
  z-index: 2;
}

.admin-user-action-menu {
  position: absolute;
  z-index: 20;
  top: calc(100% + 0.4rem);
  right: 0;
  display: grid;
  width: 10.5rem;
  overflow: hidden;
  border: 1px solid var(--border-main);
  border-radius: 0.9rem;
  background: color-mix(in srgb, var(--bg-card) 98%, transparent);
  box-shadow: 0 16px 42px rgba(15, 23, 42, 0.24);
  padding: 0.35rem;
}

.admin-user-action-menu button {
  border: 0;
  border-radius: 0.65rem;
  background: transparent;
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  font-weight: 850;
  padding: 0.65rem 0.75rem;
  text-align: left;
}

.admin-user-action-menu button:hover {
  background: color-mix(in srgb, var(--accent) 12%, var(--bg-main));
}

.admin-user-action-menu button.danger {
  color: rgb(239, 68, 68);
}

.admin-danger-action {
  border-color: rgba(239, 68, 68, 0.38) !important;
  background: rgba(239, 68, 68, 0.13) !important;
  color: rgb(239, 68, 68) !important;
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

.admin-approval-modal {
  width: min(48rem, 100%);
}

.admin-approval-modal :deep(.admin-panel) {
  margin-top: 1rem;
  border-radius: 1rem;
  box-shadow: none;
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

</style>
