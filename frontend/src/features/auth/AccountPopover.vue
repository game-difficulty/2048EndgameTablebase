<template>
    <div
      v-if="open && authUser"
      class="account-popover fixed right-3 top-[3.75rem] z-[300] w-[18rem] rounded-2xl border border-border-main p-4 text-text-main shadow-[0_20px_70px_rgba(15,23,42,0.35)]"
      data-account-menu
    >
      <div class="account-menu-head">
        <AccountAvatar
          :user="authUser"
          :supporter="hasSupporterPresentation"
          size="large"
          editable
          @edit="$emit('avatar')"
        />
        <div class="min-w-0">
          <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('auth.account.title') }}</div>
          <button
            type="button"
            class="account-name-button mt-1"
            :title="$t('profile.displayName.change')"
            @click="$emit('name')"
          >
            <span class="truncate">{{ accountDisplayName }}</span>
            <span class="account-name-edit" aria-hidden="true">✎</span>
          </button>
          <div class="truncate text-[0.72rem] font-bold text-text-secondary">{{ authUser.email }}</div>
        </div>
      </div>
      <div class="mt-4 grid gap-2 rounded-xl border border-border-main bg-bg-main/55 p-3">
        <div class="flex items-center justify-between gap-3">
          <span class="ui-caption font-black text-text-secondary">{{ $t('auth.account.bonusTokens') }}</span>
          <span class="ui-caption font-black text-text-main">{{ formatTokens(authUser.token_balance?.bonus) }}</span>
        </div>
        <div class="flex items-center justify-between gap-3">
          <span class="ui-caption font-black text-text-secondary">{{ $t('auth.account.paidTokens') }}</span>
          <span class="ui-caption font-black text-text-main">{{ formatTokens(authUser.token_balance?.paid) }}</span>
        </div>
        <div class="flex items-center justify-between gap-3 border-t border-border-main pt-2">
          <span class="ui-caption font-black text-text-secondary">{{ $t('auth.account.totalTokens') }}</span>
          <span class="ui-body font-black text-accent">{{ formatTokens(authUser.token_balance?.total) }}</span>
        </div>
      </div>
      <button
        type="button"
        class="action-btn-small mt-4 w-full justify-center"
        @click="$emit('quota')"
      >
        {{ $t('billing.quotaGuide.open') }}
      </button>
      <button
        type="button"
        class="action-btn-small mt-2 w-full justify-center"
        @click="$emit('sponsor')"
      >
        {{ $t('billing.open') }}
      </button>
      <button
        v-if="canOpenAdmin"
        type="button"
        class="action-btn-small mt-2 w-full justify-center"
        @click="$emit('admin')"
      >
        {{ $t('admin.open') }}
      </button>
      <div class="mt-2 grid grid-cols-2 gap-2">
        <button type="button" class="action-btn-small justify-center" @click="$emit('security', 'changePassword')">
          {{ $t('auth.account.changePassword') }}
        </button>
        <button type="button" class="action-btn-small justify-center !border-red-400/40 !text-red-500" @click="$emit('security', 'deactivate')">
          {{ $t('auth.account.deactivateAccount') }}
        </button>
      </div>
      <button type="button" class="action-btn-small mt-2 w-full justify-center" @click="$emit('logout')">
        {{ $t('auth.actions.logout') }}
      </button>
    </div>

</template>
<script setup>
import AccountAvatar from './AccountAvatar.vue';
defineProps({ open: Boolean, authUser: Object, hasSupporterPresentation: Boolean, accountDisplayName: String, canOpenAdmin: Boolean });
defineEmits(['avatar', 'name', 'quota', 'sponsor', 'admin', 'security', 'logout']);
const formatTokens = value => {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toLocaleString(undefined, { minimumFractionDigits: number % 1 === 0 ? 0 : 1, maximumFractionDigits: 3 }) : '0';
};
</script>
<style scoped>
.account-popover { background:linear-gradient(var(--bg-card),var(--bg-card)),var(--bg-main);max-height:calc(100dvh - 4.5rem);overflow-y:auto; }
.account-menu-head {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}

.account-menu-head > div {
  flex: 1 1 auto;
}

.account-name-button {
  display: flex;
  width: 100%;
  min-width: 0;
  align-items: center;
  gap: 0.4rem;
  color: var(--text-main);
  font-size: 0.9rem;
  font-weight: 900;
  line-height: 1.2;
  text-align: left;
}

.account-name-button:hover,
.account-name-button:focus-visible {
  color: var(--accent);
  outline: none;
}

.account-name-edit {
  flex: 0 0 auto;
  color: var(--text-secondary);
  font-size: 0.72rem;
  opacity: 0.7;
}
</style>
