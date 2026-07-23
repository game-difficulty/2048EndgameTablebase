<template>
  <div class="app-shell h-screen w-screen flex flex-col overflow-hidden">
    <div class="flex items-center gap-2 overflow-x-auto bg-bg-main/80 p-2 shadow-sm z-50 border-b border-border-main backdrop-blur-md transition-colors duration-300">
      <div
        v-for="tab in openTabDefinitions"
        :key="tab.id"
        :class="[
          'flex items-center rounded-lg border transition-all duration-300',
          tab.id !== TAB_IDS.MAIN_MENU ? 'cursor-grab active:cursor-grabbing' : '',
          draggedTabId === tab.id ? 'scale-[0.98] opacity-45' : '',
          dragTargetTabId === tab.id && draggedTabId && draggedTabId !== tab.id
            ? 'ring-2 ring-accent/50 bg-accent/8'
            : '',
          activeTab === tab.id
            ? 'surface-prominent text-white scale-[1.02]'
            : 'border-transparent bg-transparent text-text-secondary hover:bg-btn-bg/10 hover:text-text-main'
        ]"
        :draggable="tab.id !== TAB_IDS.MAIN_MENU"
        @dragstart="handleTabDragStart(tab.id, $event)"
        @dragenter.prevent
        @dragover="handleTabDragOver(tab.id, $event)"
        @drop="handleTabDrop(tab.id, $event)"
        @dragend="handleTabDragEnd"
      >
        <button
          type="button"
          @click="handleActivateTab(tab.id, $event)"
              class="select-none px-4 py-2 font-black uppercase tracking-tighter ui-control outline-none active:scale-95 focus:outline-none focus-visible:outline-none"
        >
          {{ getTabLabel(tab) }}
        </button>
        <button
          v-if="tab.closable"
          type="button"
          @click.stop="handleCloseTab(tab.id, $event)"
                class="mr-1 select-none rounded-md px-2 py-1 ui-caption font-black uppercase opacity-70 outline-none transition-opacity hover:opacity-100 focus:outline-none focus-visible:outline-none"
          aria-label="Close tab"
        >
          ×
        </button>
      </div>
      <div class="relative ml-auto flex items-center gap-2 whitespace-nowrap pl-3" data-account-menu>
        <template v-if="authUser">
          <button
            type="button"
            :class="['action-btn-small account-trigger flex items-center gap-2', hasSupporterPresentation ? 'supporter' : '']"
            :aria-expanded="accountMenuOpen ? 'true' : 'false'"
            @click="accountMenuOpen = !accountMenuOpen"
          >
            <span :class="['account-avatar', hasSupporterPresentation ? 'supporter' : '']">
              {{ accountInitials }}
            </span>
            <span class="hidden max-w-[9rem] truncate sm:inline">{{ accountDisplayName }}</span>
          </button>
        </template>
        <template v-else>
          <button type="button" class="action-btn-small" @click="openAuthDialog('login')">
            {{ $t('auth.actions.login') }}
          </button>
          <button type="button" class="action-btn-small" @click="openAuthDialog('register')">
            {{ $t('auth.actions.register') }}
          </button>
        </template>
      </div>
    </div>

    <div
      v-if="accountMenuOpen && authUser"
      class="fixed right-3 top-[3.75rem] z-[110] w-[18rem] rounded-2xl border border-border-main bg-bg-card/98 p-4 text-text-main shadow-[0_20px_70px_rgba(15,23,42,0.35)]"
      data-account-menu
    >
      <div class="account-menu-head">
        <span :class="['account-avatar large', hasSupporterPresentation ? 'supporter' : '']">
          {{ accountInitials }}
        </span>
        <div class="min-w-0">
          <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('auth.account.title') }}</div>
          <div class="mt-1 truncate ui-body font-black">{{ accountDisplayName }}</div>
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
        @click="openSponsorDialog"
      >
        {{ $t('billing.open') }}
      </button>
      <button
        v-if="canOpenAdmin"
        type="button"
        class="action-btn-small mt-2 w-full justify-center"
        @click="openAdminPage"
      >
        {{ $t('admin.open') }}
      </button>
      <div class="mt-2 grid grid-cols-2 gap-2">
        <button type="button" class="action-btn-small justify-center" @click="openAccountSecurity('changePassword')">
          {{ $t('auth.account.changePassword') }}
        </button>
        <button type="button" class="action-btn-small justify-center !border-red-400/40 !text-red-500" @click="openAccountSecurity('deactivate')">
          {{ $t('auth.account.deactivateAccount') }}
        </button>
      </div>
      <button type="button" class="action-btn-small mt-2 w-full justify-center" @click="handleLogout">
        {{ $t('auth.actions.logout') }}
      </button>
    </div>

    <div class="flex-1 relative overflow-hidden">
      <div
        v-if="isTabOpen(TAB_IDS.MAIN_MENU)"
        class="absolute inset-0"
        v-show="activeTab === TAB_IDS.MAIN_MENU"
      >
        <MainMenuView :active="activeTab === TAB_IDS.MAIN_MENU" @selectTab="openTab" />
      </div>
      <div
        v-if="isTabOpen(TAB_IDS.GAMER)"
        class="absolute inset-0"
        v-show="activeTab === TAB_IDS.GAMER"
      >
        <GamerView :active="activeTab === TAB_IDS.GAMER" />
      </div>
      <div
        v-if="isTabOpen(TAB_IDS.TRAINER)"
        class="absolute inset-0"
        v-show="activeTab === TAB_IDS.TRAINER"
      >
        <TrainerView :active="activeTab === TAB_IDS.TRAINER" />
      </div>
      <div
        v-if="isTabOpen(TAB_IDS.TESTER)"
        class="absolute inset-0"
        v-show="activeTab === TAB_IDS.TESTER"
      >
        <TesterView
          :active="activeTab === TAB_IDS.TESTER"
          @navigate-tab="openTab"
          @open-analysis="openAnalysisDialog"
        />
      </div>
      <div
        v-if="isTabOpen(TAB_IDS.MINIGAMES)"
        class="absolute inset-0"
        v-show="activeTab === TAB_IDS.MINIGAMES"
      >
        <MinigamesView :active="activeTab === TAB_IDS.MINIGAMES" />
      </div>
      <div
        v-if="isTabOpen(TAB_IDS.REPLAY)"
        class="absolute inset-0"
        v-show="activeTab === TAB_IDS.REPLAY"
      >
        <ReplayReviewView
          :active="activeTab === TAB_IDS.REPLAY"
          @navigate-tab="openTab"
          @open-analysis="openAnalysisDialog"
        />
      </div>
      <div
        v-if="isTabOpen(TAB_IDS.SETTINGS)"
        class="absolute inset-0"
        v-show="activeTab === TAB_IDS.SETTINGS"
      >
        <SettingsView :active="activeTab === TAB_IDS.SETTINGS" />
      </div>
      <div
        v-if="isTabOpen(TAB_IDS.HELP)"
        class="absolute inset-0"
        v-show="activeTab === TAB_IDS.HELP"
      >
        <HelpView :active="activeTab === TAB_IDS.HELP" />
      </div>
      <div
        v-if="isTabOpen(TAB_IDS.ADMIN)"
        class="absolute inset-0"
        v-show="activeTab === TAB_IDS.ADMIN"
      >
        <AdminView :active="activeTab === TAB_IDS.ADMIN" />
      </div>
    </div>

    <ReplayAnalysisDialog
      :open="analysisDialogOpen"
      :context="analysisDialogContext"
      @close="closeAnalysisDialog"
    />

    <AccountSecurityDialog
      :open="accountSecurityDialog.open"
      :mode="accountSecurityDialog.mode"
      @close="closeAccountSecurity"
      @success="handleAccountSecuritySuccess"
      @deactivated="handleAccountDeactivated"
    />

    <SponsorDialog
      :open="sponsorDialogOpen"
      :user="authUser"
      @close="closeSponsorDialog"
    />

    <div
      v-if="authDialogOpen"
      class="absolute inset-0 z-[115] flex items-center justify-center p-6"
    >
      <div class="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" @click="closeAuthDialog" />
      <div class="relative z-10 w-full max-w-[28rem]">
        <AuthPage :initial-mode="authDialogMode" @authenticated="handleAuthenticated" />
      </div>
    </div>

    <div
      v-if="globalErrorDialog.open"
      class="absolute inset-0 z-[120] flex items-center justify-center p-6"
    >
      <div class="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" @click="closeGlobalErrorDialog" />
      <div class="relative z-10 w-full max-w-3xl rounded-[28px] border border-border-main bg-bg-card/96 p-6 shadow-[0_24px_80px_rgba(15,23,42,0.32)]">
        <div class="flex items-start justify-between gap-4">
          <div class="space-y-1">
            <div class="ui-metric font-black tracking-tight text-text-main">
              {{ globalErrorDialog.title || $t('appError.title') }}
            </div>
            <div class="ui-body text-text-secondary">
              {{ $t('appError.note') }}
            </div>
          </div>
            <div class="flex items-center gap-2">
            <button
              type="button"
              class="action-btn-small"
              @click="globalErrorExpanded = !globalErrorExpanded"
            >
              {{ globalErrorExpanded ? $t('appError.hideDetails') : $t('appError.showDetails') }}
            </button>
            <button
              type="button"
              class="action-btn-small"
              @click="copyGlobalErrorDetails"
            >
              {{ globalErrorCopied ? $t('appError.copied') : $t('appError.copy') }}
            </button>
            <button
              type="button"
              class="action-btn-small"
              @click="closeGlobalErrorDialog"
            >
              {{ $t('common.close') }}
            </button>
          </div>
        </div>
        <div class="mt-4 rounded-2xl border border-border-main bg-bg-main/80 p-4">
          <div class="ui-body font-black text-text-main">
            {{ globalErrorSummary }}
          </div>
          <pre
            v-if="globalErrorExpanded"
            class="mt-3 max-h-[52vh] overflow-auto whitespace-pre-wrap break-words font-mono text-[0.78rem] leading-6 text-text-main"
          >{{ globalErrorDialog.message }}</pre>
        </div>
      </div>
    </div>

    <div
      v-if="tokenRequiredDialog.open"
      class="absolute inset-0 z-[121] flex items-center justify-center p-6"
    >
      <div class="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" @click="closeTokenRequiredDialog" />
      <div class="relative z-10 w-full max-w-md rounded-2xl border border-border-main bg-bg-card/96 p-6 shadow-[0_24px_80px_rgba(15,23,42,0.32)]">
        <div class="ui-metric font-black tracking-tight text-text-main">
          {{ $t('auth.tokens.insufficientTitle') }}
        </div>
        <div class="mt-2 ui-body text-text-secondary">
          {{ $t('auth.tokens.insufficientMessage', tokenRequiredDialog) }}
        </div>
        <div class="mt-5 flex justify-end">
          <button type="button" class="action-btn-small" @click="closeTokenRequiredDialog">
            {{ $t('common.close') }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, defineAsyncComponent, onMounted, onUnmounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';

import { useAppSettingsStore } from './app/useAppSettings';
import { TAB_IDS } from './app/tabRegistry';
import { useTabManager } from './app/useTabManager';
import MainMenuView from './components/MainMenuView.vue';
import AccountSecurityDialog from './features/auth/AccountSecurityDialog.vue';
import AuthPage from './features/auth/AuthPage.vue';
import SponsorDialog from './features/billing/SponsorDialog.vue';
import ReplayAnalysisDialog from './features/replay/components/ReplayAnalysisDialog.vue';
import { useAuthState } from './services/auth/authState';

const GamerView = defineAsyncComponent(() => import('./features/gamer/pages/GamerPage.vue'));
const TrainerView = defineAsyncComponent(() => import('./features/trainer/pages/TrainerPage.vue'));
const TesterView = defineAsyncComponent(() => import('./features/tester/pages/TesterPage.vue'));
const MinigamesView = defineAsyncComponent(() => import('./features/minigames/pages/MinigamesPage.vue'));
const ReplayReviewView = defineAsyncComponent(() => import('./features/replay/pages/ReplayPage.vue'));
const SettingsView = defineAsyncComponent(() => import('./features/settings/pages/SettingsPage.vue'));
const HelpView = defineAsyncComponent(() => import('./features/help/pages/HelpPage.vue'));
const AdminView = defineAsyncComponent(() => import('./features/admin/pages/AdminPage.vue'));

const { t } = useI18n();
const {
  user: authUser,
  dialogOpen: authDialogOpen,
  dialogMode: authDialogMode,
  refreshAuth,
  openAuthDialog,
  closeAuthDialog,
  setAuthenticatedUser,
  logout,
} = useAuthState();
const analysisDialogOpen = ref(false);
const analysisDialogContext = ref({});
const globalErrorDialog = ref({
  open: false,
  title: '',
  message: '',
});
const globalErrorExpanded = ref(false);
const globalErrorQueue = [];
const globalErrorCopied = ref(false);
const accountMenuOpen = ref(false);
const sponsorDialogOpen = ref(false);
const accountSecurityDialog = ref({
  open: false,
  mode: 'changePassword',
});
const tokenRequiredDialog = ref({
  open: false,
  required_tokens: 0,
  balance_tokens: 0,
});
const AUTH_REFRESH_CHECK_KEY = '2048tables:last-auth-refresh-check';
const AUTH_REFRESH_INTERVAL_MS = 6 * 60 * 60 * 1000;
let scheduledAuthRefreshTimer = null;
const { start: startAppSettings, stop: stopAppSettings } = useAppSettingsStore();
const {
  activeTab,
  openTabDefinitions,
  activateTab,
  closeTab,
  isTabOpen,
  moveTabRelative,
  openTab,
} = useTabManager();
const draggedTabId = ref(null);
const dragTargetTabId = ref(null);

const getTabLabel = (tab) => (tab.titleKey ? t(tab.titleKey) : tab.title);
const accountDisplayName = computed(() => authUser.value?.display_name || authUser.value?.email || '');
const canOpenAdmin = computed(() => {
  const email = String(authUser.value?.email || '').trim().toLowerCase();
  const displayName = String(authUser.value?.display_name || '').trim().toLowerCase();
  return email === 'assweeass@163.com' || displayName === 'user0';
});
const hasSupporterPresentation = computed(() => (
  authUser.value?.entitlements?.tier === 'supporter' || canOpenAdmin.value
));
const accountInitials = computed(() => {
  const name = accountDisplayName.value.trim();
  if (!name) return '?';
  return name.slice(0, 2).toUpperCase();
});
const formatTokens = (value) => {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) return '0';
  return number.toLocaleString(undefined, {
    minimumFractionDigits: number % 1 === 0 ? 0 : 1,
    maximumFractionDigits: 3,
  });
};

const readLastScheduledAuthRefresh = () => {
  try {
    return Number.parseInt(window.localStorage.getItem(AUTH_REFRESH_CHECK_KEY) || '0', 10) || 0;
  } catch (error) {
    return 0;
  }
};

const writeLastScheduledAuthRefresh = (value = Date.now()) => {
  try {
    window.localStorage.setItem(AUTH_REFRESH_CHECK_KEY, String(value));
  } catch (error) {
    // localStorage may be unavailable in strict privacy modes; the server still guards grants.
  }
};

const shouldRunScheduledAuthRefresh = (now = Date.now()) => (
  now - readLastScheduledAuthRefresh() >= AUTH_REFRESH_INTERVAL_MS
);

const runScheduledAuthRefresh = async () => {
  if (!authUser.value) {
    return;
  }
  const now = Date.now();
  if (!shouldRunScheduledAuthRefresh(now)) {
    return;
  }
  writeLastScheduledAuthRefresh(now);
  await refreshAuth();
};

const handleAuthVisibilityChange = () => {
  if (document.visibilityState === 'visible') {
    runScheduledAuthRefresh();
  }
};

const openAnalysisDialog = (context = {}) => {
  analysisDialogContext.value = { ...(context || {}) };
  analysisDialogOpen.value = true;
};

const closeAnalysisDialog = () => {
  analysisDialogOpen.value = false;
  analysisDialogContext.value = {};
};

const globalErrorSummary = computed(() => {
  const message = globalErrorDialog.value.message || '';
  const lines = message
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index];
    if (!line.startsWith('Traceback')) {
      return line;
    }
  }

  return lines[0] || '';
});

const readPendingGlobalErrors = () => {
  if (!Array.isArray(window.__appGlobalErrors)) {
    return [];
  }

  const pending = [...window.__appGlobalErrors];
  window.__appGlobalErrors.length = 0;
  return pending;
};

const showNextGlobalError = () => {
  const next = globalErrorQueue.shift();
  if (!next) {
    globalErrorDialog.value = {
      open: false,
      title: '',
      message: '',
    };
    return;
  }

  globalErrorCopied.value = false;
  globalErrorExpanded.value = false;
  globalErrorDialog.value = {
    open: true,
    title: next.title || '',
    message: next.message || '',
  };
};

const enqueueGlobalError = (payload = {}) => {
  const title = typeof payload.title === 'string' ? payload.title : '';
  const message = typeof payload.message === 'string' ? payload.message : '';
  if (!message.trim()) {
    return;
  }

  globalErrorQueue.push({ title, message });
  if (!globalErrorDialog.value.open) {
    showNextGlobalError();
  }
};

const handleGlobalErrorEvent = (event) => {
  enqueueGlobalError(event?.detail || {});
};

const closeGlobalErrorDialog = () => {
  showNextGlobalError();
};

const copyText = async (text) => {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'absolute';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand('copy');
  document.body.removeChild(textarea);
};

const copyGlobalErrorDetails = async () => {
  if (!globalErrorDialog.value.message) {
    return;
  }

  try {
    await copyText(globalErrorDialog.value.message);
    globalErrorCopied.value = true;
  } catch (error) {
    console.error('Failed to copy global error details', error);
  }
};

const BOARD_HOTKEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd', 'W', 'A', 'S', 'D']);

const isTextEntryElement = (element) => {
  if (!(element instanceof HTMLElement)) {
    return false;
  }

  const tagName = element.tagName;
  return tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT' || element.isContentEditable;
};

const findButtonLikeElement = (target) => {
  if (!(target instanceof Element)) {
    return null;
  }

  const match = target.closest('button, [role="button"]');
  return match instanceof HTMLElement ? match : null;
};

const blurButtonTarget = (event) => {
  const target = event?.currentTarget;
  if (target instanceof HTMLElement) {
    target.blur();
  }
};

const handleGlobalPointerUp = (event) => {
  const buttonLike = findButtonLikeElement(event.target);
  if (!buttonLike) {
    return;
  }

  requestAnimationFrame(() => {
    buttonLike.blur();
  });
};

const handleGlobalBoardHotkeyFocus = (event) => {
  if (!BOARD_HOTKEYS.has(event.key)) {
    return;
  }

  const activeElement = document.activeElement;
  if (!(activeElement instanceof HTMLElement) || isTextEntryElement(activeElement)) {
    return;
  }

  const buttonLike = findButtonLikeElement(activeElement);
  if (buttonLike) {
    buttonLike.blur();
  }
};

const handleActivateTab = (tabId, event) => {
  activateTab(tabId);
  blurButtonTarget(event);
};

const handleCloseTab = (tabId, event) => {
  closeTab(tabId);
  blurButtonTarget(event);
};

const clearTabDragState = () => {
  draggedTabId.value = null;
  dragTargetTabId.value = null;
};

const handleTabDragStart = (tabId, event) => {
  if (tabId === TAB_IDS.MAIN_MENU) {
    event.preventDefault();
    return;
  }

  draggedTabId.value = tabId;
  dragTargetTabId.value = tabId;
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('text/plain', tabId);
  }
};

const handleTabDragOver = (tabId, event) => {
  if (!draggedTabId.value || draggedTabId.value === tabId) {
    return;
  }

  event.preventDefault();
  dragTargetTabId.value = tabId;

  const currentTarget = event.currentTarget;
  const isMainMenu = tabId === TAB_IDS.MAIN_MENU;
  let placeAfter = true;

  if (!isMainMenu && currentTarget instanceof HTMLElement) {
    const bounds = currentTarget.getBoundingClientRect();
    placeAfter = event.clientX >= bounds.left + bounds.width / 2;
  }

  moveTabRelative(draggedTabId.value, tabId, placeAfter);

  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = 'move';
  }
};

const handleTabDrop = (_tabId, event) => {
  if (!draggedTabId.value) {
    return;
  }
  event.preventDefault();
  clearTabDragState();
};

const handleTabDragEnd = () => {
  clearTabDragState();
};

const handleAuthenticated = (authenticatedUser) => {
  writeLastScheduledAuthRefresh();
  setAuthenticatedUser(authenticatedUser);
  closeAuthDialog();
};

const handleLogout = async () => {
  try {
    await logout();
    accountMenuOpen.value = false;
  } catch (error) {
    console.error('Failed to log out', error);
  }
};

const openAdminPage = () => {
  accountMenuOpen.value = false;
  openTab(TAB_IDS.ADMIN);
};

const openSponsorDialog = () => {
  accountMenuOpen.value = false;
  sponsorDialogOpen.value = true;
};

const closeSponsorDialog = () => {
  sponsorDialogOpen.value = false;
};

const openAccountSecurity = (mode) => {
  accountMenuOpen.value = false;
  accountSecurityDialog.value = {
    open: true,
    mode: mode === 'deactivate' ? 'deactivate' : 'changePassword',
  };
};

const closeAccountSecurity = () => {
  accountSecurityDialog.value = {
    open: false,
    mode: 'changePassword',
  };
};

const handleAccountSecuritySuccess = async () => {
  await refreshAuth();
  closeAccountSecurity();
};

const handleAccountDeactivated = async () => {
  await refreshAuth();
  closeAccountSecurity();
};

const handleAuthRequired = () => {
  openAuthDialog('login');
};

const handleTokenRequired = (event) => {
  const detail = event?.detail || {};
  refreshAuth();
  tokenRequiredDialog.value = {
    open: true,
    required_tokens: Number(detail.required_tokens || 0),
    balance_tokens: Number(detail.balance_tokens || 0),
  };
};

const closeTokenRequiredDialog = () => {
  tokenRequiredDialog.value = {
    open: false,
    required_tokens: 0,
    balance_tokens: 0,
  };
};

const handleAccountMenuPointerDown = (event) => {
  if (!accountMenuOpen.value) {
    return;
  }
  const target = event.target;
  if (target instanceof Element && target.closest('[data-account-menu]')) {
    return;
  }
  accountMenuOpen.value = false;
};

onMounted(async () => {
  startAppSettings();
  refreshAuth().then((nextUser) => {
    if (nextUser) {
      writeLastScheduledAuthRefresh();
    }
  });
  scheduledAuthRefreshTimer = window.setInterval(runScheduledAuthRefresh, AUTH_REFRESH_INTERVAL_MS);

  for (const payload of readPendingGlobalErrors()) {
    enqueueGlobalError(payload);
  }
  window.addEventListener('app-global-error', handleGlobalErrorEvent);
  window.addEventListener('auth-required', handleAuthRequired);
  window.addEventListener('token-required', handleTokenRequired);
  document.addEventListener('visibilitychange', handleAuthVisibilityChange);
  document.addEventListener('pointerdown', handleAccountMenuPointerDown, true);
  document.addEventListener('pointerup', handleGlobalPointerUp, true);
  document.addEventListener('keydown', handleGlobalBoardHotkeyFocus, true);
});

onUnmounted(() => {
  window.removeEventListener('app-global-error', handleGlobalErrorEvent);
  window.removeEventListener('auth-required', handleAuthRequired);
  window.removeEventListener('token-required', handleTokenRequired);
  document.removeEventListener('visibilitychange', handleAuthVisibilityChange);
  document.removeEventListener('pointerdown', handleAccountMenuPointerDown, true);
  document.removeEventListener('pointerup', handleGlobalPointerUp, true);
  document.removeEventListener('keydown', handleGlobalBoardHotkeyFocus, true);
  if (scheduledAuthRefreshTimer !== null) {
    window.clearInterval(scheduledAuthRefreshTimer);
    scheduledAuthRefreshTimer = null;
  }
  stopAppSettings();
});
</script>

<style scoped>
.app-shell {
  background-color: var(--bg-main);
  background-image: var(--bg-main-gradient);
}

.account-trigger.supporter {
  border-color: color-mix(in srgb, var(--accent) 58%, var(--border-main));
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 28%, transparent);
}

.account-avatar {
  display: inline-flex;
  position: relative;
  width: 1.5rem;
  height: 1.5rem;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  border: 1px solid color-mix(in srgb, var(--accent) 26%, transparent);
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 18%, transparent);
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 950;
  line-height: 1;
}

.account-avatar.large {
  width: 2.25rem;
  height: 2.25rem;
  font-size: 0.82rem;
}

.account-avatar.supporter {
  border-color: color-mix(in srgb, var(--accent) 70%, var(--border-main));
  background:
    linear-gradient(135deg,
      color-mix(in srgb, var(--accent) 26%, transparent),
      color-mix(in srgb, var(--success) 16%, transparent));
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 14%, transparent);
}

.account-avatar.supporter::after {
  content: "";
  position: absolute;
  right: -0.12rem;
  bottom: -0.12rem;
  width: 0.58rem;
  height: 0.58rem;
  border: 1px solid color-mix(in srgb, var(--bg-card) 92%, white);
  border-radius: 999px;
  background:
    radial-gradient(circle at 50% 50%, var(--bg-card) 0 16%, transparent 18%),
    conic-gradient(from 45deg,
      var(--accent) 0 25%,
      color-mix(in srgb, var(--success) 76%, var(--accent)) 25% 50%,
      var(--accent) 50% 75%,
      color-mix(in srgb, var(--success) 76%, var(--accent)) 75% 100%);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 28%, transparent);
}

.account-avatar.large.supporter::after {
  right: -0.08rem;
  bottom: -0.08rem;
  width: 0.68rem;
  height: 0.68rem;
}

.account-menu-head {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}
</style>
