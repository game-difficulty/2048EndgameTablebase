<template>
  <div
    ref="fixedViewport"
    :class="[
      'fixed-layout-viewport',
      trainerBottomDockActive ? 'fixed-layout-viewport--dock-bottom' : '',
    ]"
  >
    <div class="fixed-layout-frame" :style="fixedLayoutFrameStyle">
      <div
        class="app-shell flex flex-col overflow-hidden"
        :style="fixedLayoutSurfaceStyle"
        @pointerdown.capture="handleWorkspaceFocus"
        @focusin.capture="handleWorkspaceFocus"
      >
    <div ref="appTopBar" class="flex items-center gap-2 overflow-x-auto bg-bg-main/80 p-2 shadow-sm z-50 border-b border-border-main backdrop-blur-md transition-colors duration-300">
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
          isTabPresented(tab.id)
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
            <AccountAvatar :user="authUser" :supporter="hasSupporterPresentation" />
            <span class="inline max-w-[9rem] truncate">{{ accountDisplayName }}</span>
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
      class="fixed right-3 top-[3.75rem] z-[300] w-[18rem] rounded-2xl border border-border-main bg-bg-card/98 p-4 text-text-main shadow-[0_20px_70px_rgba(15,23,42,0.35)]"
      data-account-menu
    >
      <div class="account-menu-head">
        <AccountAvatar
          :user="authUser"
          :supporter="hasSupporterPresentation"
          size="large"
          editable
          @edit="openAvatarEditor"
        />
        <div class="min-w-0">
          <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('auth.account.title') }}</div>
          <button
            type="button"
            class="account-name-button mt-1"
            :title="$t('profile.displayName.change')"
            @click="openDisplayNameEditor"
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
        @click="openQuotaGuideDialog"
      >
        {{ $t('billing.quotaGuide.open') }}
      </button>
      <button
        type="button"
        class="action-btn-small mt-2 w-full justify-center"
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

    <div :class="['app-workspace', trainerWorkspaceClass]">
      <div class="app-primary-pane" data-primary-pane>
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
          <GamerView
            :active="activeTab === TAB_IDS.GAMER"
            @navigate-tab="handleNavigateTab"
          />
        </div>
        <div
          v-if="isTabOpen(TAB_IDS.TESTER)"
          class="absolute inset-0"
          v-show="activeTab === TAB_IDS.TESTER"
        >
          <TesterView
            :active="activeTab === TAB_IDS.TESTER"
            @navigate-tab="handleNavigateTab"
            @open-analysis="openAnalysisDialog"
          />
        </div>
        <div
          v-if="isTabOpen(TAB_IDS.MINIGAMES)"
          class="absolute inset-0"
          v-show="activeTab === TAB_IDS.MINIGAMES"
        >
          <MinigamesView
            :active="activeTab === TAB_IDS.MINIGAMES"
            @navigate-tab="handleNavigateTab"
          />
        </div>
        <div
          v-if="isTabOpen(TAB_IDS.LEADERBOARDS)"
          class="absolute inset-0"
          v-show="activeTab === TAB_IDS.LEADERBOARDS"
        >
          <LeaderboardsView
            :active="activeTab === TAB_IDS.LEADERBOARDS"
            :requested-key="leaderboardRequestedKey"
            :request-serial="leaderboardRequestSerial"
            :requested-game-id="leaderboardRequestedGameId"
            :requested-difficulty="leaderboardRequestedDifficulty"
          />
        </div>
        <div
          v-if="isTabOpen(TAB_IDS.REPLAY)"
          class="absolute inset-0"
          v-show="activeTab === TAB_IDS.REPLAY"
        >
          <ReplayReviewView
            :active="activeTab === TAB_IDS.REPLAY"
            @navigate-tab="handleNavigateTab"
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
          <HelpView
            :active="activeTab === TAB_IDS.HELP"
            @navigate-tab="handleNavigateTab"
          />
        </div>
        <div
          v-if="isTabOpen(TAB_IDS.ADMIN)"
          class="absolute inset-0"
          v-show="activeTab === TAB_IDS.ADMIN"
        >
          <AdminView :active="activeTab === TAB_IDS.ADMIN" />
        </div>
      </div>

      <div
        v-if="isTabOpen(TAB_IDS.TRAINER)"
        v-show="trainerVisible"
        :class="['app-trainer-pane', trainerPaneClass]"
        data-trainer-pane
      >
        <TrainerView
          :active="trainerSessionActive"
          :hotkeys-enabled="trainerHotkeysEnabled"
          :dock-placement="trainerEffectiveDockPlacement"
        />
        <div
          v-if="trainerDockAvailable"
          class="trainer-dock-toolbar"
          role="toolbar"
          :aria-label="$t('trainer.dock.toolbar')"
        >
          <button
            type="button"
            class="trainer-dock-button"
            :class="{ 'trainer-dock-button--active': trainerEffectiveDockPlacement === TRAINER_DOCK_PLACEMENTS.RIGHT }"
            :title="$t('trainer.dock.right')"
            :aria-label="$t('trainer.dock.right')"
            :aria-pressed="trainerEffectiveDockPlacement === TRAINER_DOCK_PLACEMENTS.RIGHT"
            data-trainer-dock="right"
            @click="handleTrainerDockChange(TRAINER_DOCK_PLACEMENTS.RIGHT)"
          >
            <span aria-hidden="true">◧</span>
          </button>
          <button
            type="button"
            class="trainer-dock-button"
            :class="{ 'trainer-dock-button--active': trainerEffectiveDockPlacement === TRAINER_DOCK_PLACEMENTS.BOTTOM }"
            :title="$t('trainer.dock.bottom')"
            :aria-label="$t('trainer.dock.bottom')"
            :aria-pressed="trainerEffectiveDockPlacement === TRAINER_DOCK_PLACEMENTS.BOTTOM"
            data-trainer-dock="bottom"
            @click="handleTrainerDockChange(TRAINER_DOCK_PLACEMENTS.BOTTOM)"
          >
            <span aria-hidden="true">⬒</span>
          </button>
          <button
            v-if="trainerDockActive"
            type="button"
            class="trainer-dock-button"
            :title="$t('trainer.dock.full')"
            :aria-label="$t('trainer.dock.full')"
            data-trainer-dock="none"
            @click="handleTrainerDockChange(TRAINER_DOCK_PLACEMENTS.NONE)"
          >
            <span aria-hidden="true">□</span>
          </button>
        </div>
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

    <AvatarEditorDialog
      :open="avatarEditorOpen"
      :user="authUser"
      :supporter="hasSupporterPresentation"
      @close="avatarEditorOpen = false"
      @saved="handleProfileSaved"
    />

    <DisplayNameEditorDialog
      :open="displayNameEditorOpen"
      :user="authUser"
      @close="displayNameEditorOpen = false"
      @saved="handleProfileSaved"
    />

    <SponsorDialog
      :open="sponsorDialogOpen"
      :user="authUser"
      @close="closeSponsorDialog"
    />

    <QuotaGuideDialog
      :open="quotaGuideDialogOpen"
      @close="closeQuotaGuideDialog"
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
        <div class="mt-3 ui-body text-text-secondary">
          {{ $t('auth.tokens.rechargeHint') }}
        </div>
        <div class="mt-5 flex flex-wrap justify-end gap-2">
          <button type="button" class="action-btn-small surface-prominent text-white" @click="openSponsorFromTokenRequired">
            {{ $t('auth.tokens.rechargeAction') }}
          </button>
          <button type="button" class="action-btn-small" @click="closeTokenRequiredDialog">
            {{ $t('common.close') }}
          </button>
        </div>
      </div>
    </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, defineAsyncComponent, nextTick, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { useAppSettingsStore } from './app/useAppSettings';
import { TAB_IDS } from './app/tabRegistry';
import {
  TRAINER_DOCK_LAYOUT,
  TRAINER_DOCK_PLACEMENTS,
  isTrainerDocked,
  normalizeTrainerDockPlacement,
  resolveTrainerJumpDockPlacement,
  resolveTrainerDockSurfaceHeight,
} from './app/trainerDock';
import { useTabManager } from './app/useTabManager';
import MainMenuView from './components/MainMenuView.vue';
import AccountAvatar from './features/auth/AccountAvatar.vue';
import AccountSecurityDialog from './features/auth/AccountSecurityDialog.vue';
import AuthPage from './features/auth/AuthPage.vue';
import SponsorDialog from './features/billing/SponsorDialog.vue';
import ReplayAnalysisDialog from './features/replay/components/ReplayAnalysisDialog.vue';
import { queueTrainerPracticeJump } from './features/trainer/services/trainerPracticeJump';
import { useAuthState } from './services/auth/authState';

const GamerView = defineAsyncComponent(() => import('./features/gamer/pages/GamerPage.vue'));
const TrainerView = defineAsyncComponent(() => import('./features/trainer/pages/TrainerPage.vue'));
const TesterView = defineAsyncComponent(() => import('./features/tester/pages/TesterPage.vue'));
const MinigamesView = defineAsyncComponent(() => import('./features/minigames/pages/MinigamesPage.vue'));
const LeaderboardsView = defineAsyncComponent(() => import('./features/leaderboards/pages/LeaderboardPage.vue'));
const ReplayReviewView = defineAsyncComponent(() => import('./features/replay/pages/ReplayPage.vue'));
const SettingsView = defineAsyncComponent(() => import('./features/settings/pages/SettingsPage.vue'));
const HelpView = defineAsyncComponent(() => import('./features/help/pages/HelpPage.vue'));
const AdminView = defineAsyncComponent(() => import('./features/admin/pages/AdminPage.vue'));
const QuotaGuideDialog = defineAsyncComponent(() => import('./features/billing/QuotaGuideDialog.vue'));
const AvatarEditorDialog = defineAsyncComponent(() => import('./features/auth/AvatarEditorDialog.vue'));
const DisplayNameEditorDialog = defineAsyncComponent(() => import('./features/auth/DisplayNameEditorDialog.vue'));

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
const avatarEditorOpen = ref(false);
const displayNameEditorOpen = ref(false);
const sponsorDialogOpen = ref(false);
const quotaGuideDialogOpen = ref(false);
const accountSecurityDialog = ref({
  open: false,
  mode: 'changePassword',
});
const tokenRequiredDialog = ref({
  open: false,
  required_tokens: 0,
  balance_tokens: 0,
});
const FIXED_LAYOUT_MIN_WIDTH = 1280;
const FIXED_LAYOUT_MAX_WIDTH = 1600;
const FIXED_LAYOUT_HEIGHT = 800;
const fixedViewport = ref(null);
const appTopBar = ref(null);
const appTopBarHeight = ref(0);
const fixedLayoutWidth = ref(FIXED_LAYOUT_MIN_WIDTH);
const leaderboardRequestedKey = ref('');
const leaderboardRequestedGameId = ref('');
const leaderboardRequestedDifficulty = ref(1);
const leaderboardRequestSerial = ref(0);
const fixedLayoutScale = ref(1);
const viewportWidth = ref(0);
const viewportHeight = ref(0);
let fixedViewportObserver = null;
const AUTH_REFRESH_CHECK_KEY = '2048tables:last-auth-refresh-check';
const AUTH_REFRESH_INTERVAL_MS = 6 * 60 * 60 * 1000;
const TRAINER_DOCK_PREFERENCE_KEY = '2048tables:trainer-dock-placement';
let scheduledAuthRefreshTimer = null;
const { start: startAppSettings, stop: stopAppSettings } = useAppSettingsStore();
const {
  activeTab,
  openTabs,
  openTabDefinitions,
  activateTab,
  closeTab,
  isTabOpen,
  moveTabRelative,
  openTab,
  openTabInBackground,
} = useTabManager();
const draggedTabId = ref(null);
const dragTargetTabId = ref(null);
const trainerDockPlacement = ref(TRAINER_DOCK_PLACEMENTS.NONE);
const trainerDockFocused = ref(false);
const lastPrimaryTab = ref(TAB_IDS.MAIN_MENU);

const readTrainerDockPreference = () => {
  try {
    const placement = normalizeTrainerDockPlacement(
      window.localStorage.getItem(TRAINER_DOCK_PREFERENCE_KEY),
    );
    return isTrainerDocked(placement) ? placement : TRAINER_DOCK_PLACEMENTS.RIGHT;
  } catch (error) {
    return TRAINER_DOCK_PLACEMENTS.RIGHT;
  }
};

const trainerDockPreference = ref(readTrainerDockPreference());
const trainerDockAvailable = computed(() => viewportWidth.value >= 900 && viewportHeight.value >= 600);
const trainerDockActive = computed(() => (
  trainerDockAvailable.value
  && isTrainerDocked(trainerDockPlacement.value)
  && isTabOpen(TAB_IDS.TRAINER)
));
const trainerBottomDockActive = computed(() => (
  trainerDockActive.value
  && trainerDockPlacement.value === TRAINER_DOCK_PLACEMENTS.BOTTOM
));
const trainerEffectiveDockPlacement = computed(() => (
  trainerDockActive.value
    ? trainerDockPlacement.value
    : TRAINER_DOCK_PLACEMENTS.NONE
));
const trainerSessionActive = computed(() => (
  activeTab.value === TAB_IDS.TRAINER || trainerDockActive.value
));
const trainerVisible = computed(() => trainerSessionActive.value);
const trainerHotkeysEnabled = computed(() => (
  activeTab.value === TAB_IDS.TRAINER
  || (trainerDockActive.value && trainerDockFocused.value)
));
const trainerWorkspaceClass = computed(() => (
  trainerDockActive.value ? `app-workspace--dock-${trainerDockPlacement.value}` : ''
));
const trainerPaneClass = computed(() => (
  trainerDockActive.value ? `app-trainer-pane--dock-${trainerDockPlacement.value}` : 'app-trainer-pane--full'
));

const fixedLayoutSurfaceHeight = computed(() => resolveTrainerDockSurfaceHeight({
  placement: trainerEffectiveDockPlacement.value,
  baseHeight: FIXED_LAYOUT_HEIGHT,
  topBarHeight: appTopBarHeight.value,
}));

const fixedLayoutFrameStyle = computed(() => ({
  width: `${fixedLayoutWidth.value * fixedLayoutScale.value}px`,
  height: `${fixedLayoutSurfaceHeight.value * fixedLayoutScale.value}px`,
}));

const fixedLayoutSurfaceStyle = computed(() => ({
  width: `${fixedLayoutWidth.value}px`,
  height: `${fixedLayoutSurfaceHeight.value}px`,
  transform: `scale(${fixedLayoutScale.value})`,
  '--trainer-dock-right-width': `${TRAINER_DOCK_LAYOUT.RIGHT_WIDTH_PX}px`,
  '--trainer-dock-right-board-width': `${TRAINER_DOCK_LAYOUT.RIGHT_BOARD_WIDTH_PX}px`,
}));

const updateFixedLayoutScale = () => {
  const viewport = fixedViewport.value;
  if (!viewport) return;
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  if (width <= 0 || height <= 0) return;
  viewportWidth.value = width;
  viewportHeight.value = height;
  appTopBarHeight.value = appTopBar.value?.offsetHeight || 0;
  const nextScale = Math.min(
    width / FIXED_LAYOUT_MIN_WIDTH,
    height / FIXED_LAYOUT_HEIGHT,
  );
  fixedLayoutScale.value = nextScale;
  fixedLayoutWidth.value = Math.min(
    FIXED_LAYOUT_MAX_WIDTH,
    Math.max(FIXED_LAYOUT_MIN_WIDTH, width / nextScale),
  );
};

const getTabLabel = (tab) => (tab.titleKey ? t(tab.titleKey) : tab.title);
const isTabPresented = (tabId) => (
  activeTab.value === tabId || (tabId === TAB_IDS.TRAINER && trainerDockActive.value)
);
const accountDisplayName = computed(() => authUser.value?.display_name || authUser.value?.email || '');
const canOpenAdmin = computed(() => {
  const email = String(authUser.value?.email || '').trim().toLowerCase();
  const displayName = String(authUser.value?.display_name || '').trim().toLowerCase();
  return email === 'assweeass@163.com' || displayName === 'user0';
});
const hasSupporterPresentation = computed(() => (
  authUser.value?.entitlements?.tier === 'supporter' || canOpenAdmin.value
));
const formatTokens = (value) => {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) return '0';
  return number.toLocaleString(undefined, {
    minimumFractionDigits: number % 1 === 0 ? 0 : 1,
    maximumFractionDigits: 3,
  });
};

const handleNavigateTab = (tabId, detail = null) => {
  if (tabId === TAB_IDS.TRAINER && detail?.hex) {
    queueTrainerPracticeJump(detail);
    const requestedDockPlacement = resolveTrainerJumpDockPlacement({
      placement: trainerDockActive.value
        ? trainerDockPlacement.value
        : activeTab.value === TAB_IDS.HELP
          ? trainerDockPreference.value
          : TRAINER_DOCK_PLACEMENTS.NONE,
      dockAvailable: trainerDockAvailable.value,
      sourceIsHelp: activeTab.value === TAB_IDS.HELP,
    });
    if (isTrainerDocked(requestedDockPlacement)) {
      openTabInBackground(tabId);
      trainerDockPlacement.value = requestedDockPlacement;
      trainerDockFocused.value = false;
    } else {
      trainerDockPlacement.value = TRAINER_DOCK_PLACEMENTS.NONE;
      openTab(tabId);
    }
    return;
  }

  openTab(tabId);
  if (tabId === TAB_IDS.LEADERBOARDS && detail?.boardKey) {
    leaderboardRequestedKey.value = String(detail.boardKey);
    leaderboardRequestedGameId.value = String(detail.gameId || '');
    leaderboardRequestedDifficulty.value = Number(detail.difficulty) ? 1 : 0;
    leaderboardRequestSerial.value += 1;
    return;
  }
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
  if (tabId === TAB_IDS.TRAINER && isTrainerDocked(trainerDockPlacement.value)) {
    trainerDockPlacement.value = TRAINER_DOCK_PLACEMENTS.NONE;
  }
  activateTab(tabId);
  blurButtonTarget(event);
};

const handleCloseTab = (tabId, event) => {
  if (tabId === TAB_IDS.TRAINER) {
    trainerDockPlacement.value = TRAINER_DOCK_PLACEMENTS.NONE;
    trainerDockFocused.value = false;
  }
  closeTab(tabId);
  if (trainerDockActive.value && activeTab.value === TAB_IDS.TRAINER) {
    activateTab(getTrainerCompanionTab());
  }
  blurButtonTarget(event);
};

const writeTrainerDockPreference = (placement) => {
  try {
    window.localStorage.setItem(TRAINER_DOCK_PREFERENCE_KEY, placement);
  } catch (error) {
    // A stored preference is optional; docking remains available for this session.
  }
};

const getTrainerCompanionTab = () => {
  if (lastPrimaryTab.value !== TAB_IDS.TRAINER && isTabOpen(lastPrimaryTab.value)) {
    return lastPrimaryTab.value;
  }
  return openTabs.value.find((tabId) => tabId !== TAB_IDS.TRAINER) || TAB_IDS.MAIN_MENU;
};

const handleTrainerDockChange = (placement) => {
  const normalized = normalizeTrainerDockPlacement(placement);
  if (!isTrainerDocked(normalized)) {
    trainerDockPlacement.value = TRAINER_DOCK_PLACEMENTS.NONE;
    activateTab(TAB_IDS.TRAINER);
    trainerDockFocused.value = true;
    return;
  }
  if (!trainerDockAvailable.value) {
    return;
  }
  openTabInBackground(TAB_IDS.TRAINER);
  trainerDockPlacement.value = normalized;
  trainerDockPreference.value = normalized;
  writeTrainerDockPreference(normalized);
  if (activeTab.value === TAB_IDS.TRAINER) {
    activateTab(getTrainerCompanionTab());
  }
  trainerDockFocused.value = true;
};

const handleWorkspaceFocus = (event) => {
  if (!trainerDockActive.value) {
    return;
  }
  const target = event?.target;
  trainerDockFocused.value = target instanceof Element && !!target.closest('[data-trainer-pane]');
};

watch(activeTab, (tabId) => {
  if (tabId !== TAB_IDS.TRAINER) {
    lastPrimaryTab.value = tabId;
  }
});

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

const openQuotaGuideDialog = () => {
  accountMenuOpen.value = false;
  quotaGuideDialogOpen.value = true;
};

const closeQuotaGuideDialog = () => {
  quotaGuideDialogOpen.value = false;
};

const openAccountSecurity = (mode) => {
  accountMenuOpen.value = false;
  accountSecurityDialog.value = {
    open: true,
    mode: mode === 'deactivate' ? 'deactivate' : 'changePassword',
  };
};

const openAvatarEditor = () => {
  accountMenuOpen.value = false;
  avatarEditorOpen.value = true;
};

const openDisplayNameEditor = () => {
  accountMenuOpen.value = false;
  displayNameEditorOpen.value = true;
};

const handleProfileSaved = (updatedUser) => {
  if (updatedUser) {
    setAuthenticatedUser(updatedUser);
  }
  avatarEditorOpen.value = false;
  displayNameEditorOpen.value = false;
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

const openSponsorFromTokenRequired = () => {
  closeTokenRequiredDialog();
  if (!authUser.value) {
    openAuthDialog('login');
    return;
  }
  openSponsorDialog();
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
  updateFixedLayoutScale();
  if (typeof ResizeObserver === 'function') {
    fixedViewportObserver = new ResizeObserver(updateFixedLayoutScale);
    fixedViewportObserver.observe(fixedViewport.value);
    fixedViewportObserver.observe(appTopBar.value);
  } else {
    window.addEventListener('resize', updateFixedLayoutScale);
  }
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
  if (fixedViewportObserver) {
    fixedViewportObserver.disconnect();
    fixedViewportObserver = null;
  } else {
    window.removeEventListener('resize', updateFixedLayoutScale);
  }
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
.fixed-layout-viewport {
  position: fixed;
  inset:
    env(safe-area-inset-top, 0px)
    env(safe-area-inset-right, 0px)
    env(safe-area-inset-bottom, 0px)
    env(safe-area-inset-left, 0px);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background-color: var(--bg-main);
  background-image: var(--bg-main-gradient);
}

.fixed-layout-viewport--dock-bottom {
  align-items: flex-start;
  overflow-x: hidden;
  overflow-y: auto;
}

.fixed-layout-frame {
  position: relative;
  flex: 0 0 auto;
  overflow: visible;
}

.app-shell {
  transform-origin: top left;
  background-color: var(--bg-main);
  background-image: var(--bg-main-gradient);
}

.app-workspace {
  position: relative;
  display: grid;
  flex: 1 1 auto;
  min-width: 0;
  min-height: 0;
  grid-template-columns: minmax(0, 1fr);
  grid-template-rows: minmax(0, 1fr);
  overflow: hidden;
}

.app-primary-pane,
.app-trainer-pane {
  position: relative;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

.app-trainer-pane {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
}

.app-primary-pane,
.app-trainer-pane--full {
  grid-column: 1;
  grid-row: 1;
}

.app-workspace--dock-right {
  grid-template-columns: minmax(0, 1fr) var(--trainer-dock-right-width, 450px);
}

.app-trainer-pane--dock-right {
  grid-column: 2;
  grid-row: 1;
  border-left: 1px solid var(--border-main);
}

.app-workspace--dock-bottom {
  grid-template-rows: repeat(2, minmax(0, 1fr));
}

.app-trainer-pane--dock-bottom {
  grid-column: 1;
  grid-row: 2;
  border-top: 1px solid var(--border-main);
}

.trainer-dock-toolbar {
  z-index: 190;
  display: flex;
  grid-row: 2;
  justify-self: end;
  gap: 0.25rem;
  margin: 0.4rem 0.75rem 0.6rem;
  padding: 0.25rem;
  border: 1px solid var(--border-main);
  border-radius: 0.5rem;
  background: color-mix(in srgb, var(--bg-card) 94%, transparent);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
  backdrop-filter: blur(8px);
}

.trainer-dock-button {
  display: inline-flex;
  width: 2rem;
  height: 2rem;
  align-items: center;
  justify-content: center;
  border-radius: 0.35rem;
  color: var(--text-secondary);
  font-size: 1.1rem;
  font-weight: 900;
  line-height: 1;
  transition: background-color 150ms ease, color 150ms ease;
}

.trainer-dock-button:hover,
.trainer-dock-button:focus-visible,
.trainer-dock-button--active {
  background: var(--btn-bg);
  color: white;
  outline: none;
}

.app-trainer-pane--dock-right .trainer-dock-toolbar {
  gap: 0.2rem;
  margin: 0.2rem 0.55rem 0.4rem;
  padding: 0.15rem;
}

.app-trainer-pane--dock-right .trainer-dock-button {
  width: 1.7rem;
  height: 1.7rem;
  font-size: 1rem;
}

.account-trigger.supporter {
  border-color: color-mix(in srgb, var(--accent) 58%, var(--border-main));
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 28%, transparent);
}

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
