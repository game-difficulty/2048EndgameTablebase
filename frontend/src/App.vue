<template>
  <div
    class="app-shell h-screen w-screen flex flex-col overflow-hidden"
    @pointerdown.capture="handleWorkspaceFocus"
    @focusin.capture="handleWorkspaceFocus"
  >
    <div class="flex items-center gap-2 overflow-x-auto overflow-y-hidden bg-bg-main/80 p-2 shadow-sm z-50 border-b border-border-main backdrop-blur-md transition-colors duration-300">
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
    </div>

    <div :class="['app-workspace', trainerWorkspaceClass]">
      <main class="app-primary-pane" data-primary-pane>
        <div v-if="isTabOpen(TAB_IDS.MAIN_MENU)" class="absolute inset-0" v-show="activeTab === TAB_IDS.MAIN_MENU">
          <MainMenuView :active="activeTab === TAB_IDS.MAIN_MENU" @selectTab="handleNavigateTab" />
        </div>
        <div v-if="isTabOpen(TAB_IDS.GAMER)" class="absolute inset-0" v-show="activeTab === TAB_IDS.GAMER">
          <GamerView :active="activeTab === TAB_IDS.GAMER" />
        </div>
        <div v-if="isTabOpen(TAB_IDS.TESTER)" class="absolute inset-0" v-show="activeTab === TAB_IDS.TESTER">
          <TesterView
            :active="activeTab === TAB_IDS.TESTER"
            @navigate-tab="handleNavigateTab"
            @open-analysis="openAnalysisDialog"
          />
        </div>
        <div v-if="isTabOpen(TAB_IDS.MINIGAMES)" class="absolute inset-0" v-show="activeTab === TAB_IDS.MINIGAMES">
          <MinigamesView :active="activeTab === TAB_IDS.MINIGAMES" />
        </div>
        <div v-if="isTabOpen(TAB_IDS.REPLAY)" class="absolute inset-0" v-show="activeTab === TAB_IDS.REPLAY">
          <ReplayReviewView
            :active="activeTab === TAB_IDS.REPLAY"
            @navigate-tab="handleNavigateTab"
            @open-analysis="openAnalysisDialog"
          />
        </div>
        <div v-if="isTabOpen(TAB_IDS.NOTEBOOK)" class="absolute inset-0" v-show="activeTab === TAB_IDS.NOTEBOOK">
          <NotebookView :active="activeTab === TAB_IDS.NOTEBOOK" @navigate-tab="handleNavigateTab" />
        </div>
        <div v-if="isTabOpen(TAB_IDS.SETTINGS)" class="absolute inset-0" v-show="activeTab === TAB_IDS.SETTINGS">
          <SettingsView :active="activeTab === TAB_IDS.SETTINGS" />
        </div>
        <div v-if="isTabOpen(TAB_IDS.HELP)" class="absolute inset-0" v-show="activeTab === TAB_IDS.HELP">
          <HelpView :active="activeTab === TAB_IDS.HELP" @navigate-tab="handleNavigateTab" />
        </div>
      </main>

      <section
        v-if="isTabOpen(TAB_IDS.TRAINER)"
        v-show="trainerVisible"
        :class="['app-trainer-pane', trainerPaneActive ? 'app-pane--keyboard-active' : '']"
        data-trainer-pane
      >
        <div v-if="trainerDocked" class="trainer-dock-toolbar" role="toolbar" :aria-label="$t('trainer.dock.title')">
          <button
            type="button"
            :class="['trainer-dock-button', trainerDockPlacement === TRAINER_DOCK_PLACEMENTS.RIGHT ? 'is-active' : '']"
            :title="$t('trainer.dock.right')"
            @click="setTrainerDockPlacement(TRAINER_DOCK_PLACEMENTS.RIGHT)"
          >
            {{ $t('trainer.dock.right') }}
          </button>
          <button
            type="button"
            :class="['trainer-dock-button', trainerDockPlacement === TRAINER_DOCK_PLACEMENTS.BOTTOM ? 'is-active' : '']"
            :title="$t('trainer.dock.bottom')"
            @click="setTrainerDockPlacement(TRAINER_DOCK_PLACEMENTS.BOTTOM)"
          >
            {{ $t('trainer.dock.bottom') }}
          </button>
          <button type="button" class="trainer-dock-button" :title="$t('trainer.dock.full')" @click="showTrainerFullPage">
            {{ $t('trainer.dock.full') }}
          </button>
        </div>
        <TrainerView
          :active="trainerSessionActive"
          :hotkeys-enabled="trainerHotkeysEnabled"
          :dock-placement="trainerDockPlacement"
        />
      </section>
    </div>

    <ReplayAnalysisDialog
      :open="analysisDialogOpen"
      :context="analysisDialogContext"
      @close="closeAnalysisDialog"
    />

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
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { useAppSettingsStore } from './app/useAppSettings';
import {
  KEYBOARD_OWNERS,
  keyboardOwner,
  resetKeyboardOwnership,
  setKeyboardOwner,
  setSplitKeyboardMode,
} from './app/keyboardOwnership';
import { TAB_IDS } from './app/tabRegistry';
import {
  TRAINER_DOCK_PLACEMENTS,
  isTrainerDocked,
  normalizeTrainerDockPlacement,
  resolveTrainerJumpDockPlacement,
  trainerDockAvailable,
} from './app/trainerDock';
import { useTabManager } from './app/useTabManager';
import MainMenuView from './components/MainMenuView.vue';
import GamerView from './features/gamer/pages/GamerPage.vue';
import HelpView from './features/help/pages/HelpPage.vue';
import MinigamesView from './features/minigames/pages/MinigamesPage.vue';
import NotebookView from './features/notebook/pages/NotebookPage.vue';
import ReplayAnalysisDialog from './features/replay/components/ReplayAnalysisDialog.vue';
import ReplayReviewView from './features/replay/pages/ReplayPage.vue';
import SettingsView from './features/settings/pages/SettingsPage.vue';
import TesterView from './features/tester/pages/TesterPage.vue';
import TrainerView from './features/trainer/pages/TrainerPage.vue';
import { queueTrainerPracticeJump } from './features/trainer/services/trainerPracticeJump';

const { t } = useI18n();
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
const { start: startAppSettings, stop: stopAppSettings } = useAppSettingsStore();
const {
  activeTab,
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
const viewportWidth = ref(window.innerWidth);
const viewportHeight = ref(window.innerHeight);
const lastPrimaryTab = ref(TAB_IDS.MAIN_MENU);
const trainerDockPreference = ref(normalizeTrainerDockPlacement(
  window.localStorage.getItem('2048tables:trainer-dock-placement'),
));

const dockAvailable = computed(() => trainerDockAvailable(
  viewportWidth.value,
  viewportHeight.value,
));
const trainerDockPlacement = computed(() => (
  dockAvailable.value
    ? trainerDockPreference.value
    : TRAINER_DOCK_PLACEMENTS.NONE
));
const trainerDocked = computed(() => isTrainerDocked(trainerDockPlacement.value));
const trainerVisible = computed(() => (
  isTabOpen(TAB_IDS.TRAINER)
  && (trainerDocked.value || activeTab.value === TAB_IDS.TRAINER)
));
const trainerSessionActive = computed(() => trainerVisible.value);
const trainerHotkeysEnabled = computed(() => (
  activeTab.value === TAB_IDS.TRAINER
  || (trainerDocked.value && keyboardOwner.value === KEYBOARD_OWNERS.TRAINER)
));
const trainerPaneActive = computed(() => trainerVisible.value && trainerHotkeysEnabled.value);
const trainerWorkspaceClass = computed(() => {
  if (trainerDockPlacement.value === TRAINER_DOCK_PLACEMENTS.RIGHT) {
    return 'app-workspace--trainer-right';
  }
  if (trainerDockPlacement.value === TRAINER_DOCK_PLACEMENTS.BOTTOM) {
    return 'app-workspace--trainer-bottom';
  }
  return 'app-workspace--single';
});

watch(activeTab, (tabId) => {
  if (tabId !== TAB_IDS.TRAINER) {
    lastPrimaryTab.value = tabId;
  }
});

watch(trainerDocked, (docked) => {
  setSplitKeyboardMode(docked && isTabOpen(TAB_IDS.TRAINER));
  if (!docked) {
    setKeyboardOwner(
      activeTab.value === TAB_IDS.TRAINER
        ? KEYBOARD_OWNERS.TRAINER
        : KEYBOARD_OWNERS.PRIMARY,
    );
  }
}, { immediate: true });

const persistTrainerDockPreference = (placement) => {
  trainerDockPreference.value = normalizeTrainerDockPlacement(placement);
  window.localStorage.setItem(
    '2048tables:trainer-dock-placement',
    trainerDockPreference.value,
  );
};

const setTrainerDockPlacement = (placement) => {
  const normalized = normalizeTrainerDockPlacement(placement);
  if (normalized !== TRAINER_DOCK_PLACEMENTS.NONE && !dockAvailable.value) return;
  persistTrainerDockPreference(normalized);
  if (isTrainerDocked(normalized) && activeTab.value === TAB_IDS.TRAINER) {
    activateTab(lastPrimaryTab.value);
  }
  setSplitKeyboardMode(isTrainerDocked(normalized) && isTabOpen(TAB_IDS.TRAINER));
  setKeyboardOwner(isTrainerDocked(normalized) ? KEYBOARD_OWNERS.TRAINER : KEYBOARD_OWNERS.PRIMARY);
};

const showTrainerFullPage = () => {
  persistTrainerDockPreference(TRAINER_DOCK_PLACEMENTS.NONE);
  activateTab(TAB_IDS.TRAINER);
  setSplitKeyboardMode(false);
  setKeyboardOwner(KEYBOARD_OWNERS.TRAINER);
};

const isTabPresented = (tabId) => (
  activeTab.value === tabId
  || (tabId === TAB_IDS.TRAINER && trainerDocked.value && isTabOpen(tabId))
);

const getTabLabel = (tab) => (tab.titleKey ? t(tab.titleKey) : tab.title);
const openAnalysisDialog = (context = {}) => {
  analysisDialogContext.value = { ...(context || {}) };
  analysisDialogOpen.value = true;
};

const closeAnalysisDialog = () => {
  analysisDialogOpen.value = false;
  analysisDialogContext.value = {};
};

const handleNavigateTab = (tabId, detail = null) => {
  if (tabId === TAB_IDS.TRAINER && detail?.hex) {
    queueTrainerPracticeJump(detail);
    const requestedPlacement = resolveTrainerJumpDockPlacement({
      placement: trainerDockPreference.value,
      dockAvailable: dockAvailable.value,
      sourceIsHelp: Boolean(detail.sourceDocumentId),
    });
    if (isTrainerDocked(requestedPlacement)) {
      openTabInBackground(tabId);
      persistTrainerDockPreference(requestedPlacement);
      setSplitKeyboardMode(true);
    } else {
      persistTrainerDockPreference(TRAINER_DOCK_PLACEMENTS.NONE);
      openTab(tabId);
    }
    setKeyboardOwner(KEYBOARD_OWNERS.TRAINER);
    nextTick(() => setKeyboardOwner(KEYBOARD_OWNERS.TRAINER));
    return;
  }
  if (tabId === TAB_IDS.TRAINER) {
    persistTrainerDockPreference(TRAINER_DOCK_PLACEMENTS.NONE);
    setKeyboardOwner(KEYBOARD_OWNERS.TRAINER);
  }
  openTab(tabId);
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
  if (event.key === 'Escape' && trainerDocked.value) {
    setKeyboardOwner(KEYBOARD_OWNERS.PRIMARY);
  }
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
  if (tabId === TAB_IDS.TRAINER && trainerDocked.value) {
    setKeyboardOwner(KEYBOARD_OWNERS.TRAINER);
    blurButtonTarget(event);
    return;
  }
  activateTab(tabId);
  setKeyboardOwner(
    tabId === TAB_IDS.TRAINER ? KEYBOARD_OWNERS.TRAINER : KEYBOARD_OWNERS.PRIMARY,
  );
  blurButtonTarget(event);
};

const handleCloseTab = (tabId, event) => {
  closeTab(tabId);
  if (tabId === TAB_IDS.TRAINER) {
    persistTrainerDockPreference(TRAINER_DOCK_PLACEMENTS.NONE);
    resetKeyboardOwnership();
  }
  blurButtonTarget(event);
};

const handleWorkspaceFocus = (event) => {
  if (!trainerDocked.value || !(event.target instanceof Element)) return;
  setKeyboardOwner(
    event.target.closest('[data-trainer-pane]')
      ? KEYBOARD_OWNERS.TRAINER
      : KEYBOARD_OWNERS.PRIMARY,
  );
};

const updateViewportSize = () => {
  viewportWidth.value = window.innerWidth;
  viewportHeight.value = window.innerHeight;
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

onMounted(() => {
  startAppSettings();
  for (const payload of readPendingGlobalErrors()) {
    enqueueGlobalError(payload);
  }
  window.addEventListener('app-global-error', handleGlobalErrorEvent);
  window.addEventListener('resize', updateViewportSize);
  document.addEventListener('pointerup', handleGlobalPointerUp, true);
  document.addEventListener('keydown', handleGlobalBoardHotkeyFocus, true);
});

onUnmounted(() => {
  window.removeEventListener('app-global-error', handleGlobalErrorEvent);
  window.removeEventListener('resize', updateViewportSize);
  document.removeEventListener('pointerup', handleGlobalPointerUp, true);
  document.removeEventListener('keydown', handleGlobalBoardHotkeyFocus, true);
  resetKeyboardOwnership();
  stopAppSettings();
});
</script>

<style scoped>
.app-shell {
  background-color: var(--bg-main);
  background-image: var(--bg-main-gradient);
}

.app-workspace {
  position: relative;
  flex: 1 1 auto;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

.app-primary-pane,
.app-trainer-pane {
  position: relative;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

.app-workspace--single .app-primary-pane {
  width: 100%;
  height: 100%;
}

.app-workspace--single .app-trainer-pane {
  position: absolute;
  inset: 0;
}

.app-workspace--trainer-right {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 450px;
}

.app-workspace--trainer-right .app-trainer-pane {
  border-left: 1px solid var(--border-main);
}

.app-workspace--trainer-bottom {
  overflow-y: auto;
}

.app-workspace--trainer-bottom .app-primary-pane,
.app-workspace--trainer-bottom .app-trainer-pane {
  width: 100%;
  height: 100%;
  min-height: 100%;
}

.app-workspace--trainer-bottom .app-trainer-pane {
  border-top: 1px solid var(--border-main);
}

.app-trainer-pane {
  background: var(--bg-main);
  box-shadow: inset 0 0 0 2px transparent;
  transition: box-shadow 120ms ease;
}

.app-pane--keyboard-active {
  box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--accent) 70%, transparent);
}

.trainer-dock-toolbar {
  position: absolute;
  top: 0.65rem;
  right: 0.65rem;
  z-index: 140;
  display: flex;
  gap: 0.3rem;
  border: 1px solid var(--border-main);
  border-radius: 6px;
  background: color-mix(in srgb, var(--bg-card) 94%, transparent);
  padding: 0.25rem;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.trainer-dock-button {
  border-radius: 4px;
  padding: 0.3rem 0.5rem;
  color: var(--text-secondary);
  font-size: calc(0.7rem * var(--ui-scale));
  font-weight: 800;
}

.trainer-dock-button:hover,
.trainer-dock-button.is-active {
  background: var(--btn-bg);
  color: white;
}
</style>
