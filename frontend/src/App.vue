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
        v-if="isTabOpen(TAB_IDS.NOTEBOOK)"
        class="absolute inset-0"
        v-show="activeTab === TAB_IDS.NOTEBOOK"
      >
        <NotebookView
          :active="activeTab === TAB_IDS.NOTEBOOK"
          @navigate-tab="openTab"
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
import { computed, onMounted, onUnmounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';

import { useAppSettingsStore } from './app/useAppSettings';
import { TAB_IDS } from './app/tabRegistry';
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
} = useTabManager();
const draggedTabId = ref(null);
const dragTargetTabId = ref(null);

const getTabLabel = (tab) => (tab.titleKey ? t(tab.titleKey) : tab.title);
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

onMounted(() => {
  startAppSettings();
  for (const payload of readPendingGlobalErrors()) {
    enqueueGlobalError(payload);
  }
  window.addEventListener('app-global-error', handleGlobalErrorEvent);
  document.addEventListener('pointerup', handleGlobalPointerUp, true);
  document.addEventListener('keydown', handleGlobalBoardHotkeyFocus, true);
});

onUnmounted(() => {
  window.removeEventListener('app-global-error', handleGlobalErrorEvent);
  document.removeEventListener('pointerup', handleGlobalPointerUp, true);
  document.removeEventListener('keydown', handleGlobalBoardHotkeyFocus, true);
  stopAppSettings();
});
</script>

<style scoped>
.app-shell {
  background-color: var(--bg-main);
  background-image: var(--bg-main-gradient);
}
</style>
