import { computed, ref } from 'vue';

import { MAIN_TAB_ID, TAB_ORDER, TAB_REGISTRY } from './tabRegistry';

export function useTabManager() {
  const openTabs = ref([MAIN_TAB_ID]);
  const activeTab = ref(MAIN_TAB_ID);

  const hasTab = (tabId) => TAB_ORDER.includes(tabId);
  const isTabOpen = (tabId) => openTabs.value.includes(tabId);

  const ensureTabOpen = (tabId) => {
    if (!hasTab(tabId)) {
      return MAIN_TAB_ID;
    }
    if (!isTabOpen(tabId)) {
      openTabs.value = [...openTabs.value, tabId];
    }
    return tabId;
  };

  const openTab = (tabId) => {
    activeTab.value = ensureTabOpen(tabId);
  };

  const activateTab = (tabId) => {
    if (isTabOpen(tabId)) {
      activeTab.value = tabId;
      return;
    }
    openTab(tabId);
  };

  const closeTab = (tabId) => {
    const tab = TAB_REGISTRY[tabId];
    if (!tab || !tab.closable || !isTabOpen(tabId)) {
      return;
    }

    const closingIndex = openTabs.value.indexOf(tabId);
    const nextOpenTabs = openTabs.value.filter((id) => id !== tabId);
    openTabs.value = nextOpenTabs.length ? nextOpenTabs : [MAIN_TAB_ID];

    if (activeTab.value === tabId) {
      const fallbackIndex = Math.max(0, closingIndex - 1);
      activeTab.value = openTabs.value[fallbackIndex] || MAIN_TAB_ID;
    }
  };

  const moveTabRelative = (tabId, targetTabId, placeAfter = false) => {
    if (
      tabId === MAIN_TAB_ID ||
      targetTabId === tabId ||
      !isTabOpen(tabId) ||
      !isTabOpen(targetTabId)
    ) {
      return;
    }

    const nextOpenTabs = [...openTabs.value];
    const draggingIndex = nextOpenTabs.indexOf(tabId);
    const targetIndex = nextOpenTabs.indexOf(targetTabId);

    if (draggingIndex < 1 || targetIndex < 0) {
      return;
    }

    const [movedTabId] = nextOpenTabs.splice(draggingIndex, 1);

    let insertIndex = 1;
    if (targetTabId !== MAIN_TAB_ID) {
      const normalizedTargetIndex = nextOpenTabs.indexOf(targetTabId);
      insertIndex = normalizedTargetIndex + (placeAfter ? 1 : 0);
    }

    insertIndex = Math.max(1, Math.min(insertIndex, nextOpenTabs.length));
    nextOpenTabs.splice(insertIndex, 0, movedTabId);

    if (nextOpenTabs.some((id, index) => id !== openTabs.value[index])) {
      openTabs.value = nextOpenTabs;
    }
  };

  const openTabDefinitions = computed(() =>
    openTabs.value
      .map((tabId) => TAB_REGISTRY[tabId])
      .filter(Boolean)
  );

  return {
    activeTab,
    openTabs,
    openTabDefinitions,
    activateTab,
    closeTab,
    isTabOpen,
    moveTabRelative,
    openTab,
  };
}
