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
    openTab,
  };
}
