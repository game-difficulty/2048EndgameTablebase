import { ref, watch } from 'vue';

import { useAppSettingsStore } from '../../../app/useAppSettings';

export function useSettingsSession(activeRef) {
  const activeSubTab = ref('game');
  const {
    wsStatus,
    loaded,
    config,
    themes,
    currentPalette,
    refreshSettings,
    saveSetting,
    saveCustomColors,
    setTheme,
    setCustomMode,
    changeLanguage,
    start,
  } = useAppSettingsStore();

  const saveCustomColor = () => {
    saveCustomColors(config.value.custom_colors);
  };

  watch(
    activeRef,
    (isActive) => {
      if (isActive) {
        start();
        refreshSettings();
      }
    },
    { immediate: true }
  );

  return {
    activeSubTab,
    settingsLoaded: loaded,
    wsStatus,
    config,
    themes,
    currentPalette,
    saveSetting,
    saveCustomColor,
    setTheme,
    changeLanguage,
    setCustomMode,
  };
}
