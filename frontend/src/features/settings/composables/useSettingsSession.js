import { computed, onUnmounted, ref, watch } from 'vue';

import { createWsClient } from '../../../services/ws/createWsClient';
import { useAppSettingsStore } from '../../../app/useAppSettings';

export function useSettingsSession(activeRef) {
  const activeSubTab = ref('builder');
  const {
    wsStatus,
    loaded,
    config,
    categories,
    targetTiles,
    themes,
    currentPalette,
    refreshSettings,
    saveSetting,
    saveCustomColors,
    setTheme,
    setCustomMode,
    changeLanguage,
  } = useAppSettingsStore();

  const selectedCategory = ref('');
  const selectedPattern = ref('');
  const selectedTarget = ref('2048');
  const buildPath = ref('C:/2048_tables/');
  const buildProgressCurrent = ref(0);
  const buildProgressTotal = ref(0);
  const isBuilding = ref(false);

  const formatDeletionThreshold = (value) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(6) : '0.000000';
  };

  const deletionThresholdInput = ref(
    formatDeletionThreshold(config.value.deletion_threshold ?? 0)
  );

  watch(
    () => config.value.advanced_algo,
    (newVal) => {
      if (newVal) {
        config.value.optimal_branch_only = false;
        saveSetting('optimal_branch_only');
      }
    }
  );

  watch(selectedCategory, (newCategory) => {
    if (categories.value[newCategory] && categories.value[newCategory].length > 0) {
      selectedPattern.value = categories.value[newCategory][0];
    }
  });

  watch(
    () => config.value.deletion_threshold,
    (newValue) => {
      deletionThresholdInput.value = formatDeletionThreshold(newValue);
    },
    { immediate: true }
  );

  watch(
    categories,
    (nextCategories) => {
      if (Object.keys(nextCategories || {}).length > 0 && !selectedCategory.value) {
        selectedCategory.value = Object.keys(nextCategories)[0];
      }
    },
    { immediate: true, deep: true }
  );

  const filteredPatterns = computed(
    () => categories.value[selectedCategory.value] || []
  );

  const buildProgressPercent = computed(() => {
    if (buildProgressTotal.value <= 0) {
      return 8;
    }
    return Math.max(
      0,
      Math.min(
        100,
        (buildProgressCurrent.value / buildProgressTotal.value) * 100
      )
    );
  });

  const buildProgressDisplay = computed(() => {
    if (buildProgressTotal.value > 0) {
      return `${buildProgressCurrent.value.toLocaleString()}/${buildProgressTotal.value.toLocaleString()}`;
    }
    return '0/...';
  });

  let buildClient = null;

  const handleMessage = (data) => {
    if (data.type === 'BUILD_STARTED') {
      isBuilding.value = true;
      buildProgressCurrent.value = 0;
      buildProgressTotal.value = 0;
      return;
    }

    if (data.type === 'BUILD_PROGRESS') {
      const nextCurrent = Math.max(0, Number(data.payload?.current) || 0);
      const nextTotal = Math.max(nextCurrent, Number(data.payload?.total) || 0);
      buildProgressCurrent.value = nextCurrent;
      buildProgressTotal.value = nextTotal;
      isBuilding.value = !(nextTotal > 0 && nextCurrent >= nextTotal);
      return;
    }

    if (data.type === 'BUILD_FAILED') {
      isBuilding.value = false;
      buildProgressCurrent.value = 0;
      buildProgressTotal.value = 0;
      console.error('Build failed:', data.payload?.message || 'Unknown error');
      return;
    }
  };

  const connect = () => {
    if (buildClient) {
      return;
    }
    buildClient = createWsClient({
      clientId: `settings_${Math.random().toString(36).slice(2, 9)}`,
      onMessage: handleMessage,
    });
    buildClient.connect();
  };

  const disconnect = () => {
    buildClient?.disconnect();
    buildClient = null;
  };

  const commitDeletionThreshold = () => {
    const parsed = Number.parseFloat(deletionThresholdInput.value);
    const clamped = Number.isFinite(parsed) ? Math.min(1, Math.max(0, parsed)) : 0;
    config.value.deletion_threshold = Number(clamped.toFixed(6));
    deletionThresholdInput.value = formatDeletionThreshold(
      config.value.deletion_threshold
    );
    saveSetting('deletion_threshold');
  };

  const handleDeletionThresholdInput = (event) => {
    deletionThresholdInput.value = event.target.value;
  };

  const handleDeletionThresholdChange = () => {
    commitDeletionThreshold();
  };

  const stepDeletionThreshold = (direction) => {
    const currentValue = Number.isFinite(Number(config.value.deletion_threshold))
      ? Number(config.value.deletion_threshold)
      : 0;
    const steppedValue = currentValue + direction * 0.01;
    const clampedValue = Math.min(1, Math.max(0, steppedValue));
    config.value.deletion_threshold = Number(clampedValue.toFixed(6));
    deletionThresholdInput.value = formatDeletionThreshold(
      config.value.deletion_threshold
    );
    saveSetting('deletion_threshold');
  };

  const saveCustomColor = () => {
    saveCustomColors(config.value.custom_colors);
  };

  const browseFolder = async () => {
    if (window.pywebview && window.pywebview.api) {
      const path = await window.pywebview.api.select_folder();
      if (path) {
        buildPath.value = path;
      }
    }
  };

  const startBuild = () => {
    if (!selectedPattern.value || !selectedTarget.value || !buildPath.value) {
      return;
    }

    isBuilding.value = true;
    buildProgressCurrent.value = 0;
    buildProgressTotal.value = 0;

    const targetTileValue = parseInt(selectedTarget.value, 10);
    const targetExponent =
      Number.isFinite(targetTileValue) && targetTileValue > 0
        ? Math.round(Math.log2(targetTileValue))
        : targetTileValue;

    [
      'advanced_algo',
      'compress',
      'compress_temp_files',
      'optimal_branch_only',
      'chunked_solve',
      'deletion_threshold',
      'success_rate_dtype',
      'SmallTileSumLimit',
    ].forEach(saveSetting);

    buildClient?.send('START_BUILD', {
      pattern: selectedPattern.value,
      target: targetExponent,
      target_tile: selectedTarget.value,
      folder_path: buildPath.value,
      pathname: `${buildPath.value}/${selectedPattern.value}_${selectedTarget.value}_`,
    });
  };

  watch(
    activeRef,
    (isActive) => {
      if (isActive) {
        connect();
        refreshSettings();
      } else {
        disconnect();
      }
    },
    { immediate: true }
  );

  onUnmounted(() => {
    disconnect();
  });

  return {
    activeSubTab,
    settingsLoaded: loaded,
    wsStatus,
    categories,
    targetTiles,
    config,
    themes,
    currentPalette,
    selectedCategory,
    selectedPattern,
    selectedTarget,
    buildPath,
    isBuilding,
    deletionThresholdInput,
    filteredPatterns,
    buildProgressPercent,
    buildProgressDisplay,
    saveSetting,
    handleDeletionThresholdInput,
    handleDeletionThresholdChange,
    stepDeletionThreshold,
    saveCustomColor,
    setTheme,
    changeLanguage,
    setCustomMode,
    browseFolder,
    startBuild,
  };
}
