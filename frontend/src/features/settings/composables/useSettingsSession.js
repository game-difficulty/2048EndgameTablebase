import { computed, onUnmounted, ref, watch } from 'vue';

import { createWsClient } from '../../../services/ws/createWsClient';
import { useAppSettingsStore } from '../../../app/useAppSettings';
import { tryDesktopDialog } from '../../../services/runtime/desktopDialogs';

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
  const selectedTarget = ref('512');
  const buildPath = ref('C:/2048_tables/');
  const buildProgressCurrent = ref(0);
  const buildProgressTotal = ref(0);
  const isBuilding = ref(false);
  const builderAdvancedAlgo = ref(false);
  const builderCompress = ref(false);
  const builderCompressTempFiles = ref(false);
  const builderOptimalBranchOnly = ref(false);
  const builderChunkedSolve = ref(false);
  const builderSuccessRateDtype = ref('uint32');
  const builderSmallTileSumLimit = ref(96);
  const MAX_DELETION_THRESHOLD = 0.999999;
  const DEFAULT_DELETION_THRESHOLD_DECIMALS = 6;

  const countFractionDigits = (value) => {
    const decimalPart = String(value).split('.')[1];
    return decimalPart ? decimalPart.length : 0;
  };

  const toPlainNumberString = (value) => {
    const stringValue = String(value);
    if (!/[eE]/.test(stringValue)) {
      return stringValue;
    }

    const [coefficient, exponentPart] = stringValue.toLowerCase().split('e');
    const exponent = Number.parseInt(exponentPart, 10);
    if (!Number.isFinite(exponent)) {
      return stringValue;
    }

    const isNegative = coefficient.startsWith('-');
    const unsignedCoefficient = isNegative ? coefficient.slice(1) : coefficient;
    const [integerPart, fractionPart = ''] = unsignedCoefficient.split('.');
    const digits = `${integerPart}${fractionPart}`;
    const sign = isNegative ? '-' : '';

    if (exponent >= 0) {
      const wholeLength = integerPart.length + exponent;
      if (fractionPart.length <= exponent) {
        return `${sign}${digits}${'0'.repeat(exponent - fractionPart.length)}`;
      }
      return `${sign}${digits.slice(0, wholeLength)}.${digits.slice(wholeLength)}`;
    }

    const decimalIndex = integerPart.length + exponent;
    if (decimalIndex > 0) {
      return `${sign}${digits.slice(0, decimalIndex)}.${digits.slice(decimalIndex)}`;
    }

    return `${sign}0.${'0'.repeat(Math.abs(decimalIndex))}${digits}`;
  };

  const padFractionDigits = (value, minimumFractionDigits) => {
    if (minimumFractionDigits <= 0) {
      return value;
    }

    const [integerPart, fractionPart = ''] = String(value).split('.');
    return `${integerPart}.${fractionPart.padEnd(minimumFractionDigits, '0')}`;
  };

  const formatDeletionThreshold = (value) => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return `0.${'0'.repeat(DEFAULT_DELETION_THRESHOLD_DECIMALS)}`;
    }

    const plainValue = toPlainNumberString(parsed);
    return padFractionDigits(
      plainValue,
      Math.max(
        DEFAULT_DELETION_THRESHOLD_DECIMALS,
        countFractionDigits(plainValue)
      )
    );
  };

  const normalizeDeletionThreshold = (value) => {
    const parsed = Number.parseFloat(value);
    if (!Number.isFinite(parsed)) {
      return 0;
    }
    return Math.min(MAX_DELETION_THRESHOLD, Math.max(0, parsed));
  };

  const deletionThresholdInput = ref(
    formatDeletionThreshold(config.value.deletion_threshold ?? 0)
  );

  watch(selectedCategory, (newCategory) => {
    if (categories.value[newCategory] && categories.value[newCategory].length > 0) {
      selectedPattern.value = categories.value[newCategory][0];
    }
  });

  const syncBuilderStateFromConfig = () => {
    builderAdvancedAlgo.value = Boolean(config.value.advanced_algo);
    builderCompress.value = Boolean(config.value.compress);
    builderCompressTempFiles.value = Boolean(config.value.compress_temp_files);
    builderOptimalBranchOnly.value = builderAdvancedAlgo.value
      ? false
      : Boolean(config.value.optimal_branch_only);
    builderChunkedSolve.value = Boolean(config.value.chunked_solve);
    builderSuccessRateDtype.value =
      config.value.success_rate_dtype || 'uint32';
    builderSmallTileSumLimit.value =
      Number(config.value.SmallTileSumLimit) || 96;
    deletionThresholdInput.value = formatDeletionThreshold(
      normalizeDeletionThreshold(config.value.deletion_threshold ?? 0)
    );
  };

  watch(
    () => [
      config.value.advanced_algo,
      config.value.compress,
      config.value.compress_temp_files,
      config.value.optimal_branch_only,
      config.value.chunked_solve,
      config.value.success_rate_dtype,
      config.value.SmallTileSumLimit,
      config.value.deletion_threshold,
    ],
    syncBuilderStateFromConfig,
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

  const applyBuildState = (buildState = {}) => {
    const nextCurrent = Math.max(0, Number(buildState.current) || 0);
    const nextTotal = Math.max(nextCurrent, Number(buildState.total) || 0);
    buildProgressCurrent.value = nextCurrent;
    buildProgressTotal.value = nextTotal;
    isBuilding.value = Boolean(buildState.is_building);
  };

  const handleMessage = (data) => {
    if (data.type === 'SETTINGS_DATA') {
      applyBuildState(data.payload?.build_state || {});
      return;
    }

    if (data.type === 'FOLDER_SELECTED') {
      const path = String(data.payload?.path || '').trim();
      if (path) {
        buildPath.value = path;
      }
      return;
    }

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
      onOpen: () => {
        buildClient?.send('GET_SETTINGS');
      },
      onMessage: handleMessage,
    });
    buildClient.connect();
  };

  const disconnect = () => {
    buildClient?.disconnect();
    buildClient = null;
  };

  const commitDeletionThreshold = () => {
    const normalized = normalizeDeletionThreshold(deletionThresholdInput.value);
    deletionThresholdInput.value = formatDeletionThreshold(normalized);
    saveSetting('deletion_threshold', normalized);
  };

  const handleDeletionThresholdInput = (event) => {
    deletionThresholdInput.value = event.target.value;
  };

  const handleDeletionThresholdChange = () => {
    commitDeletionThreshold();
  };

  const stepDeletionThreshold = (direction) => {
    const currentValue = normalizeDeletionThreshold(deletionThresholdInput.value);
    const steppedValue = currentValue + direction * 0.01;
    const clampedValue = Number(
      normalizeDeletionThreshold(steppedValue).toFixed(6)
    );
    deletionThresholdInput.value = formatDeletionThreshold(clampedValue);
    saveSetting('deletion_threshold', clampedValue);
  };

  const handleAdvancedAlgoChange = () => {
    const nextValue = Boolean(builderAdvancedAlgo.value);
    saveSetting('advanced_algo', nextValue);
    if (nextValue) {
      builderOptimalBranchOnly.value = false;
      saveSetting('optimal_branch_only', false);
    }
  };

  const handleCompressChange = () => {
    saveSetting('compress', Boolean(builderCompress.value));
  };

  const handleCompressTempFilesChange = () => {
    saveSetting('compress_temp_files', Boolean(builderCompressTempFiles.value));
  };

  const handleOptimalBranchOnlyChange = () => {
    saveSetting('optimal_branch_only', Boolean(builderOptimalBranchOnly.value));
  };

  const handleChunkedSolveChange = () => {
    saveSetting('chunked_solve', Boolean(builderChunkedSolve.value));
  };

  const handleSuccessRateDtypeChange = () => {
    saveSetting('success_rate_dtype', builderSuccessRateDtype.value);
  };

  const handleSmallTileSumLimitChange = () => {
    saveSetting(
      'SmallTileSumLimit',
      Number(builderSmallTileSumLimit.value) || 96
    );
  };

  const saveCustomColor = () => {
    saveCustomColors(config.value.custom_colors);
  };

  const browseFolder = async () => {
    const { handled, value } = await tryDesktopDialog('select_folder');
    if (handled) {
      if (value) {
        buildPath.value = value;
      }
      return;
    }
    buildClient?.send('SETTINGS_TRIGGER_SELECT_FOLDER');
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

    const normalizedDeletionThreshold = normalizeDeletionThreshold(
      deletionThresholdInput.value
    );
    deletionThresholdInput.value = formatDeletionThreshold(
      normalizedDeletionThreshold
    );
    saveSetting('advanced_algo', Boolean(builderAdvancedAlgo.value));
    saveSetting('compress', Boolean(builderCompress.value));
    saveSetting(
      'compress_temp_files',
      Boolean(builderCompressTempFiles.value)
    );
    saveSetting(
      'optimal_branch_only',
      builderAdvancedAlgo.value ? false : Boolean(builderOptimalBranchOnly.value)
    );
    saveSetting('chunked_solve', Boolean(builderChunkedSolve.value));
    saveSetting('deletion_threshold', normalizedDeletionThreshold);
    saveSetting('success_rate_dtype', builderSuccessRateDtype.value);
    saveSetting(
      'SmallTileSumLimit',
      Number(builderSmallTileSumLimit.value) || 96
    );

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
    builderAdvancedAlgo,
    builderCompress,
    builderCompressTempFiles,
    builderOptimalBranchOnly,
    builderChunkedSolve,
    builderSuccessRateDtype,
    builderSmallTileSumLimit,
    deletionThresholdInput,
    filteredPatterns,
    buildProgressPercent,
    buildProgressDisplay,
    saveSetting,
    handleAdvancedAlgoChange,
    handleCompressChange,
    handleCompressTempFilesChange,
    handleOptimalBranchOnlyChange,
    handleChunkedSolveChange,
    handleSuccessRateDtypeChange,
    handleDeletionThresholdInput,
    handleDeletionThresholdChange,
    stepDeletionThreshold,
    handleSmallTileSumLimitChange,
    saveCustomColor,
    setTheme,
    changeLanguage,
    setCustomMode,
    browseFolder,
    startBuild,
  };
}
