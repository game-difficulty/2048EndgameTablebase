import { computed, onUnmounted, ref, watch } from 'vue';

import { createWsClient } from '../../../services/ws/createWsClient';
import { useAppSettingsStore } from '../../../app/useAppSettings';
import { tryDesktopDialog } from '../../../services/runtime/desktopDialogs';
import { normalizeBCFamilyModulus } from '../../../utils/bcFamilyModulus';
import { isVariantPattern } from '../../../utils/patternCategories';

export function useSettingsSession(activeRef) {
  const DEFAULT_BUILD_PATH = 'C:/2048_tables/';
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
    buildProgressCurrent,
    buildProgressTotal,
    isBuilding,
    applyBuildState,
    applyBuildStarted,
    applyBuildProgress,
    applyBuildFailed,
  } = useAppSettingsStore();

  const selectedCategory = ref('');
  const selectedPattern = ref('');
  const selectedTarget = ref('512');
  const buildPath = ref(DEFAULT_BUILD_PATH);
  const builderAlgorithm = ref('ex');
  const builderAdvancedAlgo = ref(false);
  const builderZMaskAlgo = ref(true);
  const builderCompress = ref(false);
  const builderCompressTempFiles = ref(false);
  const builderOptimalBranchOnly = ref(false);
  const builderChunkedSolve = ref(false);
  const builderBCFamilyModulus = ref(29);
  const builderSuccessRateDtype = ref('uint32');
  const builderSmallTileSumLimit = ref(96);
  const builderDeletionThresholdMode = ref('absolute');
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

  const normalizeDeletionThresholdMode = (value) =>
    value === 'relative' || value === 'off' ? value : 'absolute';

  const normalizeBuildPathList = (value) => {
    const rawItems = Array.isArray(value)
      ? value
      : String(value || '')
          .split(/\r?\n/);
    const seen = new Set();
    const paths = [];
    for (const rawItem of rawItems) {
      const path = String(rawItem || '').trim();
      const key = path;
      if (!path || seen.has(key)) {
        continue;
      }
      seen.add(key);
      paths.push(path);
    }
    return paths;
  };

  const normalizePathKey = (path) =>
    String(path || '')
      .trim()
      .replace(/\\/g, '/')
      .replace(/\/+$/, '')
      .toLowerCase();

  const normalizedBuildPaths = computed(() => normalizeBuildPathList(buildPath.value));

  const setBuildPaths = (paths) => {
    buildPath.value = normalizeBuildPathList(paths).join('\n');
  };

  const appendBuildPath = (path) => {
    setBuildPaths([...normalizedBuildPaths.value, path]);
  };

  const addSelectedBuildPath = (path) => {
    const currentPaths = normalizedBuildPaths.value;
    const selectedPath = String(path || '').trim();
    if (!selectedPath) {
      return;
    }

    const defaultPathKey = normalizePathKey(DEFAULT_BUILD_PATH);
    const currentIsOnlyDefault =
      currentPaths.length === 1 && normalizePathKey(currentPaths[0]) === defaultPathKey;
    if (currentIsOnlyDefault) {
      setBuildPaths([selectedPath]);
      return;
    }

    appendBuildPath(selectedPath);
  };

  const normalizeBuilderAlgorithm = (value) => {
    if (value === 'ad' || value === 'ex' || value === 'exad' || value === 'bc') {
      return value;
    }
    return 'classic';
  };

  const algorithmFromFlags = (advanced, ex) => {
    if (advanced && ex) {
      return 'exad';
    }
    if (advanced) {
      return 'ad';
    }
    if (ex) {
      return 'ex';
    }
    return 'classic';
  };

  const flagsFromAlgorithm = (algorithm, isVariant = false) => {
    const normalized = normalizeBuilderAlgorithm(algorithm);
    if (isVariant) {
      if (normalized === 'ad') {
        return { algorithm: 'classic', advanced: false, ex: false };
      }
      if (normalized === 'exad') {
        return { algorithm: 'ex', advanced: false, ex: true };
      }
    }
    return {
      algorithm: normalized,
      advanced: normalized === 'ad' || normalized === 'exad',
      ex: normalized === 'ex' || normalized === 'exad',
      bc: normalized === 'bc',
    };
  };

  const applyBuilderAlgorithm = (algorithm, { persist = true } = {}) => {
    const flags = flagsFromAlgorithm(algorithm, selectedPatternIsVariant.value);
    builderAlgorithm.value = flags.algorithm;
    builderAdvancedAlgo.value = flags.advanced;
    builderZMaskAlgo.value = flags.ex;
    if (!flags.advanced) {
      builderChunkedSolve.value = false;
    }
    if (flags.advanced || flags.bc) {
      builderOptimalBranchOnly.value = false;
    }

    if (!persist) {
      return flags;
    }

    saveSetting('algorithm_mode', flags.algorithm);
    saveSetting('advanced_algo', flags.advanced);
    saveSetting('zmask_algo', flags.ex);
    if (!flags.advanced) {
      saveSetting('chunked_solve', false);
    }
    if (flags.advanced || flags.bc) {
      saveSetting('optimal_branch_only', false);
    }
    if (flags.bc) {
      const normalizedModulus = normalizeBCFamilyModulus(builderBCFamilyModulus.value);
      builderBCFamilyModulus.value = normalizedModulus;
      saveSetting('bc_family_modulus', normalizedModulus);
    }
    return flags;
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
    const configuredAlgorithm = normalizeBuilderAlgorithm(config.value.algorithm_mode);
    const algorithmFlags = flagsFromAlgorithm(
      configuredAlgorithm,
      isVariantPattern(selectedPattern.value, categories.value)
    );
    builderAlgorithm.value = algorithmFlags.algorithm;
    builderAdvancedAlgo.value = algorithmFlags.advanced;
    builderZMaskAlgo.value = algorithmFlags.ex;
    builderCompress.value = Boolean(config.value.compress);
    builderCompressTempFiles.value = Boolean(config.value.compress_temp_files);
    builderOptimalBranchOnly.value = Boolean(config.value.optimal_branch_only);
    builderChunkedSolve.value = Boolean(config.value.chunked_solve);
    builderBCFamilyModulus.value = normalizeBCFamilyModulus(
      config.value.bc_family_modulus
    );
    builderSuccessRateDtype.value =
      config.value.success_rate_dtype || 'uint32';
    builderSmallTileSumLimit.value =
      Number(config.value.SmallTileSumLimit) || 96;
    builderDeletionThresholdMode.value = normalizeDeletionThresholdMode(
      config.value.deletion_threshold_mode
    );
    deletionThresholdInput.value = formatDeletionThreshold(
      normalizeDeletionThreshold(config.value.deletion_threshold ?? 0)
    );
  };

  watch(
    () => [
      config.value.algorithm_mode,
      config.value.advanced_algo,
      config.value.zmask_algo,
      config.value.compress,
      config.value.compress_temp_files,
      config.value.optimal_branch_only,
      config.value.chunked_solve,
      config.value.bc_family_modulus,
      config.value.success_rate_dtype,
      config.value.SmallTileSumLimit,
      config.value.deletion_threshold,
      config.value.deletion_threshold_mode,
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
  const selectedPatternIsVariant = computed(() => (
    isVariantPattern(selectedPattern.value, categories.value)
  ));

  watch(selectedPatternIsVariant, (isVariant) => {
    if (!isVariant || (builderAlgorithm.value !== 'ad' && builderAlgorithm.value !== 'exad')) {
      return;
    }
    applyBuilderAlgorithm(builderAlgorithm.value);
  });

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
    if (data.type === 'SETTINGS_DATA') {
      if (Object.prototype.hasOwnProperty.call(data.payload || {}, 'build_state')) {
        applyBuildState(data.payload?.build_state || {});
      }
      return;
    }

    if (data.type === 'FOLDER_SELECTED') {
      const path = String(data.payload?.path || '').trim();
      if (path) {
        addSelectedBuildPath(path);
      }
      return;
    }

    if (data.type === 'BUILD_STARTED') {
      applyBuildStarted(data.payload || {});
      return;
    }

    if (data.type === 'BUILD_PROGRESS') {
      applyBuildProgress(data.payload || {});
      return;
    }

    if (data.type === 'BUILD_FAILED') {
      applyBuildFailed();
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

  const handleDeletionThresholdModeChange = (mode = builderDeletionThresholdMode.value) => {
    const normalized = normalizeDeletionThresholdMode(mode);
    builderDeletionThresholdMode.value = normalized;
    saveSetting('deletion_threshold_mode', normalized);
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

  const handleBuilderAlgorithmChange = (algorithm) => {
    applyBuilderAlgorithm(algorithm);
  };

  const handleAdvancedAlgoChange = () => {
    if (selectedPatternIsVariant.value) {
      applyBuilderAlgorithm(builderZMaskAlgo.value ? 'ex' : 'classic');
      return;
    }
    const nextValue = Boolean(builderAdvancedAlgo.value);
    builderAlgorithm.value = algorithmFromFlags(nextValue, builderZMaskAlgo.value);
    saveSetting('algorithm_mode', builderAlgorithm.value);
    saveSetting('advanced_algo', nextValue);
    if (!nextValue) {
      builderChunkedSolve.value = false;
      saveSetting('chunked_solve', false);
    }
    if (nextValue) {
      builderOptimalBranchOnly.value = false;
      saveSetting('optimal_branch_only', false);
    }
  };

  const handleZMaskAlgoChange = () => {
    const nextValue = Boolean(builderZMaskAlgo.value);
    builderAlgorithm.value = algorithmFromFlags(builderAdvancedAlgo.value, nextValue);
    saveSetting('algorithm_mode', builderAlgorithm.value);
    saveSetting('zmask_algo', nextValue);
    if (builderAdvancedAlgo.value) {
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
    if (builderAdvancedAlgo.value || builderAlgorithm.value === 'bc') {
      builderOptimalBranchOnly.value = false;
      saveSetting('optimal_branch_only', false);
      return;
    }
    saveSetting('optimal_branch_only', Boolean(builderOptimalBranchOnly.value));
  };

  const handleChunkedSolveChange = () => {
    if (builderAlgorithm.value === 'bc') {
      builderChunkedSolve.value = false;
      saveSetting('chunked_solve', false);
      return;
    }
    saveSetting('chunked_solve', Boolean(builderChunkedSolve.value));
  };

  const handleBCFamilyModulusChange = () => {
    const normalized = normalizeBCFamilyModulus(builderBCFamilyModulus.value);
    builderBCFamilyModulus.value = normalized;
    saveSetting('bc_family_modulus', normalized);
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
        addSelectedBuildPath(value);
      }
      return;
    }
    buildClient?.send('SETTINGS_TRIGGER_SELECT_FOLDER');
  };

  const startBuild = () => {
    const buildPaths = normalizedBuildPaths.value;
    if (!selectedPattern.value || !selectedTarget.value || !buildPaths.length) {
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
    const normalizedDeletionThresholdMode = normalizeDeletionThresholdMode(
      builderDeletionThresholdMode.value
    );
    builderDeletionThresholdMode.value = normalizedDeletionThresholdMode;
    const algorithmFlags = flagsFromAlgorithm(
      builderAlgorithm.value,
      selectedPatternIsVariant.value
    );
    builderAlgorithm.value = algorithmFlags.algorithm;
    builderAdvancedAlgo.value = algorithmFlags.advanced;
    builderZMaskAlgo.value = algorithmFlags.ex;
    const advancedEnabled = algorithmFlags.advanced;
    const bcEnabled = algorithmFlags.bc;
    if (!advancedEnabled) {
      builderChunkedSolve.value = false;
    }
    if (advancedEnabled || bcEnabled) {
      builderOptimalBranchOnly.value = false;
    }
    if (bcEnabled) {
      builderChunkedSolve.value = false;
    }
    const normalizedBCFamilyModulus = normalizeBCFamilyModulus(
      builderBCFamilyModulus.value
    );
    builderBCFamilyModulus.value = normalizedBCFamilyModulus;
    saveSetting('algorithm_mode', algorithmFlags.algorithm);
    saveSetting('advanced_algo', advancedEnabled);
    saveSetting('zmask_algo', algorithmFlags.ex);
    saveSetting('compress', Boolean(builderCompress.value));
    saveSetting(
      'compress_temp_files',
      Boolean(builderCompressTempFiles.value)
    );
    saveSetting(
      'optimal_branch_only',
      advancedEnabled || bcEnabled ? false : Boolean(builderOptimalBranchOnly.value)
    );
    saveSetting(
      'chunked_solve',
      advancedEnabled && !bcEnabled ? Boolean(builderChunkedSolve.value) : false
    );
    saveSetting('bc_family_modulus', normalizedBCFamilyModulus);
    saveSetting('deletion_threshold_mode', normalizedDeletionThresholdMode);
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
      folder_paths: buildPaths,
      folder_path: buildPaths[0],
      pathname: `${buildPaths[0]}/${selectedPattern.value}_${selectedTarget.value}_`,
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
    selectedPatternIsVariant,
    buildPath,
    normalizedBuildPaths,
    isBuilding,
    builderAlgorithm,
    builderAdvancedAlgo,
    builderZMaskAlgo,
    builderCompress,
    builderCompressTempFiles,
    builderOptimalBranchOnly,
    builderChunkedSolve,
    builderBCFamilyModulus,
    builderSuccessRateDtype,
    builderSmallTileSumLimit,
    builderDeletionThresholdMode,
    deletionThresholdInput,
    filteredPatterns,
    buildProgressPercent,
    buildProgressDisplay,
    saveSetting,
    handleBuilderAlgorithmChange,
    handleAdvancedAlgoChange,
    handleZMaskAlgoChange,
    handleCompressChange,
    handleCompressTempFilesChange,
    handleOptimalBranchOnlyChange,
    handleChunkedSolveChange,
    handleBCFamilyModulusChange,
    handleSuccessRateDtypeChange,
    handleDeletionThresholdModeChange,
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
