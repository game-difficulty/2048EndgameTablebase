import { computed, onMounted, onUnmounted, ref, watch } from 'vue';

import { useAppSettingsStore } from '../../../app/useAppSettings';
import { useAuthState } from '../../../services/auth/authState';
import {
  fetchTablebaseCatalog,
  getCatalogVersion,
  getCatalogTargets,
  getCatalogTargetsForPattern,
  groupTablebasePatternsByCategory,
} from '../../../services/tablebases/catalogClient';
import {
  clearTablebaseResultCache,
  getCachedTablebaseResult,
  setCachedTablebaseResult,
} from '../../../services/tablebases/tablebaseResultCache';
import { createWsClient } from '../../../services/ws/createWsClient';
import { getStableWsClientId } from '../../../services/ws/clientIds';
import { isVariantPattern } from '../../../utils/patternCategories';
import { createResultBarGradient } from '../../../utils/resultBars';
import {
  restoreSuccessRate,
  formatSuccessRate,
  successRateSortValue,
  successRateRelativeLoss,
  resultValueFontSize,
} from '../../../utils/successRate';

export function useTrainerSession(activeRef) {
  const RESULT_REFRESH_GRACE_MS = 180;
  const RESULT_REFRESH_PLACEHOLDER_MS = 1400;
  const DEFAULT_TABLEBASE_PATTERN = '442t';
  const DEFAULT_TABLEBASE_TARGET = '512';
  const {
    config: appConfig,
    categories: appCategories,
    targetTiles: appTargetTiles,
    refreshSettings,
    saveSetting,
  } = useAppSettingsStore();
  const { isAuthenticated, requireAuth } = useAuthState();

  const fallbackPatternCategories = {
    basic: ['L3', 'L4', 'I3', 'I4', 'LL', 'free8', 'free9', 'free10', '444'],
  };
  const wsStatus = ref('connecting');
  const clientId = getStableWsClientId('trainer');

  const board = ref([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const metadata = ref(null);
  const hexInput = ref('');
  const cellPalette = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
  const currentPaletteValue = ref(null);
  const dis32k = ref(false);
  const awaitingSpawn = ref(false);

  const tablebasePath = ref('');
  const loadedTablebaseFullPattern = ref('');
  const patternType = ref('');
  const targetValue = ref('');
  const tableResult = ref({ dtype: '?', results: {} });
  const currentBoardHex = ref('');
  const resultsBoardHex = ref('');
  const patternCategories = ref(fallbackPatternCategories);
  const availableTargets = ref(['64', '128', '256', '512', '1024', '2048', '4096', '8192']);
  const catalogTables = ref([]);
  const catalogVersion = ref('');
  const patternMenuOpen = ref(false);
  const activePatternCategory = ref(Object.keys(fallbackPatternCategories)[0] || '');
  const patternMenuRoot = ref(null);
  const dirLabels = computed(() => (
    String(appConfig.value?.language || 'en').startsWith('zh')
      ? { left: '左', right: '右', down: '下', up: '上' }
      : { left: 'L', right: 'R', down: 'D', up: 'U' }
  ));

  const spawnMode = ref(0);
  const spawnModes = ['Random', 'Best', 'Worst', 'Manual'];
  const demoActive = ref(false);
  const showResults = ref(true);
  const recordingState = ref(false);
  const replayResultsActive = ref(false);
  const recordPlaybackLoaded = ref(false);
  const queuedStepCount = ref(0);
  const stepExecutionPending = ref(false);
  const demoSpeed = ref(40);
  let nextResultsRequestId = 0;
  const pendingResultsRequests = new Map();
  const pendingTrainerJump = ref(null);
  let demoTimer = null;
  let resultsStaleTimer = null;
  let resultsPlaceholderTimer = null;
  let tablebaseRetryTimer = null;
  const resultsRefreshPhase = ref('idle');

  const recordStep = ref(0);
  const recordMax = ref(0);
  const recordingLength = ref(0);
  const fullHistory = ref([]);
  const fullMoves = ref([]);

  let client = null;
  let initialStateSeen = false;
  let defaultTablebaseAutoApplyAttempted = false;

  const currentPatternDisplay = computed(() =>
    patternType.value && targetValue.value ? `${patternType.value}_${targetValue.value}` : ''
  );
  const isVariant = computed(() => isVariantPattern(patternType.value, patternCategories.value));
  const patternGroups = computed(() =>
    Object.entries(patternCategories.value || {}).map(([category, patterns]) => ({
      category,
      patterns: Array.isArray(patterns) ? patterns : [],
    }))
  );
  const flatPatterns = computed(() => patternGroups.value.flatMap((group) => group.patterns));
  const activePatternOptions = computed(() => patternCategories.value[activePatternCategory.value] || []);
  const availableTargetsForPattern = computed(() => {
    if (!catalogTables.value.length) {
      return availableTargets.value;
    }
    return getCatalogTargetsForPattern(catalogTables.value, patternType.value);
  });
  const hasUsableResults = computed(() =>
    Object.values(tableResult.value.results || {}).some((val) => typeof val === 'number' && Number.isFinite(val))
  );
  const hasPlayableResults = computed(() =>
    Object.values(tableResult.value.results || {}).some((val) => {
      const restored = restoreSuccessRate(val, tableResult.value.dtype || '');
      return typeof restored === 'number' && Number.isFinite(restored) && restored > 0;
    })
  );
  const recordOpen = computed(() => recordPlaybackLoaded.value && !recordingState.value);
  const recordPlaybackActive = computed(() => recordOpen.value && recordMax.value > 1);
  const recordPlaybackAtEnd = computed(() =>
    recordPlaybackActive.value && recordStep.value >= recordMax.value - 1
  );
  const bestResultMove = computed(() => {
    const dtype = tableResult.value.dtype || '';
    const orderedMoves = Object.entries(tableResult.value.results || {})
      .map(([dir, rawVal]) => ({
        dir,
        val: restoreSuccessRate(rawVal, dtype),
        sortVal: successRateSortValue(rawVal, dtype),
      }))
      .filter((item) => (
        typeof item.val === 'number' &&
        Number.isFinite(item.val) &&
        item.val > 0 &&
        typeof item.sortVal === 'number' &&
        Number.isFinite(item.sortVal)
      ))
      .sort((a, b) => b.sortVal - a.sortVal);
    return orderedMoves[0]?.dir || null;
  });

  const togglePalette = (val) => {
    currentPaletteValue.value = currentPaletteValue.value === val ? null : val;
  };

  const resultPrecision = computed(() => {
    const d = tableResult.value.dtype || '';
    return d.includes('64') ? 15 : 8;
  });

  const lerpColor = (c1, c2, r) => {
    const f = (x, y) => Math.round(x + (y - x) * r);
    const parse = (c) => c.slice(1).match(/.{2}/g).map((x) => parseInt(x, 16));
    const [r1, g1, b1] = parse(c1);
    const [r2, g2, b2] = parse(c2);
    return `rgb(${f(r1, r2)}, ${f(g1, g2)}, ${f(b1, b2)})`;
  };

  const sortedResults = computed(() => {
    const dirs = ['left', 'right', 'down', 'up'];
    const r = tableResult.value.results || {};
    const dtype = tableResult.value.dtype || '';
    const items = dirs.map((dir) => {
      const rawVal = r[dir];
      const val = restoreSuccessRate(rawVal, dtype);
      const sortVal = successRateSortValue(rawVal, dtype);
      return {
        dir,
        rawVal: (rawVal == null || typeof rawVal !== 'number') ? null : rawVal,
        val: val == null ? null : val,
        sortVal: sortVal == null ? null : sortVal,
      };
    });
    items.sort((a, b) => {
      if (a.sortVal == null && b.sortVal == null) return 0;
      if (a.sortVal == null) return 1;
      if (b.sortVal == null) return -1;
      return b.sortVal - a.sortVal;
    });
    const bestItem = items.find((i) => i.val != null && i.rawVal != null);
    const bestVal = bestItem?.val || 0;
    const prec = resultPrecision.value;

    const COLOR_GREEN = '#4caf50';
    const COLOR_YG = '#8bc34a';
    const COLOR_ORANGE = '#ff9800';
    const COLOR_RED = '#f44336';

    return items.map((item, idx) => {
      let pct = 0;
      let color = 'var(--border-main)';
      if (item.val != null && bestVal > 0) {
        const loss = successRateRelativeLoss(item.rawVal, bestItem?.rawVal, dtype);
        if (idx === 0) {
          pct = 100;
          color = COLOR_GREEN;
        } else if (loss == null || loss > 0.10) {
          pct = 0;
          color = COLOR_RED;
        } else {
          pct = (1 - loss / 0.10) * 100;
          if (loss <= 0.001) {
            color = COLOR_GREEN;
          } else if (loss <= 0.01) {
            const ratio = (loss - 0.001) / (0.01 - 0.001);
            color = lerpColor(COLOR_GREEN, COLOR_YG, ratio);
          } else if (loss <= 0.03) {
            const ratio = (loss - 0.01) / (0.03 - 0.01);
            color = lerpColor(COLOR_YG, COLOR_ORANGE, ratio);
          } else if (loss <= 0.10) {
            const ratio = (loss - 0.03) / (0.10 - 0.03);
            color = lerpColor(COLOR_ORANGE, COLOR_RED, ratio);
          }
        }
      }
      return {
        dir: item.dir,
        val: item.val,
        pct,
        display: item.rawVal == null ? '—' : formatSuccessRate(item.rawVal, dtype, prec),
        color,
        gradient: createResultBarGradient(color),
        textColor: item.val == null ? 'var(--text-secondary)' : 'var(--text-main)',
      };
    });
  });

  const createPlaceholderResults = () => ['left', 'right', 'down', 'up'].map((dir) => ({
    dir,
    val: null,
    pct: 0,
    display: '--',
    color: 'var(--border-main)',
    gradient: 'transparent',
    textColor: 'var(--text-secondary)',
  }));

  const displayedResults = computed(() => (
    !recordOpen.value && resultsRefreshPhase.value === 'placeholder'
      ? createPlaceholderResults()
      : sortedResults.value
  ));

  const resultsRefreshing = computed(() => !recordOpen.value && resultsRefreshPhase.value !== 'idle');
  const resultsUpdatingVisible = computed(
    () => !recordOpen.value && (
      resultsRefreshPhase.value === 'stale' || resultsRefreshPhase.value === 'placeholder'
    )
  );

  const getResultRowStyle = (item) => ({
    background: item.val != null ? 'var(--bg-main)' : 'transparent',
    opacity: item.val == null ? 0.5 : 1,
  });

  const resultFontSize = computed(() => resultValueFontSize(displayedResults.value.map((item) => item.display)));

  const getResultValueStyle = (item) => {
    return {
      color: item.textColor,
      fontSize: resultFontSize.value,
    };
  };

  const syncActivePatternCategory = () => {
    const matchedGroup = patternGroups.value.find((group) => group.patterns.includes(patternType.value));
    activePatternCategory.value = matchedGroup?.category || patternGroups.value[0]?.category || '';
  };

  const preferredTargetFrom = (targets) => (
    targets.includes(DEFAULT_TABLEBASE_TARGET) ? DEFAULT_TABLEBASE_TARGET : (targets[0] || '')
  );

  const chooseDefaultCatalogSelection = (patterns) => {
    const nextPattern = patterns.includes(DEFAULT_TABLEBASE_PATTERN)
      ? DEFAULT_TABLEBASE_PATTERN
      : (patterns[0] || '');
    const nextTargets = getCatalogTargetsForPattern(catalogTables.value, nextPattern);
    return {
      pattern: nextPattern,
      target: nextTargets.includes(DEFAULT_TABLEBASE_TARGET)
        ? DEFAULT_TABLEBASE_TARGET
        : preferredTargetFrom(nextTargets),
    };
  };

  const ensureTargetForCurrentPattern = () => {
    const targets = availableTargetsForPattern.value;
    if (!targets.length) {
      targetValue.value = '';
      return;
    }
    if (!targets.includes(targetValue.value)) {
      targetValue.value = preferredTargetFrom(targets);
    }
  };

  const parseFullPattern = (fullPattern) => {
    const raw = String(fullPattern || '').trim();
    const splitIndex = raw.lastIndexOf('_');
    if (splitIndex <= 0 || splitIndex >= raw.length - 1) return null;
    return {
      pattern: raw.slice(0, splitIndex),
      target: raw.slice(splitIndex + 1),
    };
  };

  const syncSelectionFromFullPattern = (fullPattern) => {
    const parsed = parseFullPattern(fullPattern);
    if (!parsed || !flatPatterns.value.includes(parsed.pattern)) {
      return false;
    }
    const targets = catalogTables.value.length
      ? getCatalogTargetsForPattern(catalogTables.value, parsed.pattern)
      : availableTargets.value;
    if (!targets.includes(parsed.target)) {
      return false;
    }
    patternType.value = parsed.pattern;
    targetValue.value = parsed.target;
    syncActivePatternCategory();
    return true;
  };

  const selectedTablebaseExists = () => {
    if (!patternType.value || !targetValue.value || !flatPatterns.value.includes(patternType.value)) {
      return false;
    }
    const targets = catalogTables.value.length
      ? getCatalogTargetsForPattern(catalogTables.value, patternType.value)
      : availableTargetsForPattern.value;
    return targets.includes(targetValue.value);
  };

  const maybeAutoApplyDefaultTablebase = () => {
    if (
      !activeRef?.value ||
      wsStatus.value !== 'connected' ||
      !isAuthenticated.value ||
      !initialStateSeen ||
      defaultTablebaseAutoApplyAttempted ||
      pendingTrainerJump.value ||
      tablebasePath.value === 'loaded' ||
      !selectedTablebaseExists()
    ) {
      return;
    }
    defaultTablebaseAutoApplyAttempted = true;
    applyTablebase({ loadDefault: true });
  };

  const protectedActions = new Set([
    'TRAINER_SET_FILEPATH',
    'TRAINER_GET_RESULTS',
    'TABLEBASE_QUERY',
    'TRAINER_DEFAULT',
    'TRAINER_MOVE',
    'TRAINER_MANUAL_SPAWN',
    'TRAINER_STEP',
    'SET_BOARD',
    'SET_CELL',
    'UNDO',
  ]);

  const triggerAction = (action, payload = {}) => {
    if (protectedActions.has(action) && !requireAuth()) {
      return false;
    }
    client?.send(action, payload);
    return true;
  };

  const loadCatalog = async ({ preserveSelection = false } = {}) => {
    try {
      const selectedPattern = patternType.value;
      const selectedTarget = targetValue.value;
      const tables = await fetchTablebaseCatalog();
      catalogTables.value = tables;
      catalogVersion.value = tables.catalogVersion || getCatalogVersion();
      const nextCategories = groupTablebasePatternsByCategory(tables);
      const patterns = Object.values(nextCategories).flat();
      if (patterns.length) {
        patternCategories.value = nextCategories;
        availableTargets.value = getCatalogTargets(tables);
        if (!patternType.value || (!preserveSelection && !patterns.includes(patternType.value))) {
          const defaults = chooseDefaultCatalogSelection(patterns);
          patternType.value = defaults.pattern;
          targetValue.value = defaults.target;
        }
        if (
          preserveSelection
          && selectedPattern
          && selectedTarget
          && !getCatalogTargetsForPattern(tables, selectedPattern).includes(selectedTarget)
        ) {
          patternType.value = selectedPattern;
          targetValue.value = selectedTarget;
        } else {
          ensureTargetForCurrentPattern();
        }
        if (loadedTablebaseFullPattern.value) {
          syncSelectionFromFullPattern(loadedTablebaseFullPattern.value);
        }
        syncActivePatternCategory();
        applyTrainerJump();
        maybeAutoApplyDefaultTablebase();
        if (
          tablebasePath.value === 'loaded'
          && currentBoardHex.value
          && resultsBoardHex.value !== currentBoardHex.value
        ) {
          queryResults('auto');
        }
      }
    } catch (error) {
      console.error(error);
    }
  };

  const applyTrainerJump = () => {
    const pending = pendingTrainerJump.value;
    if (!pending || wsStatus.value !== 'connected') return;

    hexInput.value = pending.hex;
    currentBoardHex.value = pending.hex;
    triggerAction('SET_BOARD', { hex_str: pending.hex });

    const parsed = parseFullPattern(pending.fullPattern);
    if (
      parsed &&
      flatPatterns.value.includes(parsed.pattern) &&
      (
        catalogTables.value.length
          ? getCatalogTargetsForPattern(catalogTables.value, parsed.pattern).includes(parsed.target)
          : availableTargets.value.includes(parsed.target)
      )
    ) {
      const shouldSwitchPattern = currentPatternDisplay.value !== pending.fullPattern;
      if (shouldSwitchPattern) {
        patternType.value = parsed.pattern;
        targetValue.value = parsed.target;
        syncActivePatternCategory();
        applyTablebase({ loadDefault: false });
      }
    }

    pendingTrainerJump.value = null;
  };

  const handleTrainerPracticeJump = (event) => {
    const detail = event?.detail || {};
    const hex = String(detail.hex || '').trim();
    if (!hex) return;
    pendingTrainerJump.value = {
      fullPattern: String(detail.fullPattern || '').trim(),
      hex,
    };
    applyTrainerJump();
  };

  const togglePatternMenu = () => {
    syncActivePatternCategory();
    patternMenuOpen.value = !patternMenuOpen.value;
  };

  const selectPattern = (pattern) => {
    if (patternType.value === pattern) {
      patternMenuOpen.value = false;
      return;
    }
    patternType.value = pattern;
    syncActivePatternCategory();
    patternMenuOpen.value = false;
    ensureTargetForCurrentPattern();
    if (targetValue.value) {
      onPatternChange();
    }
  };

  const closePatternMenuOnClick = (event) => {
    if (!patternMenuOpen.value || !patternMenuRoot.value) return;
    if (!patternMenuRoot.value.contains(event.target)) {
      patternMenuOpen.value = false;
    }
  };

  const hasPendingResultsForBoard = (boardHex) => {
    for (const request of pendingResultsRequests.values()) {
      if (request.boardHex === boardHex) {
        return true;
      }
    }
    return false;
  };

  const clearResultsRefreshTimers = () => {
    if (resultsStaleTimer) {
      window.clearTimeout(resultsStaleTimer);
      resultsStaleTimer = null;
    }
    if (resultsPlaceholderTimer) {
      window.clearTimeout(resultsPlaceholderTimer);
      resultsPlaceholderTimer = null;
    }
  };

  const startResultsRefresh = () => {
    clearResultsRefreshTimers();
    resultsRefreshPhase.value = 'grace';
    resultsStaleTimer = window.setTimeout(() => {
      resultsRefreshPhase.value = 'stale';
      resultsStaleTimer = null;
    }, RESULT_REFRESH_GRACE_MS);
    resultsPlaceholderTimer = window.setTimeout(() => {
      resultsRefreshPhase.value = 'placeholder';
      resultsPlaceholderTimer = null;
    }, RESULT_REFRESH_PLACEHOLDER_MS);
  };

  const finishResultsRefresh = () => {
    clearResultsRefreshTimers();
    resultsRefreshPhase.value = 'idle';
  };

  const invalidateResults = ({ clearDisplay = false } = {}) => {
    resultsBoardHex.value = '';
    if (clearDisplay) {
      tableResult.value = { dtype: '?', results: {} };
      replayResultsActive.value = false;
    }
  };

  const clearDemoTimer = () => {
    if (demoTimer) {
      window.clearTimeout(demoTimer);
      demoTimer = null;
    }
  };

  const getDemoDelayMs = () => Math.max(1, Math.round(Number(demoSpeed.value) || 40));

  const scheduleDemoStep = (delayMs = getDemoDelayMs()) => {
    clearDemoTimer();
    if (!demoActive.value || awaitingSpawn.value) return;
    if (recordPlaybackAtEnd.value) {
      demoActive.value = false;
      clearStepQueue();
      return;
    }
    demoTimer = window.setTimeout(() => {
      demoTimer = null;
      if (!demoActive.value || awaitingSpawn.value) return;
      if (recordPlaybackActive.value) {
        playRecordStep(1, { fromDemo: true });
      } else {
        trainerStep();
      }
    }, Math.max(1, delayMs));
  };

  const queryResults = (reason = 'manual') => {
    if (recordOpen.value || (reason !== 'step' && !showResults.value) || awaitingSpawn.value) return null;
    const boardHex = currentBoardHex.value || hexInput.value;
    if (!boardHex) return null;
    if (hasPendingResultsForBoard(boardHex)) return null;
    if (!isAuthenticated.value) {
      if (reason === 'auto') return null;
      if (!requireAuth()) return null;
    }

    const requestId = `${clientId}_${++nextResultsRequestId}`;
    if (tablebaseRetryTimer) {
      window.clearTimeout(tablebaseRetryTimer);
      tablebaseRetryTimer = null;
    }
    pendingResultsRequests.clear();
    pendingResultsRequests.set(requestId, { boardHex, reason });
    const fullPattern = loadedTablebaseFullPattern.value || currentPatternDisplay.value;
    const version = catalogVersion.value || getCatalogVersion();
    const cached = version && fullPattern
      ? getCachedTablebaseResult({ catalogVersion: version, fullPattern, boardHex })
      : null;
    if (cached) {
      tableResult.value = {
        dtype: cached.dtype || '?',
        results: cached.results || {},
      };
      resultsBoardHex.value = boardHex;
      finishResultsRefresh();
      if (queuedStepCount.value > 0 || demoActive.value) {
        window.queueMicrotask(pumpQueuedSteps);
      }
    } else {
      startResultsRefresh();
    }
    triggerAction('TABLEBASE_QUERY', {
      page: 'trainer',
      query_id: requestId,
      catalog_version: version,
      full_pattern: fullPattern,
      board_hex: boardHex,
    });
    return requestId;
  };

  const clearStepQueue = () => {
    queuedStepCount.value = 0;
    stepExecutionPending.value = false;
  };

  const pumpQueuedSteps = () => {
    if (!queuedStepCount.value || stepExecutionPending.value || awaitingSpawn.value) return;

    const boardHex = currentBoardHex.value || hexInput.value;
    const resultsAreFresh = resultsBoardHex.value === boardHex && hasUsableResults.value;
    if (!resultsAreFresh) {
      queryResults('step');
      return;
    }

    const move = bestResultMove.value;
    if (!move) {
      demoActive.value = false;
      clearDemoTimer();
      clearStepQueue();
      return;
    }

    queuedStepCount.value -= 1;
    stepExecutionPending.value = true;
    triggerAction('TRAINER_MOVE', { dir: move });
  };

  const handleMessage = async (data) => {
    if (
      ['TOKEN_REQUIRED', 'AUTH_REQUIRED'].includes(data.action)
      && pendingResultsRequests.size > 0
    ) {
      pendingResultsRequests.clear();
      clearTablebaseResultCache();
      invalidateResults({ clearDisplay: true });
      finishResultsRefresh();
      clearStepQueue();
      return;
    }

    if (data.action === 'RECORDING_STARTED') {
      recordingState.value = true;
      recordPlaybackLoaded.value = false;
      replayResultsActive.value = false;
      pendingResultsRequests.clear();
      finishResultsRefresh();
      invalidateResults({ clearDisplay: true });
      recordingLength.value = data.data?.recording_length ?? 1;
      return;
    }

    if (data.action === 'RECORDING_STOPPED') {
      recordingState.value = false;
      recordingLength.value = data.data?.recording_length ?? 0;
      return;
    }

    if (data.action === 'UPDATE_STATE') {
      metadata.value = data.data.animation;
      board.value = data.data.board;
      initialStateSeen = true;
      if (typeof data.data.tablebase_status === 'string') {
        tablebasePath.value = data.data.tablebase_status;
      }
      loadedTablebaseFullPattern.value = data.data.tablebase_full_pattern || '';
      if (loadedTablebaseFullPattern.value) {
        syncSelectionFromFullPattern(loadedTablebaseFullPattern.value);
      }
      const nextBoardHex = data.data.hex_str || hexInput.value;
      const boardChanged = !!nextBoardHex && nextBoardHex !== currentBoardHex.value;
      if (nextBoardHex) {
        currentBoardHex.value = nextBoardHex;
        hexInput.value = nextBoardHex;
      }
      if (data.data.record_step !== undefined) {
        recordStep.value = data.data.record_step || 0;
        recordMax.value = data.data.record_max || 0;
        recordingLength.value = data.data.recording_length || 0;
        recordPlaybackLoaded.value = !!data.data.record_playback_loaded;
        fullHistory.value = data.data.history || [];
        fullMoves.value = data.data.moves || [];
      }
      const nextSpawnMode = Number(data.data.spawn_mode);
      if (Number.isInteger(nextSpawnMode) && nextSpawnMode >= 0 && nextSpawnMode < spawnModes.length) {
        spawnMode.value = nextSpawnMode;
      }
      awaitingSpawn.value = !!data.data.awaiting_spawn;
      if (recordOpen.value) {
        pendingResultsRequests.clear();
        finishResultsRefresh();
        clearStepQueue();
        replayResultsActive.value = true;
        tableResult.value = {
          dtype: data.data.record_results_dtype || 'recorded',
          results: data.data.record_results_mode === 'embedded'
            ? (data.data.record_results || {})
            : {},
        };
        resultsBoardHex.value = currentBoardHex.value;
        if (awaitingSpawn.value) {
          demoActive.value = false;
          clearDemoTimer();
        } else if (demoActive.value) {
          if (recordPlaybackAtEnd.value) {
            demoActive.value = false;
            clearDemoTimer();
          } else {
            scheduleDemoStep();
          }
        }
        return;
      }

      replayResultsActive.value = false;
      if (boardChanged) {
        invalidateResults();
      }
      if (awaitingSpawn.value) {
        finishResultsRefresh();
        demoActive.value = false;
        clearDemoTimer();
        clearStepQueue();
      } else if (boardChanged) {
        stepExecutionPending.value = false;
        queryResults(queuedStepCount.value > 0 || demoActive.value ? 'step' : 'auto');
      }
      maybeAutoApplyDefaultTablebase();
      return;
    }

    if (data.action === 'TRAINER_RESULTS' || (
      data.action === 'TABLEBASE_QUERY_RESULT' && data.data?.page === 'trainer'
    )) {
      const requestId = data.data.request_id;
      const resultBoardHex = data.data.board_hex || currentBoardHex.value;
      if (requestId && pendingResultsRequests.has(requestId)) {
        pendingResultsRequests.delete(requestId);
      }
      if (data.data?.code) {
        if (
          data.data.code === 'REMOTE_TABLEBASE_OFFLINE'
          || data.data.code === 'REMOTE_TABLEBASE_TIMEOUT'
        ) {
          tablebasePath.value = 'temporarily_unavailable';
          clearTablebaseResultCache();
          invalidateResults({ clearDisplay: true });
        }
        if (resultBoardHex === currentBoardHex.value) finishResultsRefresh();
        return;
      }
      if (resultBoardHex !== currentBoardHex.value) {
        return;
      }
      if (recordOpen.value) {
        finishResultsRefresh();
        return;
      }
      const resultCatalogVersion = data.data.catalog_version || catalogVersion.value;
      const resultFullPattern = data.data.full_pattern
        || loadedTablebaseFullPattern.value
        || currentPatternDisplay.value;
      const activeFullPattern = loadedTablebaseFullPattern.value || currentPatternDisplay.value;
      if (resultCatalogVersion && resultCatalogVersion !== catalogVersion.value) {
        loadCatalog();
        return;
      }
      if (resultFullPattern && activeFullPattern && resultFullPattern !== activeFullPattern) {
        return;
      }
      if (resultCatalogVersion && resultFullPattern && resultBoardHex) {
        setCachedTablebaseResult({
          catalogVersion: resultCatalogVersion,
          fullPattern: resultFullPattern,
          boardHex: resultBoardHex,
        }, data.data);
      }
      finishResultsRefresh();
      replayResultsActive.value = false;
      tableResult.value = {
        dtype: data.data.dtype || '?',
        results: data.data.results || {},
      };
      resultsBoardHex.value = resultBoardHex;
      if (!hasPlayableResults.value && !recordPlaybackActive.value) {
        demoActive.value = false;
        clearDemoTimer();
        clearStepQueue();
        return;
      }
      if (demoActive.value && !queuedStepCount.value && !stepExecutionPending.value) {
        scheduleDemoStep();
      }
      if (!stepExecutionPending.value || queuedStepCount.value > 0) {
        pumpQueuedSteps();
      }
      return;
    }

    if (data.action === 'DO_AI_MOVE_CMD') {
      if (data.data.dir) {
        stepExecutionPending.value = true;
        triggerAction('TRAINER_MOVE', { dir: data.data.dir });
      }
      return;
    }

    if (data.action === 'TRAINER_STEP_FAILED') {
      demoActive.value = false;
      clearStepQueue();
      return;
    }

    if (data.action === 'TABLEBASE_PREFETCH' && data.data?.page === 'trainer') {
      const resultCatalogVersion = String(data.data.catalog_version || '');
      const resultFullPattern = String(data.data.full_pattern || '');
      const activeFullPattern = loadedTablebaseFullPattern.value || currentPatternDisplay.value;
      if (
        !resultCatalogVersion
        || !resultFullPattern
        || resultCatalogVersion !== catalogVersion.value
        || resultFullPattern !== activeFullPattern
      ) {
        return;
      }
      for (const entry of data.data.entries || []) {
        if (!entry?.board_hex) continue;
        setCachedTablebaseResult({
          catalogVersion: resultCatalogVersion,
          fullPattern: resultFullPattern,
          boardHex: entry.board_hex,
        }, entry);
      }
      return;
    }

    if (data.action === 'TABLEBASE_BUSY' && data.data?.page === 'trainer') {
      const requestId = data.data?.query_id;
      if (requestId) pendingResultsRequests.delete(requestId);
      const retryBoardHex = String(data.data?.board_hex || currentBoardHex.value || '');
      const retryAfterMs = Math.max(250, Number(data.data?.retry_after_ms) || 750);
      if (retryBoardHex && retryBoardHex === currentBoardHex.value) {
        if (tablebaseRetryTimer) window.clearTimeout(tablebaseRetryTimer);
        tablebaseRetryTimer = window.setTimeout(() => {
          tablebaseRetryTimer = null;
          if (retryBoardHex === currentBoardHex.value) queryResults('auto');
        }, retryAfterMs);
      } else {
        finishResultsRefresh();
      }
      return;
    }

    if (data.action === 'TABLEBASE_CATALOG_UPDATED') {
      loadCatalog({ preserveSelection: true });
      return;
    }

  };

  const connect = () => {
    if (client) {
      return;
    }
    client = createWsClient({
      clientId,
      onOpen: () => {
        wsStatus.value = 'connected';
        initialStateSeen = false;
        defaultTablebaseAutoApplyAttempted = false;
        loadCatalog({ preserveSelection: true });
        triggerAction('GET_STATE');
        applyTrainerJump();
      },
      onMessage: handleMessage,
      onClose: () => {
        wsStatus.value = 'disconnected';
        demoActive.value = false;
        clearDemoTimer();
        finishResultsRefresh();
        pendingResultsRequests.clear();
        clearStepQueue();
      },
    });
    wsStatus.value = 'connecting';
    client.connect();
  };

  const disconnect = () => {
    demoActive.value = false;
    clearDemoTimer();
    finishResultsRefresh();
    pendingResultsRequests.clear();
    clearStepQueue();
    if (tablebaseRetryTimer) {
      window.clearTimeout(tablebaseRetryTimer);
      tablebaseRetryTimer = null;
    }
    client?.disconnect();
    client = null;
    wsStatus.value = 'disconnected';
  };

  const setBoard = () => {
    if (!hexInput.value) return;
    demoActive.value = false;
    clearDemoTimer();
    finishResultsRefresh();
    clearStepQueue();
    triggerAction('SET_BOARD', { hex_str: hexInput.value });
  };

  const TILE_SEQUENCE = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

  const getCycledTileValue = (cellVal, delta) => {
    const idx = TILE_SEQUENCE.indexOf(cellVal);
    if (idx < 0) {
      return cellVal;
    }
    const nextIndex = (idx + delta + TILE_SEQUENCE.length) % TILE_SEQUENCE.length;
    return TILE_SEQUENCE[nextIndex];
  };

  const handleCellClick = (row, col, btn) => {
    if (awaitingSpawn.value) {
      const cellVal = board.value[row * 4 + col];
      if (cellVal === 0) {
        const spawnVal = (btn === 2) ? 4 : 2;
        awaitingSpawn.value = false;
        demoActive.value = false;
        clearDemoTimer();
        clearStepQueue();
        triggerAction('TRAINER_MANUAL_SPAWN', { row, col, val: spawnVal });
      }
      return;
    }

    if (currentPaletteValue.value === null) {
      return;
    }

    if (btn === 0) {
      demoActive.value = false;
      clearDemoTimer();
      clearStepQueue();
      triggerAction('SET_CELL', { row, col, val: currentPaletteValue.value });
    } else if (btn === 2) {
      const cellVal = board.value[row * 4 + col];
      const nextVal = getCycledTileValue(cellVal, 1);
      demoActive.value = false;
      clearDemoTimer();
      clearStepQueue();
      triggerAction('SET_CELL', { row, col, val: nextVal });
    } else {
      const cellVal = board.value[row * 4 + col];
      const prevVal = getCycledTileValue(cellVal, -1);
      demoActive.value = false;
      clearDemoTimer();
      clearStepQueue();
      triggerAction('SET_CELL', { row, col, val: prevVal });
    }
  };

  const onPatternChange = () => {
    demoActive.value = false;
    clearDemoTimer();
    clearStepQueue();
    if (!patternType.value || !targetValue.value) return;
    applyTablebase({ loadDefault: true });
  };

  const selectFolder = () => applyTablebase({ loadDefault: true });

  const applyTablebase = ({ loadDefault = false } = {}) => {
    if (!patternType.value || !targetValue.value) return;
    const fullPattern = `${patternType.value}_${targetValue.value}`;
    clearTablebaseResultCache();
    recordPlaybackLoaded.value = false;
    replayResultsActive.value = false;
    pendingResultsRequests.clear();
    invalidateResults({ clearDisplay: true });
    startResultsRefresh();
    triggerAction('TRAINER_SET_FILEPATH', {
      pattern: fullPattern,
      target: targetValue.value,
      load_default: loadDefault,
    });
  };

  const trainerStep = () => {
    if (recordPlaybackActive.value) {
      playRecordStep(1);
      return;
    }

    queuedStepCount.value += 1;
    pumpQueuedSteps();
  };

  const playRecordStep = (delta, { fromDemo = false } = {}) => {
    if (!recordPlaybackActive.value) return false;
    if ((delta > 0 && recordPlaybackAtEnd.value) || (delta < 0 && recordStep.value <= 0)) {
      if (fromDemo) {
        demoActive.value = false;
        clearDemoTimer();
        clearStepQueue();
      }
      return true;
    }
    triggerAction('RECORD_STEP', { dir: delta });
    return true;
  };

  const trainerUndo = () => {
    demoActive.value = false;
    clearDemoTimer();
    clearStepQueue();
    triggerAction('UNDO');
  };

  const trainerDefault = () => {
    demoActive.value = false;
    clearDemoTimer();
    clearStepQueue();
    triggerAction('TRAINER_DEFAULT');
  };

  const toggleDemo = () => {
    demoActive.value = !demoActive.value;
    if (demoActive.value) {
      if (recordPlaybackActive.value) {
        scheduleDemoStep();
      } else if (resultsBoardHex.value === currentBoardHex.value && hasPlayableResults.value) {
        scheduleDemoStep();
      } else {
        queryResults('step');
      }
    } else {
      clearDemoTimer();
      clearStepQueue();
    }
  };

  const setSpawnMode = (mode) => {
    spawnMode.value = mode;
    triggerAction('SET_SPAWN_MODE', { mode });
  };

  const manageRecord = async (cmd) => {
    if (cmd === 'TOGGLE' || cmd === 'OPEN') {
      demoActive.value = false;
      clearDemoTimer();
      clearStepQueue();
    } else if (cmd === 'PREV' || cmd === 'NEXT') {
      demoActive.value = false;
      clearDemoTimer();
      clearStepQueue();
      const delta = cmd === 'PREV' ? -1 : 1;
      playRecordStep(delta);
    }
  };

  const moveBoard = (dir) => {
    const normalized = String(dir || '').toLowerCase();
    if (!['up', 'down', 'left', 'right'].includes(normalized) || awaitingSpawn.value) return;
    demoActive.value = false;
    clearDemoTimer();
    clearStepQueue();
    triggerAction('TRAINER_MOVE', { dir: normalized });
  };

  const handleKeydown = (event) => {
    if (!activeRef?.value) return;
    if (event.code === 'Escape' && patternMenuOpen.value) {
      patternMenuOpen.value = false;
      return;
    }
    const target = event.target;
    if (target instanceof HTMLElement) {
      if (target.closest('[data-trainer-hex-input="true"]')) {
        if (event.code === 'Enter') {
          event.preventDefault();
          setBoard();
        }
        return;
      }
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'SELECT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }
    }

    const map = {
      ArrowUp: 'up', KeyW: 'up',
      ArrowDown: 'down', KeyS: 'down',
      ArrowLeft: 'left', KeyA: 'left',
      ArrowRight: 'right', KeyD: 'right',
    };
    if (map[event.code]) {
      event.preventDefault();
      moveBoard(map[event.code]);
    } else if (event.code === 'Backspace' || event.code === 'Delete') {
      event.preventDefault();
      demoActive.value = false;
      clearDemoTimer();
      clearStepQueue();
      triggerAction('UNDO');
    } else if (event.code === 'KeyE') {
      event.preventDefault();
      togglePalette(0);
    } else if (event.code === 'KeyQ') {
      event.preventDefault();
      setSpawnMode(spawnMode.value === 3 ? 0 : 3);
    } else if (event.code === 'Enter') {
      event.preventDefault();
      if (demoActive.value) {
        demoActive.value = false;
        clearDemoTimer();
      } else {
        trainerStep();
      }
    }
  };

  const preventCtx = (event) => event.preventDefault();

  const onDis32kChange = () => {
    saveSetting('dis_32k', dis32k.value);
  };

  watch(showResults, (enabled) => {
    if (enabled) {
      queryResults('visibility');
    }
  });

  watch(
    appCategories,
    (nextCategories) => {
      if (catalogTables.value.length) return;
      patternCategories.value = Object.keys(nextCategories || {}).length > 0
        ? nextCategories
        : fallbackPatternCategories;
      if (patternType.value && !flatPatterns.value.includes(patternType.value)) {
        patternType.value = '';
      }
      syncActivePatternCategory();
      applyTrainerJump();
    },
    { immediate: true, deep: true }
  );

  watch(
    appTargetTiles,
    (nextTargets) => {
      if (catalogTables.value.length) return;
      const normalizedTargets = (nextTargets || []).map(String);
      if (normalizedTargets.length > 0) {
        availableTargets.value = normalizedTargets;
      }
      ensureTargetForCurrentPattern();
      applyTrainerJump();
    },
    { immediate: true, deep: true }
  );

  watch(
    () => appConfig.value.dis_32k,
    (value) => {
      dis32k.value = !!value;
    },
    { immediate: true }
  );

  watch(
    () => appConfig.value.demo_speed,
    (value) => {
      demoSpeed.value = Number(value) || 40;
    },
    { immediate: true }
  );

  watch(demoSpeed, () => {
    if (demoActive.value && !queuedStepCount.value && !stepExecutionPending.value) {
      scheduleDemoStep();
    }
  });

  onMounted(() => {
    syncActivePatternCategory();
    window.addEventListener('keydown', handleKeydown);
    window.addEventListener('trainer-practice-jump', handleTrainerPracticeJump);
    document.addEventListener('click', closePatternMenuOnClick);
    document.addEventListener('contextmenu', preventCtx);
  });

  watch(
    activeRef,
    (isActive) => {
      if (isActive) {
        connect();
        refreshSettings();
        loadCatalog();
        maybeAutoApplyDefaultTablebase();
      }
    },
    { immediate: true }
  );

  watch(isAuthenticated, (authenticated) => {
    if (authenticated) {
      maybeAutoApplyDefaultTablebase();
    }
  });

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown);
    window.removeEventListener('trainer-practice-jump', handleTrainerPracticeJump);
    document.removeEventListener('click', closePatternMenuOnClick);
    document.removeEventListener('contextmenu', preventCtx);
    disconnect();
  });

  return {
    currentPatternDisplay,
    isVariant,
    wsStatus,
    tablebasePath,
    togglePatternMenu,
    selectPattern,
    patternType,
    patternMenuOpen,
    patternGroups,
    activePatternCategory,
    activePatternOptions,
    targetValue,
    availableTargets,
    availableTargetsForPattern,
    onPatternChange,
    selectFolder,
    applyTablebase,
    hexInput,
    setBoard,
    board,
    metadata,
    dis32k,
    handleCellClick,
    awaitingSpawn,
    currentPaletteValue,
    togglePalette,
    cellPalette,
    replayResultsActive,
    showResults,
    queryResults,
    sortedResults,
    displayedResults,
    resultsRefreshing,
    resultsUpdatingVisible,
    getResultRowStyle,
    dirLabels,
    getResultValueStyle,
    tableResult,
    toggleDemo,
    demoActive,
    trainerStep,
    trainerUndo,
    trainerDefault,
    moveBoard,
    spawnModes,
    spawnMode,
    setSpawnMode,
    triggerAction,
    recordStep,
    recordMax,
    recordingState,
    manageRecord,
    onDis32kChange,
    patternMenuRoot,
  };
}
