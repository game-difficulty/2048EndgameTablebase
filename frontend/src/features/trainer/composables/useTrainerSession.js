import { computed, onMounted, onUnmounted, ref, watch } from 'vue';

import { useAppSettingsStore } from '../../../app/useAppSettings';
import {
  createSnapshotBoardFrame,
  createTransitionBoardFrame,
} from '../../../components/boardFrame.js';
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
import {
  buildTesterPrefetchPayload as buildTrainerPrefetchPayload,
  createTesterPrefetchState as createTrainerPrefetchState,
  createTesterSpawnRandomSource as createTrainerSpawnRandomSource,
} from '../../../services/tablebases/testerPrefetchRng';
import { createWsClient } from '../../../services/ws/createWsClient';
import { getStableWsClientId } from '../../../services/ws/clientIds';
import { isVariantPattern } from '../../../utils/patternCategories';
import { createResultBarGradient } from '../../../utils/resultBars';
import {
  canApplyPracticeSeed,
  createPracticeSession,
  reducePracticeSession,
  selectRandomSpawn,
} from '../../practice/engine/practiceSession.js';
import {
  restoreSuccessRate,
  formatSuccessRate,
  successRateSortValue,
  successRateRelativeLoss,
  resultValueFontSize,
} from '../../../utils/successRate';
import {
  buildTrainerBoardEdit,
  normalizeTrainerBoardHex,
  transformTrainerBoard,
} from '../engine/trainerBoardState.js';
import {
  EMPTY_PATTERN_CATEGORY,
  EMPTY_PATTERN_ID,
  isEmptyTrainerPattern,
  stripTrainerQueryPayload,
  withEmptyPatternGroup,
} from '../engine/trainerEmptyPattern.js';
import { registerTrainerPracticeJumpConsumer } from '../services/trainerPracticeJump';
import {
  clearTrainerPracticeState,
  restoreTrainerPracticeState,
  saveTrainerPracticeState,
} from '../services/trainerPracticeStore.js';

export function useTrainerSession(activeRef, hotkeysEnabledRef = activeRef) {
  const RESULT_REFRESH_GRACE_MS = 180;
  const RESULT_REFRESH_PLACEHOLDER_MS = 1400;
  const DEFAULT_TABLEBASE_PATTERN = '442t';
  const DEFAULT_TABLEBASE_TARGET = '512';
  const {
    config: appConfig,
    refreshSettings,
    saveSetting,
  } = useAppSettingsStore();
  const { isAuthenticated, requireAuth, user: authUser } = useAuthState();

  const wsStatus = ref('connecting');
  const clientId = getStableWsClientId('trainer');

  const board = ref([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const boardFrame = ref(createSnapshotBoardFrame(0, board.value));
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
  const patternCategories = ref({});
  const availableTargets = ref([]);
  const catalogTables = ref([]);
  const catalogVersion = ref('');
  const patternMenuOpen = ref(false);
  const activePatternCategory = ref(EMPTY_PATTERN_CATEGORY);
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
  let paletteEditDirty = false;
  let trainerPrefetchState = createTrainerPrefetchState();
  let localPracticeSession = createPracticeSession();
  let nextTablebaseRequestId = 0;
  let pendingTablebaseRequestId = '';
  let pendingTablebaseBoardRevision = null;
  let nextSpawnRequestId = 0;
  let pendingSpawnQuery = null;
  let tablebaseReadyForConnection = false;
  const queuedMoveDirections = [];
  let persistPracticeTimer = null;
  let practiceRestoreAttempted = false;
  let unregisterTrainerPracticeJumpConsumer = null;
  const resultsRefreshPhase = ref('idle');

  const recordStep = ref(0);
  const recordMax = ref(0);
  const recordingLength = ref(0);
  const fullHistory = ref([]);
  const fullMoves = ref([]);

  let client = null;
  let boardFrameRevision = 0;
  let initialStateSeen = false;
  let defaultTablebaseAutoApplyAttempted = false;

  const spawnRate4 = () => Math.max(
    0,
    Math.min(1, Number(appConfig.value['4_spawn_rate'] ?? 0.1) || 0),
  );

  const resetTrainerPrefetchState = () => {
    trainerPrefetchState = createTrainerPrefetchState();
  };

  const persistTrainerPractice = () => {
    if (!isAuthenticated.value || !authUser.value?.id || !localPracticeSession) return false;
    return saveTrainerPracticeState({
      userId: Number(authUser.value.id),
      pattern: patternType.value,
      target: targetValue.value,
      loadedFullPattern: loadedTablebaseFullPattern.value,
      tablebaseStatus: tablebasePath.value,
      spawnMode: spawnMode.value,
      prefetchState: trainerPrefetchState,
      result: tableResult.value,
      resultsBoardHex: resultsBoardHex.value,
      practice: localPracticeSession,
    });
  };

  const persistTrainerPracticeSoon = () => {
    if (persistPracticeTimer) window.clearTimeout(persistPracticeTimer);
    persistPracticeTimer = window.setTimeout(() => {
      persistPracticeTimer = null;
      persistTrainerPractice();
    }, 50);
  };

  const syncLocalPracticeSession = ({ animate = true } = {}) => {
    board.value = [...localPracticeSession.board];
    boardFrameRevision += 1;
    boardFrame.value = animate
      ? createTransitionBoardFrame(
        boardFrameRevision,
        localPracticeSession.transition,
        localPracticeSession.board,
      )
      : createSnapshotBoardFrame(boardFrameRevision, localPracticeSession.board);
    currentBoardHex.value = localPracticeSession.boardHex;
    hexInput.value = localPracticeSession.boardHex;
    awaitingSpawn.value = localPracticeSession.phase === 'awaiting_spawn';
    fullHistory.value = localPracticeSession.history.map((entry) => entry.boardHex);
    fullMoves.value = localPracticeSession.history.map((entry) => entry.lastMove);
    recordStep.value = Math.max(0, localPracticeSession.history.length - 1);
    recordMax.value = localPracticeSession.history.length;
    if (localPracticeSession.context?.state) {
      trainerPrefetchState = localPracticeSession.context;
    }
    persistTrainerPracticeSoon();
  };

  const createLocalPracticeSession = (
    boardValue,
    { animate = false, useVariant = isVariant.value } = {},
  ) => {
    localPracticeSession = createPracticeSession({
      boardHex: normalizeTrainerBoardHex(boardValue) || '0000000000000000',
      useVariant: Boolean(useVariant),
      context: trainerPrefetchState,
    });
    syncLocalPracticeSession({ animate });
  };

  const cancelPendingLocalTurn = () => {
    if (localPracticeSession.phase !== 'awaiting_spawn') return false;
    pendingSpawnQuery = null;
    const reduced = reducePracticeSession(localPracticeSession, { type: 'UNDO' });
    if (!reduced.accepted) return false;
    localPracticeSession = reduced.state;
    syncLocalPracticeSession({ animate: false });
    return true;
  };

  const resetLocalRandomState = () => {
    resetTrainerPrefetchState();
    localPracticeSession = {
      ...localPracticeSession,
      context: trainerPrefetchState,
    };
  };

  const isEmptyPattern = computed(() => isEmptyTrainerPattern(patternType.value));
  const currentPatternDisplay = computed(() => {
    if (isEmptyPattern.value) return EMPTY_PATTERN_ID;
    return patternType.value && targetValue.value ? `${patternType.value}_${targetValue.value}` : '';
  });
  const isVariant = computed(() => isVariantPattern(patternType.value, patternCategories.value));

  const restoreTrainerPractice = () => {
    if (practiceRestoreAttempted || !authUser.value?.id) return false;
    practiceRestoreAttempted = true;
    const restored = restoreTrainerPracticeState();
    if (!restored || Number(restored.userId) !== Number(authUser.value.id)) {
      if (restored) clearTrainerPracticeState();
      return false;
    }
    patternType.value = String(restored.pattern || patternType.value);
    targetValue.value = String(restored.target || targetValue.value);
    loadedTablebaseFullPattern.value = String(restored.loadedFullPattern || '');
    tablebasePath.value = String(restored.tablebaseStatus || 'not_selected');
    spawnMode.value = Math.max(0, Math.min(3, Math.trunc(Number(restored.spawnMode) || 0)));
    trainerPrefetchState = restored.prefetchState || createTrainerPrefetchState();
    localPracticeSession = restored.practice;
    localPracticeSession = {
      ...localPracticeSession,
      useVariant: Boolean(restored.practice.useVariant),
      transition: null,
    };
    tableResult.value = restored.result && typeof restored.result === 'object'
      ? restored.result
      : { dtype: '?', results: {} };
    resultsBoardHex.value = String(restored.resultsBoardHex || '');
    syncActivePatternCategory();
    syncLocalPracticeSession({ animate: false });
    return true;
  };
  const catalogPatternGroups = computed(() =>
    Object.entries(patternCategories.value || {}).map(([category, patterns]) => ({
      category,
      patterns: Array.isArray(patterns) ? patterns : [],
    }))
  );
  const patternGroups = computed(() => withEmptyPatternGroup(catalogPatternGroups.value));
  const flatPatterns = computed(() => catalogPatternGroups.value.flatMap((group) => group.patterns));
  const activePatternOptions = computed(() => (
    activePatternCategory.value === EMPTY_PATTERN_CATEGORY
      ? [EMPTY_PATTERN_ID]
      : (patternCategories.value[activePatternCategory.value] || [])
  ));
  const availableTargetsForPattern = computed(() => {
    if (isEmptyPattern.value) return [];
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
    const previousValue = currentPaletteValue.value;
    currentPaletteValue.value = previousValue === val ? null : val;
    if (previousValue !== null && currentPaletteValue.value === null) {
      finishPaletteEditing();
    }
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
    const variantGroup = isVariantPattern(patternType.value, patternCategories.value)
      ? patternGroups.value.find((group) => group.category === 'variant')
      : null;
    activePatternCategory.value = matchedGroup?.category
      || variantGroup?.category
      || patternGroups.value[0]?.category
      || '';
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
    if (isEmptyPattern.value) {
      targetValue.value = '';
      return;
    }
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
    if (!parsed) {
      return false;
    }
    const catalogHasPattern = flatPatterns.value.includes(parsed.pattern);
    if (!catalogHasPattern) return false;
    const targets = getCatalogTargetsForPattern(catalogTables.value, parsed.pattern);
    if (!targets.includes(parsed.target)) {
      return false;
    }
    patternType.value = parsed.pattern;
    targetValue.value = parsed.target;
    syncActivePatternCategory();
    return true;
  };

  const selectedTablebaseExists = () => {
    if (
      isEmptyPattern.value
      || !patternType.value
      || !targetValue.value
      || !flatPatterns.value.includes(patternType.value)
    ) {
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
      pendingTablebaseRequestId ||
      isEmptyPattern.value ||
      tablebasePath.value === 'loaded' ||
      !selectedTablebaseExists()
    ) {
      return;
    }
    if (applyTablebase({ loadDefault: true })) {
      defaultTablebaseAutoApplyAttempted = true;
    }
  };

  const protectedActions = new Set([
    'TRAINER_SET_FILEPATH',
    'TRAINER_SET_EMPTY_PATTERN',
    'TRAINER_GET_RESULTS',
    'TABLEBASE_QUERY',
    'TRAINER_DEFAULT',
    'TRAINER_SPAWN_QUERY',
    'TRAINER_STEP',
  ]);

  const triggerAction = (action, payload = {}) => {
    if (protectedActions.has(action) && !requireAuth()) {
      return false;
    }
    client?.send(action, isEmptyPattern.value ? stripTrainerQueryPayload(payload) : payload);
    return true;
  };

  const loadCatalog = async () => {
    try {
      const previousCatalogVersion = catalogVersion.value;
      const tables = await fetchTablebaseCatalog();
      catalogTables.value = tables;
      catalogVersion.value = tables.catalogVersion || getCatalogVersion();
      if (
        previousCatalogVersion
        && catalogVersion.value
        && previousCatalogVersion !== catalogVersion.value
      ) {
        clearTablebaseResultCache();
      }
      const nextCategories = groupTablebasePatternsByCategory(tables);
      const patterns = Object.values(nextCategories).flat();
      patternCategories.value = nextCategories;
      availableTargets.value = getCatalogTargets(tables);
      if (patterns.length) {
        if (
          !patternType.value
          || (!patterns.includes(patternType.value) && !isEmptyPattern.value)
        ) {
          const defaults = chooseDefaultCatalogSelection(patterns);
          patternType.value = defaults.pattern;
          targetValue.value = defaults.target;
        }
        if (!isEmptyPattern.value) ensureTargetForCurrentPattern();
        if (loadedTablebaseFullPattern.value) {
          syncSelectionFromFullPattern(loadedTablebaseFullPattern.value);
        }
        syncActivePatternCategory();
        applyTrainerJump();
        maybeAutoApplyDefaultTablebase();
        if (
          wsStatus.value === 'connected'
          && tablebasePath.value === 'loaded'
          && selectedTablebaseExists()
          && !tablebaseReadyForConnection
          && !pendingTablebaseRequestId
        ) {
          applyTablebase({ loadDefault: false, preserveLocalState: true });
        }
        if (
          tablebaseReadyForConnection
          && tablebasePath.value === 'loaded'
          && currentBoardHex.value
          && resultsBoardHex.value !== currentBoardHex.value
        ) {
          queryResults('auto');
        }
      } else if (!isEmptyPattern.value) {
        patternType.value = '';
        targetValue.value = '';
        syncActivePatternCategory();
      }
    } catch (error) {
      console.error(error);
    }
  };

  const applyTrainerJump = () => {
    const pending = pendingTrainerJump.value;
    if (!pending || pending.requestId || wsStatus.value !== 'connected') return;

    const parsed = parseFullPattern(pending.fullPattern);
    clearQueuedMoveDirections();
    if (parsed) {
      const requestId = `${clientId}_table_${++nextTablebaseRequestId}`;
      patternType.value = parsed.pattern;
      targetValue.value = parsed.target;
      syncActivePatternCategory();
      pending.requestId = requestId;
      pendingTablebaseRequestId = requestId;
      pendingTablebaseBoardRevision = null;
      tablebaseReadyForConnection = false;
      if (!triggerAction('TRAINER_SET_FILEPATH', {
        request_id: requestId,
        pattern: pending.fullPattern,
        target: parsed.target,
        load_default: false,
        client_local_board: true,
        preserve_client_board: true,
        client_revision: localPracticeSession.revision,
      })) {
        pending.requestId = '';
        pendingTablebaseRequestId = '';
        return;
      }
    }

    createLocalPracticeSession(pending.hex);
    invalidateResults({ clearDisplay: true });
    if (!parsed) {
      pendingTrainerJump.value = null;
      defaultTablebaseAutoApplyAttempted = true;
      queryResults('practice-jump');
    }
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

  const activateEmptyPattern = () => {
    const alreadyEmpty = isEmptyPattern.value;
    demoActive.value = false;
    clearDemoTimer();
    clearStepQueue();
    clearQueuedMoveDirections();
    finishPaletteEditing({ query: false });
    clearPaletteSyncTimer();
    if (!alreadyEmpty) paletteEditDirty = false;
    currentPaletteValue.value = null;
    cancelPendingLocalTurn();
    resetLocalRandomState();
    pendingResultsRequests.clear();
    loadedTablebaseFullPattern.value = '';
    tablebasePath.value = 'not_selected';
    localPracticeSession = { ...localPracticeSession, useVariant: false };
    patternType.value = EMPTY_PATTERN_ID;
    targetValue.value = '';
    activePatternCategory.value = EMPTY_PATTERN_CATEGORY;
    patternMenuOpen.value = false;
    if ([1, 2].includes(Number(spawnMode.value))) {
      spawnMode.value = 0;
    }
    invalidateResults({ clearDisplay: true });
    finishResultsRefresh();
    const requestId = `${clientId}_table_${++nextTablebaseRequestId}`;
    pendingTablebaseRequestId = requestId;
    pendingTablebaseBoardRevision = null;
    tablebaseReadyForConnection = false;
    if (!triggerAction('TRAINER_SET_EMPTY_PATTERN', {
      request_id: requestId,
      client_local_board: true,
      client_revision: localPracticeSession.revision,
    })) pendingTablebaseRequestId = '';
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
    if (isEmptyTrainerPattern(pattern)) {
      activateEmptyPattern();
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

  const applyCachedResultsForBoard = (targetBoardHex, { clearOnMiss = false } = {}) => {
    const fullPattern = loadedTablebaseFullPattern.value || currentPatternDisplay.value;
    const version = catalogVersion.value || getCatalogVersion();
    const cached = version && fullPattern
      ? getCachedTablebaseResult({
        catalogVersion: version,
        fullPattern,
        boardHex: targetBoardHex,
      })
      : null;
    if (cached) {
      tableResult.value = {
        dtype: cached.dtype || '?',
        results: cached.results || {},
      };
      resultsBoardHex.value = targetBoardHex;
      finishResultsRefresh();
      if (queuedStepCount.value > 0 || demoActive.value) {
        window.queueMicrotask(pumpQueuedSteps);
      }
      return true;
    }
    if (clearOnMiss) {
      invalidateResults({ clearDisplay: true });
      finishResultsRefresh();
    }
    return false;
  };

  const trackResultsRequest = (requestId, targetBoardHex, reason) => {
    if (tablebaseRetryTimer) {
      window.clearTimeout(tablebaseRetryTimer);
      tablebaseRetryTimer = null;
    }
    pendingResultsRequests.clear();
    pendingResultsRequests.set(requestId, { boardHex: targetBoardHex, reason });
    const fullPattern = loadedTablebaseFullPattern.value || currentPatternDisplay.value;
    const version = catalogVersion.value || getCatalogVersion();
    const cacheHit = applyCachedResultsForBoard(targetBoardHex);
    if (!cacheHit) {
      startResultsRefresh();
    }
    return { fullPattern, version, cacheHit };
  };

  const prepareResultsRequest = (boardHex, reason = 'manual') => {
    if (isEmptyPattern.value) return null;
    if (recordOpen.value || (reason !== 'step' && !showResults.value) || awaitingSpawn.value) return null;
    if (wsStatus.value !== 'connected' || !tablebaseReadyForConnection) return null;
    if (!boardHex) return null;
    if (hasPendingResultsForBoard(boardHex)) return null;
    if (!isAuthenticated.value) {
      if (reason === 'auto') return null;
      if (!requireAuth()) return null;
    }

    const requestId = `${clientId}_${++nextResultsRequestId}`;
    const { fullPattern, version, cacheHit } = trackResultsRequest(requestId, boardHex, reason);
    const prefetchRng = Number(spawnMode.value) === 0
      ? buildTrainerPrefetchPayload(trainerPrefetchState, spawnRate4())
      : null;
    return { requestId, fullPattern, version, cacheHit, prefetchRng };
  };

  const queryResults = (reason = 'manual') => {
    if (isEmptyPattern.value) return null;
    const boardHex = currentBoardHex.value || hexInput.value;
    const prepared = prepareResultsRequest(boardHex, reason);
    if (!prepared) return null;
    triggerAction('TABLEBASE_QUERY', {
      page: 'trainer',
      client_local_board: true,
      query_id: prepared.requestId,
      catalog_version: prepared.version,
      full_pattern: prepared.fullPattern,
      board_hex: boardHex,
      prefetch_rng: prepared.prefetchRng,
    });
    return prepared.requestId;
  };

  const clearStepQueue = () => {
    queuedStepCount.value = 0;
    stepExecutionPending.value = false;
  };

  const clearQueuedMoveDirections = () => {
    queuedMoveDirections.length = 0;
  };

  const requestSpawnForCurrentBoard = (mode = Number(spawnMode.value)) => {
    if (![1, 2].includes(Number(mode)) || localPracticeSession.phase !== 'awaiting_spawn') return false;
    const requestId = `${clientId}_spawn_${++nextSpawnRequestId}`;
    pendingSpawnQuery = {
      requestId,
      revision: localPracticeSession.revision,
    };
    const sent = triggerAction('TRAINER_SPAWN_QUERY', {
      request_id: requestId,
      revision: localPracticeSession.revision,
      board_hex: localPracticeSession.boardHex,
      full_pattern: loadedTablebaseFullPattern.value || currentPatternDisplay.value,
      mode: Number(mode),
    });
    if (!sent) pendingSpawnQuery = null;
    return sent;
  };

  const executeTrainerMove = (direction) => {
    const normalized = String(direction || '').toLowerCase();
    const currentSpawnMode = Number(spawnMode.value) || 0;
    const currentHex = currentBoardHex.value || hexInput.value;
    if (
      !['up', 'down', 'left', 'right'].includes(normalized)
      || awaitingSpawn.value
      || (currentSpawnMode !== 0 && currentSpawnMode !== 3 && wsStatus.value !== 'connected')
      || !requireAuth()
    ) {
      return false;
    }
    const deterministicSpawn = currentSpawnMode === 0
      ? createTrainerSpawnRandomSource(trainerPrefetchState)
      : null;
    const reduced = reducePracticeSession(localPracticeSession, {
      type: currentSpawnMode === 0 ? 'MOVE_RANDOM' : 'MOVE_ONLY',
      direction: normalized,
      spawnRate4: spawnRate4(),
      randomSource: deterministicSpawn?.randomSource,
      nextContext: deterministicSpawn?.nextState,
    });
    if (!reduced.accepted || !currentHex) {
      stepExecutionPending.value = false;
      return false;
    }
    localPracticeSession = reduced.state;
    syncLocalPracticeSession();
    invalidateResults();

    const queryReason = queuedStepCount.value > 0 || demoActive.value ? 'step' : 'auto';
    const preparedQuery = currentSpawnMode === 0
      ? prepareResultsRequest(currentBoardHex.value, queryReason)
      : null;
    if (preparedQuery) {
      triggerAction('TABLEBASE_QUERY', {
        page: 'trainer',
        client_local_board: true,
        query_id: preparedQuery.requestId,
        catalog_version: preparedQuery.version,
        full_pattern: preparedQuery.fullPattern,
        board_hex: currentBoardHex.value,
        prefetch_rng: preparedQuery.prefetchRng,
      });
    } else if (currentSpawnMode === 1 || currentSpawnMode === 2) requestSpawnForCurrentBoard(currentSpawnMode);
    if (currentSpawnMode === 0 && preparedQuery?.cacheHit) {
      stepExecutionPending.value = false;
      if (demoActive.value) {
        scheduleDemoStep();
      } else if (queuedStepCount.value > 0) {
        window.queueMicrotask(pumpQueuedSteps);
      }
    }
    return true;
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
    executeTrainerMove(move);
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

    if (data.action === 'TRAINER_BOARD_SYNCED' || data.action === 'TRAINER_MOVE_ACCEPTED') return;

    if (data.action === 'TRAINER_TABLEBASE_READY') {
      const response = data.data || {};
      const requestId = String(response.request_id || '');
      if (requestId && requestId !== pendingTablebaseRequestId) return;
      const completedJump = !!requestId
        && pendingTrainerJump.value?.requestId === requestId;
      const requestedBoardRevision = pendingTablebaseBoardRevision;
      pendingTablebaseRequestId = '';
      pendingTablebaseBoardRevision = null;
      tablebaseReadyForConnection = true;
      tablebasePath.value = String(response.tablebase_status || 'not_selected');
      loadedTablebaseFullPattern.value = String(response.tablebase_full_pattern || '');
      if (loadedTablebaseFullPattern.value) syncSelectionFromFullPattern(loadedTablebaseFullPattern.value);
      localPracticeSession = {
        ...localPracticeSession,
        useVariant: Boolean(response.use_variant),
      };
      if (
        response.board_hex
        && requestedBoardRevision != null
        && canApplyPracticeSeed(
          localPracticeSession,
          requestedBoardRevision,
          response.client_revision,
        )
      ) {
        resetTrainerPrefetchState();
        createLocalPracticeSession(String(response.board_hex), {
          useVariant: Boolean(response.use_variant),
        });
      }
      if (completedJump) {
        pendingTrainerJump.value = null;
        defaultTablebaseAutoApplyAttempted = true;
      }
      finishResultsRefresh();
      persistTrainerPracticeSoon();
      if (
        tablebasePath.value === 'loaded'
        && localPracticeSession.phase === 'awaiting_spawn'
        && [1, 2].includes(Number(spawnMode.value))
      ) requestSpawnForCurrentBoard();
      else if (tablebasePath.value === 'loaded' && currentBoardHex.value) queryResults('tablebase-ready');
      else invalidateResults({ clearDisplay: true });
      maybeAutoApplyDefaultTablebase();
      return;
    }

    if (data.action === 'TRAINER_SPAWN_RESULT') {
      const response = data.data || {};
      const pending = pendingSpawnQuery;
      if (
        !pending
        || String(response.request_id || '') !== pending.requestId
        || Number(response.revision) !== pending.revision
        || String(response.board_hex || '').toLowerCase() !== localPracticeSession.boardHex
        || localPracticeSession.revision !== pending.revision
      ) return;

      pendingSpawnQuery = null;
      let spawn = response.found
        ? { index: Number(response.index), value: Number(response.value) }
        : null;
      let nextContext = localPracticeSession.context;
      if (!spawn) {
        const randomSpawn = createTrainerSpawnRandomSource(localPracticeSession.context || trainerPrefetchState);
        spawn = selectRandomSpawn(localPracticeSession.board, spawnRate4(), randomSpawn.randomSource);
        nextContext = randomSpawn.nextState();
      }
      if (!spawn) return;
      const reduced = reducePracticeSession(localPracticeSession, {
        type: 'SPAWN',
        expectedRevision: pending.revision,
        index: spawn.index,
        value: spawn.value,
        nextContext,
      });
      if (!reduced.accepted) return;
      localPracticeSession = reduced.state;
      syncLocalPracticeSession();
      invalidateResults();
      queryResults(queuedStepCount.value > 0 || demoActive.value ? 'step' : 'auto');
      window.queueMicrotask(flushQueuedTrainerMove);
      return;
    }

    if (data.action === 'TRAINER_RESULTS' || (
      data.action === 'TABLEBASE_QUERY_RESULT' && data.data?.page === 'trainer'
    )) {
      if (isEmptyPattern.value) return;
      const requestId = data.data.query_id || data.data.request_id;
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
        if (resultBoardHex === currentBoardHex.value) {
          stepExecutionPending.value = false;
          finishResultsRefresh();
        }
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
      stepExecutionPending.value = false;
      persistTrainerPracticeSoon();
      if (!hasPlayableResults.value && !recordPlaybackActive.value) {
        demoActive.value = false;
        clearDemoTimer();
        clearStepQueue();
        return;
      }
      if (
        demoActive.value
        && !queuedStepCount.value
        && !stepExecutionPending.value
        && !demoTimer
      ) {
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
        executeTrainerMove(data.data.dir);
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
      loadCatalog();
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
        tablebaseReadyForConnection = false;
        initialStateSeen = true;
        defaultTablebaseAutoApplyAttempted = false;
        loadCatalog();
        const reattachingTablebase = !isEmptyPattern.value
          && selectedTablebaseExists()
          && tablebasePath.value === 'loaded';
        if (reattachingTablebase) {
          applyTablebase({ loadDefault: false, preserveLocalState: true });
        } else {
          maybeAutoApplyDefaultTablebase();
        }
        applyTrainerJump();
      },
      onMessage: handleMessage,
      onClose: () => {
        wsStatus.value = 'disconnected';
        tablebaseReadyForConnection = false;
        pendingTablebaseRequestId = '';
        pendingTablebaseBoardRevision = null;
        if (pendingTrainerJump.value) {
          pendingTrainerJump.value.requestId = '';
        }
        demoActive.value = false;
        clearDemoTimer();
        finishResultsRefresh();
        pendingResultsRequests.clear();
        clearStepQueue();
        clearQueuedMoveDirections();
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
    pendingSpawnQuery = null;
    clearStepQueue();
    clearQueuedMoveDirections();
    if (tablebaseRetryTimer) {
      window.clearTimeout(tablebaseRetryTimer);
      tablebaseRetryTimer = null;
    }
    client?.disconnect();
    client = null;
    wsStatus.value = 'disconnected';
  };

  const applyLocalBoardSnapshot = (nextBoardHex, { clearOnCacheMiss = true } = {}) => {
    const normalized = normalizeTrainerBoardHex(nextBoardHex);
    if (!normalized) return false;
    const reduced = reducePracticeSession(localPracticeSession, {
      type: 'SET_BOARD',
      boardHex: normalized,
      nextContext: trainerPrefetchState,
    });
    if (!reduced.accepted) return false;
    localPracticeSession = reduced.state;
    syncLocalPracticeSession({ animate: false });
    replayResultsActive.value = false;
    pendingResultsRequests.clear();
    pendingSpawnQuery = null;
    stepExecutionPending.value = false;
    return applyCachedResultsForBoard(normalized, { clearOnMiss: clearOnCacheMiss });
  };

  const clearPaletteSyncTimer = () => {};

  function finishPaletteEditing({ query = true } = {}) {
    if (!paletteEditDirty) return false;
    paletteEditDirty = false;
    if (query && !isEmptyPattern.value) queryResults('palette-exit');
    return true;
  }

  const syncPendingEmptyPatternBoard = () => true;

  const setBoard = () => {
    const normalized = normalizeTrainerBoardHex(hexInput.value);
    if (!normalized || !requireAuth()) return;
    demoActive.value = false;
    clearDemoTimer();
    finishResultsRefresh();
    clearStepQueue();
    clearQueuedMoveDirections();
    clearPaletteSyncTimer();
    paletteEditDirty = false;
    currentPaletteValue.value = null;
    applyLocalBoardSnapshot(normalized);
    if (isEmptyPattern.value) {
      paletteEditDirty = true;
      return;
    }
    queryResults('set-board');
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
        const spawnIndex = row * 4 + col;
        demoActive.value = false;
        clearDemoTimer();
        clearStepQueue();
        clearQueuedMoveDirections();
        pendingSpawnQuery = null;
        const reduced = reducePracticeSession(localPracticeSession, {
          type: 'SPAWN',
          expectedRevision: localPracticeSession.revision,
          index: spawnIndex,
          value: spawnVal,
        });
        if (!reduced.accepted) return;
        localPracticeSession = reduced.state;
        syncLocalPracticeSession();
        invalidateResults();
        queryResults('auto');
      }
      return;
    }

    if (currentPaletteValue.value === null) {
      return;
    }

    if (!requireAuth()) return;
    const cellIndex = row * 4 + col;
    const cellVal = board.value[cellIndex];
    const nextVal = btn === 0
      ? currentPaletteValue.value
      : getCycledTileValue(cellVal, btn === 2 ? 1 : -1);
    if (nextVal === cellVal) return;

    demoActive.value = false;
    clearDemoTimer();
    clearStepQueue();
    clearQueuedMoveDirections();
    const edit = buildTrainerBoardEdit(board.value, row, col, nextVal);
    if (!edit) return;
    applyLocalBoardSnapshot(edit.boardHex);
    paletteEditDirty = true;
    invalidateResults();
  };

  const onPatternChange = () => {
    demoActive.value = false;
    clearDemoTimer();
    clearStepQueue();
    finishPaletteEditing({ query: false });
    paletteEditDirty = false;
    currentPaletteValue.value = null;
    if (isEmptyPattern.value) {
      activateEmptyPattern();
      return;
    }
    if (!patternType.value || !targetValue.value) return;
    applyTablebase({ loadDefault: true });
  };

  const selectFolder = () => applyTablebase({ loadDefault: true });

  const applyTablebase = ({ loadDefault = false, preserveLocalState = false } = {}) => {
    if (isEmptyPattern.value || !patternType.value || !targetValue.value) return;
    const fullPattern = `${patternType.value}_${targetValue.value}`;
    if (!preserveLocalState) {
      clearQueuedMoveDirections();
      cancelPendingLocalTurn();
      resetLocalRandomState();
      createLocalPracticeSession(currentBoardHex.value || '0000000000000000');
      clearTablebaseResultCache();
    }
    recordPlaybackLoaded.value = false;
    replayResultsActive.value = false;
    pendingResultsRequests.clear();
    if (!preserveLocalState) {
      invalidateResults({ clearDisplay: true });
      startResultsRefresh();
    }
    const requestId = `${clientId}_table_${++nextTablebaseRequestId}`;
    pendingTablebaseRequestId = requestId;
    pendingTablebaseBoardRevision = loadDefault ? localPracticeSession.revision : null;
    tablebaseReadyForConnection = false;
    const sent = triggerAction('TRAINER_SET_FILEPATH', {
      request_id: requestId,
      pattern: fullPattern,
      target: targetValue.value,
      load_default: loadDefault,
      client_local_board: true,
      preserve_client_board: !loadDefault,
      client_revision: localPracticeSession.revision,
    });
    if (!sent) {
      pendingTablebaseRequestId = '';
      pendingTablebaseBoardRevision = null;
    }
    return sent;
  };

  const trainerStep = () => {
    if (isEmptyPattern.value) return;
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
    clearQueuedMoveDirections();
    if (!requireAuth()) return;
    pendingSpawnQuery = null;
    const reduced = reducePracticeSession(localPracticeSession, { type: 'UNDO' });
    if (!reduced.accepted) return;
    localPracticeSession = reduced.state;
    syncLocalPracticeSession({ animate: false });
    invalidateResults();
    if (!isEmptyPattern.value) queryResults('undo');
  };

  const trainerDefault = () => {
    if (isEmptyPattern.value) return;
    demoActive.value = false;
    clearDemoTimer();
    clearStepQueue();
    clearQueuedMoveDirections();
    resetLocalRandomState();
    const requestId = `${clientId}_table_${++nextTablebaseRequestId}`;
    pendingTablebaseRequestId = requestId;
    pendingTablebaseBoardRevision = localPracticeSession.revision;
    if (!triggerAction('TRAINER_DEFAULT', {
      request_id: requestId,
      client_local_board: true,
      client_revision: localPracticeSession.revision,
    })) {
      pendingTablebaseRequestId = '';
      pendingTablebaseBoardRevision = null;
    }
  };

  const toggleDemo = () => {
    if (isEmptyPattern.value) return;
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
    if (isEmptyPattern.value && [1, 2].includes(Number(mode))) return;
    const normalizedMode = Math.max(0, Math.min(3, Math.trunc(Number(mode) || 0)));
    clearQueuedMoveDirections();
    spawnMode.value = normalizedMode;
    persistTrainerPracticeSoon();
    if (localPracticeSession.phase !== 'awaiting_spawn') return;
    pendingSpawnQuery = null;
    if (normalizedMode === 0) {
      const randomSpawn = createTrainerSpawnRandomSource(localPracticeSession.context || trainerPrefetchState);
      const spawn = selectRandomSpawn(localPracticeSession.board, spawnRate4(), randomSpawn.randomSource);
      if (!spawn) return;
      const reduced = reducePracticeSession(localPracticeSession, {
        type: 'SPAWN',
        index: spawn.index,
        value: spawn.value,
        nextContext: randomSpawn.nextState(),
      });
      if (!reduced.accepted) return;
      localPracticeSession = reduced.state;
      syncLocalPracticeSession();
      queryResults('auto');
    } else if (normalizedMode === 1 || normalizedMode === 2) {
      requestSpawnForCurrentBoard(normalizedMode);
    }
  };

  const transformBoard = (type) => {
    const transformed = transformTrainerBoard(localPracticeSession.board, type);
    if (!transformed) return false;
    const reduced = reducePracticeSession(localPracticeSession, {
      type: 'SET_BOARD',
      board: transformed,
    });
    if (!reduced.accepted) return false;
    localPracticeSession = reduced.state;
    syncLocalPracticeSession({ animate: false });
    invalidateResults();
    if (!isEmptyPattern.value) queryResults('transform');
    return true;
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
    if (!['up', 'down', 'left', 'right'].includes(normalized)) return false;
    if (awaitingSpawn.value) {
      if ([1, 2].includes(Number(spawnMode.value)) && queuedMoveDirections.length < 4) {
        queuedMoveDirections.push(normalized);
        return true;
      }
      return false;
    }
    demoActive.value = false;
    clearDemoTimer();
    clearStepQueue();
    return executeTrainerMove(normalized);
  };

  function flushQueuedTrainerMove() {
    if (awaitingSpawn.value || !queuedMoveDirections.length) return;
    const direction = queuedMoveDirections.shift();
    window.queueMicrotask(() => {
      if (!executeTrainerMove(direction)) flushQueuedTrainerMove();
    });
  }

  const handleKeydown = (event) => {
    if (!hotkeysEnabledRef?.value) return;
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
      trainerUndo();
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
    if (enabled && !isEmptyPattern.value) {
      queryResults('visibility');
    }
  });

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
    restoreTrainerPractice();
    syncActivePatternCategory();
    window.addEventListener('keydown', handleKeydown);
    window.addEventListener('trainer-practice-jump', handleTrainerPracticeJump);
    unregisterTrainerPracticeJumpConsumer = registerTrainerPracticeJumpConsumer(
      (detail) => handleTrainerPracticeJump({ detail }),
    );
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
      restoreTrainerPractice();
      if (isEmptyPattern.value) activateEmptyPattern();
      applyTrainerJump();
      maybeAutoApplyDefaultTablebase();
    } else {
      if (persistPracticeTimer) {
        window.clearTimeout(persistPracticeTimer);
        persistPracticeTimer = null;
      }
      practiceRestoreAttempted = false;
      clearTrainerPracticeState();
      resetTrainerPrefetchState();
      localPracticeSession = createPracticeSession({ context: trainerPrefetchState });
      syncLocalPracticeSession({ animate: false });
      invalidateResults({ clearDisplay: true });
    }
  });

  onUnmounted(() => {
    if (persistPracticeTimer) {
      window.clearTimeout(persistPracticeTimer);
      persistPracticeTimer = null;
    }
    persistTrainerPractice();
    unregisterTrainerPracticeJumpConsumer?.();
    unregisterTrainerPracticeJumpConsumer = null;
    finishPaletteEditing({ query: false });
    clearPaletteSyncTimer();
    window.removeEventListener('keydown', handleKeydown);
    window.removeEventListener('trainer-practice-jump', handleTrainerPracticeJump);
    document.removeEventListener('click', closePatternMenuOnClick);
    document.removeEventListener('contextmenu', preventCtx);
    disconnect();
  });

  return {
    currentPatternDisplay,
    isEmptyPattern,
    emptyPatternId: EMPTY_PATTERN_ID,
    emptyPatternCategory: EMPTY_PATTERN_CATEGORY,
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
    boardFrame,
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
    transformBoard,
    triggerAction,
    recordStep,
    recordMax,
    recordingState,
    manageRecord,
    onDis32kChange,
    patternMenuRoot,
  };
}
