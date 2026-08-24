import { computed, onMounted, onUnmounted, ref, watch } from 'vue';

import { useAppSettingsStore } from '../../../app/useAppSettings';
import { useAuthState } from '../../../services/auth/authState';
import { downloadBlob, downloadText } from '../../../services/files/browserFiles';
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
  buildTesterPrefetchPayload,
  createTesterPrefetchState,
  createTesterSpawnRandomSource,
} from '../../../services/tablebases/testerPrefetchRng';
import { createWsClient } from '../../../services/ws/createWsClient';
import { getStableWsClientId } from '../../../services/ws/clientIds';
import { buildOptimisticMoveTransition } from '../../replay/engine/replayTransition';
import { buildOptimisticTesterLastStep } from '../engine/testerOptimisticFeedback';
import { isVariantPattern } from '../../../utils/patternCategories';
import { createResultBarGradient } from '../../../utils/resultBars';
import {
  restoreSuccessRate,
  formatSuccessRate,
  successRateSortValue,
  successRateRelativeLoss,
  resultValueFontSize,
} from '../../../utils/successRate';

export function useTesterSession(activeRef) {
  const DEFAULT_TABLEBASE_PATTERN = '442t';
  const DEFAULT_TABLEBASE_TARGET = '512';
  const { config: appConfig } = useAppSettingsStore();
  const { isAuthenticated, requireAuth } = useAuthState();

  const fallbackPatternCategories = {
    basic: ['L3', 'LL', 'free8', 'free9', 'free10', '444'],
  };
  const fallbackPerformanceLabels = ['Perfect!', 'Excellent!', 'Nice try!', 'Not bad!', 'Mistake!', 'Blunder!', 'Terrible!'];
  const performanceLabels = ref([...fallbackPerformanceLabels]);
  const dirLabels = computed(() => (
    isZh()
      ? { left: '左', right: '右', down: '下', up: '上' }
      : { left: 'L', right: 'R', down: 'D', up: 'U' }
  ));
  const COLOR_GREEN = '#2e7d32';
  const COLOR_YG = '#8bc34a';
  const COLOR_ORANGE = '#ff9800';
  const COLOR_RED = '#f44336';
  const evaluationColors = {
    'Perfect!': '#2e7d32',
    'Excellent!': '#7cb342',
    'Nice try!': '#c0ca33',
    'Not bad!': '#fb8c00',
    'Mistake!': '#f4511e',
    'Blunder!': '#e53935',
    'Terrible!': '#b71c1c',
  };
  const zhEvaluationLabels = {
    'Perfect!': 'Perfect!',
    'Excellent!': 'Excellent!',
    'Nice try!': 'Nice try!',
    'Not bad!': 'Not bad!',
    'Mistake!': 'Mistake!',
    'Blunder!': 'Blunder!',
    'Terrible!': 'Terrible!',
  };
  const evaluationColorPalette = [
    '#2e7d32',
    '#7cb342',
    '#c0ca33',
    '#fb8c00',
    '#f4511e',
    '#e53935',
    '#b71c1c',
  ];

  const wsStatus = ref('connecting');
  const clientId = getStableWsClientId('tester');
  const board = ref(new Array(16).fill(0));
  const metadata = ref({});
  const dis32k = ref(false);
  const currentLanguage = ref('en');
  const showInsights = ref(true);
  const patternCategories = ref(fallbackPatternCategories);
  const availableTargets = ref(['64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384']);
  const selectedPattern = ref(DEFAULT_TABLEBASE_PATTERN);
  const selectedTarget = ref(DEFAULT_TABLEBASE_TARGET);
  const activePatternCategory = ref(Object.keys(fallbackPatternCategories)[0] || '');
  const patternMenuOpen = ref(false);
  const currentBoardHex = ref('0000000000000000');
  const hexInput = ref('0000000000000000');
  const resultDtype = ref('?');
  const results = ref({});
  const logs = ref([]);
  const tableFound = ref(false);
  const ready = ref(false);
  const lookupPending = ref(false);
  const queryInFlight = ref(false);
  const queuedMoveDirection = ref('');
  const statusMessage = ref('');
  const recordLength = ref(0);
  const pendingPracticeJump = ref(null);
  const lastStep = ref({
    board_lines: [],
    result_lines: [],
    results: {},
    dtype: '?',
    message_lines: [],
    evaluation: '',
    direction: null,
    best_move: null,
    loss: null,
    goodness_of_fit: null,
  });
  const metrics = ref({
    combo: 0,
    max_combo: 0,
    goodness_of_fit: 1,
    performance_stats: {},
    score: 0,
    best_score: 0,
  });
  const patternMenuRoot = ref(null);
  const catalogTables = ref([]);
  const catalogVersion = ref('');

  let client = null;
  let bootstrapSelectionSent = false;
  let initialStateSeen = false;
  let nextQueryId = 0;
  let lastQueryScope = '';
  let activeQuery = null;
  let queryRetryTimer = null;
  let testerPrefetchState = createTesterPrefetchState();
  let pendingMovePrefetch = null;

  const spawnRate4 = () => Math.max(
    0,
    Math.min(1, Number(appConfig.value['4_spawn_rate'] ?? 0.1) || 0),
  );

  const resetTesterPrefetchState = () => {
    testerPrefetchState = createTesterPrefetchState();
    pendingMovePrefetch = null;
  };

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
    return getCatalogTargetsForPattern(catalogTables.value, selectedPattern.value);
  });
  const currentPatternDisplay = computed(() => (
    selectedPattern.value && selectedTarget.value ? `${selectedPattern.value}_${selectedTarget.value}` : 'Select Pattern'
  ));
  const isVariant = computed(() => isVariantPattern(selectedPattern.value, patternCategories.value));
  const canMove = computed(() => (
    ready.value
    && tableFound.value
    && !lookupPending.value
    && Object.values(results.value).some((value) => typeof value === 'number')
    && wsStatus.value === 'connected'
  ));
  const goodnessDisplay = computed(() => Number(metrics.value.goodness_of_fit ?? 1).toFixed(4));
  const resultPrecision = computed(() => String(resultDtype.value || '').includes('64') ? 15 : 8);
  const displayedResultDtype = computed(() => (
    insightsActive.value ? (lastStep.value?.dtype || resultDtype.value || '?') : (resultDtype.value || '?')
  ));
  const connectionBadgeClass = computed(() => (
    wsStatus.value === 'connected'
      ? 'badge-base badge-connection-connected'
      : 'badge-base badge-connection-disconnected'
  ));
  const hasLastStep = computed(() => !!(lastStep.value?.direction && lastStep.value?.best_move));
  const insightsActive = computed(() => showInsights.value && hasLastStep.value);
  const perfectLabel = computed(() => performanceLabels.value[0] || fallbackPerformanceLabels[0]);
  const getEvaluationColor = (label) => {
    if (evaluationColors[label]) return evaluationColors[label];
    const index = performanceLabels.value.indexOf(label);
    if (index >= 0) return evaluationColorPalette[index % evaluationColorPalette.length];
    return 'var(--accent)';
  };
  const evaluationTotal = computed(() => performanceLabels.value.reduce((sum, label) => sum + Number(metrics.value.performance_stats?.[label] || 0), 0));

  const isZh = () => String(currentLanguage.value || 'en').startsWith('zh');
  const getEvaluationLabel = (label) => (isZh() ? (zhEvaluationLabels[label] || label) : label);

  const evaluationSegments = computed(() => performanceLabels.value.map((label) => {
    const count = Number(metrics.value.performance_stats?.[label] || 0);
    const total = evaluationTotal.value || 1;
    const percent = evaluationTotal.value ? (count / total) * 100 : 0;
    const name = getEvaluationLabel(label);
    return {
      label,
      shortLabel: name,
      count,
      percent,
      color: getEvaluationColor(label),
      tooltip: `${name}: ${count}/${evaluationTotal.value || 0} (${percent.toFixed(1)}%)`,
    };
  }));

  const displayedEvaluationSegments = computed(() => (
    insightsActive.value
      ? evaluationSegments.value
      : performanceLabels.value.map((label) => ({
        label,
        shortLabel: getEvaluationLabel(label),
        count: 0,
        percent: 0,
        color: getEvaluationColor(label),
        tooltip: `${getEvaluationLabel(label)}: 0`,
      }))
  ));

  const moveLabels = computed(() => (
    isZh()
      ? { left: '左', right: '右', up: '上', down: '下' }
      : { left: 'Left', right: 'Right', up: 'Up', down: 'Down' }
  ));

  const feedbackEvaluation = computed(() => insightsActive.value ? (lastStep.value.evaluation || perfectLabel.value) : 'waiting');
  const feedbackBadgeText = computed(() => {
    if (!showInsights.value) return '--';
    if (!hasLastStep.value) return isZh() ? '等待落子' : 'Waiting';
    return getEvaluationLabel(lastStep.value.evaluation || perfectLabel.value);
  });
  const feedbackBadgeStyle = computed(() => {
    if (!showInsights.value || !hasLastStep.value) {
      return { color: 'var(--text-secondary)' };
    }
    const color = getEvaluationColor(lastStep.value.evaluation);
    return { color };
  });
  const feedbackLossText = computed(() => {
    if (!hasLastStep.value || !showInsights.value) return '';
    if (feedbackEvaluation.value === perfectLabel.value) return '';
    const loss = Number(lastStep.value?.loss ?? 0);
    return isZh()
      ? `单步损失 ${(loss * 100).toFixed(2)}%`
      : `One-step loss ${(loss * 100).toFixed(2)}%`;
  });
  const feedbackPressedLabel = computed(() => (isZh() ? '你走的是' : 'You pressed'));
  const feedbackBestLabel = computed(() => (isZh() ? '最优解' : 'Best move'));
  const feedbackConnector = computed(() => (isZh() ? '·' : 'and'));
  const feedbackPressedMove = computed(() => insightsActive.value ? (moveLabels.value[lastStep.value.direction] || '?') : '--');
  const feedbackBestMove = computed(() => insightsActive.value ? (moveLabels.value[lastStep.value.best_move] || '?') : '--');
  const feedbackPressedMoveStyle = computed(() => {
    const evaluation = lastStep.value?.evaluation;
    const color = insightsActive.value ? getEvaluationColor(evaluation) : 'var(--text-secondary)';
    return { color };
  });
  const feedbackBestMoveStyle = computed(() => {
    const color = insightsActive.value ? COLOR_GREEN : 'var(--text-secondary)';
    return { color };
  });

  const lerpColor = (c1, c2, ratio) => {
    const parseRgbColor = (color) => color.slice(1).match(/.{2}/g).map((part) => parseInt(part, 16));
    const mix = (a, b) => Math.round(a + (b - a) * ratio);
    const [r1, g1, b1] = parseRgbColor(c1);
    const [r2, g2, b2] = parseRgbColor(c2);
    return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`;
  };

  const resultSource = computed(() => (
    insightsActive.value ? (lastStep.value?.results || {}) : results.value
  ));

  const sortedResults = computed(() => {
    const items = ['left', 'right', 'down', 'up']
      .map((dir) => {
        const rawVal = resultSource.value?.[dir];
        const val = restoreSuccessRate(rawVal, displayedResultDtype.value || '');
        const sortVal = successRateSortValue(rawVal, displayedResultDtype.value || '');
        return {
          dir,
          rawVal: typeof rawVal === 'number' ? rawVal : null,
          val: val == null ? null : val,
          sortVal: sortVal == null ? null : sortVal,
        };
      })
      .sort((a, b) => {
        if (a.sortVal == null && b.sortVal == null) return 0;
        if (a.sortVal == null) return 1;
        if (b.sortVal == null) return -1;
        return b.sortVal - a.sortVal;
      });

    const bestItem = items.find((item) => item.val != null && item.rawVal != null);
    const bestVal = bestItem?.val || 0;
    return items.map((item, index) => {
      let pct = 0;
      let color = 'var(--border-main)';
      if (item.val != null && bestVal > 0) {
        const loss = successRateRelativeLoss(item.rawVal, bestItem?.rawVal, displayedResultDtype.value || '');
        if (index === 0) {
          pct = 100;
          color = COLOR_GREEN;
        } else if (loss != null && loss <= 0.10) {
          pct = (1 - loss / 0.10) * 100;
          color = loss <= 0.001
            ? COLOR_GREEN
            : (loss <= 0.01
              ? lerpColor(COLOR_GREEN, COLOR_YG, (loss - 0.001) / 0.009)
              : (loss <= 0.03
                ? lerpColor(COLOR_YG, COLOR_ORANGE, (loss - 0.01) / 0.02)
                : lerpColor(COLOR_ORANGE, COLOR_RED, (loss - 0.03) / 0.07)));
        } else {
          color = COLOR_RED;
        }
      }

      return {
        ...item,
        pct,
        gradient: color.startsWith('#') ? createResultBarGradient(color) : color,
        display: item.rawVal == null ? '--' : formatSuccessRate(item.rawVal, displayedResultDtype.value || '', resultPrecision.value),
        textColor: item.val == null ? 'var(--text-secondary)' : 'var(--text-main)',
      };
    });
  });

  const displayedResults = computed(() => (
    insightsActive.value
      ? sortedResults.value
      : ['up', 'down', 'right', 'left'].map((dir) => ({
        dir,
        rawVal: 0,
        val: 0,
        pct: 0,
        gradient: 'transparent',
        display: '0',
        textColor: 'var(--text-secondary)',
      }))
  ));

  const getResultRowStyle = (item) => ({
    background: item.val != null ? 'var(--bg-main)' : 'transparent',
    opacity: item.val == null ? 0.55 : 1,
  });

  const resultFontSize = computed(() => resultValueFontSize(displayedResults.value.map((item) => item.display)));

  const getResultValueStyle = (item) => {
    return { color: item.textColor, fontSize: resultFontSize.value };
  };

  const formatBoardValue = (value) => {
    if (!value) return '_';
    if (value === 32768) return '';
    if (value >= 1024) return `${Math.floor(value / 1024)}k`;
    return String(value);
  };

  const parseBoardToken = (token) => {
    const normalized = String(token || '_').trim().toLowerCase();
    if (!normalized || normalized === '_') return { value: 0, label: '_' };
    if (normalized === 'x') return { value: 32768, label: '' };
    if (normalized.endsWith('k')) {
      const thousands = Number.parseInt(normalized.slice(0, -1), 10);
      if (Number.isFinite(thousands) && thousands > 0) {
        return { value: thousands * 1024, label: `${thousands}k` };
      }
    }
    const numeric = Number.parseInt(normalized, 10);
    if (Number.isFinite(numeric) && numeric > 0) return { value: numeric, label: String(numeric) };
    return { value: 0, label: '_' };
  };

  const flatBoardToTiles = (flatBoard) => {
    const safeBoard = Array.isArray(flatBoard) ? flatBoard : [];
    return Array.from({ length: 16 }, (_, index) => {
      const value = Number(safeBoard[index] || 0);
      return {
        key: `board_${index}`,
        value,
        label: formatBoardValue(value),
      };
    });
  };

  const resultConsoleBoardLines = computed(() => {
    if (!insightsActive.value) return [];
    const lines = lastStep.value?.board_lines;
    if (Array.isArray(lines) && lines.length === 4) return lines;
    return [];
  });

  const resultConsoleTiles = computed(() => {
    if (Array.isArray(lastStep.value?.board) && lastStep.value.board.length === 16) {
      return flatBoardToTiles(lastStep.value.board);
    }
    if (resultConsoleBoardLines.value.length === 4) {
      return resultConsoleBoardLines.value.flatMap((line, rowIndex) => {
        const tokens = String(line || '').trim().split(/\s+/u).slice(0, 4);
        while (tokens.length < 4) tokens.push('_');
        return tokens.map((token, colIndex) => {
          const tile = parseBoardToken(token);
          return {
            key: `last_${rowIndex}_${colIndex}`,
            value: tile.value,
            label: tile.label,
          };
        });
      });
    }
    return flatBoardToTiles(board.value);
  });

  const getResultMiniTileStyle = (tile) => {
    if (!tile || !tile.value) {
      return {
        backgroundColor: 'var(--color-empty)',
        color: 'transparent',
      };
    }
    // Set 32768 tile to be invisible (board background color) when isVariant is true
    if (isVariant.value && Number(tile.value) === 32768) {
      return {
        backgroundColor: 'var(--color-board-bg)',
        color: 'transparent',
        boxShadow: 'none',
      };
    }
    return {
      backgroundColor: `var(--color-tile-${tile.value})`,
      color: `var(--color-text-${tile.value})`,
    };
  };

  const syncCategoryFromPattern = (pattern) => {
    const match = patternGroups.value.find((group) => group.patterns.includes(pattern));
    activePatternCategory.value = match?.category || patternGroups.value[0]?.category || '';
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

  const preferredTargetFrom = (targets) => (
    targets.includes(DEFAULT_TABLEBASE_TARGET) ? DEFAULT_TABLEBASE_TARGET : (targets[0] || '')
  );

  const ensureDefaultSelection = ({ preserveSelection = false } = {}) => {
    const groups = patternGroups.value;
    if (!groups.length) return;
    if (
      preserveSelection
      && selectedPattern.value
      && selectedTarget.value
      && !getCatalogTargetsForPattern(catalogTables.value, selectedPattern.value).includes(selectedTarget.value)
    ) {
      syncCategoryFromPattern(selectedPattern.value);
      return;
    }
    selectedPattern.value = flatPatterns.value.includes(selectedPattern.value)
      ? selectedPattern.value
      : (
        flatPatterns.value.includes(DEFAULT_TABLEBASE_PATTERN)
          ? DEFAULT_TABLEBASE_PATTERN
          : (groups[0].patterns[0] || '')
      );
    const targets = availableTargetsForPattern.value;
    selectedTarget.value = targets.includes(selectedTarget.value)
      ? selectedTarget.value
      : preferredTargetFrom(targets);
    syncCategoryFromPattern(selectedPattern.value);
  };

  const loadCatalog = async ({ preserveSelection = false } = {}) => {
    try {
      const tables = await fetchTablebaseCatalog();
      catalogTables.value = tables;
      catalogVersion.value = tables.catalogVersion || getCatalogVersion();
      const nextCategories = groupTablebasePatternsByCategory(tables);
      if (Object.values(nextCategories).some((patterns) => patterns.length)) {
        patternCategories.value = nextCategories;
        availableTargets.value = getCatalogTargets(tables);
        ensureDefaultSelection({ preserveSelection });
        maybeApplyInitialPatternSelection();
        if (
          ready.value
          && tableFound.value
          && currentBoardHex.value !== '0000000000000000'
          && !Object.values(results.value).some((value) => typeof value === 'number')
        ) {
          queryTablebase(currentBoardHex.value);
        }
      }
    } catch (error) {
      console.error(error);
    }
  };

  const togglePatternMenu = () => {
    syncCategoryFromPattern(selectedPattern.value);
    patternMenuOpen.value = !patternMenuOpen.value;
  };

  const protectedActions = new Set([
    'TESTER_SELECT_PATTERN',
    'TESTER_RESET_RANDOM',
    'TESTER_MOVE',
    'TESTER_SET_BOARD',
    'TESTER_EXPORT_LOG',
    'TESTER_EXPORT_REPLAY',
    'TABLEBASE_QUERY',
  ]);

  const triggerAction = (action, payload = {}) => {
    if (protectedActions.has(action) && !requireAuth()) {
      return false;
    }
    client?.send(action, payload);
    return true;
  };

  const selectPattern = (pattern) => {
    selectedPattern.value = pattern;
    syncCategoryFromPattern(pattern);
    ensureDefaultSelection();
    patternMenuOpen.value = false;
    applyPatternSelection();
  };

  const closePatternMenuOnClick = (event) => {
    if (!patternMenuOpen.value || !patternMenuRoot.value) return;
    if (!patternMenuRoot.value.contains(event.target)) patternMenuOpen.value = false;
  };

  const applyPatternSelection = () => {
    if (!selectedPattern.value || !selectedTarget.value) return;
    clearTablebaseResultCache();
    resetTesterPrefetchState();
    lastQueryScope = '';
    activeQuery = null;
    syncCategoryFromPattern(selectedPattern.value);
    triggerAction('TESTER_SELECT_PATTERN', { pattern: selectedPattern.value, target: selectedTarget.value });
  };

  const hasLocalPracticeState = () => (
    ready.value ||
    tableFound.value ||
    currentBoardHex.value !== '0000000000000000' ||
    logs.value.length > 0 ||
    recordLength.value > 0
  );

  const maybeApplyInitialPatternSelection = () => {
    if (
      !activeRef?.value ||
      wsStatus.value !== 'connected' ||
      !isAuthenticated.value ||
      !initialStateSeen ||
      bootstrapSelectionSent ||
      !selectedPattern.value ||
      !selectedTarget.value ||
      hasLocalPracticeState()
    ) {
      return;
    }
    bootstrapSelectionSent = true;
    applyPatternSelection();
  };

  const resetRandom = () => {
    resetTesterPrefetchState();
    return triggerAction('TESTER_RESET_RANDOM');
  };
  const applyManualBoard = () => {
    if (hexInput.value.trim()) {
      resetTesterPrefetchState();
      triggerAction('TESTER_SET_BOARD', { hex_str: hexInput.value.trim() });
    }
  };
  const toggleInsights = () => { showInsights.value = !showInsights.value; };

  const tablebaseCacheKey = (boardHex) => ({
    catalogVersion: catalogVersion.value || getCatalogVersion(),
    fullPattern: currentPatternDisplay.value,
    boardHex,
  });

  const applyCachedTablebaseResult = (boardHex) => {
    if (!boardHex || !catalogVersion.value || !currentPatternDisplay.value) return false;
    const cached = getCachedTablebaseResult(tablebaseCacheKey(boardHex));
    if (!cached) return false;
    resultDtype.value = cached.dtype || '?';
    results.value = cached.results || {};
    lookupPending.value = false;
    return true;
  };

  const prepareTablebaseQuery = (boardHex, { useCache = true } = {}) => {
    const normalizedBoard = String(boardHex || '').trim().toLowerCase();
    if (
      !normalizedBoard
      || !catalogVersion.value
      || !currentPatternDisplay.value
      || !tableFound.value
      || wsStatus.value !== 'connected'
      || !isAuthenticated.value
    ) {
      return false;
    }
    const cacheHit = useCache && applyCachedTablebaseResult(normalizedBoard);
    if (queryRetryTimer) {
      window.clearTimeout(queryRetryTimer);
      queryRetryTimer = null;
    }
    if (!cacheHit) {
      resultDtype.value = '?';
      results.value = {};
      lookupPending.value = true;
    }
    const queryId = `${clientId}_${++nextQueryId}`;
    lastQueryScope = `${currentPatternDisplay.value}:${normalizedBoard}`;
    activeQuery = {
      queryId,
      boardHex: normalizedBoard,
      fullPattern: currentPatternDisplay.value,
    };
    queryInFlight.value = true;
    return {
      queryId,
      normalizedBoard,
      cacheHit,
      prefetchRng: buildTesterPrefetchPayload(
        testerPrefetchState,
        spawnRate4(),
      ),
    };
  };

  const queryTablebase = (boardHex, { useCache = true } = {}) => {
    const prepared = prepareTablebaseQuery(boardHex, { useCache });
    if (!prepared) return false;
    client?.send('TABLEBASE_QUERY', {
      page: 'tester',
      query_id: prepared.queryId,
      catalog_version: catalogVersion.value,
      full_pattern: currentPatternDisplay.value,
      board_hex: prepared.normalizedBoard,
      prefetch_rng: prepared.prefetchRng,
    });
    return prepared.cacheHit;
  };

  const move = (dir) => {
    const normalizedDirection = String(dir || '').toLowerCase();
    if (!['up', 'down', 'left', 'right'].includes(normalizedDirection) || !requireAuth()) {
      return false;
    }
    if (!canMove.value) {
      if (
        ready.value
        && tableFound.value
        && wsStatus.value === 'connected'
        && (lookupPending.value || queryInFlight.value)
      ) {
        queuedMoveDirection.value = normalizedDirection;
        return true;
      }
      return false;
    }
    queuedMoveDirection.value = '';
    const fromBoardHex = currentBoardHex.value;
    const optimisticLastStep = buildOptimisticTesterLastStep({
      board: board.value,
      results: results.value,
      dtype: resultDtype.value,
      direction: normalizedDirection,
      goodnessOfFit: metrics.value.goodness_of_fit,
    });
    const prefetchStateBeforeMove = {
      state: [...testerPrefetchState.state],
      turn: testerPrefetchState.turn,
    };
    const deterministicSpawn = createTesterSpawnRandomSource(testerPrefetchState);
    const transition = buildOptimisticMoveTransition(
      board.value,
      normalizedDirection,
      isVariant.value,
      spawnRate4(),
      deterministicSpawn.randomSource,
    );
    if (!transition) return false;

    if (optimisticLastStep) lastStep.value = optimisticLastStep;

    testerPrefetchState = deterministicSpawn.nextState();
    pendingMovePrefetch = {
      boardHex: transition.hex,
      previousState: prefetchStateBeforeMove,
    };
    board.value = transition.board;
    metadata.value = transition.metadata;
    currentBoardHex.value = transition.hex;
    hexInput.value = transition.hex;
    const preparedQuery = prepareTablebaseQuery(transition.hex);
    client?.send('TESTER_MOVE', {
      dir: normalizedDirection,
      from_board_hex: fromBoardHex,
      board_hex: transition.hex,
      spawn_index: transition.spawnIndex,
      spawn_value: transition.spawnValue,
      query_id: preparedQuery?.queryId,
      prefetch_rng: preparedQuery?.prefetchRng,
    });
    return true;
  };

  const flushQueuedMove = () => {
    const direction = queuedMoveDirection.value;
    if (!direction || !canMove.value) return;
    queuedMoveDirection.value = '';
    window.queueMicrotask(() => {
      if (!move(direction) && (lookupPending.value || queryInFlight.value)) {
        queuedMoveDirection.value = direction;
      }
    });
  };

  const saveLog = () => {
    if (!logs.value.length) return;
    triggerAction('TESTER_EXPORT_LOG');
  };

  const saveReplay = () => {
    if (recordLength.value < 1) return;
    triggerAction('TESTER_EXPORT_REPLAY');
  };

  const handlePracticeJump = (event) => {
    const detail = event?.detail || {};
    const parsed = parseFullPattern(detail.fullPattern);
    const hex = String(detail.hex || '').trim();
    if (!parsed || !hex) return;

    selectedPattern.value = parsed.pattern;
    selectedTarget.value = (
      catalogTables.value.length
        ? getCatalogTargetsForPattern(catalogTables.value, parsed.pattern).includes(parsed.target)
        : availableTargets.value.includes(parsed.target)
    )
      ? parsed.target
      : selectedTarget.value;
    ensureDefaultSelection();
    syncCategoryFromPattern(parsed.pattern);
    pendingPracticeJump.value = {
      pattern: selectedPattern.value,
      target: selectedTarget.value,
      hex,
    };
    resetTesterPrefetchState();
    if (wsStatus.value === 'connected') {
      triggerAction('TESTER_SELECT_PATTERN', {
        pattern: selectedPattern.value,
        target: selectedTarget.value,
      });
    }
  };

  const handleTesterBootstrap = (payload) => {
    if (!catalogTables.value.length) {
      patternCategories.value = payload?.categories || fallbackPatternCategories;
      availableTargets.value = (payload?.target_tiles || []).map(String);
    }
    ensureDefaultSelection();
    maybeApplyInitialPatternSelection();
  };

  const handleTesterState = (payload) => {
    const previousBoardHex = currentBoardHex.value;
    const incomingBoardHex = String(payload?.hex_str || currentBoardHex.value).toLowerCase();
    if (pendingMovePrefetch) {
      if (incomingBoardHex !== pendingMovePrefetch.boardHex) {
        testerPrefetchState = pendingMovePrefetch.previousState;
      }
      pendingMovePrefetch = null;
    } else if (incomingBoardHex && incomingBoardHex !== previousBoardHex) {
      resetTesterPrefetchState();
    }
    const incomingLabels = payload?.metrics?.performance_labels;
    performanceLabels.value = Array.isArray(incomingLabels) && incomingLabels.length
      ? [...incomingLabels]
      : [...fallbackPerformanceLabels];
    board.value = Array.isArray(payload?.board) ? payload.board : new Array(16).fill(0);
    metadata.value = payload?.animation || {};
    currentBoardHex.value = incomingBoardHex;
    hexInput.value = currentBoardHex.value;
    if (activeQuery && activeQuery.boardHex !== currentBoardHex.value) {
      activeQuery = null;
      queryInFlight.value = false;
    }
    resultDtype.value = payload?.dtype || '?';
    results.value = payload?.results || {};
    if (Array.isArray(payload?.logs)) {
      logs.value = payload.logs;
    } else if (Array.isArray(payload?.logs_delta)) {
      logs.value = [...logs.value, ...payload.logs_delta];
    }
    lastStep.value = payload?.last_step || {
      board_lines: [],
      result_lines: [],
      results: {},
      dtype: '?',
      message_lines: [],
      evaluation: '',
      direction: null,
      best_move: null,
      loss: null,
      goodness_of_fit: null,
    };
    ready.value = !!payload?.ready;
    lookupPending.value = !!payload?.lookup_pending;
    tableFound.value = !!payload?.table_found;
    statusMessage.value = payload?.status || '';
    recordLength.value = payload?.record?.length || 0;
    metrics.value = {
      combo: payload?.metrics?.combo ?? 0,
      max_combo: payload?.metrics?.max_combo ?? 0,
      goodness_of_fit: payload?.metrics?.goodness_of_fit ?? 1,
      performance_stats: payload?.metrics?.performance_stats || {},
      score: payload?.metrics?.score ?? 0,
      best_score: payload?.metrics?.best_score ?? 0,
    };
    if (payload?.pattern && payload.pattern !== '?' && flatPatterns.value.includes(payload.pattern)) {
      selectedPattern.value = payload.pattern;
      syncCategoryFromPattern(payload.pattern);
    }
    if (
      payload?.target &&
      payload.target !== '?' &&
      availableTargetsForPattern.value.includes(String(payload.target))
    ) {
      selectedTarget.value = String(payload.target);
    }
    if (
      pendingPracticeJump.value &&
      payload?.pattern === pendingPracticeJump.value.pattern &&
      String(payload?.target) === pendingPracticeJump.value.target
    ) {
      const { hex } = pendingPracticeJump.value;
      pendingPracticeJump.value = null;
      triggerAction('TESTER_SET_BOARD', { hex_str: hex });
    }
    initialStateSeen = true;
    maybeApplyInitialPatternSelection();
    const hasServerResults = Object.values(results.value).some((value) => typeof value === 'number');
    if (hasServerResults && catalogVersion.value && currentPatternDisplay.value) {
      setCachedTablebaseResult(tablebaseCacheKey(currentBoardHex.value), {
        found: true,
        dtype: resultDtype.value,
        results: results.value,
      });
      lookupPending.value = false;
      queryInFlight.value = false;
      flushQueuedMove();
    } else if (
      ready.value
      && tableFound.value
      && currentBoardHex.value !== '0000000000000000'
      && (
        previousBoardHex !== currentBoardHex.value
        || lastQueryScope !== `${currentPatternDisplay.value}:${currentBoardHex.value}`
      )
    ) {
      queryTablebase(currentBoardHex.value);
    }
  };

  const handleTesterResults = (payload) => {
    const resultBoardHex = String(payload?.board_hex || '');
    if (!resultBoardHex || resultBoardHex !== currentBoardHex.value) return;
    resultDtype.value = payload?.dtype || '?';
    results.value = payload?.results || {};
    lookupPending.value = !!payload?.lookup_pending;
    if (!lookupPending.value) {
      queryInFlight.value = false;
      flushQueuedMove();
    }
    if (Array.isArray(payload?.logs_delta)) {
      logs.value = [...logs.value, ...payload.logs_delta];
    }
  };

  const handleTesterMoveAccepted = (payload) => {
    const acceptedBoardHex = String(payload?.board_hex || '');
    if (!acceptedBoardHex || acceptedBoardHex !== currentBoardHex.value) return;
    if (pendingMovePrefetch?.boardHex === acceptedBoardHex) {
      pendingMovePrefetch = null;
    }
    lastStep.value = payload?.last_step || lastStep.value;
    lookupPending.value = !!payload?.lookup_pending
      && !Object.values(results.value).some((value) => typeof value === 'number');
    recordLength.value = payload?.record?.length ?? recordLength.value;
    metrics.value = {
      combo: payload?.metrics?.combo ?? metrics.value.combo,
      max_combo: payload?.metrics?.max_combo ?? metrics.value.max_combo,
      goodness_of_fit: payload?.metrics?.goodness_of_fit ?? metrics.value.goodness_of_fit,
      performance_stats: payload?.metrics?.performance_stats || metrics.value.performance_stats,
      score: payload?.metrics?.score ?? metrics.value.score,
      best_score: payload?.metrics?.best_score ?? metrics.value.best_score,
    };
    if (Array.isArray(payload?.logs_delta)) {
      logs.value = [...logs.value, ...payload.logs_delta];
    }
  };

  const handleTablebaseQueryResult = (payload) => {
    if (payload?.page !== 'tester') return;
    if (payload?.code) {
      if (payload?.query_id && payload.query_id === activeQuery?.queryId) {
        activeQuery = null;
        queryInFlight.value = false;
      }
      if (
        payload.code !== 'STALE_TABLEBASE_QUERY'
        && String(payload?.board_hex || '').toLowerCase() === currentBoardHex.value
      ) {
        lookupPending.value = false;
        statusMessage.value = (
          payload.code === 'REMOTE_TABLEBASE_OFFLINE'
          || payload.code === 'REMOTE_TABLEBASE_TIMEOUT'
        )
          ? (String(appConfig.value?.language || '').toLowerCase().startsWith('zh')
            ? '所选定式暂不可用。'
            : 'The selected tablebase is temporarily unavailable.')
          : (payload?.message || statusMessage.value);
      }
      return;
    }
    const resultBoardHex = String(payload?.board_hex || '').toLowerCase();
    const fullPattern = String(payload?.full_pattern || '');
    const resultCatalogVersion = String(payload?.catalog_version || catalogVersion.value);
    if (!resultBoardHex || !fullPattern || !resultCatalogVersion) return;
    if (resultCatalogVersion !== catalogVersion.value) {
      loadCatalog();
      return;
    }
    if (fullPattern !== currentPatternDisplay.value) return;
    setCachedTablebaseResult({
      catalogVersion: resultCatalogVersion,
      fullPattern,
      boardHex: resultBoardHex,
    }, payload);
    if (
      resultBoardHex !== currentBoardHex.value
    ) {
      return;
    }
    resultDtype.value = payload?.dtype || '?';
    results.value = payload?.results || {};
    lookupPending.value = false;
    queryInFlight.value = false;
    if (!payload?.query_id || payload.query_id === activeQuery?.queryId) {
      activeQuery = null;
    }
    statusMessage.value = '';
    if (Array.isArray(payload?.logs_delta)) {
      logs.value = [...logs.value, ...payload.logs_delta];
    }
    flushQueuedMove();
  };

  const handleTablebasePrefetch = (payload) => {
    if (payload?.page !== 'tester') return;
    const fullPattern = String(payload?.full_pattern || '');
    const resultCatalogVersion = String(payload?.catalog_version || '');
    if (
      !fullPattern
      || !resultCatalogVersion
      || resultCatalogVersion !== catalogVersion.value
      || fullPattern !== currentPatternDisplay.value
    ) return;
    for (const entry of payload?.entries || []) {
      if (!entry?.board_hex) continue;
      setCachedTablebaseResult({
        catalogVersion: resultCatalogVersion,
        fullPattern,
        boardHex: entry.board_hex,
      }, entry);
    }
  };

  const handleWSMessage = (message) => {
    if (message.action === 'TESTER_BOOTSTRAP') handleTesterBootstrap(message.data);
    else if (message.action === 'TESTER_STATE') handleTesterState(message.data);
    else if (message.action === 'TESTER_MOVE_ACCEPTED') handleTesterMoveAccepted(message.data);
    else if (message.action === 'TESTER_RESULTS') handleTesterResults(message.data);
    else if (message.action === 'TABLEBASE_QUERY_RESULT') handleTablebaseQueryResult(message.data);
    else if (message.action === 'TABLEBASE_PREFETCH') handleTablebasePrefetch(message.data);
    else if (message.action === 'TABLEBASE_CATALOG_UPDATED') {
      loadCatalog({ preserveSelection: true });
    }
    else if (message.action === 'TABLEBASE_BUSY' && message.data?.page === 'tester') {
      const retryBoard = currentBoardHex.value;
      const retryPattern = currentPatternDisplay.value;
      lookupPending.value = !Object.values(results.value).some((value) => typeof value === 'number');
      statusMessage.value = message.data?.message || statusMessage.value;
      queryRetryTimer = window.setTimeout(() => {
        queryRetryTimer = null;
        if (
          currentBoardHex.value === retryBoard
          && currentPatternDisplay.value === retryPattern
          && wsStatus.value === 'connected'
        ) {
          queryTablebase(retryBoard);
        }
      }, Math.max(250, Number(message.data?.retry_after_ms) || 1000));
    }
    else if (message.action === 'TOKEN_REQUIRED' && activeQuery) {
      activeQuery = null;
      queryInFlight.value = false;
      clearTablebaseResultCache();
      resultDtype.value = '?';
      results.value = {};
      lookupPending.value = false;
      if (queryRetryTimer) {
        window.clearTimeout(queryRetryTimer);
        queryRetryTimer = null;
      }
    }
    else if (
      (lookupPending.value || activeQuery)
      && ['AUTH_REQUIRED', 'TOKEN_REQUIRED', 'ERROR'].includes(message.action)
    ) {
      activeQuery = null;
      queryInFlight.value = false;
      client?.send('TESTER_GET_INIT');
    }
    else if (message.action === 'TESTER_EXPORT_LOG') {
      const payload = message.data || {};
      downloadText(payload.text || '', payload.filename || 'tester_log.txt', payload.mime || 'text/plain;charset=utf-8');
    } else if (message.action === 'TESTER_EXPORT_REPLAY') {
      const payload = message.data || {};
      const binary = atob(payload.base64 || '');
      const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
      downloadBlob(new Blob([bytes], { type: payload.mime || 'application/octet-stream' }), payload.filename || 'tester_replay.rpl');
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
        bootstrapSelectionSent = false;
        initialStateSeen = false;
        loadCatalog({ preserveSelection: true });
        triggerAction('TESTER_GET_INIT');
      },
      onMessage: handleWSMessage,
      onClose: () => {
        wsStatus.value = 'disconnected';
      },
    });
    wsStatus.value = 'connecting';
    client.connect();
  };

  const disconnect = () => {
    if (queryRetryTimer) {
      window.clearTimeout(queryRetryTimer);
      queryRetryTimer = null;
    }
    activeQuery = null;
    queryInFlight.value = false;
    queuedMoveDirection.value = '';
    if (pendingMovePrefetch) {
      testerPrefetchState = pendingMovePrefetch.previousState;
      pendingMovePrefetch = null;
    }
    client?.disconnect();
    client = null;
    wsStatus.value = 'disconnected';
  };

  const handleKeyDown = (event) => {
    if (!activeRef?.value) return;
    if (event.key === 'Escape' && patternMenuOpen.value) {
      patternMenuOpen.value = false;
      return;
    }
    const code = event.code;
    const arrowMove = {
      ArrowUp: 'up',
      ArrowDown: 'down',
      ArrowLeft: 'left',
      ArrowRight: 'right',
    }[code];
    const target = event.target;
    if (target instanceof HTMLElement) {
      if (target.isContentEditable) return;
      if (target.closest('[data-tester-text-input="true"]')) {
        if (arrowMove) {
          event.preventDefault();
          target.blur();
          move(arrowMove);
        }
        return;
      }
    }
    if (code === 'ArrowUp' || code === 'KeyW') { event.preventDefault(); move('up'); }
    else if (code === 'ArrowDown' || code === 'KeyS') { event.preventDefault(); move('down'); }
    else if (code === 'ArrowLeft' || code === 'KeyA') { event.preventDefault(); move('left'); }
    else if (code === 'ArrowRight' || code === 'KeyD') { event.preventDefault(); move('right'); }
    else if (code === 'KeyR') { event.preventDefault(); resetRandom(); }
    else if (code === 'KeyF') { event.preventDefault(); toggleInsights(); }
  };

  onMounted(() => {
    syncCategoryFromPattern(selectedPattern.value);
    loadCatalog();
    window.addEventListener('keydown', handleKeyDown, true);
    window.addEventListener('tester-practice-jump', handlePracticeJump);
    document.addEventListener('click', closePatternMenuOnClick);
  });

  watch(
    () => appConfig.value.dis_32k,
    (value) => {
      dis32k.value = !!value;
    },
    { immediate: true }
  );

  watch(
    () => appConfig.value.language,
    (value) => {
      currentLanguage.value = value || 'en';
    },
    { immediate: true }
  );

  watch(
    activeRef,
    (isActive) => {
      if (isActive) {
        connect();
        maybeApplyInitialPatternSelection();
      }
    },
    { immediate: true }
  );

  watch(isAuthenticated, (authenticated) => {
    if (authenticated) {
      maybeApplyInitialPatternSelection();
    }
  });

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeyDown, true);
    window.removeEventListener('tester-practice-jump', handlePracticeJump);
    document.removeEventListener('click', closePatternMenuOnClick);
    disconnect();
  });

  return {
    wsStatus,
    board,
    metadata,
    dis32k,
    showInsights,
    availableTargets,
    availableTargetsForPattern,
    selectedPattern,
    selectedTarget,
    activePatternCategory,
    patternMenuOpen,
    hexInput,
    currentBoardHex,
    logs,
    recordLength,
    metrics,
    patternMenuRoot,
    patternGroups,
    activePatternOptions,
    currentPatternDisplay,
    isVariant,
    displayedResultDtype,
    connectionBadgeClass,
    insightsActive,
    evaluationTotal,
    displayedEvaluationSegments,
    dirLabels,
    feedbackBadgeText,
    feedbackBadgeStyle,
    feedbackLossText,
    feedbackPressedLabel,
    feedbackPressedMove,
    feedbackPressedMoveStyle,
    feedbackConnector,
    feedbackBestLabel,
    feedbackBestMove,
    feedbackBestMoveStyle,
    goodnessDisplay,
    displayedResults,
    resultConsoleTiles,
    togglePatternMenu,
    selectPattern,
    applyPatternSelection,
    resetRandom,
    applyManualBoard,
    toggleInsights,
    move,
    saveLog,
    saveReplay,
    getResultRowStyle,
    getResultValueStyle,
    getResultMiniTileStyle,
  };
}
