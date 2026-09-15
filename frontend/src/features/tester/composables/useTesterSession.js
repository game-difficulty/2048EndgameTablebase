import { computed, onMounted, onUnmounted, ref, watch } from 'vue';

import { KEYBOARD_OWNERS, keyboardInputAllowed } from '../../../app/keyboardOwnership';
import { useAppSettingsStore } from '../../../app/useAppSettings';
import {
  createSnapshotBoardFrame,
  createTransitionBoardFrame,
} from '../../../components/boardFrame.js';
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
import { saveLocalTesterReplay } from '../../replay/services/localTesterReplayStore';
import { canApplyPracticeSeed } from '../../practice/engine/practiceSession.js';
import {
  clearTesterPracticeState,
  restoreTesterPracticeState,
  saveTesterPracticeState,
} from '../services/testerPracticeStore';
import {
  applyTesterLocalMove,
  createTesterLocalSession,
  encodeTesterReplay,
  replaceTesterLocalBoard,
  testerReplayFilename,
} from '../engine/testerLocalSession';
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
  const { isAuthenticated, requireAuth, user: authUser } = useAuthState();

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
  const boardFrame = ref(createSnapshotBoardFrame(0, board.value));
  const dis32k = ref(false);
  const currentLanguage = ref('en');
  const showInsights = ref(true);
  const patternCategories = ref({});
  const availableTargets = ref([]);
  const selectedPattern = ref('');
  const selectedTarget = ref('');
  const activePatternCategory = ref('');
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
  const queuedMoveDirections = [];
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
  let boardFrameRevision = 0;
  let bootstrapSelectionSent = false;
  let initialStateSeen = false;
  let nextQueryId = 0;
  let lastQueryScope = '';
  let activeQuery = null;
  let queryRetryTimer = null;
  let testerPrefetchState = createTesterPrefetchState();
  let localTesterSession = null;
  let nextBoardLoadId = 0;
  let pendingBoardLoad = null;
  let nextTablebaseAttachId = 0;
  let pendingTablebaseAttachId = '';
  let persistPracticeTimer = null;
  let practiceRestoreAttempted = false;

  const spawnRate4 = () => Math.max(
    0,
    Math.min(1, Number(appConfig.value['4_spawn_rate'] ?? 0.1) || 0),
  );

  const resetTesterPrefetchState = () => {
    testerPrefetchState = createTesterPrefetchState();
  };

  const persistTesterPractice = () => {
    if (!localTesterSession || !authUser.value?.id) return false;
    return saveTesterPracticeState({
      userId: Number(authUser.value.id),
      pattern: selectedPattern.value,
      target: selectedTarget.value,
      ready: ready.value,
      tableFound: tableFound.value,
      statusMessage: statusMessage.value,
      resultDtype: resultDtype.value,
      results: results.value,
      prefetchState: testerPrefetchState,
      session: localTesterSession,
    });
  };

  const persistTesterPracticeSoon = () => {
    if (persistPracticeTimer) window.clearTimeout(persistPracticeTimer);
    persistPracticeTimer = window.setTimeout(() => {
      persistPracticeTimer = null;
      persistTesterPractice();
    }, 50);
  };

  const syncLocalTesterSession = ({ animate = true } = {}) => {
    if (!localTesterSession?.practice) return false;
    const practice = localTesterSession.practice;
    board.value = [...practice.board];
    boardFrameRevision += 1;
    boardFrame.value = animate
      ? createTransitionBoardFrame(boardFrameRevision, practice.transition, practice.board)
      : createSnapshotBoardFrame(boardFrameRevision, practice.board);
    currentBoardHex.value = practice.boardHex;
    hexInput.value = practice.boardHex;
    lastStep.value = localTesterSession.lastStep;
    metrics.value = { ...localTesterSession.metrics };
    logs.value = [...localTesterSession.logs];
    recordLength.value = localTesterSession.records.length;
    persistTesterPracticeSoon();
    return true;
  };

  const beginLocalTesterSession = ({ board: nextBoard, boardHex, openingLogs = [] } = {}) => {
    localTesterSession = createTesterLocalSession({
      board: nextBoard,
      boardHex,
      useVariant: isVariant.value,
      context: testerPrefetchState,
      performanceLabels: performanceLabels.value,
      openingLogs,
    });
    syncLocalTesterSession({ animate: false });
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
      return [];
    }
    return getCatalogTargetsForPattern(catalogTables.value, selectedPattern.value);
  });
  const currentPatternDisplay = computed(() => (
    selectedPattern.value && selectedTarget.value ? `${selectedPattern.value}_${selectedTarget.value}` : 'Select Pattern'
  ));
  const isVariant = computed(() => isVariantPattern(selectedPattern.value, patternCategories.value));

  const restoreTesterPractice = () => {
    if (practiceRestoreAttempted || !authUser.value?.id) return false;
    practiceRestoreAttempted = true;
    const restored = restoreTesterPracticeState();
    if (!restored || Number(restored.userId) !== Number(authUser.value.id)) {
      if (restored) clearTesterPracticeState();
      return false;
    }
    selectedPattern.value = String(restored.pattern || selectedPattern.value);
    selectedTarget.value = String(restored.target || selectedTarget.value);
    testerPrefetchState = restored.prefetchState || createTesterPrefetchState();
    localTesterSession = restored.session;
    ready.value = Boolean(restored.ready);
    tableFound.value = Boolean(restored.tableFound);
    statusMessage.value = String(restored.statusMessage || '');
    resultDtype.value = String(restored.resultDtype || '?');
    results.value = restored.results && typeof restored.results === 'object'
      ? restored.results
      : {};
    syncCategoryFromPattern(selectedPattern.value);
    syncLocalTesterSession({ animate: false });
    lookupPending.value = false;
    return true;
  };
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

  const ensureDefaultSelection = () => {
    const groups = patternGroups.value;
    if (!groups.length) {
      selectedPattern.value = '';
      selectedTarget.value = '';
      activePatternCategory.value = '';
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

  const loadCatalog = async () => {
    try {
      const tables = await fetchTablebaseCatalog();
      catalogTables.value = tables;
      catalogVersion.value = tables.catalogVersion || getCatalogVersion();
      const nextCategories = groupTablebasePatternsByCategory(tables);
      patternCategories.value = nextCategories;
      availableTargets.value = getCatalogTargets(tables);
      if (Object.values(nextCategories).some((patterns) => patterns.length)) {
        ensureDefaultSelection();
        maybeApplyInitialPatternSelection();
        if (
          ready.value
          && tableFound.value
          && currentBoardHex.value !== '0000000000000000'
          && !Object.values(results.value).some((value) => typeof value === 'number')
        ) {
          queryTablebase(currentBoardHex.value);
        }
      } else {
        ensureDefaultSelection();
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
    'TABLEBASE_QUERY',
  ]);

  const triggerAction = (action, payload = {}) => {
    if (protectedActions.has(action) && !requireAuth()) {
      return false;
    }
    client?.send(action, payload);
    return true;
  };

  const requestTesterBoard = (action, payload = {}) => {
    const requestId = `${clientId}_board_${++nextBoardLoadId}`;
    const clientRevision = Number(localTesterSession?.practice?.revision || 0);
    pendingBoardLoad = { requestId, clientRevision };
    queuedMoveDirections.length = 0;
    results.value = {};
    resultDtype.value = '?';
    lookupPending.value = true;
    ready.value = false;
    if (!triggerAction(action, {
      ...payload,
      request_id: requestId,
      client_revision: clientRevision,
      client_local_board: true,
    })) {
      pendingBoardLoad = null;
      lookupPending.value = false;
      return false;
    }
    return true;
  };

  const reattachTesterTablebase = () => {
    if (!selectedPattern.value || !selectedTarget.value || wsStatus.value !== 'connected') {
      return false;
    }
    const requestId = `${clientId}_attach_${++nextTablebaseAttachId}`;
    pendingTablebaseAttachId = requestId;
    if (!triggerAction('TESTER_SELECT_PATTERN', {
      pattern: selectedPattern.value,
      target: selectedTarget.value,
      request_id: requestId,
      client_local_board: true,
      preserve_client_board: true,
    })) {
      pendingTablebaseAttachId = '';
      return false;
    }
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
    requestTesterBoard('TESTER_SELECT_PATTERN', {
      pattern: selectedPattern.value,
      target: selectedTarget.value,
    });
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
      !selectedTarget.value
    ) {
      return;
    }
    bootstrapSelectionSent = true;
    if (hasLocalPracticeState()) reattachTesterTablebase();
    else applyPatternSelection();
  };

  const resetRandom = () => {
    resetTesterPrefetchState();
    return requestTesterBoard('TESTER_RESET_RANDOM');
  };
  const applyManualBoard = () => {
    const normalized = String(hexInput.value || '').trim().replace(/^0x/iu, '').toLowerCase();
    if (!/^[0-9a-f]{1,16}$/u.test(normalized) || !requireAuth()) return false;
    resetTesterPrefetchState();
    if (!localTesterSession) {
      beginLocalTesterSession({ boardHex: normalized.padStart(16, '0') });
    } else {
      localTesterSession = replaceTesterLocalBoard(localTesterSession, {
        boardHex: normalized.padStart(16, '0'),
        context: testerPrefetchState,
      });
      syncLocalTesterSession({ animate: false });
    }
    results.value = {};
    resultDtype.value = '?';
    lookupPending.value = true;
    const cacheHit = queryTablebase(currentBoardHex.value);
    if (cacheHit && queuedMoveDirections.length) window.queueMicrotask(flushQueuedMove);
    return true;
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
      || !ready.value
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
      client_local_board: true,
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
        if (queuedMoveDirections.length < 4) queuedMoveDirections.push(normalizedDirection);
        return true;
      }
      return false;
    }
    if (!localTesterSession) return false;
    const deterministicSpawn = createTesterSpawnRandomSource(testerPrefetchState);
    const moved = applyTesterLocalMove(localTesterSession, {
      direction: normalizedDirection,
      results: results.value,
      dtype: resultDtype.value,
      spawnRate4: spawnRate4(),
      randomSource: deterministicSpawn.randomSource,
      nextContext: deterministicSpawn.nextState,
    });
    if (!moved.accepted) return false;
    testerPrefetchState = deterministicSpawn.nextState();
    localTesterSession = moved.session;
    syncLocalTesterSession();
    results.value = {};
    resultDtype.value = '?';
    lookupPending.value = true;
    const cacheHit = queryTablebase(currentBoardHex.value);
    persistLatestReplay();
    if (cacheHit && queuedMoveDirections.length) window.queueMicrotask(flushQueuedMove);
    return true;
  };

  const flushQueuedMove = () => {
    const direction = queuedMoveDirections.shift();
    if (!direction || !canMove.value) return;
    window.queueMicrotask(() => {
      if (!move(direction) && (lookupPending.value || queryInFlight.value)) {
        queuedMoveDirections.unshift(direction);
      }
    });
  };

  const saveLog = () => {
    if (!logs.value.length || !requireAuth()) return;
    downloadText(
      logs.value.join('\n'),
      `tester_log_${Math.floor(Date.now() / 1000)}.txt`,
      'text/plain;charset=utf-8',
    );
  };

  const persistLatestReplay = () => {
    if (recordLength.value < 1 || !localTesterSession || !requireAuth()) return;
    const replay = encodeTesterReplay(localTesterSession);
    if (!replay) return;
    const filename = testerReplayFilename(
      currentPatternDisplay.value,
      metrics.value.goodness_of_fit,
    );
    saveLocalTesterReplay({
      buffer: replay,
      filename,
      pattern: currentPatternDisplay.value,
      useVariant: isVariant.value,
    });
    return { replay, filename };
  };

  const saveReplay = () => {
    const latest = persistLatestReplay();
    if (!latest) return;
    downloadBlob(
      new Blob([latest.replay], { type: 'application/octet-stream' }),
      latest.filename,
    );
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
    pendingPracticeJump.value = null;
    resetTesterPrefetchState();
    beginLocalTesterSession({ boardHex: hex, openingLogs: [] });
    results.value = {};
    resultDtype.value = '?';
    ready.value = false;
    tableFound.value = false;
    lookupPending.value = false;
    if (wsStatus.value === 'connected') {
      reattachTesterTablebase();
    }
  };

  const handleTesterBootstrap = (payload) => {
    initialStateSeen = true;
    ensureDefaultSelection();
    maybeApplyInitialPatternSelection();
  };

  const handleTesterBoardSeed = (payload) => {
    const loadRequestId = String(payload?.load_request_id || '');
    const pending = pendingBoardLoad;
    if (!pending || loadRequestId !== pending.requestId) return;
    pendingBoardLoad = null;
    const seedIsCurrent = localTesterSession?.practice
      ? canApplyPracticeSeed(
        localTesterSession.practice,
        pending.clientRevision,
        payload?.client_revision,
      )
      : (
        Number.isInteger(payload?.client_revision)
        && payload.client_revision === pending.clientRevision
      );
    const incomingBoardHex = String(payload?.hex_str || '0000000000000000').toLowerCase();
    const incomingLabels = payload?.metrics?.performance_labels;
    performanceLabels.value = Array.isArray(incomingLabels) && incomingLabels.length
      ? [...incomingLabels]
      : [...fallbackPerformanceLabels];
    ready.value = !!payload?.ready;
    tableFound.value = !!payload?.table_found;
    statusMessage.value = payload?.status || '';
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
    const jumpHex = (
      pendingPracticeJump.value
      && payload?.pattern === pendingPracticeJump.value.pattern
      && String(payload?.target) === pendingPracticeJump.value.target
    ) ? String(pendingPracticeJump.value.hex || '') : '';
    pendingPracticeJump.value = null;
    activeQuery = null;
    queryInFlight.value = false;
    queuedMoveDirections.length = 0;
    results.value = {};
    resultDtype.value = '?';
    if (seedIsCurrent) {
      beginLocalTesterSession({
        board: jumpHex ? undefined : payload?.board,
        boardHex: jumpHex || incomingBoardHex,
        openingLogs: Array.isArray(payload?.logs) ? payload.logs : [],
      });
    }
    lookupPending.value = ready.value && tableFound.value;
    if (lookupPending.value && currentBoardHex.value !== '0000000000000000') {
      queryTablebase(currentBoardHex.value);
    }
  };

  const handleTesterTablebaseReady = (payload) => {
    const requestId = String(payload?.request_id || '');
    if (!pendingTablebaseAttachId || requestId !== pendingTablebaseAttachId) return;
    pendingTablebaseAttachId = '';
    ready.value = !!payload?.ready;
    tableFound.value = !!payload?.table_found;
    statusMessage.value = payload?.status || '';
    if (ready.value && tableFound.value && localTesterSession) {
      results.value = {};
      resultDtype.value = '?';
      lookupPending.value = true;
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
    persistTesterPracticeSoon();
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
    else if (message.action === 'TESTER_BOARD_SEED') handleTesterBoardSeed(message.data);
    else if (message.action === 'TESTER_TABLEBASE_READY') handleTesterTablebaseReady(message.data);
    else if (message.action === 'TESTER_RESULTS') handleTesterResults(message.data);
    else if (message.action === 'TABLEBASE_QUERY_RESULT') handleTablebaseQueryResult(message.data);
    else if (message.action === 'TABLEBASE_PREFETCH') handleTablebasePrefetch(message.data);
    else if (message.action === 'TABLEBASE_CATALOG_UPDATED') {
      loadCatalog();
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
      lookupPending.value = false;
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
        loadCatalog();
        triggerAction('TESTER_GET_INIT', { client_local_board: true });
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
    pendingBoardLoad = null;
    pendingTablebaseAttachId = '';
    queuedMoveDirections.length = 0;
    client?.disconnect();
    client = null;
    wsStatus.value = 'disconnected';
  };

  const handleKeyDown = (event) => {
    if (!activeRef?.value || !keyboardInputAllowed(KEYBOARD_OWNERS.PRIMARY)) return;
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
    if (code === 'ArrowUp' || code === 'KeyW' || code === 'KeyK') { event.preventDefault(); move('up'); }
    else if (code === 'ArrowDown' || code === 'KeyS' || code === 'KeyJ') { event.preventDefault(); move('down'); }
    else if (code === 'ArrowLeft' || code === 'KeyA' || code === 'KeyH') { event.preventDefault(); move('left'); }
    else if (code === 'ArrowRight' || code === 'KeyD' || code === 'KeyL') { event.preventDefault(); move('right'); }
    else if (code === 'KeyR') { event.preventDefault(); resetRandom(); }
    else if (code === 'KeyF') { event.preventDefault(); toggleInsights(); }
  };

  onMounted(() => {
    restoreTesterPractice();
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
      restoreTesterPractice();
      maybeApplyInitialPatternSelection();
    } else {
      if (persistPracticeTimer) {
        window.clearTimeout(persistPracticeTimer);
        persistPracticeTimer = null;
      }
      practiceRestoreAttempted = false;
      clearTesterPracticeState();
      localTesterSession = null;
      pendingBoardLoad = null;
      board.value = new Array(16).fill(0);
      boardFrameRevision += 1;
      boardFrame.value = createSnapshotBoardFrame(boardFrameRevision, board.value);
      currentBoardHex.value = '0000000000000000';
      hexInput.value = currentBoardHex.value;
      results.value = {};
      resultDtype.value = '?';
      logs.value = [];
      recordLength.value = 0;
      ready.value = false;
      tableFound.value = false;
      lookupPending.value = false;
    }
  });

  onUnmounted(() => {
    if (persistPracticeTimer) {
      window.clearTimeout(persistPracticeTimer);
      persistPracticeTimer = null;
      persistTesterPractice();
    }
    window.removeEventListener('keydown', handleKeyDown, true);
    window.removeEventListener('tester-practice-jump', handlePracticeJump);
    document.removeEventListener('click', closePatternMenuOnClick);
    disconnect();
  });

  return {
    wsStatus,
    board,
    boardFrame,
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
