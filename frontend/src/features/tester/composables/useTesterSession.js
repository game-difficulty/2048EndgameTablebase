import { computed, onMounted, onUnmounted, ref, watch } from 'vue';

import { useAppSettingsStore } from '../../../app/useAppSettings';
import { createWsClient } from '../../../services/ws/createWsClient';
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
  const { config: appConfig } = useAppSettingsStore();

  const fallbackPatternCategories = {
    basic: ['L3', 'L4', 'I3', 'I4', 'LL', 'free8', 'free9', 'free10', '444'],
  };
  const performanceLabels = ['Perfect!', 'Excellent!', 'Nice try!', 'Not bad!', 'Mistake!', 'Blunder!', 'Terrible!'];
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

  const wsStatus = ref('connecting');
  const board = ref(new Array(16).fill(0));
  const metadata = ref({});
  const dis32k = ref(false);
  const currentLanguage = ref('en');
  const showInsights = ref(true);
  const patternCategories = ref(fallbackPatternCategories);
  const availableTargets = ref(['64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384']);
  const selectedPattern = ref('L3');
  const selectedTarget = ref('512');
  const activePatternCategory = ref(Object.keys(fallbackPatternCategories)[0] || '');
  const patternMenuOpen = ref(false);
  const currentBoardHex = ref('0000000000000000');
  const hexInput = ref('0000000000000000');
  const resultDtype = ref('?');
  const results = ref({});
  const logs = ref([]);
  const tableFound = ref(false);
  const ready = ref(false);
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

  let client = null;
  let bootstrapSelectionSent = false;

  const patternGroups = computed(() =>
    Object.entries(patternCategories.value || {}).map(([category, patterns]) => ({
      category,
      patterns: Array.isArray(patterns) ? patterns : [],
    }))
  );
  const flatPatterns = computed(() => patternGroups.value.flatMap((group) => group.patterns));
  const activePatternOptions = computed(() => patternCategories.value[activePatternCategory.value] || []);
  const currentPatternDisplay = computed(() => (
    selectedPattern.value && selectedTarget.value ? `${selectedPattern.value}_${selectedTarget.value}` : 'Select Pattern'
  ));
  const isVariant = computed(() => isVariantPattern(selectedPattern.value, patternCategories.value));
  const canMove = computed(() => ready.value && tableFound.value && wsStatus.value === 'connected');
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
  const evaluationTotal = computed(() => performanceLabels.reduce((sum, label) => sum + Number(metrics.value.performance_stats?.[label] || 0), 0));

  const isZh = () => String(currentLanguage.value || 'en').startsWith('zh');
  const getEvaluationLabel = (label) => (isZh() ? (zhEvaluationLabels[label] || label) : label);

  const evaluationSegments = computed(() => performanceLabels.map((label) => {
    const count = Number(metrics.value.performance_stats?.[label] || 0);
    const total = evaluationTotal.value || 1;
    const percent = evaluationTotal.value ? (count / total) * 100 : 0;
    const name = getEvaluationLabel(label);
    return {
      label,
      shortLabel: name,
      count,
      percent,
      color: evaluationColors[label] || 'var(--border-main)',
      tooltip: `${name}: ${count}/${evaluationTotal.value || 0} (${percent.toFixed(1)}%)`,
    };
  }));

  const displayedEvaluationSegments = computed(() => (
    insightsActive.value
      ? evaluationSegments.value
      : performanceLabels.map((label) => ({
        label,
        shortLabel: getEvaluationLabel(label),
        count: 0,
        percent: 0,
        color: evaluationColors[label] || 'var(--border-main)',
        tooltip: `${getEvaluationLabel(label)}: 0`,
      }))
  ));

  const moveLabels = computed(() => (
    isZh()
      ? { left: '左', right: '右', up: '上', down: '下' }
      : { left: 'Left', right: 'Right', up: 'Up', down: 'Down' }
  ));

  const feedbackEvaluation = computed(() => insightsActive.value ? (lastStep.value.evaluation || 'Perfect!') : 'waiting');
  const feedbackBadgeText = computed(() => {
    if (!showInsights.value) return '--';
    if (!hasLastStep.value) return isZh() ? '等待落子' : 'Waiting';
    return getEvaluationLabel(lastStep.value.evaluation || 'Perfect!');
  });
  const feedbackBadgeStyle = computed(() => {
    if (!showInsights.value || !hasLastStep.value) {
      return { color: 'var(--text-secondary)' };
    }
    const color = evaluationColors[lastStep.value.evaluation] || 'var(--accent)';
    return { color };
  });
  const feedbackLossText = computed(() => {
    if (!hasLastStep.value || !showInsights.value) return '';
    if (feedbackEvaluation.value === 'Perfect!') return '';
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
    const color = insightsActive.value ? (evaluationColors[evaluation] || 'var(--text-main)') : 'var(--text-secondary)';
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

  const ensureDefaultSelection = () => {
    const groups = patternGroups.value;
    if (!groups.length) return;
    selectedPattern.value = flatPatterns.value.includes(selectedPattern.value)
      ? selectedPattern.value
      : (groups[0].patterns[0] || '');
    selectedTarget.value = availableTargets.value.includes(selectedTarget.value)
      ? selectedTarget.value
      : (availableTargets.value.includes('512') ? '512' : (availableTargets.value[0] || ''));
    syncCategoryFromPattern(selectedPattern.value);
  };

  const togglePatternMenu = () => {
    syncCategoryFromPattern(selectedPattern.value);
    patternMenuOpen.value = !patternMenuOpen.value;
  };

  const triggerAction = (action, payload = {}) => {
    client?.send(action, payload);
  };

  const selectPattern = (pattern) => {
    selectedPattern.value = pattern;
    syncCategoryFromPattern(pattern);
    patternMenuOpen.value = false;
    applyPatternSelection();
  };

  const closePatternMenuOnClick = (event) => {
    if (!patternMenuOpen.value || !patternMenuRoot.value) return;
    if (!patternMenuRoot.value.contains(event.target)) patternMenuOpen.value = false;
  };

  const applyPatternSelection = () => {
    if (!selectedPattern.value || !selectedTarget.value) return;
    syncCategoryFromPattern(selectedPattern.value);
    triggerAction('TESTER_SELECT_PATTERN', { pattern: selectedPattern.value, target: selectedTarget.value });
  };

  const resetRandom = () => triggerAction('TESTER_RESET_RANDOM');
  const applyManualBoard = () => {
    if (hexInput.value.trim()) {
      triggerAction('TESTER_SET_BOARD', { hex_str: hexInput.value.trim() });
    }
  };
  const toggleInsights = () => { showInsights.value = !showInsights.value; };
  const move = (dir) => canMove.value && triggerAction('TESTER_MOVE', { dir });

  const saveLog = async () => {
    if (!logs.value.length || !window.pywebview?.api?.select_save_tester_log) return;
    const path = await window.pywebview.api.select_save_tester_log();
    if (path) triggerAction('TESTER_SAVE_LOG', { path });
  };

  const saveReplay = async () => {
    if (recordLength.value < 1 || !window.pywebview?.api?.select_save_tester_replay) return;
    const path = await window.pywebview.api.select_save_tester_replay();
    if (path) triggerAction('TESTER_SAVE_REPLAY', { path });
  };

  const handlePracticeJump = (event) => {
    const detail = event?.detail || {};
    const parsed = parseFullPattern(detail.fullPattern);
    const hex = String(detail.hex || '').trim();
    if (!parsed || !hex) return;

    pendingPracticeJump.value = { ...parsed, hex };
    selectedPattern.value = parsed.pattern;
    selectedTarget.value = parsed.target;
    syncCategoryFromPattern(parsed.pattern);
    if (wsStatus.value === 'connected') {
      triggerAction('TESTER_SELECT_PATTERN', { pattern: parsed.pattern, target: parsed.target });
    }
  };

  const handleTesterBootstrap = (payload) => {
    patternCategories.value = payload?.categories || fallbackPatternCategories;
    availableTargets.value = (payload?.target_tiles || []).map(String);
    ensureDefaultSelection();
    if (!bootstrapSelectionSent && selectedPattern.value && selectedTarget.value) {
      bootstrapSelectionSent = true;
      applyPatternSelection();
    }
  };

  const handleTesterState = (payload) => {
    board.value = Array.isArray(payload?.board) ? payload.board : new Array(16).fill(0);
    metadata.value = payload?.animation || {};
    currentBoardHex.value = payload?.hex_str || currentBoardHex.value;
    hexInput.value = currentBoardHex.value;
    resultDtype.value = payload?.dtype || '?';
    results.value = payload?.results || {};
    logs.value = Array.isArray(payload?.logs) ? payload.logs : [];
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
    if (payload?.target && payload.target !== '?' && availableTargets.value.includes(String(payload.target))) {
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
  };

  const handleWSMessage = (message) => {
    if (message.action === 'TESTER_BOOTSTRAP') handleTesterBootstrap(message.data);
    else if (message.action === 'TESTER_STATE') handleTesterState(message.data);
  };

  const connect = () => {
    if (client) {
      return;
    }
    client = createWsClient({
      clientId: `tester_${Math.random().toString(36).slice(2, 9)}`,
      onOpen: () => {
        wsStatus.value = 'connected';
        bootstrapSelectionSent = false;
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
    const target = event.target;
    if (target instanceof HTMLElement) {
      if (target.isContentEditable) return;
      if (target.closest('[data-tester-text-input="true"]')) return;
    }
    const code = event.code;
    if (code === 'ArrowUp' || code === 'KeyW') { event.preventDefault(); move('up'); }
    else if (code === 'ArrowDown' || code === 'KeyS') { event.preventDefault(); move('down'); }
    else if (code === 'ArrowLeft' || code === 'KeyA') { event.preventDefault(); move('left'); }
    else if (code === 'ArrowRight' || code === 'KeyD') { event.preventDefault(); move('right'); }
    else if (code === 'KeyR') { event.preventDefault(); resetRandom(); }
    else if (code === 'KeyF') { event.preventDefault(); toggleInsights(); }
  };

  onMounted(() => {
    syncCategoryFromPattern(selectedPattern.value);
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
      }
    },
    { immediate: true }
  );

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
    selectedPattern,
    selectedTarget,
    activePatternCategory,
    patternMenuOpen,
    hexInput,
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
    saveLog,
    saveReplay,
    getResultRowStyle,
    getResultValueStyle,
    getResultMiniTileStyle,
  };
}
