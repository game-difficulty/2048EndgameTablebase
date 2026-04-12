import { computed, onMounted, onUnmounted, ref, watch } from 'vue';

import { useAppSettingsStore } from '../../../app/useAppSettings';
import { createWsClient } from '../../../services/ws/createWsClient';
import { isVariantPattern } from '../../../utils/patternCategories';
import { createResultBarGradient } from '../../../utils/resultBars';
import { restoreSuccessRate, formatSuccessRate } from '../../../utils/successRate';

export function useTrainerSession(activeRef) {
  const {
    config: appConfig,
    categories: appCategories,
    targetTiles: appTargetTiles,
    refreshSettings,
    saveSetting,
  } = useAppSettingsStore();

  const fallbackPatternCategories = {
    basic: ['L3', 'L4', 'I3', 'I4', 'LL', 'free8', 'free9', 'free10', '444'],
  };
  const wsStatus = ref('connecting');
  const clientId = `trainer_${Math.random().toString(36).substring(2, 9)}`;

  const board = ref([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const metadata = ref(null);
  const hexInput = ref('');
  const cellPalette = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
  const currentPaletteValue = ref(null);
  const dis32k = ref(false);
  const awaitingSpawn = ref(false);

  const tablebasePath = ref('');
  const patternType = ref('L3');
  const targetValue = ref('512');
  const tableResult = ref({ dtype: '?', results: {} });
  const currentBoardHex = ref('');
  const resultsBoardHex = ref('');
  const patternCategories = ref(fallbackPatternCategories);
  const availableTargets = ref(['64', '128', '256', '512', '1024', '2048', '4096', '8192']);
  const patternMenuOpen = ref(false);
  const activePatternCategory = ref(Object.keys(fallbackPatternCategories)[0] || '');
  const patternMenuRoot = ref(null);
  const dirLabels = { left: 'L', right: 'R', down: 'D', up: 'U' };

  const spawnMode = ref(0);
  const spawnModes = ['Random', 'Best', 'Worst', 'Manual'];
  const demoActive = ref(false);
  const showResults = ref(true);
  const recordingState = ref(false);
  const replayResultsActive = ref(false);
  const queuedStepCount = ref(0);
  const stepExecutionPending = ref(false);
  const demoSpeed = ref(40);
  let nextResultsRequestId = 0;
  const pendingResultsRequests = new Map();
  const pendingTrainerJump = ref(null);
  let demoTimer = null;

  const recordStep = ref(0);
  const recordMax = ref(0);
  const recordingLength = ref(0);
  const fullHistory = ref([]);
  const fullMoves = ref([]);

  let client = null;

  const currentPatternDisplay = computed(() => `${patternType.value}_${targetValue.value}`);
  const isVariant = computed(() => isVariantPattern(patternType.value, patternCategories.value));
  const patternGroups = computed(() =>
    Object.entries(patternCategories.value || {}).map(([category, patterns]) => ({
      category,
      patterns: Array.isArray(patterns) ? patterns : [],
    }))
  );
  const flatPatterns = computed(() => patternGroups.value.flatMap((group) => group.patterns));
  const activePatternOptions = computed(() => patternCategories.value[activePatternCategory.value] || []);
  const hasUsableResults = computed(() =>
    Object.values(tableResult.value.results || {}).some((val) => typeof val === 'number' && Number.isFinite(val))
  );
  const hasPlayableResults = computed(() =>
    Object.values(tableResult.value.results || {}).some((val) => {
      const restored = restoreSuccessRate(val, tableResult.value.dtype || '');
      return typeof restored === 'number' && Number.isFinite(restored) && restored > 0;
    })
  );
  const bestResultMove = computed(() => {
    const dtype = tableResult.value.dtype || '';
    const orderedMoves = Object.entries(tableResult.value.results || {})
      .map(([dir, val]) => [dir, restoreSuccessRate(val, dtype)])
      .filter(([, val]) => typeof val === 'number' && Number.isFinite(val) && val > 0)
      .sort(([, left], [, right]) => right - left);
    return orderedMoves[0]?.[0] || null;
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
      return {
        dir,
        rawVal: (rawVal == null || typeof rawVal !== 'number') ? null : rawVal,
        val: val == null ? null : val,
      };
    });
    items.sort((a, b) => {
      if (a.val == null && b.val == null) return 0;
      if (a.val == null) return 1;
      if (b.val == null) return -1;
      return b.val - a.val;
    });
    const bestVal = items.find((i) => i.val != null)?.val || 0;
    const prec = resultPrecision.value;

    const COLOR_GREEN = '#4caf50';
    const COLOR_YG = '#8bc34a';
    const COLOR_ORANGE = '#ff9800';
    const COLOR_RED = '#f44336';

    return items.map((item, idx) => {
      let pct = 0;
      let color = 'var(--border-main)';
      if (item.val != null && bestVal > 0) {
        const loss = 1 - item.val / bestVal;
        if (idx === 0) {
          pct = 100;
          color = COLOR_GREEN;
        } else if (loss > 0.10) {
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

  const getResultRowStyle = (item) => ({
    background: item.val != null ? 'var(--bg-main)' : 'transparent',
    opacity: item.val == null ? 0.5 : 1,
  });

  const getResultValueStyle = (item) => {
    const length = item.display?.length || 0;
    let fontSize = '19px';
    if (length >= 21) {
      fontSize = '15px';
    } else if (length >= 19) {
      fontSize = '16px';
    } else if (length >= 17) {
      fontSize = '17px';
    }

    return {
      color: item.textColor,
      fontSize,
    };
  };

  const syncActivePatternCategory = () => {
    const matchedGroup = patternGroups.value.find((group) => group.patterns.includes(patternType.value));
    activePatternCategory.value = matchedGroup?.category || patternGroups.value[0]?.category || '';
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

  const triggerAction = (action, payload = {}) => {
    client?.send(action, payload);
  };

  const applyTrainerJump = () => {
    const pending = pendingTrainerJump.value;
    if (!pending || wsStatus.value !== 'connected') return;

    hexInput.value = pending.hex;
    currentBoardHex.value = pending.hex;
    triggerAction('SET_BOARD', { hex_str: pending.hex });

    const parsed = parseFullPattern(pending.fullPattern);
    if (parsed && flatPatterns.value.includes(parsed.pattern) && availableTargets.value.includes(parsed.target)) {
      const shouldSwitchPattern = currentPatternDisplay.value !== pending.fullPattern;
      if (shouldSwitchPattern) {
        patternType.value = parsed.pattern;
        targetValue.value = parsed.target;
        syncActivePatternCategory();
        applyTablebase();
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
    onPatternChange();
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

  const invalidateResults = () => {
    resultsBoardHex.value = '';
    if (!replayResultsActive.value) {
      tableResult.value = { dtype: '?', results: {} };
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
    demoTimer = window.setTimeout(() => {
      demoTimer = null;
      if (!demoActive.value || awaitingSpawn.value) return;
      trainerStep();
    }, Math.max(1, delayMs));
  };

  const queryResults = (reason = 'manual') => {
    if ((reason !== 'step' && !showResults.value) || awaitingSpawn.value || replayResultsActive.value) return null;
    const boardHex = currentBoardHex.value || hexInput.value;
    if (!boardHex) return null;
    if (resultsBoardHex.value === boardHex || hasPendingResultsForBoard(boardHex)) return null;

    const requestId = `${clientId}_${++nextResultsRequestId}`;
    pendingResultsRequests.set(requestId, { boardHex, reason });
    triggerAction('TRAINER_GET_RESULTS', { request_id: requestId });
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
    if (data.action === 'RECORDING_STARTED') {
      recordingState.value = true;
      recordingLength.value = data.data?.recording_length ?? 1;
      return;
    }

    if (data.action === 'RECORDING_STOPPED') {
      recordingState.value = false;
      recordingLength.value = data.data?.recording_length ?? 0;
      return;
    }

    if (data.action === 'RECORD_SAVE_REQUIRED') {
      let path = null;
      if (window.pywebview && window.pywebview.api) {
        path = await window.pywebview.api.select_save_record();
      } else {
        triggerAction('TRIGGER_RECORD_SAVE');
        return;
      }
      if (path) triggerAction('STOP_RECORDING', { path });
      return;
    }

    if (data.action === 'UPDATE_STATE') {
      metadata.value = data.data.animation;
      board.value = data.data.board;
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
        fullHistory.value = data.data.history || [];
        fullMoves.value = data.data.moves || [];
      }
      if (data.data.record_results_mode === 'embedded') {
        replayResultsActive.value = true;
        tableResult.value = {
          dtype: data.data.record_results_dtype || 'recorded',
          results: data.data.record_results || {},
        };
        resultsBoardHex.value = currentBoardHex.value;
      } else if (replayResultsActive.value) {
        replayResultsActive.value = false;
        invalidateResults();
      }
      awaitingSpawn.value = !!data.data.awaiting_spawn;
      if (boardChanged && !replayResultsActive.value) {
        invalidateResults();
      }
      if (awaitingSpawn.value) {
        demoActive.value = false;
        clearDemoTimer();
        clearStepQueue();
      } else if (boardChanged) {
        stepExecutionPending.value = false;
        queryResults(queuedStepCount.value > 0 || demoActive.value ? 'step' : 'auto');
      }

      return;
    }

    if (data.action === 'TRAINER_RESULTS') {
      const requestId = data.data.request_id;
      const resultBoardHex = data.data.board_hex || currentBoardHex.value;
      if (requestId && pendingResultsRequests.has(requestId)) {
        pendingResultsRequests.delete(requestId);
      }
      if (resultBoardHex !== currentBoardHex.value) {
        return;
      }
      replayResultsActive.value = false;
      tableResult.value = {
        dtype: data.data.dtype,
        results: data.data.results,
      };
      resultsBoardHex.value = resultBoardHex;
      if (!hasPlayableResults.value) {
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

    if (data.action === 'FOLDER_SELECTED') {
      if (data.data.path) {
        tablebasePath.value = data.data.path;
        applyTablebase();
      }
      return;
    }

    if (data.action === 'DO_API_CALLBACK') {
      const type = data.data.type;
      if (type === 'RECORD_OPEN') triggerAction('RECORD_OPEN', { path: data.data.path });
      if (type === 'RECORD_SAVE') triggerAction('RECORD_SAVE', { path: data.data.path });
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
        triggerAction('GET_STATE');
      },
      onMessage: handleMessage,
      onClose: () => {
        wsStatus.value = 'disconnected';
        demoActive.value = false;
        clearDemoTimer();
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
    pendingResultsRequests.clear();
    clearStepQueue();
    client?.disconnect();
    client = null;
    wsStatus.value = 'disconnected';
  };

  const setBoard = () => {
    if (!hexInput.value) return;
    demoActive.value = false;
    clearDemoTimer();
    clearStepQueue();
    triggerAction('SET_BOARD', { hex_str: hexInput.value });
  };

  const TILE_SEQUENCE = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

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

    if (btn === 0) {
      if (currentPaletteValue.value !== null) {
        demoActive.value = false;
        clearDemoTimer();
        clearStepQueue();
        triggerAction('SET_CELL', { row, col, val: currentPaletteValue.value });
      }
    } else if (btn === 2) {
      const cellVal = board.value[row * 4 + col];
      const idx = TILE_SEQUENCE.indexOf(cellVal);
      const nextVal = (idx >= 0 && idx < TILE_SEQUENCE.length - 1) ? TILE_SEQUENCE[idx + 1] : cellVal;
      demoActive.value = false;
      clearDemoTimer();
      clearStepQueue();
      triggerAction('SET_CELL', { row, col, val: nextVal });
    } else {
      const cellVal = board.value[row * 4 + col];
      const idx = TILE_SEQUENCE.indexOf(cellVal);
      const prevVal = idx > 0 ? TILE_SEQUENCE[idx - 1] : 0;
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
    applyTablebase();
  };

  const selectFolder = async () => {
    try {
      if (window.pywebview && window.pywebview.api) {
        const path = await window.pywebview.api.select_folder();
        if (path) {
          tablebasePath.value = path;
          applyTablebase();
        }
      } else {
        triggerAction('TRIGGER_SELECT_FOLDER');
      }
    } catch (error) {
      console.error(error);
    }
  };

  const applyTablebase = () => {
    const fullPattern = `${patternType.value}_${targetValue.value}`;
    triggerAction('TRAINER_SET_FILEPATH', {
      filepath: tablebasePath.value,
      pattern: fullPattern,
      target: targetValue.value,
    });
  };

  const trainerStep = () => {
    queuedStepCount.value += 1;
    pumpQueuedSteps();
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
      if (resultsBoardHex.value === currentBoardHex.value && hasPlayableResults.value) {
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
    if (cmd === 'TOGGLE') {
      if (recordingState.value) {
        triggerAction('PREPARE_STOP_RECORDING');
      } else {
        triggerAction('START_RECORDING');
      }
    } else if (cmd === 'OPEN') {
      if (window.pywebview && window.pywebview.api) {
        const path = await window.pywebview.api.select_open_record();
        if (path) triggerAction('RECORD_OPEN', { path });
      } else {
        triggerAction('TRIGGER_RECORD_OPEN');
      }
    } else if (cmd === 'PREV' || cmd === 'NEXT') {
      triggerAction('RECORD_STEP', { dir: cmd === 'PREV' ? -1 : 1 });
    }
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
      if (awaitingSpawn.value) return;
      demoActive.value = false;
      clearDemoTimer();
      clearStepQueue();
      triggerAction('TRAINER_MOVE', { dir: map[event.code] });
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
      patternCategories.value = Object.keys(nextCategories || {}).length > 0
        ? nextCategories
        : fallbackPatternCategories;
      if (!flatPatterns.value.includes(patternType.value)) {
        patternType.value = flatPatterns.value[0] || patternType.value;
      }
      syncActivePatternCategory();
      applyTrainerJump();
    },
    { immediate: true, deep: true }
  );

  watch(
    appTargetTiles,
    (nextTargets) => {
      const normalizedTargets = (nextTargets || []).map(String);
      if (normalizedTargets.length > 0) {
        availableTargets.value = normalizedTargets;
      }
      if (!availableTargets.value.includes(targetValue.value)) {
        targetValue.value = availableTargets.value[0] || targetValue.value;
      }
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
      }
    },
    { immediate: true }
  );

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
    togglePatternMenu,
    selectPattern,
    patternType,
    patternMenuOpen,
    patternGroups,
    activePatternCategory,
    activePatternOptions,
    targetValue,
    availableTargets,
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
    getResultRowStyle,
    dirLabels,
    getResultValueStyle,
    tableResult,
    toggleDemo,
    demoActive,
    trainerStep,
    trainerUndo,
    trainerDefault,
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
