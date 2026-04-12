import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { useAppSettingsStore } from '../../../app/useAppSettings';
import { createWsClient } from '../../../services/ws/createWsClient';
import { isVariantPattern } from '../../../utils/patternCategories';
import { createResultBarGradient } from '../../../utils/resultBars';

export function useReplaySession(activeRef, emit) {
  const RESULT_REFRESH_GRACE_MS = 600;
  const RESULT_REFRESH_PLACEHOLDER_MS = 1800;
  const { config: appConfig, categories: appCategories, saveSetting } = useAppSettingsStore();
  const { t } = useI18n();

  const COLOR_GREEN = '#2e7d32';
  const COLOR_YG = '#8bc34a';
  const COLOR_ORANGE = '#ff9800';
  const COLOR_RED = '#f44336';
  const dirLabels = { left: 'L', right: 'R', up: 'U', down: 'D' };
  const evaluationColors = {
    'Perfect!': '#2e7d32',
    'Excellent!': '#7cb342',
    'Nice try!': '#c0ca33',
    'Not bad!': '#fb8c00',
    'Mistake!': '#f4511e',
    'Blunder!': '#e53935',
    'Terrible!': '#b71c1c',
  };
  const performanceLabels = ['Perfect!', 'Excellent!', 'Nice try!', 'Not bad!', 'Mistake!', 'Blunder!', 'Terrible!'];
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
  const currentHex = ref('0000000000000000');
  const loaded = ref(false);
  const replayStatus = ref('');
  const replayPattern = ref('');
  const replaySource = ref('');
  const currentStep = ref(0);
  const totalSteps = ref(0);
  const replayResults = ref({});
  const currentMove = ref(null);
  const bestMove = ref(null);
  const loss = ref(null);
  const goodnessOfFit = ref(null);
  const combo = ref(0);
  const summary = ref({ total_moves: 0, final_gof: 0, max_combo: 0, counts: {} });
  const losses = ref([]);
  const sliderThreshold = ref(1);
  const demoSpeed = ref(40);
  const dis32k = ref(false);
  const currentLanguage = ref('en');
  const menuOpen = ref(false);
  const menuRoot = ref(null);
  const demoActive = ref(false);

  let client = null;
  let demoTimer = null;
  let resultsStaleTimer = null;
  let resultsPlaceholderTimer = null;
  const resultsRefreshPhase = ref('idle');

  const makePlaceholderResults = () => ['up', 'down', 'right', 'left'].map((dir) => ({
    dir,
    pct: 0,
    gradient: 'transparent',
    display: '0',
    textColor: 'var(--text-secondary)',
    val: null,
  }));

  const isZh = () => String(currentLanguage.value || 'en').startsWith('zh');
  const getEvaluationLabel = (label) => (isZh() ? (zhEvaluationLabels[label] || label) : label);
  const trimTrailingZeros = (value) =>
    value.replace(/(\.\d*?[1-9])0+$/u, '$1').replace(/\.0+$/u, '').replace(/\.$/u, '');
  const formatReplayRate = (value) => trimTrailingZeros(Number(value || 0).toFixed(9));

  const lerpColor = (c1, c2, ratio) => {
    const parseRgbColor = (color) => color.slice(1).match(/.{2}/g).map(part => parseInt(part, 16));
    const mix = (a, b) => Math.round(a + (b - a) * ratio);
    const [r1, g1, b1] = parseRgbColor(c1);
    const [r2, g2, b2] = parseRgbColor(c2);
    return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`;
  };

  const fileDisplay = computed(() => {
    if (replaySource.value) {
      const source = String(replaySource.value);
      const parts = source.split(/[\\/]/u);
      const display = parts[parts.length - 1] || source;
      const normalizedDisplay = display.toLowerCase().replace(/[\s-]+/gu, '_');
      if (normalizedDisplay === 'tester_session') return t('replay.status.testerSession');
      return display;
    }
    if (replayPattern.value) return replayPattern.value;
    return t('replay.status.noReplayLoaded');
  });

  const goodnessDisplay = computed(() => Number(goodnessOfFit.value ?? 0).toFixed(4));
  const summaryMaxCombo = computed(() => Number(summary.value?.max_combo ?? 0));

  const sortedResults = computed(() => {
    const entries = ['left', 'right', 'down', 'up']
      .map((dir) => {
        const val = replayResults.value?.[dir];
        return { dir, val: typeof val === 'number' ? val : null };
      })
      .sort((a, b) => {
        if (a.val == null && b.val == null) return 0;
        if (a.val == null) return 1;
        if (b.val == null) return -1;
        return b.val - a.val;
      });
    const bestVal = entries.find((item) => item.val != null)?.val || 0;
    return entries.map((item, index) => {
      let pct = 0;
      let color = 'var(--border-main)';
      if (item.val != null && bestVal > 0) {
        const relLoss = 1 - item.val / bestVal;
        if (index === 0) {
          pct = 100;
          color = COLOR_GREEN;
        } else if (relLoss <= 0.10) {
          pct = (1 - relLoss / 0.10) * 100;
          color = relLoss <= 0.001
            ? COLOR_GREEN
            : (relLoss <= 0.01
              ? lerpColor(COLOR_GREEN, COLOR_YG, (relLoss - 0.001) / 0.009)
              : (relLoss <= 0.03
                ? lerpColor(COLOR_YG, COLOR_ORANGE, (relLoss - 0.01) / 0.02)
                : lerpColor(COLOR_ORANGE, COLOR_RED, (relLoss - 0.03) / 0.07)));
        } else {
          color = COLOR_RED;
        }
      }
      return {
        ...item,
        pct,
        gradient: color.startsWith('#') ? createResultBarGradient(color) : color,
        display: item.val == null ? '--' : formatReplayRate(item.val),
        textColor: item.val == null ? 'var(--text-secondary)' : 'var(--text-main)',
      };
    });
  });

  const displayedResults = computed(() => {
    if (loaded.value && currentStep.value < totalSteps.value) {
      return resultsRefreshPhase.value === 'placeholder' ? makePlaceholderResults() : sortedResults.value;
    }
    return makePlaceholderResults();
  });

  const resultsRefreshing = computed(() => resultsRefreshPhase.value !== 'idle');

  const currentEvaluation = computed(() => {
    if (loss.value == null) return null;
    const val = Number(loss.value);
    if (val > 1 - 3e-10) return 'Perfect!';
    if (val >= 0.999) return 'Excellent!';
    if (val >= 0.99) return 'Nice try!';
    if (val >= 0.975) return 'Not bad!';
    if (val >= 0.9) return 'Mistake!';
    if (val >= 0.75) return 'Blunder!';
    return 'Terrible!';
  });

  const feedbackBadgeText = computed(() => {
    if (!loaded.value) return replayStatus.value || t('replay.status.noReplayLoaded');
    if (currentStep.value >= totalSteps.value) return t('replay.status.replayComplete');
    return getEvaluationLabel(currentEvaluation.value || 'Perfect!');
  });

  const feedbackBadgeStyle = computed(() => {
    if (!loaded.value || currentStep.value >= totalSteps.value) {
      return { color: 'var(--text-secondary)' };
    }
    return { color: evaluationColors[currentEvaluation.value] || 'var(--accent)' };
  });

  const feedbackLossText = computed(() => {
    if (
      !loaded.value ||
      currentStep.value >= totalSteps.value ||
      currentEvaluation.value === 'Perfect!' ||
      loss.value == null
    ) {
      return '';
    }
    return isZh()
      ? `单步损失 ${((1 - Number(loss.value)) * 100).toFixed(2)}%`
      : `One-step loss ${((1 - Number(loss.value)) * 100).toFixed(2)}%`;
  });

  const moveLabels = computed(() => (
    isZh()
      ? { left: '左', right: '右', up: '上', down: '下' }
      : { left: 'Left', right: 'Right', up: 'Up', down: 'Down' }
  ));
  const feedbackPressedLabel = computed(() => (isZh() ? '你走的是' : 'You pressed'));
  const feedbackBestLabel = computed(() => (isZh() ? '最优解' : 'Best move'));
  const feedbackConnector = computed(() => (isZh() ? '·' : 'and'));
  const feedbackPressedMove = computed(() => (
    loaded.value && currentStep.value < totalSteps.value
      ? (moveLabels.value[currentMove.value] || '--')
      : '--'
  ));
  const feedbackBestMove = computed(() => (
    loaded.value && currentStep.value < totalSteps.value
      ? (moveLabels.value[bestMove.value] || '--')
      : '--'
  ));
  const feedbackPressedMoveStyle = computed(() => ({
    color: loaded.value && currentStep.value < totalSteps.value
      ? (evaluationColors[currentEvaluation.value] || 'var(--text-main)')
      : 'var(--text-secondary)',
  }));
  const feedbackBestMoveStyle = computed(() => ({
    color: loaded.value && currentStep.value < totalSteps.value ? COLOR_GREEN : 'var(--text-secondary)',
  }));

  const evaluationTotal = computed(() => Number(summary.value?.total_moves || 0));
  const evaluationSegments = computed(() => performanceLabels.map((label) => {
    const count = Number(summary.value?.counts?.[label] || 0);
    const total = evaluationTotal.value || 1;
    return {
      label,
      shortLabel: getEvaluationLabel(label),
      count,
      percent: evaluationTotal.value ? (count / total) * 100 : 0,
      color: evaluationColors[label] || 'var(--border-main)',
    };
  }));

  const markerIndices = computed(() => {
    const arr = Array.isArray(losses.value) ? losses.value.map(Number).filter(Number.isFinite) : [];
    if (!arr.length) return [];
    const sorted = [...arr].sort((a, b) => a - b);
    const qIndex = Math.max(
      0,
      Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * 0.1))
    );
    const threshold = Math.min(sorted[qIndex], Number(sliderThreshold.value) || 1);
    return arr
      .map((item, index) => ({ item, index }))
      .filter(({ item }) => item < 1 && item < threshold)
      .map(({ index }) => index);
  });

  const hasNextPoint = computed(() =>
    markerIndices.value.some((point) => point > currentStep.value)
  );

  const getResultValueStyle = (item) => {
    const length = item.display?.length || 0;
    const fontSize = length >= 21 ? '15px' : (length >= 19 ? '16px' : (length >= 17 ? '17px' : '19px'));
    return { color: item.textColor, fontSize };
  };

  const stopDemo = () => {
    if (demoTimer) window.clearTimeout(demoTimer);
    demoTimer = null;
    demoActive.value = false;
  };

  const triggerAction = (action, payload = {}) => {
    client?.send(action, payload);
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

  const scheduleDemo = () => {
    if (demoTimer) window.clearTimeout(demoTimer);
    if (!demoActive.value) return;
    if (!loaded.value || currentStep.value >= totalSteps.value) {
      stopDemo();
      return;
    }
    const delayMs = Math.max(1, Math.round(Number(demoSpeed.value) || 40));
    demoTimer = window.setTimeout(() => {
      if (!demoActive.value) return;
      startResultsRefresh();
      triggerAction('REPLAY_STEP', { delta: 1 });
    }, delayMs);
  };

  const toggleDemo = () => {
    if (demoActive.value) {
      stopDemo();
      return;
    }
    if (!loaded.value) return;
    demoActive.value = true;
    scheduleDemo();
  };

  const stepReplay = (delta) => {
    stopDemo();
    if (!loaded.value) return;
    startResultsRefresh();
    triggerAction('REPLAY_STEP', { delta });
  };

  const handleSliderStep = (step) => {
    stopDemo();
    if (!loaded.value) return;
    startResultsRefresh();
    triggerAction('REPLAY_SET_STEP', { step });
  };

  const updateSliderThreshold = (value) => {
    sliderThreshold.value = value;
    saveSetting('record_player_slider_threshold', value);
  };

  const nextInaccuracy = () => {
    const point = markerIndices.value.find((item) => item > currentStep.value);
    if (point == null) return;
    stopDemo();
    startResultsRefresh();
    triggerAction('REPLAY_SET_STEP', { step: point });
  };

  const openReplayFile = async () => {
    menuOpen.value = false;
    if (!window.pywebview?.api?.select_open_replay_file) return;
    const path = await window.pywebview.api.select_open_replay_file();
    if (path) {
      stopDemo();
      startResultsRefresh();
      triggerAction('REPLAY_LOAD_FILE', { path });
    }
  };

  const loadLatestReplay = () => {
    menuOpen.value = false;
    stopDemo();
    startResultsRefresh();
    triggerAction('REPLAY_LOAD_LATEST');
  };

  const guessFullPattern = () => {
    if (replayPattern.value && String(replayPattern.value).includes('_')) return replayPattern.value;
    const source = String(replaySource.value || '');
    const fileName = source.split(/[\\/]/u).pop() || '';
    const match = fileName.match(/^([A-Za-z0-9]+_\d+)/u);
    return match ? match[1] : '';
  };

  const isVariant = computed(() => (
    isVariantPattern(replayPattern.value || guessFullPattern(), appCategories.value)
  ));

  const jumpToPractice = () => {
    if (!loaded.value || !currentHex.value) return;
    window.dispatchEvent(new CustomEvent('trainer-practice-jump', {
      detail: {
        fullPattern: guessFullPattern(),
        hex: currentHex.value,
      },
    }));
    emit('navigate-tab', 'TrainerView');
  };

  const closeMenuOnClick = (event) => {
    if (!menuOpen.value || !menuRoot.value) return;
    if (!menuRoot.value.contains(event.target)) menuOpen.value = false;
  };

  const handleReplayState = (payload) => {
    finishResultsRefresh();
    board.value = Array.isArray(payload?.board) ? payload.board : new Array(16).fill(0);
    metadata.value = payload?.animation || {};
    currentHex.value = payload?.hex_str || '0000000000000000';
    loaded.value = !!payload?.loaded;
    replayStatus.value = payload?.status || '';
    replayPattern.value = payload?.pattern || '';
    replaySource.value = payload?.source || '';
    currentStep.value = Number(payload?.current_step || 0);
    totalSteps.value = Number(payload?.total_steps || 0);
    replayResults.value = payload?.results || {};
    currentMove.value = payload?.current_move || null;
    bestMove.value = payload?.best_move || null;
    loss.value = typeof payload?.loss === 'number' ? payload.loss : null;
    goodnessOfFit.value = typeof payload?.goodness_of_fit === 'number' ? payload.goodness_of_fit : null;
    combo.value = Number(payload?.combo || 0);
    summary.value = payload?.summary || { total_moves: 0, final_gof: 0, max_combo: 0, counts: {} };
    losses.value = Array.isArray(payload?.losses) ? payload.losses : [];
    if (demoActive.value) scheduleDemo();
  };

  const handleWSMessage = (message) => {
    if (message.action === 'REPLAY_STATE') handleReplayState(message.data);
  };

  const connect = () => {
    if (client) {
      return;
    }
    client = createWsClient({
      clientId: `replay_${Math.random().toString(36).slice(2, 9)}`,
      onOpen: () => {
        wsStatus.value = 'connected';
        triggerAction('REPLAY_GET_INIT');
      },
      onMessage: handleWSMessage,
      onClose: () => {
        wsStatus.value = 'disconnected';
        finishResultsRefresh();
        stopDemo();
      },
    });
    wsStatus.value = 'connecting';
    client.connect();
  };

  const disconnect = () => {
    finishResultsRefresh();
    stopDemo();
    client?.disconnect();
    client = null;
    wsStatus.value = 'disconnected';
  };

  const handleKeyDown = (event) => {
    if (!activeRef?.value) return;
    const target = event.target;
    if (target instanceof HTMLElement && (target.isContentEditable || target.closest('input, textarea, select'))) return;
    if (event.key === 'Escape' && menuOpen.value) {
      menuOpen.value = false;
      return;
    }
    if (event.ctrlKey && event.code === 'KeyN') {
      event.preventDefault();
      openReplayFile();
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      stepReplay(1);
    } else if (event.key === 'Backspace' || event.key === 'Delete') {
      event.preventDefault();
      stepReplay(-1);
    }
  };

  onMounted(() => {
    window.addEventListener('keydown', handleKeyDown, true);
    document.addEventListener('click', closeMenuOnClick);
  });

  watch(demoSpeed, () => {
    if (demoActive.value) scheduleDemo();
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
    () => appConfig.value.demo_speed,
    (value) => {
      demoSpeed.value = Number(value) || 40;
    },
    { immediate: true }
  );

  watch(
    () => appConfig.value.record_player_slider_threshold,
    (value) => {
      sliderThreshold.value = Number(value) || 1;
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
    document.removeEventListener('click', closeMenuOnClick);
    disconnect();
  });

  return {
    board,
    metadata,
    currentHex,
    loaded,
    replayStatus,
    replayPattern,
    isVariant,
    replaySource,
    currentStep,
    totalSteps,
    losses,
    sliderThreshold,
    dis32k,
    menuOpen,
    menuRoot,
    demoActive,
    dirLabels,
    fileDisplay,
    goodnessDisplay,
    summaryMaxCombo,
    displayedResults,
    resultsRefreshing,
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
    evaluationTotal,
    evaluationSegments,
    hasNextPoint,
    combo,
    getResultValueStyle,
    toggleDemo,
    stepReplay,
    handleSliderStep,
    updateSliderThreshold,
    nextInaccuracy,
    openReplayFile,
    loadLatestReplay,
    jumpToPractice,
  };
}
