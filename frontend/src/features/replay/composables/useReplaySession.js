import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { useAppSettingsStore } from '../../../app/useAppSettings';
import { useAuthState } from '../../../services/auth/authState';
import { pickSingleBrowserFile, readFileAsArrayBuffer } from '../../../services/files/browserFiles';
import { isVariantPattern } from '../../../utils/patternCategories';
import { createResultBarGradient } from '../../../utils/resultBars';
import { resultValueFontSize } from '../../../utils/successRate';
import { PERFORMANCE_LABELS } from '../engine/replayAnalysis';
import { ReplayController } from '../engine/replayController';
import { parseReplayAsync } from '../engine/replayLoader';
import { MAX_RPL_BYTES } from '../engine/rplParser';
import {
  authorizeLocalReplayLoad,
  createReplayRequestId,
  fetchLatestReplay,
} from '../services/replayClient';
import {
  restoreReplaySession,
  saveReplayPosition,
  saveReplaySource,
} from '../services/replaySessionStore';

export function useReplaySession(activeRef, emit) {
  const { config: appConfig, categories: appCategories, saveSetting } = useAppSettingsStore();
  const { requireAuth } = useAuthState();
  const { t } = useI18n();

  const COLOR_GREEN = '#2e7d32';
  const COLOR_YG = '#8bc34a';
  const COLOR_ORANGE = '#ff9800';
  const COLOR_RED = '#f44336';
  const fallbackPerformanceLabels = [...PERFORMANCE_LABELS];
  const evaluationColors = {
    'Perfect!': '#2e7d32',
    'Excellent!': '#7cb342',
    'Nice try!': '#c0ca33',
    'Not bad!': '#fb8c00',
    'Mistake!': '#f4511e',
    'Blunder!': '#e53935',
    'Terrible!': '#b71c1c',
  };
  const evaluationColorPalette = ['#2e7d32', '#7cb342', '#c0ca33', '#fb8c00', '#f4511e', '#e53935', '#b71c1c'];

  const board = ref(new Array(16).fill(0));
  const metadata = ref({});
  const currentHex = ref('0000000000000000');
  const loaded = ref(false);
  const replayStatus = ref('');
  const replayPattern = ref('');
  const replaySource = ref('');
  const replayUseVariant = ref(false);
  const currentStep = ref(0);
  const totalSteps = ref(0);
  const replayResults = ref({});
  const currentMove = ref(null);
  const bestMove = ref(null);
  const loss = ref(null);
  const evaluation = ref(null);
  const goodnessOfFit = ref(null);
  const combo = ref(0);
  const summary = ref({ total_moves: 0, final_gof: 0, max_combo: 0, counts: {} });
  const losses = ref([]);
  const performanceLabels = ref([...fallbackPerformanceLabels]);
  const sliderThreshold = ref(1);
  const demoSpeed = ref(40);
  const dis32k = ref(false);
  const currentLanguage = ref('en');
  const menuOpen = ref(false);
  const menuRoot = ref(null);
  const demoActive = ref(false);
  const loadingReplay = ref(false);
  const loadError = ref('');

  let controller = null;
  let demoTimer = null;
  let positionTimer = null;
  let restoreStarted = false;

  const isZh = () => String(currentLanguage.value || 'en').startsWith('zh');
  const dirLabels = computed(() => (
    isZh()
      ? { left: '左', right: '右', up: '上', down: '下' }
      : { left: 'L', right: 'R', up: 'U', down: 'D' }
  ));
  const zhEvaluationLabels = Object.fromEntries(fallbackPerformanceLabels.map((label) => [label, label]));
  const getEvaluationLabel = (label) => (isZh() ? (zhEvaluationLabels[label] || label) : label);
  const perfectLabel = computed(() => performanceLabels.value[0] || fallbackPerformanceLabels[0]);
  const getEvaluationColor = (label) => {
    if (evaluationColors[label]) return evaluationColors[label];
    const index = performanceLabels.value.indexOf(label);
    return index >= 0 ? evaluationColorPalette[index % evaluationColorPalette.length] : 'var(--accent)';
  };
  const trimTrailingZeros = (value) => value
    .replace(/(\.\d*?[1-9])0+$/u, '$1')
    .replace(/\.0+$/u, '')
    .replace(/\.$/u, '');
  const formatReplayRate = (value) => trimTrailingZeros(Number(value || 0).toFixed(9));

  const lerpColor = (c1, c2, ratio) => {
    const parseRgbColor = (color) => color.slice(1).match(/.{2}/gu).map((part) => parseInt(part, 16));
    const mix = (a, b) => Math.round(a + (b - a) * ratio);
    const [r1, g1, b1] = parseRgbColor(c1);
    const [r2, g2, b2] = parseRgbColor(c2);
    return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`;
  };

  const fileDisplay = computed(() => {
    if (replaySource.value) {
      const source = String(replaySource.value);
      const display = source.split(/[\\/]/u).pop() || source;
      if (display.toLowerCase().replace(/[\s-]+/gu, '_') === 'tester_session') {
        return replayPattern.value || t('replay.status.testerSession');
      }
      return display;
    }
    if (replayPattern.value) return replayPattern.value;
    return t('replay.status.noReplayLoaded');
  });
  const goodnessDisplay = computed(() => Number(goodnessOfFit.value ?? 0).toFixed(4));
  const summaryMaxCombo = computed(() => Number(summary.value?.max_combo ?? 0));

  const makePlaceholderResults = () => ['up', 'down', 'right', 'left'].map((dir) => ({
    dir,
    pct: 0,
    gradient: 'transparent',
    display: '0',
    textColor: 'var(--text-secondary)',
    val: null,
  }));

  const sortedResults = computed(() => {
    const entries = ['left', 'right', 'down', 'up']
      .map((dir) => ({
        dir,
        val: typeof replayResults.value?.[dir] === 'number' ? replayResults.value[dir] : null,
      }))
      .sort((left, right) => {
        if (left.val == null && right.val == null) return 0;
        if (left.val == null) return 1;
        if (right.val == null) return -1;
        return right.val - left.val;
      });
    const bestVal = entries.find((item) => item.val != null)?.val || 0;
    return entries.map((item, index) => {
      let pct = 0;
      let color = 'var(--border-main)';
      if (item.val != null && bestVal > 0) {
        const relativeLoss = 1 - item.val / bestVal;
        if (index === 0) {
          pct = 100;
          color = COLOR_GREEN;
        } else if (relativeLoss <= 0.10) {
          pct = (1 - relativeLoss / 0.10) * 100;
          color = relativeLoss <= 0.001
            ? COLOR_GREEN
            : relativeLoss <= 0.01
              ? lerpColor(COLOR_GREEN, COLOR_YG, (relativeLoss - 0.001) / 0.009)
              : relativeLoss <= 0.03
                ? lerpColor(COLOR_YG, COLOR_ORANGE, (relativeLoss - 0.01) / 0.02)
                : lerpColor(COLOR_ORANGE, COLOR_RED, (relativeLoss - 0.03) / 0.07);
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
  const displayedResults = computed(() => (
    loaded.value && currentStep.value < totalSteps.value ? sortedResults.value : makePlaceholderResults()
  ));
  const resultsRefreshing = computed(() => loadingReplay.value);
  const resultsUpdatingVisible = computed(() => loadingReplay.value);
  const currentEvaluation = computed(() => evaluation.value || (loss.value == null ? null : perfectLabel.value));

  const feedbackBadgeText = computed(() => {
    if (!loaded.value) return replayStatus.value || t('replay.status.noReplayLoaded');
    if (currentStep.value >= totalSteps.value) return t('replay.status.replayComplete');
    return getEvaluationLabel(currentEvaluation.value || perfectLabel.value);
  });
  const feedbackBadgeStyle = computed(() => ({
    color: !loaded.value || currentStep.value >= totalSteps.value
      ? 'var(--text-secondary)'
      : getEvaluationColor(currentEvaluation.value),
  }));
  const feedbackLossText = computed(() => {
    if (
      !loaded.value
      || currentStep.value >= totalSteps.value
      || currentEvaluation.value === perfectLabel.value
      || loss.value == null
    ) return '';
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
    loaded.value && currentStep.value < totalSteps.value ? (moveLabels.value[currentMove.value] || '--') : '--'
  ));
  const feedbackBestMove = computed(() => (
    loaded.value && currentStep.value < totalSteps.value ? (moveLabels.value[bestMove.value] || '--') : '--'
  ));
  const feedbackPressedMoveStyle = computed(() => ({
    color: loaded.value && currentStep.value < totalSteps.value
      ? getEvaluationColor(currentEvaluation.value)
      : 'var(--text-secondary)',
  }));
  const feedbackBestMoveStyle = computed(() => ({
    color: loaded.value && currentStep.value < totalSteps.value ? COLOR_GREEN : 'var(--text-secondary)',
  }));
  const evaluationTotal = computed(() => Number(summary.value?.total_moves || 0));
  const evaluationSegments = computed(() => performanceLabels.value.map((label) => {
    const count = Number(summary.value?.counts?.[label] || 0);
    return {
      label,
      shortLabel: getEvaluationLabel(label),
      count,
      percent: evaluationTotal.value ? (count / evaluationTotal.value) * 100 : 0,
      color: getEvaluationColor(label),
    };
  }));

  const markerIndices = computed(() => {
    const values = Array.isArray(losses.value) ? losses.value.map(Number).filter(Number.isFinite) : [];
    if (!values.length) return [];
    const sorted = [...values].sort((left, right) => left - right);
    const position = (sorted.length - 1) * 0.1;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    const quantile = lower === upper
      ? sorted[lower]
      : sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    const threshold = Math.min(quantile, Number(sliderThreshold.value) || 1);
    return values
      .map((item, index) => ({ item, index }))
      .filter(({ item }) => item < 1 && item < threshold)
      .map(({ index }) => index);
  });
  const hasNextPoint = computed(() => markerIndices.value.some((point) => point > currentStep.value));
  const resultFontSize = computed(() => resultValueFontSize(displayedResults.value.map((item) => item.display)));
  const getResultValueStyle = (item) => ({ color: item.textColor, fontSize: resultFontSize.value });

  const stopDemo = () => {
    if (demoTimer) window.clearTimeout(demoTimer);
    demoTimer = null;
    demoActive.value = false;
  };
  const blurActiveControl = () => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
  };
  const consumePendingLatestReplayLoad = () => {
    if (!window.__pendingReplayLatestLoad) return false;
    window.__pendingReplayLatestLoad = false;
    return true;
  };

  const persistPositionSoon = () => {
    if (positionTimer) window.clearTimeout(positionTimer);
    positionTimer = window.setTimeout(() => {
      positionTimer = null;
      saveReplayPosition(currentStep.value);
    }, 250);
  };

  const applyState = (state) => {
    board.value = state.board;
    metadata.value = state.animation || {};
    currentHex.value = state.hex_str;
    loaded.value = !!state.loaded;
    replayStatus.value = state.status || '';
    replayPattern.value = state.pattern || '';
    replaySource.value = state.source || '';
    currentStep.value = Number(state.current_step || 0);
    totalSteps.value = Number(state.total_steps || 0);
    replayResults.value = state.results || {};
    currentMove.value = state.current_move || null;
    bestMove.value = state.best_move || null;
    loss.value = typeof state.loss === 'number' ? state.loss : null;
    evaluation.value = state.evaluation || null;
    goodnessOfFit.value = typeof state.goodness_of_fit === 'number' ? state.goodness_of_fit : null;
    combo.value = Number(state.combo || 0);
  };

  const installReplay = async (buffer, replayMetadata, { step = 0, persist = true } = {}) => {
    const parsed = await parseReplayAsync(buffer, sliderThreshold.value);
    controller = new ReplayController({
      replay: parsed.replay,
      analysis: parsed.analysis,
      pattern: replayMetadata.pattern,
      source: replayMetadata.source || replayMetadata.filename,
      useVariant: replayMetadata.useVariant,
    });
    replayUseVariant.value = !!replayMetadata.useVariant;
    losses.value = Array.from(parsed.analysis.losses);
    summary.value = parsed.analysis.summary;
    performanceLabels.value = [...PERFORMANCE_LABELS];
    applyState(controller.setStep(step));
    if (persist) {
      saveReplaySource(parsed.rawBuffer, replayMetadata);
      saveReplayPosition(currentStep.value);
    }
  };

  const formatLoadError = (error) => {
    if (error?.status === 401 || error?.status === 402 || error?.code === 'INSUFFICIENT_TOKENS') {
      return '';
    }
    if (error?.code === 'NO_LATEST_REPLAY' || error?.payload?.detail?.code === 'NO_LATEST_REPLAY') {
      return t('replay.status.noLatest');
    }
    if (error?.code === 'REPLAY_TOO_LARGE') return t('replay.status.tooLarge');
    if (error?.code === 'NETWORK_ERROR' || /failed to fetch|network/iu.test(String(error?.message || ''))) {
      return t('replay.status.networkError');
    }
    return t('replay.status.invalidFile');
  };

  const guessPatternFromFilename = (filename) => {
    const name = String(filename || '').split(/[\\/]/u).pop() || '';
    return name.match(/^([A-Za-z0-9]+_\d+)/u)?.[1] || '';
  };
  const shouldRelaxFileAccept = () => {
    if (typeof navigator === 'undefined') return false;
    return /iPad|iPhone|iPod/u.test(navigator.userAgent || '')
      || (navigator.platform === 'MacIntel' && Number(navigator.maxTouchPoints || 0) > 1);
  };

  const openReplayFile = async () => {
    menuOpen.value = false;
    if (!requireAuth() || loadingReplay.value) return;
    stopDemo();
    loadError.value = '';
    try {
      const file = await pickSingleBrowserFile(shouldRelaxFileAccept() ? {} : { accept: '.rpl' });
      if (!file) return;
      if (!String(file.name || '').toLowerCase().endsWith('.rpl')) {
        loadError.value = t('replay.status.invalidFile');
        return;
      }
      if (file.size <= 0 || file.size > MAX_RPL_BYTES) {
        loadError.value = file.size > MAX_RPL_BYTES
          ? t('replay.status.tooLarge')
          : t('replay.status.invalidFile');
        return;
      }
      loadingReplay.value = true;
      replayStatus.value = t('replay.status.loading');
      const buffer = await readFileAsArrayBuffer(file);
      await authorizeLocalReplayLoad({
        requestId: createReplayRequestId(),
        filename: file.name,
        size: file.size,
      });
      const pattern = guessPatternFromFilename(file.name);
      await installReplay(buffer, {
        filename: file.name,
        source: file.name,
        pattern,
        useVariant: isVariantPattern(pattern, appCategories.value),
      });
    } catch (error) {
      console.error('Failed to load replay file', error);
      loadError.value = formatLoadError(error);
      if (!loaded.value) replayStatus.value = loadError.value || '';
    } finally {
      loadingReplay.value = false;
    }
  };

  const requestLatestReplayLoad = async () => {
    if (!requireAuth() || loadingReplay.value) return;
    stopDemo();
    loadError.value = '';
    loadingReplay.value = true;
    replayStatus.value = t('replay.status.loading');
    try {
      const latest = await fetchLatestReplay({ requestId: createReplayRequestId() });
      await installReplay(latest.buffer, {
        filename: latest.filename,
        source: latest.source,
        pattern: latest.pattern,
        useVariant: latest.useVariant,
      });
    } catch (error) {
      console.error('Failed to load latest replay', error);
      loadError.value = formatLoadError(error);
      if (!loaded.value) replayStatus.value = loadError.value || '';
    } finally {
      loadingReplay.value = false;
    }
  };

  const loadLatestReplay = () => {
    menuOpen.value = false;
    requestLatestReplayLoad();
  };

  const scheduleDemo = () => {
    if (demoTimer) window.clearTimeout(demoTimer);
    if (!demoActive.value || !controller) return;
    if (currentStep.value >= totalSteps.value) {
      stopDemo();
      return;
    }
    const delayMs = Math.max(1, Math.round(Number(demoSpeed.value) || 40));
    demoTimer = window.setTimeout(() => {
      if (!demoActive.value || !controller) return;
      applyState(controller.step(1));
      persistPositionSoon();
      scheduleDemo();
    }, delayMs);
  };

  const toggleDemo = () => {
    if (demoActive.value) {
      stopDemo();
      return;
    }
    if (!controller) return;
    demoActive.value = true;
    scheduleDemo();
  };
  const stepReplay = (delta) => {
    stopDemo();
    blurActiveControl();
    if (!controller) return;
    applyState(controller.step(delta));
    persistPositionSoon();
  };
  const handleSliderStep = (step) => {
    stopDemo();
    blurActiveControl();
    if (!controller) return;
    applyState(controller.setStep(step));
    persistPositionSoon();
  };
  const updateSliderThreshold = (value) => {
    sliderThreshold.value = value;
    saveSetting('record_player_slider_threshold', value);
  };
  const nextInaccuracy = () => {
    const point = markerIndices.value.find((item) => item > currentStep.value);
    if (point != null) handleSliderStep(point);
  };

  const guessFullPattern = () => replayPattern.value || guessPatternFromFilename(replaySource.value);
  const isVariant = computed(() => replayUseVariant.value || isVariantPattern(guessFullPattern(), appCategories.value));
  const jumpToPractice = () => {
    if (!loaded.value || !currentHex.value) return;
    emit('navigate-tab', 'TrainerView', { fullPattern: guessFullPattern(), hex: currentHex.value });
  };
  const closeMenuOnClick = (event) => {
    if (menuOpen.value && menuRoot.value && !menuRoot.value.contains(event.target)) menuOpen.value = false;
  };
  const handleKeyDown = (event) => {
    if (!activeRef?.value) return;
    const target = event.target;
    if (target instanceof HTMLElement) {
      const slider = target.matches('[data-replay-slider-range="true"]');
      if (!slider && (target.isContentEditable || target.closest('input, textarea, select'))) return;
    }
    if (event.key === 'Escape' && menuOpen.value) {
      menuOpen.value = false;
    } else if (event.ctrlKey && event.code === 'KeyN') {
      event.preventDefault();
      openReplayFile();
    } else if (event.key === 'Enter') {
      event.preventDefault();
      stepReplay(1);
    } else if (event.key === 'Backspace' || event.key === 'Delete') {
      event.preventDefault();
      stepReplay(-1);
    }
  };

  const restorePreviousReplay = async () => {
    if (restoreStarted) return;
    restoreStarted = true;
    const stored = restoreReplaySession();
    if (!stored) return;
    loadingReplay.value = true;
    try {
      await installReplay(stored.buffer, stored, { step: stored.step, persist: false });
    } catch (error) {
      console.error('Failed to restore replay session', error);
    } finally {
      loadingReplay.value = false;
    }
  };

  onMounted(async () => {
    window.addEventListener('keydown', handleKeyDown, true);
    document.addEventListener('click', closeMenuOnClick);
    const loadLatest = activeRef?.value && consumePendingLatestReplayLoad();
    await restorePreviousReplay();
    if (loadLatest) requestLatestReplayLoad();
  });
  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeyDown, true);
    document.removeEventListener('click', closeMenuOnClick);
    stopDemo();
    if (positionTimer) window.clearTimeout(positionTimer);
    if (loaded.value) saveReplayPosition(currentStep.value);
  });

  watch(demoSpeed, () => {
    if (demoActive.value) scheduleDemo();
  });
  watch(() => appConfig.value.dis_32k, (value) => { dis32k.value = !!value; }, { immediate: true });
  watch(() => appConfig.value.language, (value) => { currentLanguage.value = value || 'en'; }, { immediate: true });
  watch(() => appConfig.value.demo_speed, (value) => { demoSpeed.value = Number(value) || 40; }, { immediate: true });
  watch(() => appConfig.value.record_player_slider_threshold, (value) => {
    const parsed = Number(value);
    sliderThreshold.value = Number.isFinite(parsed) ? parsed : 1;
  }, { immediate: true });
  watch(activeRef, (isActive) => {
    if (isActive && consumePendingLatestReplayLoad()) requestLatestReplayLoad();
  });

  return {
    board,
    metadata,
    currentHex,
    loaded,
    replayStatus,
    replayPattern,
    replaySource,
    currentStep,
    totalSteps,
    losses,
    sliderThreshold,
    dis32k,
    menuOpen,
    menuRoot,
    demoActive,
    loadingReplay,
    loadError,
    dirLabels,
    fileDisplay,
    goodnessDisplay,
    summaryMaxCombo,
    displayedResults,
    resultsRefreshing,
    resultsUpdatingVisible,
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
    isVariant,
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
