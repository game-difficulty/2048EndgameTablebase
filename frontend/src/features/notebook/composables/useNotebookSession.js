import { computed, onMounted, onUnmounted, ref, watch } from 'vue';

import { useAppSettingsStore } from '../../../app/useAppSettings';
import { createWsClient } from '../../../services/ws/createWsClient';
import { isVariantPattern } from '../../../utils/patternCategories';

const DEFAULT_BOARD = new Array(16).fill(0);
const SAMPLE_MODE_KEYS = ['mistakeCount', 'totalLoss', 'combined'];
const AUTO_NEXT_DELAY_MS = 1500;
const DEFAULT_NOTEBOOK_THRESHOLD = 0.999;

function clampNotebookThreshold(value) {
  return Math.min(1, Math.max(0, value));
}

function parseNotebookThresholdInput(value) {
  const parsed = Number(String(value ?? '').trim());
  if (!Number.isFinite(parsed)) return null;
  return clampNotebookThreshold(parsed);
}

function formatNotebookThreshold(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_NOTEBOOK_THRESHOLD.toFixed(6);
  return clampNotebookThreshold(parsed).toFixed(6);
}

export function useNotebookSession(activeRef) {
  const { config: appConfig, categories: appCategories } = useAppSettingsStore();

  const wsStatus = ref('connecting');
  const dis32k = ref(false);
  const board = ref([...DEFAULT_BOARD]);
  const metadata = ref({});
  const availablePatterns = ref([]);
  const selectedPattern = ref('');
  const patternMenuOpen = ref(false);
  const patternMenuRoot = ref(null);
  const combo = ref(0);
  const remaining = ref(0);
  const correct = ref(0);
  const incorrect = ref(0);
  const sampleMode = ref(0);
  const lastDirection = ref(null);
  const currentHex = ref('0000000000000000');
  const answered = ref(false);
  const answerCorrect = ref(null);
  const bestMove = ref(null);
  const notebookThreshold = ref(DEFAULT_NOTEBOOK_THRESHOLD);
  const notebookThresholdInput = ref(formatNotebookThreshold(DEFAULT_NOTEBOOK_THRESHOLD));
  const nextCountdownMs = ref(0);
  const backendReady = ref(false);

  let client = null;
  let nextCountdownTimer = null;

  const currentPatternDisplay = computed(() => selectedPattern.value || '--');
  const isVariant = computed(() => isVariantPattern(selectedPattern.value, appCategories.value));
  const connectionBadgeClass = computed(() => (
    wsStatus.value === 'connected'
      ? 'badge-base badge-connection-connected'
      : 'badge-base badge-connection-disconnected'
  ));
  const sampleModes = computed(() => SAMPLE_MODE_KEYS);
  const hasPatterns = computed(() => availablePatterns.value.length > 0);
  const hasProblem = computed(() => selectedPattern.value && board.value.some((value) => Number(value) > 0));
  const actionEnabled = computed(() => backendReady.value && hasProblem.value);
  const canAnswer = computed(() => actionEnabled.value && !answered.value);
  const nextCountdownActive = computed(() => nextCountdownMs.value > 0);
  const nextCountdownProgress = computed(() => (
    nextCountdownActive.value ? Math.max(0, Math.min(1, nextCountdownMs.value / AUTO_NEXT_DELAY_MS)) : 0
  ));
  const nextButtonLabel = computed(() => {
    if (!nextCountdownActive.value) return null;
    return `${(nextCountdownMs.value / 1000).toFixed(1)}s`;
  });
  const thresholdInputIsValid = computed(() => parseNotebookThresholdInput(notebookThresholdInput.value) !== null);
  const thresholdDirty = computed(() => {
    const parsed = parseNotebookThresholdInput(notebookThresholdInput.value);
    return parsed !== null && Math.abs(parsed - notebookThreshold.value) > 1e-12;
  });
  const thresholdSaveEnabled = computed(() => (
    backendReady.value
    && wsStatus.value === 'connected'
    && thresholdInputIsValid.value
    && thresholdDirty.value
  ));

  const stopNextCountdown = () => {
    if (nextCountdownTimer) {
      window.clearInterval(nextCountdownTimer);
      nextCountdownTimer = null;
    }
    nextCountdownMs.value = 0;
  };

  const startNextCountdown = () => {
    stopNextCountdown();
    nextCountdownMs.value = AUTO_NEXT_DELAY_MS;
    const startedAt = Date.now();
    nextCountdownTimer = window.setInterval(() => {
      const remaining = Math.max(0, AUTO_NEXT_DELAY_MS - (Date.now() - startedAt));
      nextCountdownMs.value = remaining;
      if (remaining <= 0) {
        stopNextCountdown();
        nextProblem();
      }
    }, 50);
  };

  const syncBoard = (rawBoard) => {
    board.value = Array.isArray(rawBoard) && rawBoard.length === 16
      ? rawBoard.map((value) => Number(value) || 0)
      : [...DEFAULT_BOARD];
  };

  const syncNotebookThreshold = (rawThreshold) => {
    const parsed = parseNotebookThresholdInput(rawThreshold);
    if (parsed === null) return;
    const shouldRefreshInput = !thresholdDirty.value || notebookThresholdInput.value.trim() === '';
    notebookThreshold.value = parsed;
    if (shouldRefreshInput) {
      notebookThresholdInput.value = formatNotebookThreshold(parsed);
    }
  };

  const handleNotebookBootstrap = (payload) => {
    backendReady.value = true;
    availablePatterns.value = Array.isArray(payload?.patterns) ? payload.patterns.map(String) : [];
    if (!selectedPattern.value || !availablePatterns.value.includes(selectedPattern.value)) {
      selectedPattern.value = availablePatterns.value[0] || '';
    }
    if (typeof payload?.weight_mode === 'number') {
      sampleMode.value = Math.max(0, Math.min(SAMPLE_MODE_KEYS.length - 1, payload.weight_mode));
    }
    syncNotebookThreshold(payload?.notebook_threshold);
  };

  const handleNotebookState = (payload) => {
    backendReady.value = true;
    if (Array.isArray(payload?.patterns)) {
      availablePatterns.value = payload.patterns.map(String);
    }
    if (payload && Object.prototype.hasOwnProperty.call(payload, 'pattern')) {
      selectedPattern.value = String(payload.pattern || '');
    } else if (selectedPattern.value && !availablePatterns.value.includes(selectedPattern.value)) {
      selectedPattern.value = availablePatterns.value[0] || '';
    }
    syncBoard(payload?.board);
    metadata.value = payload?.animation || {};
    currentHex.value = typeof payload?.hex_str === 'string' ? payload.hex_str : currentHex.value;
    combo.value = Number(payload?.feedback?.combo ?? payload?.combo ?? combo.value) || 0;
    remaining.value = Number(payload?.feedback?.remaining ?? payload?.remaining ?? remaining.value) || 0;
    correct.value = Number(payload?.feedback?.correct ?? payload?.correct ?? correct.value) || 0;
    incorrect.value = Number(payload?.feedback?.incorrect ?? payload?.incorrect ?? incorrect.value) || 0;
    answered.value = !!payload?.answered;
    lastDirection.value = payload?.last_direction || null;
    answerCorrect.value = typeof payload?.answer_correct === 'boolean' ? payload.answer_correct : null;
    bestMove.value = payload?.best_move || null;
    if (typeof payload?.weight_mode === 'number') {
      sampleMode.value = Math.max(0, Math.min(SAMPLE_MODE_KEYS.length - 1, payload.weight_mode));
    }
    syncNotebookThreshold(payload?.notebook_threshold);
  };

  const handleMessage = (message) => {
    if (message.action === 'NOTEBOOK_BOOTSTRAP') {
      handleNotebookBootstrap(message.data);
    } else if (message.action === 'NOTEBOOK_STATE') {
      handleNotebookState(message.data);
    }
  };

  const connect = () => {
    if (client) return;
    client = createWsClient({
      clientId: `notebook_${Math.random().toString(36).slice(2, 9)}`,
      onOpen: () => {
        wsStatus.value = 'connected';
        client?.send('NOTEBOOK_GET_INIT');
      },
      onMessage: handleMessage,
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

  const send = (action, data = undefined) => {
    if (!backendReady.value || !client) return false;
    return client.send(action, data);
  };

  const selectPattern = (pattern) => {
    stopNextCountdown();
    selectedPattern.value = pattern;
    patternMenuOpen.value = false;
    send('NOTEBOOK_SELECT_PATTERN', { pattern });
  };

  const setSampleMode = (index) => {
    stopNextCountdown();
    sampleMode.value = index;
    send('NOTEBOOK_SET_WEIGHT_MODE', { mode: index });
  };

  const answerDirection = (direction) => {
    if (!canAnswer.value) return;
    lastDirection.value = direction.toLowerCase();
    send('NOTEBOOK_ANSWER', { direction: direction.toLowerCase() });
  };

  const nextProblem = () => {
    if (!actionEnabled.value) return;
    stopNextCountdown();
    send('NOTEBOOK_NEXT');
  };

  const deleteCurrent = () => {
    if (!actionEnabled.value) return;
    stopNextCountdown();
    send('NOTEBOOK_DELETE');
  };

  const jumpToTrainer = (emit) => {
    stopNextCountdown();
    if (hasProblem.value && currentHex.value) {
      window.dispatchEvent(new CustomEvent('trainer-practice-jump', {
        detail: {
          fullPattern: selectedPattern.value,
          hex: currentHex.value,
        },
      }));
    }
    emit('navigate-tab', 'TrainerView');
  };

  const updateNotebookThresholdInput = (value) => {
    notebookThresholdInput.value = value;
  };

  const saveNotebookThreshold = () => {
    const parsed = parseNotebookThresholdInput(notebookThresholdInput.value);
    if (parsed === null) return;
    const normalized = formatNotebookThreshold(parsed);
    notebookThresholdInput.value = normalized;
    if (Math.abs(parsed - notebookThreshold.value) <= 1e-12) return;
    send('NOTEBOOK_SET_THRESHOLD', { threshold: parsed });
  };

  const closePatternMenuOnClick = (event) => {
    if (!patternMenuOpen.value || !patternMenuRoot.value) return;
    if (!patternMenuRoot.value.contains(event.target)) {
      patternMenuOpen.value = false;
    }
  };

  const handleKeyDown = (event) => {
    if (!activeRef?.value) return;
    const target = event.target;
    if (target instanceof HTMLElement) {
      const tag = target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || target.isContentEditable) return;
    }
    if (event.key === 'Escape' && patternMenuOpen.value) {
      patternMenuOpen.value = false;
      return;
    }
    if (event.code === 'ArrowUp' || event.code === 'KeyW') {
      event.preventDefault();
      answerDirection('Up');
    } else if (event.code === 'ArrowDown' || event.code === 'KeyS') {
      event.preventDefault();
      answerDirection('Down');
    } else if (event.code === 'ArrowLeft' || event.code === 'KeyA') {
      event.preventDefault();
      answerDirection('Left');
    } else if (event.code === 'ArrowRight' || event.code === 'KeyD') {
      event.preventDefault();
      answerDirection('Right');
    } else if (event.code === 'Enter') {
      event.preventDefault();
      nextProblem();
    } else if (event.code === 'Delete' || event.code === 'Backspace') {
      event.preventDefault();
      deleteCurrent();
    }
  };

  const getDirectionButtonClass = (direction) => ([
    'notebook-direction-btn',
    answered.value && lastDirection.value === direction.toLowerCase() && answerCorrect.value === true ? 'is-correct' : '',
    answered.value && lastDirection.value === direction.toLowerCase() && answerCorrect.value === false ? 'is-wrong' : '',
    answered.value && bestMove.value === direction.toLowerCase() && answerCorrect.value === false ? 'is-correct' : '',
    !hasProblem.value ? 'is-disabled' : '',
  ]);

  watch(
    () => appConfig.value.dis_32k,
    (value) => {
      dis32k.value = !!value;
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

  watch(
    [answered, answerCorrect, currentHex],
    ([isAnswered, isCorrect]) => {
      if (isAnswered && isCorrect) {
        startNextCountdown();
      } else {
        stopNextCountdown();
      }
    },
    { immediate: true }
  );

  onMounted(() => {
    window.addEventListener('keydown', handleKeyDown, true);
    document.addEventListener('click', closePatternMenuOnClick);
  });

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeyDown, true);
    document.removeEventListener('click', closePatternMenuOnClick);
    stopNextCountdown();
    disconnect();
  });

  return {
    wsStatus,
    board,
    metadata,
    dis32k,
    availablePatterns,
    selectedPattern,
    patternMenuOpen,
    patternMenuRoot,
    combo,
    remaining,
    correct,
    incorrect,
    sampleMode,
    notebookThresholdInput,
    thresholdInputIsValid,
    thresholdDirty,
    thresholdSaveEnabled,
    answered,
    answerCorrect,
    currentPatternDisplay,
    isVariant,
    connectionBadgeClass,
    sampleModes,
    hasPatterns,
    actionEnabled,
    canAnswer,
    nextCountdownActive,
    nextCountdownProgress,
    nextButtonLabel,
    togglePatternMenu: () => { patternMenuOpen.value = !patternMenuOpen.value; },
    selectPattern,
    setSampleMode,
    answerDirection,
    nextProblem,
    deleteCurrent,
    jumpToTrainer,
    updateNotebookThresholdInput,
    saveNotebookThreshold,
    getDirectionButtonClass,
  };
}
