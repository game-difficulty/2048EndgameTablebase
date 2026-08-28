import { onMounted, onUnmounted, ref, watch } from 'vue';

import { KEYBOARD_OWNERS, keyboardInputAllowed } from '../../../app/keyboardOwnership';
import { createWsClient } from '../../../services/ws/createWsClient';

export function useGamerSession(activeRef) {
  const VALID_DIRECTIONS = new Set(['left', 'right', 'up', 'down']);
  const AI_MOVE_ACK_TIMEOUT_MS = 2000;
  const AI_RETRY_DELAY_MS = 150;
  const AI_MAX_STALLED_MOVES = 4;
  const AI_MAX_NO_OP_RETRIES_PER_BOARD = 5;

  const board = ref([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536]);
  const metadata = ref(null);
  const score = ref({ current: 0, best: 0 });
  const wsStatus = ref('connecting');
  const aiEnabled = ref(false);
  const difficulty = ref(0);
  const aiSpeed = ref(100);
  const hexInput = ref('');
  const currentHex = ref('0000000000000000');
  const specialTiles = ref([]);

  const scoreAnimations = ref([]);
  let animIdCounter = 0;
  let client = null;
  let aiContinuationTimer = null;
  let aiMoveAckTimer = null;
  let aiPhase = null;
  let stalledAiMoveCount = 0;
  let aiNoOpRetriesByHex = new Map();

  watch(() => score.value.current, (newVal, oldVal) => {
    if (newVal > oldVal && oldVal !== undefined && oldVal !== 0) {
      const diff = newVal - oldVal;
      const id = animIdCounter++;
      scoreAnimations.value.push({ id, value: diff });
      window.setTimeout(() => {
        scoreAnimations.value = scoreAnimations.value.filter((item) => item.id !== id);
      }, 1000);
    }
  });

  const triggerAction = (action, payload = {}) => {
    return client?.send(action, payload) === true;
  };

  const clearAiContinuationTimer = () => {
    if (aiContinuationTimer !== null) {
      window.clearTimeout(aiContinuationTimer);
      aiContinuationTimer = null;
    }
  };

  const clearAiMoveAckTimer = () => {
    if (aiMoveAckTimer !== null) {
      window.clearTimeout(aiMoveAckTimer);
      aiMoveAckTimer = null;
    }
  };

  const clearAiTimers = () => {
    clearAiContinuationTimer();
    clearAiMoveAckTimer();
  };

  const stopAI = () => {
    aiEnabled.value = false;
    aiPhase = null;
    stalledAiMoveCount = 0;
    aiNoOpRetriesByHex.clear();
    clearAiTimers();
  };

  const canContinueAI = () => (
    aiEnabled.value &&
    activeRef?.value &&
    wsStatus.value === 'connected'
  );

  const requestAiStep = (delay = 0) => {
    clearAiContinuationTimer();
    if (!canContinueAI() || aiPhase !== null) {
      return;
    }

    aiContinuationTimer = window.setTimeout(() => {
      aiContinuationTimer = null;
      if (!canContinueAI() || aiPhase !== null) {
        return;
      }
      if (triggerAction('AI_STEP')) {
        aiPhase = 'ai-step';
      }
    }, delay);
  };

  const handleAiMoveStalled = () => {
    if (aiPhase !== 'user-move' || !aiEnabled.value) {
      return;
    }

    aiPhase = null;
    stalledAiMoveCount += 1;
    if (stalledAiMoveCount > AI_MAX_STALLED_MOVES) {
      console.warn('Gamer AI stopped after repeated moves without state updates.');
      stopAI();
      return;
    }

    triggerAction('GET_STATE');
  };

  const watchAiMoveAck = () => {
    clearAiMoveAckTimer();
    aiMoveAckTimer = window.setTimeout(() => {
      aiMoveAckTimer = null;
      handleAiMoveStalled();
    }, AI_MOVE_ACK_TIMEOUT_MS);
  };

  const handleMessage = (data) => {
    if (data.action === 'UPDATE_STATE') {
      const previousHex = currentHex.value;
      const completedAiMove = aiPhase === 'user-move';
      board.value = data.data.board;
      score.value = data.data.score;
      metadata.value = data.data.animation;
      currentHex.value = data.data.hex_str || currentHex.value;
      specialTiles.value = Array.isArray(data.data.gamer_special_tiles)
        ? data.data.gamer_special_tiles
        : [];

      if (data.data.settings) {
        difficulty.value = data.data.settings.difficulty;
        aiSpeed.value = data.data.settings.speed;
      }

      aiPhase = null;
      clearAiMoveAckTimer();
      if (completedAiMove || (data.data.hex_str && data.data.hex_str !== previousHex)) {
        stalledAiMoveCount = 0;
        aiNoOpRetriesByHex.clear();
      }

      if (canContinueAI()) {
        requestAiStep();
      }
      return;
    }

    if (data.action === 'DO_AI_MOVE_CMD') {
      const dir = String(data.data?.dir || '').toLowerCase();
      if (VALID_DIRECTIONS.has(dir)) {
        const fromContinuousAI = aiEnabled.value;
        aiPhase = fromContinuousAI ? 'user-move' : null;
        if (fromContinuousAI && !activeRef?.value) {
          aiPhase = null;
          return;
        }
        triggerAction('USER_MOVE', {
          dir,
          source: fromContinuousAI ? 'ai' : 'manual-ai-step',
        });
        if (fromContinuousAI) {
          watchAiMoveAck();
        }
      } else {
        if (aiEnabled.value) {
          stopAI();
        }
      }
      return;
    }

    if (data.action === 'AI_MOVE_SKIPPED') {
      aiPhase = null;
      clearAiMoveAckTimer();
      if (!aiEnabled.value) {
        return;
      }
      if (data.data?.has_legal_moves === false) {
        stopAI();
        return;
      }

      const boardHex = String(data.data?.hex_str || currentHex.value || '');
      const retryCount = (aiNoOpRetriesByHex.get(boardHex) || 0) + 1;
      aiNoOpRetriesByHex.set(boardHex, retryCount);
      if (retryCount > AI_MAX_NO_OP_RETRIES_PER_BOARD) {
        console.warn('Gamer AI stopped after repeated no-op AI moves on the same board.');
        stopAI();
        return;
      }
      requestAiStep(AI_RETRY_DELAY_MS * retryCount);
      return;
    }
  };

  const connect = () => {
    if (client) {
      return;
    }
    client = createWsClient({
      clientId: `gamer_${Math.random().toString(36).substring(2, 9)}`,
      onOpen: () => {
        wsStatus.value = 'connected';
        aiPhase = null;
        triggerAction('GET_STATE');
      },
      onMessage: handleMessage,
      onClose: () => {
        wsStatus.value = 'disconnected';
        aiPhase = null;
        clearAiTimers();
      },
    });
    wsStatus.value = 'connecting';
    client.connect();
  };

  const disconnect = () => {
    aiPhase = null;
    clearAiTimers();
    onBeforeUnload();
    client?.disconnect();
    client = null;
    wsStatus.value = 'disconnected';
  };

  const handleKeydown = (event) => {
    if (!activeRef?.value || !keyboardInputAllowed(KEYBOARD_OWNERS.PRIMARY)) return;
    const target = event.target;
    if (
      target instanceof HTMLElement &&
      (target.tagName === 'INPUT' ||
        target.tagName === 'SELECT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable)
    ) {
      return;
    }

    const map = {
      ArrowUp: 'up', KeyW: 'up',
      ArrowDown: 'down', KeyS: 'down',
      ArrowLeft: 'left', KeyA: 'left',
      ArrowRight: 'right', KeyD: 'right',
    };

    if (map[event.code]) {
      event.preventDefault();
      triggerAction('USER_MOVE', { dir: map[event.code] });
    } else if (event.code === 'Enter' || event.code === 'Space') {
      event.preventDefault();
      if (aiEnabled.value) {
        stopAI();
      } else {
        triggerAction('AI_STEP');
      }
    } else if (event.code === 'Backspace' || event.code === 'Delete') {
      event.preventDefault();
      if (aiEnabled.value) {
        stopAI();
      } else {
        triggerAction('UNDO');
      }
    }
  };

  const toggleAI = () => {
    if (aiEnabled.value) {
      stopAI();
      return;
    }

    aiEnabled.value = true;
    stalledAiMoveCount = 0;
    aiNoOpRetriesByHex.clear();
    aiPhase = null;
    requestAiStep();
  };

  const updateSettings = () => {
    triggerAction('UPDATE_SETTINGS', {
      difficulty: difficulty.value,
      speed: aiSpeed.value,
    });
  };

  const setBoard = () => {
    const val = hexInput.value.trim();
    if (!val) return;
    triggerAction('SET_BOARD', { hex_str: val });
  };

  const writeCurrentBoardToHex = () => {
    const digits = board.value.slice(0, 16).map((tile) => {
      const value = Number(tile) || 0;
      if (value <= 0) {
        return '0';
      }
      if (value > 32768) {
        return 'f';
      }
      const exponent = Math.log2(value);
      if (!Number.isInteger(exponent) || exponent < 0) {
        return '0';
      }
      return exponent.toString(16);
    });

    while (digits.length < 16) {
      digits.push('0');
    }
    hexInput.value = digits.join('');
  };

  const openBrowserAi = async () => {
    const url = 'https://2048-endgame-tablebase.netlify.app/';
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const onBeforeUnload = () => {
    client?.send('SAVE_GAME_STATE', {
      board_encoded: currentHex.value,
      score: score.value.current,
      best_score: score.value.best,
      special_tiles: specialTiles.value,
    });
  };

  onMounted(() => {
    window.addEventListener('keydown', handleKeydown);
    window.addEventListener('beforeunload', onBeforeUnload);
  });

  watch(
    activeRef,
    (isActive) => {
      if (isActive) {
        connect();
        if (aiEnabled.value) {
          aiPhase = null;
          requestAiStep();
        }
      } else {
        aiPhase = null;
        clearAiTimers();
      }
    },
    { immediate: true }
  );

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown);
    window.removeEventListener('beforeunload', onBeforeUnload);
    disconnect();
  });

  return {
    board,
    metadata,
    score,
    wsStatus,
    aiEnabled,
    difficulty,
    aiSpeed,
    hexInput,
    scoreAnimations,
    triggerAction,
    toggleAI,
    updateSettings,
    setBoard,
    writeCurrentBoardToHex,
    openBrowserAi,
  };
}
