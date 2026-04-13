import { onMounted, onUnmounted, ref, watch } from 'vue';

import { createWsClient } from '../../../services/ws/createWsClient';

export function useGamerSession(activeRef) {
  const VALID_DIRECTIONS = new Set(['left', 'right', 'up', 'down']);

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
    client?.send(action, payload);
  };

  const handleMessage = (data) => {
    if (data.action === 'UPDATE_STATE') {
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

      if (aiEnabled.value) {
        triggerAction('AI_STEP');
      }
      return;
    }

    if (data.action === 'DO_AI_MOVE_CMD') {
      const dir = String(data.data?.dir || '').toLowerCase();
      if (VALID_DIRECTIONS.has(dir)) {
        triggerAction('USER_MOVE', { dir });
      } else {
        aiEnabled.value = false;
      }
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
        triggerAction('GET_STATE');
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
    onBeforeUnload();
    client?.disconnect();
    client = null;
    wsStatus.value = 'disconnected';
  };

  const handleKeydown = (event) => {
    if (!activeRef?.value) return;
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
        aiEnabled.value = false;
      } else {
        triggerAction('AI_STEP');
      }
    } else if (event.code === 'Backspace' || event.code === 'Delete') {
      event.preventDefault();
      if (aiEnabled.value) {
        aiEnabled.value = false;
      } else {
        triggerAction('UNDO');
      }
    }
  };

  const toggleAI = () => {
    aiEnabled.value = !aiEnabled.value;
    if (aiEnabled.value) {
      triggerAction('AI_STEP');
    }
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
    try {
      if (window.pywebview?.api?.open_external_url) {
        await window.pywebview.api.open_external_url(url);
        return;
      }
    } catch (error) {
      console.error(error);
    }
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
      } else {
        disconnect();
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
