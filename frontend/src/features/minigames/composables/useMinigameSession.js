import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { createWsClient } from '../../../services/ws/createWsClient';
import { createEmptyMinigameMenu, createEmptyMinigameState } from '../model/minigameViewState';

const isTextEntryElement = (element) => {
  if (!(element instanceof HTMLElement)) {
    return false;
  }
  const tagName = element.tagName;
  return tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT' || element.isContentEditable;
};

export function useMinigameSession(activeRef) {
  const { t } = useI18n();

  const wsStatus = ref('disconnected');
  const menuData = ref(createEmptyMinigameMenu());
  const gameState = ref(createEmptyMinigameState());
  const lastMenuFocusGameId = ref('');
  const toastMessage = ref('');
  const inputLockedUntil = ref(0);
  const overlay = ref({
    open: false,
    type: 'info',
    title: '',
    message: '',
    level: '',
  });
  const pendingOverlay = ref(null);

  let client = null;
  let toastTimer = null;
  let inputLockTimer = null;

  const hasActiveGame = computed(() => Boolean(gameState.value?.gameId));
  const currentView = computed(() => (hasActiveGame.value ? 'play' : 'menu'));
  const menuSections = computed(() => menuData.value.sections || []);
  const difficulty = computed(() => Number(menuData.value.difficulty ?? 1));
  const connectionBadgeClass = computed(() => {
    if (wsStatus.value === 'connected') return 'badge-base badge-connection-connected';
    if (wsStatus.value === 'connecting') return 'badge-base badge-connection-pending';
    return 'badge-base badge-connection-disconnected';
  });

  const showToast = (message) => {
    toastMessage.value = message;
    if (toastTimer) window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      toastMessage.value = '';
      toastTimer = null;
    }, 2200);
  };

  const lockInputFor = (ms) => {
    const duration = Math.max(0, Number(ms || 0));
    if (!duration) return;
    const nextLockedUntil = Date.now() + duration;
    inputLockedUntil.value = Math.max(inputLockedUntil.value, nextLockedUntil);
    if (inputLockTimer) {
      window.clearTimeout(inputLockTimer);
    }
    inputLockTimer = window.setTimeout(() => {
      inputLockedUntil.value = 0;
      inputLockTimer = null;
    }, Math.max(0, inputLockedUntil.value - Date.now()));
  };

  const openOverlay = (payload) => {
    overlay.value = {
      open: true,
      type: payload.type || 'info',
      title: payload.title || '',
      message: payload.message || '',
      level: payload.level || '',
    };
  };

  const closeOverlay = () => {
    if (pendingOverlay.value) {
      const nextOverlay = pendingOverlay.value;
      pendingOverlay.value = null;
      overlay.value = {
        open: true,
        type: nextOverlay.type || 'info',
        title: nextOverlay.title || '',
        message: nextOverlay.message || '',
        level: nextOverlay.level || '',
      };
      return;
    }
    overlay.value = {
      open: false,
      type: 'info',
      title: '',
      message: '',
      level: '',
    };
  };

  const handleMenuData = (payload) => {
    menuData.value = {
      ...createEmptyMinigameMenu(),
      ...(payload || {}),
    };
  };

  const handleStateData = (payload) => {
    const previousStatus = String(gameState.value?.status || '');
    const nextState = {
      ...createEmptyMinigameState(),
      ...(payload || {}),
      shape: {
        ...createEmptyMinigameState().shape,
        ...(payload?.shape || {}),
      },
      view: {
        ...createEmptyMinigameState().view,
        ...(payload?.view || {}),
      },
      hud: {
        ...createEmptyMinigameState().hud,
        ...(payload?.hud || {}),
      },
      powerups: {
        ...createEmptyMinigameState().powerups,
        ...(payload?.powerups || {}),
      },
      interaction: {
        ...createEmptyMinigameState().interaction,
        ...(payload?.interaction || {}),
      },
      messages: payload?.messages || {},
    };
    gameState.value = nextState;

    const animation = gameState.value.animation || {};
    const followUp = animation.followUp || null;
    if (followUp) {
      if (followUp.lockInput !== false) {
        lockInputFor(Number(followUp.delayMs || 0) + Number(followUp.durationMs || 0));
      }
    } else {
      const effectDurations = []
        .concat(Array.isArray(animation.effects) ? animation.effects : [])
        .concat(Array.isArray(animation.pageEffects) ? animation.pageEffects : []);
      if (effectDurations.length) {
        const maxEffectDuration = effectDurations.reduce(
          (duration, effect) =>
            Math.max(
              duration,
              Number(effect?.delayMs || 0) + Number(effect?.durationMs || effect?.animDurationMs || 430)
            ),
          430
        );
        lockInputFor(maxEffectDuration);
      }
    }

    const messages = gameState.value.messages || {};
    const enteredGameOver = nextState.status === 'game_over' && previousStatus !== 'game_over';
    pendingOverlay.value = null;
    if (messages.toast) {
      showToast(messages.toast);
    }
    if (messages.infoDialog) {
      openOverlay({
        type: 'info',
        title: t('minigames.overlay.infoTitle'),
        message: String(messages.infoDialog),
      });
    } else if (messages.trophy) {
      if (messages.gameOver || enteredGameOver) {
        pendingOverlay.value = {
          type: 'gameOver',
          title: t('minigames.overlay.gameOverTitle'),
          message: String(messages.gameOver || t('minigames.overlay.gameOverMessage')),
          level: '',
        };
      }
      openOverlay({
        type: 'trophy',
        title: t('minigames.overlay.trophyTitle'),
        message: messages.trophy.message || '',
        level: messages.trophy.level || '',
      });
    } else if (messages.gameOver || enteredGameOver) {
      openOverlay({
        type: 'gameOver',
        title: t('minigames.overlay.gameOverTitle'),
        message: String(messages.gameOver || t('minigames.overlay.gameOverMessage')),
      });
    } else if (nextState.status !== 'game_over' && overlay.value.type === 'gameOver' && overlay.value.open) {
      closeOverlay();
    }
  };

  const ensureClient = () => {
    if (client) return client;
    client = createWsClient({
      clientId: 'minigames_main',
      onOpen: () => {
        wsStatus.value = 'connected';
        client?.send('MINIGAME_GET_MENU');
      },
      onMessage: (message) => {
        const action = message?.action || message?.type;
        if (action === 'MINIGAME_MENU_DATA') {
          handleMenuData(message.data || message.payload || {});
          return;
        }
        if (action === 'MINIGAME_STATE') {
          handleStateData(message.data || message.payload || {});
          return;
        }
        if (action === 'ERROR') {
          const errorMessage = message?.data?.message || message?.payload?.message || 'Unknown error';
          showToast(errorMessage);
        }
      },
      onClose: () => {
        wsStatus.value = 'disconnected';
      },
      onError: () => {
        wsStatus.value = 'disconnected';
      },
    });
    return client;
  };

  const connect = () => {
    const ws = ensureClient();
    if (wsStatus.value === 'connected' || wsStatus.value === 'connecting') return;
    wsStatus.value = 'connecting';
    ws.connect();
  };

  const sendAction = (action, data = {}) => {
    ensureClient()?.send(action, data);
  };

  const setDifficulty = (value) => {
    sendAction('MINIGAME_SET_DIFFICULTY', { difficulty: Number(value) ? 1 : 0 });
  };

  const startGame = (gameId) => {
    lastMenuFocusGameId.value = String(gameId || '');
    closeOverlay();
    sendAction('MINIGAME_START', { gameId });
  };

  const backToMenu = () => {
    lastMenuFocusGameId.value = String(gameState.value?.gameId || lastMenuFocusGameId.value || '');
    closeOverlay();
    sendAction('MINIGAME_BACK_TO_MENU');
    gameState.value = createEmptyMinigameState();
  };

  const newGame = () => {
    closeOverlay();
    sendAction('MINIGAME_NEW_GAME');
  };

  const requestInfo = () => {
    sendAction('MINIGAME_INFO');
  };

  const triggerCustomAction = ({ key, phase }) => {
    if (!key) return;
    if (Date.now() < inputLockedUntil.value) return;
    sendAction('MINIGAME_CUSTOM_ACTION', { key, phase: phase || 'trigger' });
  };

  const usePowerup = (mode) => {
    if (Date.now() < inputLockedUntil.value) return;
    sendAction('MINIGAME_USE_POWERUP', { mode });
  };

  const cancelInteraction = () => {
    if (Date.now() < inputLockedUntil.value) return;
    sendAction('MINIGAME_CANCEL_INTERACTION');
  };

  const handleBoardCellClick = (cell) => {
    if (!gameState.value?.interaction?.active) return;
    if (Date.now() < inputLockedUntil.value) return;
    sendAction('MINIGAME_TARGET_ACTION', {
      index: cell.index,
      row: cell.row,
      col: cell.col,
    });
  };

  const move = (direction) => {
    if (currentView.value !== 'play') return;
    if (overlay.value.open && overlay.value.type === 'gameOver') return;
    if (gameState.value?.interaction?.active) return;
    if (Date.now() < inputLockedUntil.value) return;
    sendAction('MINIGAME_MOVE', { direction });
  };

  const handleKeydown = (event) => {
    if (!activeRef.value || currentView.value !== 'play') return;
    if (isTextEntryElement(event.target)) return;
    const map = {
      ArrowUp: 'up',
      ArrowDown: 'down',
      ArrowLeft: 'left',
      ArrowRight: 'right',
      w: 'up',
      a: 'left',
      s: 'down',
      d: 'right',
      W: 'up',
      A: 'left',
      S: 'down',
      D: 'right',
    };
    const direction = map[event.key];
    if (!direction) return;
    event.preventDefault();
    move(direction);
  };

  onMounted(() => {
    if (activeRef.value) {
      connect();
    }
    window.addEventListener('keydown', handleKeydown, true);
  });

  onUnmounted(() => {
    if (toastTimer) window.clearTimeout(toastTimer);
    if (inputLockTimer) window.clearTimeout(inputLockTimer);
    window.removeEventListener('keydown', handleKeydown, true);
    client?.send('MINIGAME_CLOSE');
    client?.disconnect();
    client = null;
  });

  watch(
    activeRef,
    (active) => {
      if (active) {
        connect();
      }
    },
    { immediate: true }
  );

  return {
    wsStatus,
    connectionBadgeClass,
    menuSections,
    difficulty,
    currentView,
    gameState,
    lastMenuFocusGameId,
    toastMessage,
    overlay,
    closeOverlay,
    setDifficulty,
    startGame,
    backToMenu,
    newGame,
    requestInfo,
    triggerCustomAction,
    usePowerup,
    cancelInteraction,
    handleBoardCellClick,
  };
}
