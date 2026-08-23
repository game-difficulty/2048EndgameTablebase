import { computed, onMounted, onUnmounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';

import { useAuthState } from '../../../services/auth/authState';
import { createLocalStorageStore } from '../../../services/storage/localStorageStore';
import { MinigameController } from '../engine/controller';
import { createEmptyMinigameMenu, createEmptyMinigameState } from '../model/minigameViewState';
import { submitMinigameScore } from '../services/minigameRankingClient';

const isTextEntryElement = (element) => {
  if (!(element instanceof HTMLElement)) {
    return false;
  }
  const tagName = element.tagName;
  return tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT' || element.isContentEditable;
};

const normalizeHudPanels = (hud, receivedAt = Date.now()) => {
  const customPanels = Array.isArray(hud?.customPanels) ? hud.customPanels : [];
  return {
    ...(hud || {}),
    customPanels: customPanels.map((panel) => {
      if (panel?.type !== 'countdown') return panel;
      return {
        ...panel,
        syncedAt: receivedAt,
      };
    }),
  };
};

const defaultMinigameState = () => ({
  difficulty: 1,
  summaries: {},
  activeGameSnapshots: {},
});

const minigameStore = createLocalStorageStore({
  key: 'minigames',
  version: 2,
  defaultValue: defaultMinigameState(),
  migrate(value) {
    const previous = value && typeof value === 'object' ? value : {};
    return {
      ...defaultMinigameState(),
      difficulty: Number(previous.difficulty) ? 1 : 0,
      summaries: previous.summaries && typeof previous.summaries === 'object' ? previous.summaries : {},
      activeGameSnapshots: previous.activeGameSnapshots && typeof previous.activeGameSnapshots === 'object'
        ? previous.activeGameSnapshots
        : {},
    };
  },
});

const normalizeStoredState = () => ({
  ...defaultMinigameState(),
  ...(minigameStore.read() || {}),
});

const snapshotKey = (gameId, difficulty) => `${gameId || ''}:${Number(difficulty) ? 1 : 0}`;

export function useMinigameSession(activeRef) {
  const { t } = useI18n();
  const { user: authUser } = useAuthState();

  const storedState = ref(normalizeStoredState());
  const menuData = ref({
    ...createEmptyMinigameMenu(),
    difficulty: Number(storedState.value.difficulty) ? 1 : 0,
  });
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

  let controller = null;
  let toastTimer = null;
  let inputLockTimer = null;
  const submittedFinals = new Set();

  const submitFinishedGame = async (state) => {
    const snapshot = state?.snapshot;
    const engine = snapshot?.engine;
    if (!authUser.value || !snapshot?.gameId || !engine?.isOver) return;
    const board = Array.isArray(state.board) ? state.board.map((value) => Number(value)) : [];
    const rows = Number(state.shape?.rows || 0);
    const cols = Number(state.shape?.cols || 0);
    if (!board.length || rows * cols !== board.length) return;
    const fingerprint = [
      authUser.value.id,
      snapshot.gameId,
      Number(snapshot.difficulty) ? 1 : 0,
      Number(state.score || 0),
      Number(engine.isPassed || 0),
      board.join(','),
    ].join(':');
    if (submittedFinals.has(fingerprint)) return;
    submittedFinals.add(fingerprint);
    try {
      const result = await submitMinigameScore({
        game_id: snapshot.gameId,
        difficulty: Number(snapshot.difficulty) ? 1 : 0,
        score: Math.max(0, Math.trunc(Number(state.score || 0))),
        trophy_tier: Math.max(0, Math.min(4, Math.trunc(Number(engine.isPassed || 0)))),
        highest_tile_exp: Math.max(0, Math.min(63, Math.trunc(Number(engine.highestTileExp || engine.maxNum || 0)))),
        final_board: board.map((value) => Math.trunc(value)),
        board_rows: rows,
        board_cols: cols,
      });
      window.dispatchEvent(new CustomEvent('minigame-score-updated', {
        detail: {
          gameId: snapshot.gameId,
          difficulty: Number(snapshot.difficulty) ? 1 : 0,
          result,
        },
      }));
    } catch (error) {
      if (error?.status !== 401) {
        submittedFinals.delete(fingerprint);
        console.warn('Minigame score submission failed.', error);
      }
    }
  };

  const hasActiveGame = computed(() => Boolean(gameState.value?.gameId));
  const currentView = computed(() => (hasActiveGame.value ? 'play' : 'menu'));
  const menuSections = computed(() => menuData.value.sections || []);
  const difficulty = computed(() => Number(menuData.value.difficulty ?? 1));
  const ensureController = () => {
    if (!controller) {
      controller = new MinigameController({
        difficulty: Number(storedState.value.difficulty) ? 1 : 0,
        summaries: storedState.value.summaries || {},
        snapshotKey,
      });
    }
    controller.setDifficulty(Number(storedState.value.difficulty) ? 1 : 0);
    controller.setSummaries(storedState.value.summaries || {});
    return controller;
  };

  const refreshMenu = () => {
    menuData.value = {
      ...createEmptyMinigameMenu(),
      ...ensureController().menuPayload(),
    };
  };

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

  const persistState = (updater) => {
    storedState.value = minigameStore.update((current) => {
      const base = {
        ...defaultMinigameState(),
        ...(current || {}),
      };
      return updater(base);
    });
    refreshMenu();
  };

  const handleStateData = (payload) => {
    const previousStatus = String(gameState.value?.status || '');
    const receivedAt = Date.now();
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
        ...normalizeHudPanels(payload?.hud || {}, receivedAt),
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
    void submitFinishedGame(nextState);
    const snapshot = nextState.snapshot;
    if (snapshot?.gameId) {
      const key = snapshotKey(snapshot.gameId, snapshot.difficulty ?? difficulty.value);
      const boardValues = Array.isArray(nextState.board) ? nextState.board : [];
      const highestExp = boardValues.reduce((maxExp, value) => {
        const numeric = Number(value || 0);
        if (numeric <= 0) return maxExp;
        return Math.max(maxExp, numeric);
      }, 0);
      persistState((current) => {
        const previousSummary = current.summaries?.[key] || {};
        const engineHighestExp = Math.max(
          Number(snapshot?.engine?.maxNum || 0),
          Number(snapshot?.engine?.highestTileExp || 0)
        );
        const summaryHighestExp = Math.max(
          Number(previousSummary.highestExp || 0),
          Number(highestExp || 0),
          engineHighestExp
        );
        const trophy = Math.max(
          Number(previousSummary.trophy || 0),
          Number(snapshot?.engine?.isPassed || 0)
        );
        return {
          ...current,
          difficulty: Number(snapshot.difficulty) ? 1 : 0,
          activeGameSnapshots: {
            ...(current.activeGameSnapshots || {}),
            [key]: snapshot,
          },
          summaries: {
            ...(current.summaries || {}),
            [key]: {
              bestScore: Math.max(Number(previousSummary.bestScore || 0), Number(nextState.best || 0)),
              highestTile: summaryHighestExp > 0 ? 2 ** summaryHighestExp : 0,
              highestExp: summaryHighestExp,
              trophy,
            },
          },
        };
      });
    }

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

  const runLocalAction = async (callback) => {
    try {
      const payload = await callback(ensureController());
      if (payload?.gameId) {
        handleStateData(payload);
      } else {
        refreshMenu();
      }
      return true;
    } catch (error) {
      console.error('Minigame local action failed.', error);
      showToast(error?.message || 'Minigame action failed.');
      return false;
    }
  };

  const setDifficulty = (value) => {
    const nextDifficulty = Number(value) ? 1 : 0;
    persistState((current) => ({
      ...current,
      difficulty: nextDifficulty,
    }));
    menuData.value = {
      ...menuData.value,
      difficulty: nextDifficulty,
    };
    ensureController().setDifficulty(nextDifficulty);
    refreshMenu();
  };

  const startGame = async (gameId) => {
    lastMenuFocusGameId.value = String(gameId || '');
    closeOverlay();
    const key = snapshotKey(gameId, difficulty.value);
    const snapshot = storedState.value.activeGameSnapshots?.[key] || null;
    await runLocalAction((localController) => localController.startGame(gameId, snapshot));
  };

  const backToMenu = () => {
    lastMenuFocusGameId.value = String(gameState.value?.gameId || lastMenuFocusGameId.value || '');
    closeOverlay();
    ensureController().backToMenu();
    gameState.value = createEmptyMinigameState();
    refreshMenu();
  };

  const newGame = async () => {
    closeOverlay();
    if (gameState.value?.gameId) {
      const key = snapshotKey(gameState.value.gameId, difficulty.value);
      persistState((current) => {
        const activeGameSnapshots = { ...(current.activeGameSnapshots || {}) };
        delete activeGameSnapshots[key];
        return {
          ...current,
          activeGameSnapshots,
        };
      });
    }
    await runLocalAction((localController) => localController.newGame());
  };

  const requestInfo = () => {
    runLocalAction((localController) => localController.requestInfo());
  };

  const triggerCustomAction = ({ key, phase }) => {
    if (!key) return;
    if (Date.now() < inputLockedUntil.value) return;
    runLocalAction((localController) => localController.triggerCustomAction({ key, phase }));
  };

  const usePowerup = (mode) => {
    if (Date.now() < inputLockedUntil.value) return;
    runLocalAction((localController) => localController.usePowerup(mode));
  };

  const cancelInteraction = () => {
    if (Date.now() < inputLockedUntil.value) return;
    runLocalAction((localController) => localController.cancelInteraction());
  };

  const handleBoardCellClick = (cell) => {
    if (!gameState.value?.interaction?.active) return;
    if (Date.now() < inputLockedUntil.value) return;
    runLocalAction((localController) => localController.targetAction(cell.index));
  };

  const move = (direction) => {
    if (currentView.value !== 'play') return;
    if (overlay.value.open && overlay.value.type === 'gameOver') return;
    if (gameState.value?.interaction?.active) return;
    if (Date.now() < inputLockedUntil.value) return;
    runLocalAction((localController) => localController.move(direction));
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
    refreshMenu();
    window.addEventListener('keydown', handleKeydown, true);
  });

  onUnmounted(() => {
    if (toastTimer) window.clearTimeout(toastTimer);
    if (inputLockTimer) window.clearTimeout(inputLockTimer);
    window.removeEventListener('keydown', handleKeydown, true);
    controller?.close();
    controller = null;
  });

  refreshMenu();

  return {
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
    move,
  };
}
